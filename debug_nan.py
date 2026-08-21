"""Diagnose NaN in FOMAML v2 eval flow log_prob."""
import torch, sys, os, json, yaml
sys.path.insert(0, os.getcwd())

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel
from torch.utils.data import DataLoader

DEVICE = 'cuda'
torch.backends.cudnn.benchmark = False  # deterministic for debugging

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Build model, load all checkpoints
# ---------------------------------------------------------------------------
print("Loading model...")
model = TrafficPerceptionModel(config, stage=2).to(DEVICE).eval()

# 1. Perception
p_ckpt = torch.load('checkpoints/stage2_best.pt', map_location=DEVICE, weights_only=False)
p_sd = p_ckpt.get('model_state') or p_ckpt.get('model') or p_ckpt
p_loaded = 0
for k, v in p_sd.items():
    if k.startswith('flow_chain.'):
        continue
    try:
        target = model
        parts = k.split('.')
        for part in parts[:-1]:
            target = getattr(target, part)
        param = getattr(target, parts[-1])
        if isinstance(param, torch.nn.Parameter):
            param.data.copy_(v)
            p_loaded += 1
    except Exception:
        pass
print(f"  Perception: {p_loaded} params loaded")

# 2. FlowChain (frozen)
f_ckpt = torch.load('checkpoints/flowchain_best_finetuned.pt', map_location=DEVICE, weights_only=False)
f_sd = f_ckpt.get('model') or f_ckpt.get('model_state') or f_ckpt
if any(k.startswith('flow_chain.') for k in f_sd.keys()):
    f_sd = {k.replace('flow_chain.', ''): v for k, v in f_sd.items()}
if any(k.startswith('predictor.') for k in f_sd.keys()):
    f_sd = {k.replace('predictor.', ''): v for k, v in f_sd.items()}
fc_missing, fc_unexpected = model.flow_chain.load_state_dict(f_sd, strict=False)
print(f"  FlowChain: {len(fc_missing)} missing, {len(fc_unexpected)} unexpected")

# 3. FOMAML v2 trainable
fomaml = torch.load('checkpoints/fomaml_v2/best_fomaml.pt', map_location=DEVICE, weights_only=False)
trainable = fomaml['trainable_params']
print(f"  FOMAML v2: epoch={fomaml['epoch']}, {len(trainable)} trainable param keys")

# Apply trainable params
fc = model.flow_chain.model
applied = 0
for layer_idx in range(3):
    layer = fc.transformer.encoder.layers[layer_idx]
    sa = layer.self_attn
    for name in ['in_proj_weight', 'in_proj_bias']:
        k = f'enc_attn.L{layer_idx}.{name}'
        if k in trainable:
            setattr(sa, name, torch.nn.Parameter(trainable[k].to(DEVICE)))
            applied += 1
    for pname in ['weight', 'bias']:
        k = f'enc_attn.L{layer_idx}.out_proj.{pname}'
        if k in trainable:
            setattr(sa.out_proj, pname, torch.nn.Parameter(trainable[k].to(DEVICE)))
            applied += 1
    for norm_name in ['norm1', 'norm2']:
        for pname in ['weight', 'bias']:
            k = f'enc_norm.L{layer_idx}.{norm_name}.{pname}'
            if k in trainable:
                setattr(getattr(layer, norm_name), pname,
                        torch.nn.Parameter(trainable[k].to(DEVICE)))
                applied += 1
for pname in ['weight', 'bias']:
    k = f'enc_norm.final.{pname}'
    if k in trainable:
        setattr(fc.transformer.encoder.norm, pname,
                torch.nn.Parameter(trainable[k].to(DEVICE)))
        applied += 1
for blk_idx in range(4):
    for pname in ['log_gamma', 'beta']:
        k = f'flow.net.{2*blk_idx+1}.{pname}'
        if k in trainable:
            target = fc.flow.net[2*blk_idx+1]
            setattr(target, pname, torch.nn.Parameter(trainable[k].to(DEVICE)))
            applied += 1
print(f"  Applied: {applied}/{len(trainable)} trainable params")

# ---------------------------------------------------------------------------
# Also test via the full log_prob path (like the eval script does)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TESTING FULL log_prob PATH (model.flow_chain.log_prob)")
print("="*60)

with open('data/domains/domain_labels_int.json') as f:
    domain_map = json.load(f)

ds = TrajectoryDataset(
    data_dir='data/processed/trajectories', label_dir='labels',
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='trajectory_only', max_samples=200,
    domain_label_map=domain_map,
)
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=trajectory_collate_fn)

all_ok = True
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 10:
        break

    obs = batch['obs_trajectory'].to(DEVICE)
    target = batch['target_trajectory'].to(DEVICE)
    B = obs.shape[0]
    zero_cond = torch.zeros(B, model.condition_dim, device=DEVICE)

    try:
        log_prob = model.flow_chain.log_prob(
            obs_trajectory=obs, target=target, perception_c=zero_cond)
        nll = -log_prob.mean().item()
        has_nan = torch.isnan(log_prob).any().item()
        print(f"  Batch {batch_idx}: log_prob OK, nll={nll:.4f}, nan={has_nan}")
        if has_nan:
            all_ok = False
    except Exception as e:
        print(f"  Batch {batch_idx}: FAILED — {e}")
        all_ok = False
        break

# ---------------------------------------------------------------------------
# Also test forward (sampling path)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("TESTING FORWARD PATH (model.flow_chain.forward)")
print("="*60)

for batch_idx, batch in enumerate(loader):
    if batch_idx >= 5:
        break

    obs = batch['obs_trajectory'].to(DEVICE)
    target = batch['target_trajectory'].to(DEVICE)
    B = obs.shape[0]
    zero_cond = torch.zeros(B, model.condition_dim, device=DEVICE)

    try:
        pred = model.flow_chain(
            obs_trajectory=obs, perception_c=zero_cond, num_samples=2)
        mean_pred = pred["mean"]
        has_nan = torch.isnan(mean_pred).any().item()
        print(f"  Batch {batch_idx}: forward OK, mean shape={list(mean_pred.shape)}, nan={has_nan}")
        if has_nan:
            all_ok = False
    except Exception as e:
        print(f"  Batch {batch_idx}: FORWARD FAILED — {e}")
        all_ok = False

# ---------------------------------------------------------------------------
# Flow BN parameter check
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("FLOW BATCHNORM PARAMETERS")
print("="*60)
flow = model.flow_chain.model.flow
for i, module in enumerate(flow.net):
    if hasattr(module, 'log_gamma'):
        lg = module.log_gamma.data
        bt = module.beta.data
        lg_str = f"[{lg[0].item():.6f}, {lg[1].item():.6f}]"
        bt_str = f"[{bt[0].item():.6f}, {bt[1].item():.6f}]"
        print(f"  net[{i}] BatchNorm: log_gamma={lg_str}, beta={bt_str}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
if all_ok:
    print("ALL CHECKS PASSED — model works correctly in inference mode.")
    print()
    print("The NaN in eval_fomaml.py likely comes from INNER-LOOP ADAPTATION:")
    print("  5 steps of SGD (lr=0.01) on a small support set may push")
    print("  enc_attn/enc_norm/flow_bn params into NaN regime.")
    print()
    print("Suggestions:")
    print("  1. Reduce inner_lr: --inner-lr 0.001")
    print("  2. Reduce inner_steps: --inner-steps 2 or 3")
    print("  3. Add gradient clipping in compute_loss")
    print("  4. Run eval with --inner-steps 0 first to confirm base model works")
else:
    print("NaN DETECTED — see above for which step fails")
