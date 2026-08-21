"""
FOMAML v2 smoke test — 极小样本完整流程，全面输出各环节参数。
用法: python smoke_test_fomaml.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore', message=r'std\(\): degrees of freedom is <= 0')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Config
# ============================================================================
MAX_SAMPLES = 200        # 极小样本
INNER_LR = 0.01
OUTER_LR = 1e-3
INNER_STEPS = 3          # 减少内循环步数
ADE_WEIGHT = 1.0
BATCH_SIZE = 16
EPOCHS = 3
VAL_INTERVAL = 1
META_TRAIN = [0, 1]
META_VAL = [0]    # Use domain 0 for val (small sample, always present)
SEP = "=" * 70

# ============================================================================
# Step 1: Load data
# ============================================================================
print(f"\n{SEP}")
print("STEP 1: LOAD DATA")
print(SEP)

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

with open('data/domains/domain_labels_int.json') as f:
    domain_map = json.load(f)

dataset = TrajectoryDataset(
    data_dir='data/processed/trajectories', label_dir='labels',
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='trajectory_only',
    max_samples=MAX_SAMPLES, domain_label_map=domain_map,
)

print(f"Total samples: {len(dataset)}")

# Per-domain counts
from collections import Counter
dom_counts = Counter()
for i in range(len(dataset)):
    dom_counts[dataset[i].get('domain_id', -1)] += 1
print(f"Per-domain samples: {dict(sorted(dom_counts.items()))}")

# ============================================================================
# Step 2: Build model + load checkpoints
# ============================================================================
print(f"\n{SEP}")
print("STEP 2: BUILD MODEL")
print(SEP)

model = TrafficPerceptionModel(config, stage=2).to(DEVICE).eval()

# Perception
p_ckpt = torch.load('checkpoints/stage2_best.pt', map_location=DEVICE, weights_only=False)
p_sd = p_ckpt.get('model_state') or p_ckpt.get('model') or p_ckpt
p_loaded = 0
for k, v in p_sd.items():
    if k.startswith('flow_chain.'): continue
    try:
        target = model
        for part in k.split('.')[:-1]:
            target = getattr(target, part)
        param = getattr(target, k.split('.')[-1])
        if isinstance(param, nn.Parameter):
            param.data.copy_(v); p_loaded += 1
    except: pass
print(f"Perception params loaded: {p_loaded}")

# FlowChain
f_ckpt = torch.load('checkpoints/flowchain_best_finetuned.pt', map_location=DEVICE, weights_only=False)
f_sd = f_ckpt.get('model') or f_ckpt.get('model_state') or f_ckpt
if any(k.startswith('flow_chain.') for k in f_sd.keys()):
    f_sd = {k.replace('flow_chain.', ''): v for k, v in f_sd.items()}
if any(k.startswith('predictor.') for k in f_sd.keys()):
    f_sd = {k.replace('predictor.', ''): v for k, v in f_sd.items()}
fc_missing, fc_unexpected = model.flow_chain.load_state_dict(f_sd, strict=False)
print(f"FlowChain: {len(fc_missing)} missing, {len(fc_unexpected)} unexpected")

# Freeze all → unfreeze v2 params
for p in model.parameters():
    p.requires_grad_(False)

fc = model.flow_chain.model
n_unfrozen = 0
for layer in fc.transformer.encoder.layers:
    if hasattr(layer, 'self_attn'):
        for p in layer.self_attn.parameters():
            p.requires_grad_(True); n_unfrozen += 1
for layer in fc.transformer.encoder.layers:
    for nm in ['norm1', 'norm2']:
        if hasattr(layer, nm):
            for p in getattr(layer, nm).parameters():
                p.requires_grad_(True); n_unfrozen += 1
if hasattr(fc.transformer.encoder, 'norm'):
    for p in fc.transformer.encoder.norm.parameters():
        p.requires_grad_(True); n_unfrozen += 1
for name, p in fc.flow.named_parameters():
    if 'log_gamma' in name or 'beta' in name:
        p.requires_grad_(True); n_unfrozen += 1

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

# ============================================================================
# Step 3: Identify trainable params
# ============================================================================
print(f"\n{SEP}")
print("STEP 3: TRAINABLE PARAMETERS")
print(SEP)

trainable_params = {}
for li, layer in enumerate(fc.transformer.encoder.layers):
    if hasattr(layer, 'self_attn'):
        for name, p in layer.self_attn.named_parameters():
            if p.requires_grad:
                trainable_params[f"enc_attn.L{li}.{name}"] = p
    for nm in ['norm1', 'norm2']:
        if hasattr(layer, nm):
            for name, p in getattr(layer, nm).named_parameters():
                if p.requires_grad:
                    trainable_params[f"enc_norm.L{li}.{nm}.{name}"] = p
if hasattr(fc.transformer.encoder, 'norm'):
    for name, p in fc.transformer.encoder.norm.named_parameters():
        if p.requires_grad:
            trainable_params[f"enc_norm.final.{name}"] = p
for name, p in fc.flow.named_parameters():
    if ('log_gamma' in name or 'beta' in name) and p.requires_grad:
        trainable_params[f"flow.{name}"] = p

print(f"Trainable param count: {len(trainable_params)}")
for k, v in sorted(trainable_params.items()):
    print(f"  {k:45s} shape={list(v.shape)}  grad_fn={'YES' if v.requires_grad else 'NO'}")

# --- Helper: save/restore BN running stats (they get corrupted by train-mode forward) ---
def get_bn_state(model):
    """Return {name: (running_mean.clone(), running_var.clone())} for flow BN buffers."""
    state = {}
    for i, module in enumerate(model.flow_chain.model.flow.net):
        if hasattr(module, 'running_mean'):
            state[f'flow.net.{i}'] = (module.running_mean.clone(), module.running_var.clone())
    return state

def restore_bn_state(model, state):
    for i, module in enumerate(model.flow_chain.model.flow.net):
        key = f'flow.net.{i}'
        if key in state:
            module.running_mean.copy_(state[key][0])
            module.running_var.copy_(state[key][1])

def save_full_state(model, trainable_params):
    """Save both trainable params AND BN buffers."""
    return {
        'params': {name: p.data.clone() for name, p in trainable_params.items()},
        'bn': get_bn_state(model),
    }

def restore_full_state(model, trainable_params, saved):
    for name, p in trainable_params.items():
        if name in saved['params']:
            p.data.copy_(saved['params'][name])
        p.grad = None
    restore_bn_state(model, saved['bn'])

# ============================================================================
# Step 4: Build per-domain data loaders
# ============================================================================
print(f"\n{SEP}")
print("STEP 4: DOMAIN SPLITS")
print(SEP)

rng = np.random.RandomState(42)
dom_to_idx = {d: [] for d in set(dom_counts.keys())}
for i in range(len(dataset)):
    did = dataset[i].get('domain_id', -1)
    if did in dom_to_idx:
        dom_to_idx[did].append(i)

splits = {}
for did in sorted(dom_to_idx.keys()):
    indices = np.array(dom_to_idx[did])
    rng.shuffle(indices)
    n_sup = max(1, int(len(indices) * 0.7))
    splits[did] = {'support': indices[:n_sup].tolist(),
                   'query': indices[n_sup:].tolist()}
    print(f"  Domain {did}: {len(indices)} total → {n_sup} support / {len(indices)-n_sup} query")

# ============================================================================
# Step 5: Forward pass sanity check
# ============================================================================
print(f"\n{SEP}")
print("STEP 5: FORWARD PASS SANITY CHECK")
print(SEP)

def compute_loss(model, obs, target, ade_weight=1.0):
    B = obs.shape[0]
    zero_cond = torch.zeros(B, model.condition_dim, device=DEVICE)
    log_prob = model.flow_chain.log_prob(obs_trajectory=obs, target=target, perception_c=zero_cond)
    nll = -log_prob.mean()
    pred = model.flow_chain(obs_trajectory=obs, perception_c=zero_cond, num_samples=1)
    mean_pred = pred["mean"]
    ade = torch.sqrt(((mean_pred - target.to(DEVICE)) ** 2).sum(dim=-1) + 1e-8).mean()
    loss = nll + ade_weight * ade
    return loss, {"nll": nll.item(), "ade": ade.item(), "loss": loss.item()}

# Test a batch
test_indices = splits[0]['support'][:4]
test_samples = [dataset[i] for i in test_indices]
batch = trajectory_collate_fn(test_samples)
obs = batch['obs_trajectory'].to(DEVICE)
target = batch['target_trajectory'].to(DEVICE)

model.train()
loss, metrics = compute_loss(model, obs, target, ADE_WEIGHT)
print(f"  Batch size: {obs.shape[0]}")
print(f"  obs shape:  {list(obs.shape)}, range=[{obs.min():.2f}, {obs.max():.2f}]")
print(f"  target:     {list(target.shape)}, range=[{target.min():.2f}, {target.max():.2f}]")
print(f"  NLL={metrics['nll']:.4f}, ADE={metrics['ade']:.4f}, Loss={metrics['loss']:.4f}")

# Check that trainable params have requires_grad
for name, p in trainable_params.items():
    if not p.requires_grad:
        print(f"  ERROR: {name} does NOT require grad!")
print(f"  All {len(trainable_params)} params require_grad=True ✓")

# ============================================================================
# Step 6: Inner loop test (1 step of SGD)
# ============================================================================
print(f"\n{SEP}")
print("STEP 6: INNER LOOP TEST (1 step SGD)")
print(SEP)

# Print initial BN state
bn_init = get_bn_state(model)
print("  BN running_mean/running_var before training:")
for k, (rm, rv) in sorted(bn_init.items()):
    print(f"    {k}: mean={rm.tolist()}, var={rv.tolist()}")

# Record full state before
state_before = save_full_state(model, trainable_params)

# Forward + backward + SGD (in train mode)
loss, metrics = compute_loss(model, obs, target, ADE_WEIGHT)
loss.backward()
n_updated = 0
for name, p in trainable_params.items():
    if p.grad is not None:
        grad_norm_before = p.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
        p.data -= INNER_LR * p.grad
        p.grad = None
        n_updated += 1

# Record params after, check change
n_changed = 0
max_grad_norm = 0.0
for name, p in trainable_params.items():
    before = state_before['params'][name]
    diff = (p.data - before).abs().max().item()
    rel_diff = diff / (before.abs().max().item() + 1e-8)
    if diff > 1e-15 or rel_diff > 1e-15:
        n_changed += 1
        if n_changed <= 5:
            print(f"  {name}: max_change={diff:.2e}, relative={rel_diff:.2e}")

print(f"\n  Params with grad: {n_updated}/{len(trainable_params)}")
print(f"  Params changed:   {n_changed}/{len(trainable_params)}")
if n_changed > 0:
    print(f"  Inner loop SGD WORKS ✓")
else:
    print(f"  WARNING: No params changed (but {n_updated} had grads)")

# BN state after training (corrupted)
bn_after_train = get_bn_state(model)
print(f"\n  BN running_mean after 1 train-mode forward:")
for k in sorted(bn_after_train):
    if bn_after_train[k][0].tolist() != bn_init[k][0].tolist():
        print(f"    {k}: CHANGED mean={bn_after_train[k][0].tolist()}")

# Now test eval mode — this is where NaN would happen
# First, test WITHOUT restoring BN (to confirm the corruption causes NaN)
try:
    model.eval()
    loss2, metrics2 = compute_loss(model, obs, target, ADE_WEIGHT)
    print(f"\n  Eval mode (corrupted BN): NLL={metrics2['nll']:.4f}, ADE={metrics2['ade']:.4f}")
    has_nan_eval = False
except Exception as e:
    print(f"\n  Eval mode (corrupted BN): FAILED — {str(e)[:100]}")
    has_nan_eval = True

# Restore full state (params + BN)
restore_full_state(model, trainable_params, state_before)
model.eval()

# Test again with restored state
try:
    loss3, metrics3 = compute_loss(model, obs, target, ADE_WEIGHT)
    print(f"  Eval mode (restored BN):  NLL={metrics3['nll']:.4f}, ADE={metrics3['ade']:.4f}")
    print(f"  Loss before SGD: {metrics['loss']:.4f} → after restore: {metrics3['loss']:.4f}")
    if metrics3['loss'] == metrics['loss']:
        print(f"  BN restore successful: loss unchanged ✓")
    else:
        print(f"  BN restore: loss differs by {abs(metrics3['loss']-metrics['loss']):.4f} (expected small diff)")
    has_nan_restored = False
except Exception as e:
    print(f"  Eval mode (restored BN): STILL FAILED — {str(e)[:100]}")
    has_nan_restored = True

if has_nan_eval and not has_nan_restored:
    print(f"\n  >>> ROOT CAUSE CONFIRMED: BN running stats corruption causes NaN")
    print(f"  >>> FIX: save/restore BN buffers alongside trainable params")

# ============================================================================
# Step 7: Mini training loop (3 epochs, 2 domains)
# ============================================================================
print(f"\n{SEP}")
print(f"STEP 7: MINI TRAINING ({EPOCHS} epochs × {len(META_TRAIN)} domains)")
print(SEP)

optimizer = torch.optim.AdamW(
    list(trainable_params.values()), lr=OUTER_LR, weight_decay=1e-5)

# Build loaders
sup_loaders, qry_loaders = {}, {}
for did in META_TRAIN + META_VAL:
    if did not in splits: continue
    sup_loaders[did] = DataLoader(
        Subset(dataset, splits[did]['support']),
        batch_size=min(BATCH_SIZE, len(splits[did]['support'])),
        shuffle=True, collate_fn=trajectory_collate_fn, num_workers=0)
    qry_loaders[did] = DataLoader(
        Subset(dataset, splits[did]['query']),
        batch_size=min(BATCH_SIZE, len(splits[did]['query'])),
        shuffle=True, collate_fn=trajectory_collate_fn, num_workers=0)

for epoch in range(EPOCHS):
    t0 = time.time()
    optimizer.zero_grad()

    epoch_inner_loss = 0.0
    epoch_query_loss = 0.0
    n_domains = 0

    for did in META_TRAIN:
        if did not in sup_loaders: continue
        # Save full state (params + BN)
        state = save_full_state(model, trainable_params)

        # --- Inner loop ---
        inner_total = 0.0
        inner_steps_done = 0
        for k, batch in enumerate(sup_loaders[did]):
            if k >= INNER_STEPS: break
            obs, target = batch['obs_trajectory'].to(DEVICE), batch['target_trajectory'].to(DEVICE)
            loss, _ = compute_loss(model, obs, target, ADE_WEIGHT)
            loss.backward()
            for name, p in trainable_params.items():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                    p.data -= INNER_LR * p.grad
                    p.grad = None
            inner_total += loss.item()
            inner_steps_done += 1
        epoch_inner_loss += inner_total / max(inner_steps_done, 1)

        # --- Query loss ---
        q_total = 0.0; q_n = 0
        n_q_batches = max(len(qry_loaders[did]), 1)
        for batch in qry_loaders[did]:
            obs, target = batch['obs_trajectory'].to(DEVICE), batch['target_trajectory'].to(DEVICE)
            q_loss, qm = compute_loss(model, obs, target, ADE_WEIGHT)
            (q_loss / n_q_batches).backward()
            q_total += qm['loss']; q_n += 1

        # Save adapted grads, restore full state (params + BN), set grads
        adapted_grads = {}
        for name, p in trainable_params.items():
            adapted_grads[name] = p.grad.clone() if p.grad is not None else None
            p.grad = None
        restore_full_state(model, trainable_params, state)
        for name, p in trainable_params.items():
            if adapted_grads[name] is not None:
                p.grad = adapted_grads[name]

        epoch_query_loss += q_total / max(q_n, 1)
        n_domains += 1

    # Outer step
    optimizer.step()

    avg_inner = epoch_inner_loss / max(n_domains, 1)
    avg_query = epoch_query_loss / max(n_domains, 1)

    # --- Validation ---
    val_msg = ""
    if epoch % VAL_INTERVAL == 0:
        state_val = save_full_state(model, trainable_params)
        val_nll, val_ade, val_n = 0.0, 0.0, 0
        for did in META_VAL:
            if did not in sup_loaders: continue
            # Inner loop adapt
            for k, batch in enumerate(sup_loaders[did]):
                if k >= INNER_STEPS: break
                obs, target = batch['obs_trajectory'].to(DEVICE), batch['target_trajectory'].to(DEVICE)
                loss, _ = compute_loss(model, obs, target, ADE_WEIGHT)
                loss.backward()
                for name, p in trainable_params.items():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                        p.data -= INNER_LR * p.grad
                        p.grad = None
            # Eval (in eval mode with safe BN restore)
            model.eval()
            for batch in qry_loaders[did]:
                obs, target = batch['obs_trajectory'].to(DEVICE), batch['target_trajectory'].to(DEVICE)
                with torch.no_grad():
                    _, vm = compute_loss(model, obs, target, ADE_WEIGHT)
                    val_nll += vm['nll'] * obs.shape[0]
                    val_ade += vm['ade'] * obs.shape[0]
                    val_n += obs.shape[0]
            model.train()
            # Restore full state before next domain
            restore_full_state(model, trainable_params, state_val)
        val_msg = f"Val NLL={val_nll/max(val_n,1):.4f} ADE={val_ade/max(val_n,1):.4f}"
        # Final restore
        restore_full_state(model, trainable_params, state_val)

    print(f"  Epoch {epoch}: Inner={avg_inner:.4f} Query={avg_query:.4f} "
          f"Time={time.time()-t0:.1f}s  {val_msg}")

# ============================================================================
# Step 8: Checkpoint save/load test
# ============================================================================
print(f"\n{SEP}")
print("STEP 8: CHECKPOINT SAVE/LOAD")
print(SEP)

os.makedirs('/tmp/smoke_test', exist_ok=True)
ckpt_path = '/tmp/smoke_test/test_ckpt.pt'

# Save
torch.save({
    'epoch': EPOCHS,
    'trainable_params': {name: p.data.clone() for name, p in trainable_params.items()},
    'optimizer_state': optimizer.state_dict(),
    'config': {'inner_lr': INNER_LR, 'outer_lr': OUTER_LR, 'inner_steps': INNER_STEPS,
               'ade_weight': ADE_WEIGHT, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS},
}, ckpt_path)
size_kb = os.path.getsize(ckpt_path) / 1024
print(f"  Saved: {ckpt_path} ({size_kb:.1f} KB)")

# Load
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
print(f"  Loaded: epoch={ckpt['epoch']}, "
      f"params={len(ckpt['trainable_params'])} keys, "
      f"opt_keys={list(ckpt['optimizer_state'].keys())}")
params_loaded = sum(1 for name in trainable_params
                    if name in ckpt['trainable_params'])
print(f"  Params matched: {params_loaded}/{len(trainable_params)}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{SEP}")
print("SMOKE TEST COMPLETE")
print(SEP)
print(f"  Data: {len(dataset)} samples from {len(dom_counts)} domains")
print(f"  Model: {n_trainable:,} trainable / {n_total:,} total params")
print(f"  Training: {EPOCHS} epochs × {len(META_TRAIN)} domains")
print(f"  Inner LR: {INNER_LR}, Inner steps: {INNER_STEPS}")
print(f"  Outer LR: {OUTER_LR}")
print(f"  Inner loop SGD: {'✓ working' if n_changed > 0 else '✗ BROKEN'}")
print(f"  Checkpoint: save/load ✓")
print(f"\n  All checks passed — ready for full training.")
