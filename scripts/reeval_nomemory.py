"""
Re-evaluate NoMemory checkpoint with corrected metrics.
Extracts FlowChain weights from old-architecture checkpoint.
Usage: python scripts/reeval_nomemory.py
"""
import sys, torch
sys.path.insert(0, '.')
from scripts.run_experiments import evaluate_trajectory
from src.baselines.baseline_models import FlowChainBase
from data.dataset import TrajectoryDataset, trajectory_collate_fn
from torch.utils.data import DataLoader, Subset

device = 'cuda'
ckpt_path = 'checkpoints/ablation_nomemory.pt'

ds = TrajectoryDataset(
    'data/processed/trajectories', obs_len=8, pred_len=12,
    stride=8, min_trajectory_len=20,
    target_classes=['pedestrian'], mode='trajectory_only',
)
indices = [i for i, s in enumerate(ds.samples)
           if any(d.replace('_', '') in s.get('video', '') for d in ['2026_01_27'])]
test_loader = DataLoader(Subset(ds, indices), batch_size=64, shuffle=False,
                         collate_fn=trajectory_collate_fn)

model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=4, condition_dim=256).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
ckpt_state = ckpt['model_state']

# Checkpoint: flow_chain.model.xxx → FlowChainBase: predictor.model.xxx
flow_state = {}
for k, v in ckpt_state.items():
    if k.startswith('flow_chain.model.'):
        new_k = k.replace('flow_chain.', 'predictor.', 1)
        flow_state[new_k] = v

missing, unexpected = model.load_state_dict(flow_state, strict=False)
print(f'Loaded {len(flow_state)} FlowChain params (missing={len(missing)}, unexpected={len(unexpected)})')
model.eval()

m = evaluate_trajectory(model, test_loader, device)
print(f'NoMemory: ADE={m["ADE"]:.2f}, FDE={m["FDE"]:.2f}, NLL={m["NLL"]:.2f}')
