"""
Re-evaluate FullModel checkpoint with OLD metrics (best-of-100, median).
Matches the original 34.92 evaluation method.
Usage: python scripts/reeval_fullmodel_oldmetric.py
"""
import sys, torch
sys.path.insert(0, '.')
from src.baselines.baseline_models import FlowChainBase
from data.dataset import TrajectoryDataset, trajectory_collate_fn
from torch.utils.data import DataLoader, Subset

device = 'cuda'
NORM = torch.tensor([3840.0, 2160.0])
num_samples = 100

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
ckpt = torch.load('checkpoints/ablation_fullmodel.pt', map_location=device)
flow_state = {}
for k, v in ckpt['model_state'].items():
    if k.startswith('flow_chain.model.'):
        flow_state[k.replace('flow_chain.', 'predictor.', 1)] = v
model.load_state_dict(flow_state, strict=False)
model.eval()

norm = NORM.to(device)
all_ade, all_fde = [], []

for batch in test_loader:
    obs = batch["obs_trajectory"].to(device) / norm
    target = batch["target_trajectory"].to(device) / norm
    pred = model(obs_trajectory=obs, num_samples=num_samples)

    if "samples" in pred and pred["samples"].dim() >= 4:
        samples = pred["samples"].clamp(0.0, 1.0)
        samples_px = samples * norm
        target_px = target * norm
        diff = samples_px - target_px.unsqueeze(0)
        l2 = torch.sqrt((diff ** 2).sum(dim=-1))
        ade_per = l2.mean(dim=-1)
        fde_per = l2[:, :, -1]
        best = ade_per.argmin(dim=0)
        all_ade.append(ade_per.gather(0, best.unsqueeze(0)).squeeze(0).cpu())
        all_fde.append(fde_per.gather(0, best.unsqueeze(0)).squeeze(0).cpu())
    elif "mean" in pred:
        diff = pred["mean"] * norm - target * norm
        l2 = torch.sqrt((diff ** 2).sum(dim=-1))
        all_ade.append(l2.mean(dim=-1).cpu())
        all_fde.append(l2[:, -1].cpu())

ade = torch.cat(all_ade)
fde = torch.cat(all_fde)
print(f'FullModel (OLD metric): ADE={ade.median():.2f}, FDE={fde.median():.2f}')
