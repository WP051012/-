"""Quick eval of FlowChain checkpoint"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.official_wrappers import FlowChainBase
from scripts.run_experiments import load_split_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
PX_PER_M = 50.0

# Load test set
_, _, test_set, _, _, _ = load_split_datasets(
    "data/processed/trajectories/", quick=False
)

loader = DataLoader(
    test_set, batch_size=64, shuffle=False, collate_fn=trajectory_collate_fn
)

# Load trained model
model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(device)
model.load_state_dict(torch.load("checkpoints/flowchain_best.pt", map_location=device))
model.eval()
print(f"Model loaded. Test batches: {len(loader)}")

all_ade, all_fde, all_nll = [], [], []
norm_tensor = NORM.to(device)

with torch.no_grad():
    for i, batch in enumerate(loader):
        obs = batch["obs_trajectory"].to(device) / norm_tensor
        target = batch["target_trajectory"].to(device) / norm_tensor
        pred = model(obs_trajectory=obs, num_samples=100)

        if "samples" in pred and pred["samples"].dim() >= 4:
            samples = pred["samples"].clamp(0.0, 1.0)
            N = samples.shape[0]
            se = ((samples.unsqueeze(2) - target.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1).sqrt()
            mse = se.mean(dim=-1)
            best_idx = mse.argmin(dim=0)
            B, T = target.shape[:2]
            idx_t = best_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(1, B, T, 2)
            best_samples = torch.gather(samples, 0, idx_t).squeeze(0)
            ade = (best_samples - target).norm(dim=-1).mean()
            fde = (best_samples[:, -1] - target[:, -1]).norm(dim=-1).mean()
        else:
            best_samples = pred.get("mean", pred.get("best_samples", obs))
            if best_samples.dim() == 3 and best_samples.shape[0] > 1:
                best_samples = best_samples.mean(0, keepdim=True)
            ade = (best_samples - target).norm(dim=-1).mean()
            fde = (best_samples[:, -1] - target[:, -1]).norm(dim=-1).mean()

        all_ade.append(ade.item())
        all_fde.append(fde.item())

        # NLL
        if hasattr(model, "log_prob"):
            try:
                nll = -model.log_prob(
                    obs_trajectory=obs, target=target,
                    perception_c=torch.zeros(obs.shape[0], 256, device=device)
                ).mean()
                all_nll.append(nll.item())
            except Exception:
                pass

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(loader)} batches done")

avg_ade = np.mean(all_ade)
avg_fde = np.mean(all_fde)
print(f"\nFlowChain Results (10 epochs, ALL trajectories):")
print(f"  ADE: {avg_ade:.4f} px  =  {avg_ade / PX_PER_M:.4f} m")
print(f"  FDE: {avg_fde:.4f} px  =  {avg_fde / PX_PER_M:.4f} m")
if all_nll:
    print(f"  NLL: {np.mean(all_nll):.4f}")
print(f"  Batches evaluated: {len(all_ade)}")
