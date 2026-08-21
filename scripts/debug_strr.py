#!/usr/bin/env python3
"""STRR 快速诊断 — 检查 logit 输出分布和梯度"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, numpy as np
from torch.utils.data import DataLoader
from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.official_wrappers import STRROfficial

NORM = torch.tensor([3840.0, 2160.0])
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load scene data with violation labels
ds = TrajectoryDataset(
    data_dir="data/processed/trajectories/", label_dir="labels/",
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=["pedestrian"], mode="with_scene", max_scene_samples=2000,
)
scene_idx = ds.with_scene_subset()
n_pos = sum(1 for i in scene_idx if ds.samples[i].get("is_violation", False))
n_total = len(scene_idx)
print(f"Scene samples: {n_total}, positives: {n_pos} ({100*n_pos/max(1,n_total):.1f}%)")

loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=trajectory_collate_fn)

# Collect a few positive and negative samples
pos_samples, neg_samples = [], []
for batch in loader:
    label = batch.get("is_violation")
    if isinstance(label, torch.Tensor):
        lv = int(label.item()) if label.numel() > 0 else 0
    else:
        lv = int(label[0]) if isinstance(label, list) and len(label) > 0 else 0
    if lv == 1 and len(pos_samples) < 5:
        pos_samples.append(batch)
    elif lv == 0 and len(neg_samples) < 5:
        neg_samples.append(batch)
    if len(pos_samples) >= 5 and len(neg_samples) >= 5:
        break

print(f"\nCollected {len(pos_samples)} pos + {len(neg_samples)} neg samples")

# Test: random init logit range
model = STRROfficial(obs_len=8, pred_len=12).to(device)
model.eval()

print("\n--- Random init logit check ---")
with torch.no_grad():
    for label_type, samples in [("POS", pos_samples), ("NEG", neg_samples)]:
        logits = []
        for batch in samples:
            obs = batch["obs_trajectory"].to(device) / NORM.to(device)
            scene_list = batch.get("scene_list", [None])
            sd = scene_list[0] if scene_list else None
            logit = model(obs_trajectory=obs, scene_data=sd)
            logits.append(logit.item())
        print(f"  {label_type}: logits={[f'{l:.4f}' for l in logits]}")

# Quick training test (20 batches)
import torch.nn as nn, torch.optim as optim
model2 = STRROfficial(obs_len=8, pred_len=12).to(device)
model2.train()
opt = optim.AdamW(model2.parameters(), lr=1e-3)
POS_WEIGHT = torch.tensor([20.0], device=device)
losses = []

for i, batch in enumerate(loader):
    if i >= 100:
        break
    obs = batch["obs_trajectory"].to(device) / NORM.to(device)
    scene_list = batch.get("scene_list", [None])
    sd = scene_list[0] if scene_list else None
    lv = batch.get("is_violation")
    if isinstance(lv, torch.Tensor):
        label = lv.float().to(device)
    else:
        label = torch.tensor([float(lv[0]) if isinstance(lv, list) else 0.0], device=device)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    opt.zero_grad()
    logit = model2(obs_trajectory=obs, scene_data=sd)
    if logit.dim() == 0:
        logit = logit.unsqueeze(0)
    loss = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)(logit, label)
    loss.backward()
    nn.utils.clip_grad_norm_(model2.parameters(), 10.0)
    opt.step()
    losses.append((loss.item(), logit.item(), label.item()))

print("\n--- Training check (100 steps) ---")
print(f"  Initial loss: {losses[0][0]:.4f}")
print(f"  Final loss:   {losses[-1][0]:.4f}")
pos_losses = [l for l, _, lab in losses if lab == 1]
neg_losses = [l for l, _, lab in losses if lab == 0]
print(f"  Pos samples: {len(pos_losses)}, avg loss: {np.mean(pos_losses):.4f}" if pos_losses else "  NO POS SAMPLES!")
print(f"  Neg samples: {len(neg_losses)}, avg loss: {np.mean(neg_losses):.4f}")

# Check final logit distribution
final_logits = [lo for _, lo, _ in losses[-20:]]
final_labels = [la for _, _, la in losses[-20:]]
print(f"\n  Final logits: mean={np.mean(final_logits):.4f}, range=[{min(final_logits):.4f}, {max(final_logits):.4f}]")
pos_logits = [lo for lo, lab in zip(final_logits, final_labels) if lab == 1]
neg_logits = [lo for lo, lab in zip(final_logits, final_labels) if lab == 0]
print(f"  Pos logits: {[f'{l:.4f}' for l in pos_logits]}")
print(f"  Neg logits: {[f'{l:.4f}' for l in neg_logits]}")

# Check if pos_logits are higher than neg_logits
if pos_logits and neg_logits:
    if np.mean(pos_logits) > np.mean(neg_logits):
        print("  ✓ Pos > Neg logits — model is learning")
    else:
        print("  ✗ Pos ≤ Neg logits — model is NOT learning")

print("\nDone.")
