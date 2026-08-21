#!/usr/bin/env python3
"""
FlowChain 诊断脚本
==================
检查：训练loss是否下降、log_prob和sampling是否一致、scaler是否正常。
在云端项目根目录运行: python scripts/debug_flowchain.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.baseline_models import FlowChainBase

NORM = torch.tensor([3840.0, 2160.0])
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
ds = TrajectoryDataset(
    data_dir="data/processed/trajectories/",
    obs_len=8, pred_len=12,
    stride=8, min_trajectory_len=20,
    target_classes=["pedestrian"],
    mode="trajectory_only",
)
subset = Subset(ds, range(min(200, len(ds))))
loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=trajectory_collate_fn)
batch = next(iter(loader))

obs = batch["obs_trajectory"].to(device) / NORM.to(device)
target = batch["target_trajectory"].to(device) / NORM.to(device)

print("=" * 60)
print("FlowChain 诊断")
print("=" * 60)
print(f"obs range:   x=[{obs[...,0].min():.4f}, {obs[...,0].max():.4f}]  y=[{obs[...,1].min():.4f}, {obs[...,1].max():.4f}]")
print(f"target range: x=[{target[...,0].min():.4f}, {target[...,0].max():.4f}]  y=[{target[...,1].min():.4f}, {target[...,1].max():.4f}]")

# ============================================
# Test 1: Random init — check log_prob range
# ============================================
print("\n--- Test 1: Random init log_prob ---")
model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(device)
model.eval()
with torch.no_grad():
    lp = model.predictor.log_prob(
        obs_trajectory=obs, target=target,
        perception_c=torch.zeros(4, 256, device=device)
    )
print(f"Random init log_prob: mean={lp.mean().item():.2f}, std={lp.std().item():.2f}")
print(f"  → NLL (random): {-lp.mean().item():.2f}")

# ============================================
# Test 2: Check scaler values
# ============================================
print("\n--- Test 2: MeanScaler check ---")
with torch.no_grad():
    _, scale = model.predictor.model.scaler(obs)
print(f"Scaler output shape: {scale.shape}")
print(f"Scale per sample (x, y):")
for b in range(min(4, obs.shape[0])):
    print(f"  sample {b}: scale=({scale[b,0,0].item():.4f}, {scale[b,0,1].item():.4f})")
    print(f"           obs range x=[{obs[b,:,0].min():.4f}, {obs[b,:,0].max():.4f}]")
    print(f"           target range x=[{target[b,:,0].min():.4f}, {target[b,:,0].max():.4f}]")

# ============================================
# Test 3: Forward pass — check output range
# ============================================
print("\n--- Test 3: Random init forward pass ---")
with torch.no_grad():
    pred = model.predictor(obs_trajectory=obs, perception_c=torch.zeros(4, 256, device=device), num_samples=20)

print(f"samples shape: {pred['samples'].shape}")  # should be (N, B, pred, 2) = (20, 4, 12, 2)
print(f"log_probs shape: {pred['log_probs'].shape}")
print(f"mean shape: {pred['mean'].shape}")

samples = pred["samples"]  # (N, B, pred, 2)
print(f"samples range: x=[{samples[...,0].min():.4f}, {samples[...,0].max():.4f}]")
print(f"                y=[{samples[...,1].min():.4f}, {samples[...,1].max():.4f}]")

mean_pred = pred["mean"]  # (B, pred, 2)
print(f"mean range:   x=[{mean_pred[...,0].min():.4f}, {mean_pred[...,0].max():.4f}]")
print(f"              y=[{mean_pred[...,1].min():.4f}, {mean_pred[...,1].max():.4f}]")

# Check per-sample
for b in range(min(2, obs.shape[0])):
    print(f"\n  sample {b}:")
    print(f"    last_obs: ({obs[b,-1,0].item():.4f}, {obs[b,-1,1].item():.4f})")
    print(f"    mean[0]:  ({mean_pred[b,0,0].item():.4f}, {mean_pred[b,0,1].item():.4f})")
    print(f"    mean[-1]: ({mean_pred[b,-1,0].item():.4f}, {mean_pred[b,-1,1].item():.4f})")
    print(f"    target[0]:  ({target[b,0,0].item():.4f}, {target[b,0,1].item():.4f})")
    print(f"    target[-1]: ({target[b,-1,0].item():.4f}, {target[b,-1,1].item():.4f})")
    # Best-of-20 ADE
    diff_ade = ((samples[:, b] - target[b].unsqueeze(0)) ** 2).sum(dim=-1).sqrt().mean(dim=-1)
    best_idx = diff_ade.argmin()
    print(f"    best-of-20 ADE (norm): {diff_ade[best_idx].item():.4f}")
    print(f"    best-of-20 ADE (px):   {diff_ade[best_idx].item() * 3840:.0f}")

# ============================================
# Test 4: Quick training sanity check (1 epoch)
# ============================================
print("\n--- Test 4: One-epoch training check ---")
import torch.optim as optim
model2 = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(device)
model2.train()
optimizer = optim.AdamW(model2.parameters(), lr=1e-3)
n_steps = min(50, len(loader))
losses = []

for i, batch in enumerate(loader):
    if i >= n_steps:
        break
    obs_b = batch["obs_trajectory"].to(device) / NORM.to(device)
    target_b = batch["target_trajectory"].to(device) / NORM.to(device)

    optimizer.zero_grad()
    lp = model2.predictor.log_prob(
        obs_trajectory=obs_b, target=target_b,
        perception_c=torch.zeros(obs_b.shape[0], 256, device=device)
    )
    loss = -lp.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), 10.0)
    optimizer.step()
    losses.append(loss.item())

print(f"Initial loss: {losses[0]:.2f}")
print(f"Final loss (step {n_steps}): {losses[-1]:.2f}")
print(f"Loss trend: {'DECREASING ✓' if losses[-1] < losses[0] * 0.9 else 'FLAT or INCREASING ✗'}")

# Grad norm check
total_grad_norm = 0
for p in model2.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.norm().item() ** 2
total_grad_norm = total_grad_norm ** 0.5
print(f"Final total grad norm: {total_grad_norm:.2f} (clipped to 10.0)")

# Check if flow parameters are getting gradients
flow_params_with_grad = 0
flow_params_total = 0
for name, p in model2.named_parameters():
    if 'flow' in name:
        flow_params_total += 1
        if p.grad is not None and p.grad.norm() > 1e-8:
            flow_params_with_grad += 1
print(f"Flow params with grad: {flow_params_with_grad}/{flow_params_total}")

# ============================================
# Test 5: Compare teacher-forcing vs autoregressive
# ============================================
print("\n--- Test 5: Teacher-forcing NLL vs autoregressive ADE ---")
model2.eval()
with torch.no_grad():
    # Teacher-forcing NLL
    lp = model2.predictor.log_prob(
        obs_trajectory=obs[:2], target=target[:2],
        perception_c=torch.zeros(2, 256, device=device)
    )
    tf_nll = -lp.mean().item()

    # Autoregressive sampling
    pred = model2.predictor(obs_trajectory=obs[:2], perception_c=torch.zeros(2, 256, device=device), num_samples=50)
    samples = pred["samples"]  # (50, 2, 12, 2)
    diff = ((samples - target[:2].unsqueeze(0)) ** 2).sum(dim=-1).sqrt()
    ade_per_sample = diff.mean(dim=-1)  # (50, 2)
    best_ade = ade_per_sample.min(dim=0).values.mean().item()

    # Without clamping
    mean_pred = pred["mean"]
    raw_ade = ((mean_pred - target[:2]) ** 2).sum(dim=-1).sqrt().mean(dim=-1).mean().item()

print(f"Teacher-forcing NLL: {tf_nll:.2f}")
print(f"Autoregressive best-of-50 ADE (norm): {best_ade:.4f} = {best_ade*3840:.0f}px")
print(f"Autoregressive mean ADE (norm): {raw_ade:.4f} = {raw_ade*3840:.0f}px")

if tf_nll > 50 and best_ade > 0.1:
    print("\n⚠️  Both NLL and ADE are very high → model is completely random")
elif tf_nll < 10 and best_ade > 0.1:
    print("\n⚠️  NLL is reasonable but ADE is high → exposure bias (teacher-forcing gap)")
elif tf_nll > 50 and best_ade < 0.1:
    print("\n⚠️  ADE is OK but NLL is high → distribution mismatch")
else:
    print("\n✓ Both NLL and ADE are reasonable")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
