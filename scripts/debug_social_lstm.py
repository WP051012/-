#!/usr/bin/env python3
"""
SocialLSTM 诊断脚本
===================
检查预测值范围、训练loss、输入输出维度是否正确。
在云端项目根目录运行: python scripts/debug_social_lstm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.official_wrappers import SocialLSTMOfficial

NORM = torch.tensor([3840.0, 2160.0])
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a few samples
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
obs = batch["obs_trajectory"].to(device)
target = batch["target_trajectory"].to(device)

obs_norm = obs / NORM.to(device)
target_norm = target / NORM.to(device)

print("=" * 60)
print("SocialLSTM 诊断")
print("=" * 60)
print(f"Device: {device}")
print(f"obs range (px):  x=[{obs[...,0].min():.0f}, {obs[...,0].max():.0f}]  y=[{obs[...,1].min():.0f}, {obs[...,1].max():.0f}]")
print(f"obs range (norm): x=[{obs_norm[...,0].min():.4f}, {obs_norm[...,0].max():.4f}]  y=[{obs_norm[...,1].min():.4f}, {obs_norm[...,1].max():.4f}]")
print(f"target range (px): x=[{target[...,0].min():.0f}, {target[...,0].max():.0f}]  y=[{target[...,1].min():.0f}, {target[...,1].max():.0f}]")

# Test 1: Forward pass with random weights
print("\n--- Test 1: Random init forward pass ---")
model = SocialLSTMOfficial(obs_len=8, pred_len=12).to(device)
model.eval()
with torch.no_grad():
    pred = model(obs_trajectory=obs_norm)

pred_mean = pred["mean"]
print(f"pred mean shape: {pred_mean.shape}")
print(f"pred mean range: x=[{pred_mean[...,0].min():.4f}, {pred_mean[...,0].max():.4f}]  y=[{pred_mean[...,1].min():.4f}, {pred_mean[...,1].max():.4f}]")

pred_px = pred_mean * NORM.to(device)
print(f"pred mean (px) range: x=[{pred_px[...,0].min():.0f}, {pred_px[...,0].max():.0f}]  y=[{pred_px[...,1].min():.0f}, {pred_px[...,1].max():.0f}]")

# Check if predictions are in normalized space [0, 1] or raw deltas
last_obs = obs_norm[:, -1:, :]
print(f"\nlast_obs: {last_obs[0, 0, :].cpu().numpy()}")
print(f"pred step0: {pred_mean[0, 0, :].cpu().numpy()}")
print(f"pred step11: {pred_mean[0, -1, :].cpu().numpy()}")
print(f"target step0: {target_norm[0, 0, :].cpu().numpy()}")
print(f"target step11: {target_norm[0, -1, :].cpu().numpy()}")

# Is pred near [0,0] or near last_obs?
diff_from_zero = (pred_mean ** 2).mean().sqrt().item()
diff_from_last = ((pred_mean - last_obs) ** 2).mean().sqrt().item()
diff_from_target = ((pred_mean - target_norm) ** 2).mean().sqrt().item()
print(f"\nRMS pred vs 0:        {diff_from_zero:.4f}")
print(f"RMS pred vs last_obs:  {diff_from_last:.4f}")
print(f"RMS pred vs target:    {diff_from_target:.4f}")

# Test 2: Check the official model internals
print("\n--- Test 2: Official model internals ---")
print(f"model.cell input_size:  {model.model.cell.input_size}")
print(f"model.cell hidden_size: {model.model.cell.hidden_size}")
print(f"input_embedding_layer:  {model.model.input_embedding_layer}")
print(f"output_layer:           {model.model.output_layer}")

# Check if input dim matches
emb_dim = model.model.input_embedding_layer.out_features
print(f"embedding output dim: {emb_dim}")
print(f"Expected cell input:  {emb_dim * 2} (emb + social zeros)")
print(f"Actual cell input:    {model.model.cell.input_size}")
if emb_dim * 2 != model.model.cell.input_size:
    print("❌ CELL INPUT SIZE MISMATCH! This would crash during forward pass.")
    print(f"   Expected {emb_dim*2}, got {model.model.cell.input_size}")
else:
    print("✓ Cell input size matches")

# Test 3: Verify _predict_simple output accumulation
print("\n--- Test 3: Verify delta accumulation ---")
traj = obs_norm[0]  # (8, 2)
with torch.no_grad():
    traj_pred = model._predict_simple(traj, device)

print(f"traj_pred shape: {traj_pred.shape}")
print(f"traj_pred range: x=[{traj_pred[:,0].min():.4f}, {traj_pred[:,0].max():.4f}]")

# Check if values are accumulating (monotonically changing or staying near last obs)
steps_x = traj_pred[:, 0].cpu().numpy()
steps_y = traj_pred[:, 1].cpu().numpy()
print(f"\nPer-step x: {np.array2string(steps_x, precision=4)}")
print(f"Per-step y: {np.array2string(steps_y, precision=4)}")
print(f"Step-to-step Δx: {np.array2string(np.diff(steps_x), precision=4)}")
print(f"Step-to-step Δy: {np.array2string(np.diff(steps_y), precision=4)}")

# Check: are consecutive steps identical? (bug: always feeding same last_obs)
if np.allclose(np.diff(steps_x), 0) and np.allclose(np.diff(steps_y), 0):
    print("❌ ALL STEPS IDENTICAL! _predict_simple is NOT accumulating positions!")
else:
    print("✓ Steps are changing (accumulation working)")

# Expected: traj_pred should be near obs_norm values, not near zero
print(f"\nlast_obs:  ({traj[-1, 0].item():.4f}, {traj[-1, 1].item():.4f})")
print(f"pred[-1]:  ({traj_pred[-1, 0].item():.4f}, {traj_pred[-1, 1].item():.4f})")
diff = torch.sqrt(((traj_pred - traj[-1]) ** 2).sum(dim=-1))
print(f"Dist from last_obs per step: {diff.cpu().numpy()}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
