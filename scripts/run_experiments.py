#!/usr/bin/env python3
"""
实验运行脚本 — 完整对比实验 + 消融实验
=========================================
按照实验方案设计，运行所有 baseline 对比和消融实验，
输出 result CSVs。

实验1: 轨迹预测性能对比
    方法: Social-LSTM, STGCNN, FlowChain, OurMethod
    指标: ADE, FDE, NLL

实验2: 闯红灯预测性能对比
    方法: LSTM-Classifier, GRU-Classifier, STRR, OurMethod
    指标: Accuracy, Precision, Recall, F1, AUC

实验3: 消融实验 (真实OurMethod + ablation flag)
    变体: NoGraph, NoMemory, NoCogContext, NoFlowChain, NoChange vs FullModel
    每个变体从完整OurMethod出发，只关掉一个模块

数据划分:
    Train: 2026_01_15, 01_21, 01_22, 01_23
    Val:   2026_01_26
    Test:  2026_01_27

用法:
    # 运行所有实验
    python scripts/run_experiments.py --all

    # 仅运行轨迹预测实验
    python scripts/run_experiments.py --exp trajectory

    # 仅运行分类实验
    python scripts/run_experiments.py --exp classification

    # 仅运行消融实验
    python scripts/run_experiments.py --exp ablation

    # 快速测试模式 (少量数据)
    python scripts/run_experiments.py --exp trajectory --quick
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.baseline_models import (
    FlowChainBase,
    TransformerBaseline,
    RNNBaseline,
)
from src.baselines.official_wrappers import (
    SocialLSTMOfficial, SocialSTGCNNOfficial, STRROfficial,
    GaussianMCViolationClassifier,
)
from src.perception_model import TrafficPerceptionModel
from src.prediction.flow_chain import flow_chain_nll_loss, joint_nll_mse_loss
from src.evaluation import (
    compute_ade, compute_fde, compute_nll,
    compute_trajectory_metrics, compute_classification_metrics,
    export_trajectory_results_csv, export_classification_results_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
# Config
# ======================================================================

# Data split by date
TRAIN_DATES = {"2026_01_15", "2026_01_21", "2026_01_22", "2026_01_23"}
VAL_DATES = {"2026_01_26"}
TEST_DATES = {"2026_01_27"}

IMG_W, IMG_H = 3840.0, 2160.0
NORM = torch.tensor([IMG_W, IMG_H])  # anisotropic: x/3840, y/2160


def filter_samples_by_date(dataset: TrajectoryDataset, dates: set) -> Subset:
    """Filter dataset samples by video date."""
    indices = []
    for i, s in enumerate(dataset.samples):
        # Extract date from video name
        video = s.get("video", "")
        date_match = None
        for d in dates:
            if d.replace("_", "") in video.replace("-", "").replace("timing_", ""):
                date_match = d
                break
        if date_match:
            indices.append(i)
    return Subset(dataset, indices)


# ======================================================================
# Memory logging helper
# ======================================================================

def _get_cpu_rss_gb() -> float:
    """Return current process RSS in GB, or 0 if unavailable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except ImportError:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024 / (1024 ** 3)
        except Exception:
            return 0.0


# ======================================================================
# Training helpers
# ======================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_type: str = "trajectory",  # "trajectory" or "classifier"
    logger_instance=None,
) -> dict:
    """Generic training loop for any model."""
    if logger_instance is None:
        logger_instance = logger

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            obs = batch["obs_trajectory"].to(device) / NORM.to(device)
            target = batch["target_trajectory"].to(device) / NORM.to(device)

            optimizer.zero_grad()

            if model_type == "trajectory":
                # Teacher forcing: if model supports log_prob, use it (much more stable for flow models)
                if hasattr(model, 'log_prob'):
                    lp = model.log_prob(obs_trajectory=obs, target=target,
                                        perception_c=torch.zeros(obs.shape[0], getattr(model, '_cond_dim', 256), device=device))
                    loss = -lp.mean()
                elif hasattr(model, 'predictor') and hasattr(model.predictor, 'log_prob'):
                    lp = model.predictor.log_prob(obs_trajectory=obs, target=target,
                                                   perception_c=torch.zeros(obs.shape[0], getattr(model, '_cond_dim', 256), device=device))
                    loss = -lp.mean()
                else:
                    # Fallback: autoregressive sampling + NLL
                    pred = model(obs_trajectory=obs, num_samples=10)
                    try:
                        loss = flow_chain_nll_loss(pred, target)
                    except Exception:
                        loss = ((pred["mean"] - target) ** 2).mean()
            else:
                # Classifier: use full sequence (obs + target) as input
                full_traj = torch.cat([obs, target], dim=1)  # (B, obs+pred, 2)
                logits = model(full_traj)
                # Use violation labels if available
                labels = batch.get("is_violation", torch.zeros(logits.shape[0]))
                loss = nn.BCEWithLogitsLoss()(logits, labels.float().to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs_trajectory"].to(device) / NORM.to(device)
                target = batch["target_trajectory"].to(device) / NORM.to(device)

                if model_type == "trajectory":
                    if hasattr(model, 'log_prob'):
                        lp = model.log_prob(obs_trajectory=obs, target=target,
                                            perception_c=torch.zeros(obs.shape[0], getattr(model, '_cond_dim', 256), device=device))
                        loss = -lp.mean()
                    elif hasattr(model, 'predictor') and hasattr(model.predictor, 'log_prob'):
                        lp = model.predictor.log_prob(obs_trajectory=obs, target=target,
                                                       perception_c=torch.zeros(obs.shape[0], getattr(model, '_cond_dim', 256), device=device))
                        loss = -lp.mean()
                    else:
                        pred = model(obs_trajectory=obs, num_samples=10)
                        from src.prediction.flow_chain import flow_chain_nll_loss
                        try:
                            loss = flow_chain_nll_loss(pred, target)
                        except Exception:
                            loss = ((pred["mean"] - target) ** 2).mean()
                else:
                    full_traj = torch.cat([obs, target], dim=1)
                    logits = model(full_traj)
                    labels = batch.get("is_violation", torch.zeros(logits.shape[0]))
                    loss = nn.BCEWithLogitsLoss()(logits, labels.float().to(device))

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {"best_val_loss": best_val_loss}


def train_perception_model(
    model: TrafficPerceptionModel,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    start_epoch: int = 0,
    checkpoint_path: str = None,
    segment_epochs: int = 0,
    grad_accum_steps: int = 4,
) -> dict:
    """Train TrafficPerceptionModel with gradient accumulation + segment support.

    Gradient accumulation: since per-sample processing gives effective batch=1,
    gradients are accumulated over `grad_accum_steps` samples before stepping.

    Segment mode (segment_epochs > 0):
    - Trains only segment_epochs epochs, saves checkpoint, returns immediately
    - You then restart the script with --resume to continue in a fresh process
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    max_agents = 10

    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", start_epoch)
        logger.info(f"  Resuming from epoch {start_epoch}")

    # Determine how many epochs to run this segment
    actual_start = start_epoch
    if segment_epochs > 0:
        actual_end = min(start_epoch + segment_epochs, epochs)
    else:
        actual_end = epochs

    logger.info(f"  Training epochs {actual_start}→{actual_end} (total target: {epochs})")

    mem_start = _get_cpu_rss_gb()
    if mem_start > 0:
        logger.info(f"  Memory before segment: {mem_start:.1f} GB")

    for epoch in range(actual_start, actual_end):
        model.train()
        total_loss = 0.0
        n = 0
        sample_count = 0  # global counter for gradient accumulation
        optimizer.zero_grad()

        epoch_iter = tqdm(train_loader, desc=f"PM-E{epoch}", leave=False)

        for batch_idx, batch in enumerate(epoch_iter):
            B = batch["obs_trajectory"].shape[0]

            for b in range(B):
                model.reset_state()

                obs = batch["obs_trajectory"][b:b+1].to(device) / NORM.to(device)
                target = batch["target_trajectory"][b:b+1].to(device) / NORM.to(device)
                scene = batch.get("scene_list", [None] * B)[b]

                scene_data = None
                if scene is not None:
                    n_agents = len(scene["class_names"]) if "class_names" in scene else 0
                    if n_agents > max_agents:
                        target_pos = scene["positions"][0]
                        dists = torch.norm(scene["positions"] - target_pos, dim=1)
                        _, top_idx = dists.topk(min(max_agents, n_agents))
                        scene_data = {
                            "bboxes": scene["bboxes"][top_idx].unsqueeze(0).to(device),
                            "positions": scene["positions"][top_idx].unsqueeze(0).to(device),
                            "class_names": [scene["class_names"][i] for i in top_idx.tolist()],
                            "track_ids": [scene["track_ids"][i] for i in top_idx.tolist()] if "track_ids" in scene else [],
                            "target_idx": 0,
                        }
                    else:
                        scene_data = {
                            "bboxes": scene["bboxes"].unsqueeze(0).to(device),
                            "positions": scene["positions"].unsqueeze(0).to(device),
                            "class_names": scene["class_names"],
                            "track_ids": scene.get("track_ids", []),
                            "target_idx": 0,
                        }

                try:
                    perception_c = model.compute_perception_context(
                        obs_trajectory=obs.squeeze(0), scene_data=scene_data)

                    # --- Loss: NLL (FlowChain) or MSE (no_flowchain) ---
                    if getattr(model, 'ablation', None) == "no_flowchain":
                        # MLP deterministic: predict DELTAS, convert to absolute
                        h = model._last_gru_hidden
                        if h is None:
                            h = torch.zeros(obs.shape[0], model.gru_hidden_dim,
                                            device=device)
                        if h.dim() == 1:
                            h = h.unsqueeze(0)
                        deltas = model.mlp_decoder(
                            torch.cat([h, perception_c], dim=-1))
                        deltas = deltas.view(1, model.pred_len, model.trajectory_dim)
                        pred = obs[:, -1:] + deltas  # delta → absolute
                        loss = ((pred - target) ** 2).mean()
                    else:
                        lp = model.flow_chain.log_prob(
                            obs_trajectory=obs.squeeze(0),
                            target=target.squeeze(0),
                            perception_c=perception_c)
                        loss = -lp.mean()

                    if torch.isfinite(loss):
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / grad_accum_steps
                        scaled_loss.backward()
                        sample_count += 1
                        total_loss += loss.item()
                        n += 1

                        # Step only after accumulating enough gradients
                        if sample_count % grad_accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                            optimizer.step()
                            optimizer.zero_grad()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.warning(f"OOM at sample {n}, skipping")
                        continue
                    raise
                finally:
                    try: del loss
                    except NameError: pass
                    try: del perception_c
                    except NameError: pass
                    try: del lp
                    except NameError: pass
                    try: del obs
                    except NameError: pass
                    try: del target
                    except NameError: pass
                    try: del scene_data
                    except NameError: pass
                    try: del scene
                    except NameError: pass

            # Cleanup after each batch
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    del batch[k]
            if batch_idx % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Flush remaining accumulated gradients
        if sample_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

        # End-of-epoch cleanup
        gc.collect()
        torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        mem_now = _get_cpu_rss_gb()
        if mem_start > 0 and mem_now > 0:
            delta = mem_now - mem_start
            logger.info(f"  PM E{epoch}: loss={avg_loss:.4f} | CPU={mem_now:.1f}GB (Δ={delta:+.1f})")

    # Save checkpoint (always save final model)
    ckpt_path = checkpoint_path or "checkpoints/perception_segment.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt_data = {
        "epoch": actual_end,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(ckpt_data, ckpt_path)
    logger.info(f"  Checkpoint saved: {ckpt_path} (epoch {actual_end}/{epochs})")

    # Also save as ablation_fullmodel.pt for classification experiment
    fullmodel_path = "checkpoints/ablation_fullmodel.pt"
    os.makedirs(os.path.dirname(fullmodel_path), exist_ok=True)
    torch.save(ckpt_data, fullmodel_path)
    logger.info(f"  Also saved: {fullmodel_path}")

    if segment_epochs > 0 and actual_end < epochs:
        logger.info(f"  Resume with: --resume {ckpt_path} --segment {segment_epochs}")
        return {"status": "segment_complete", "next_epoch": actual_end, "checkpoint": ckpt_path}

    return {"final_loss": total_loss / max(n, 1)}


# ======================================================================
# Experiment 1: Trajectory Prediction
# ======================================================================

def run_trajectory_experiment(
    train_set, val_set, test_set,
    train_set_scene=None, val_set_scene=None, test_set_scene=None,
    config: dict = None, device: str = "cuda", quick: bool = False,
    epochs_override: int = None,
    skip_models: set = None,
    resume_from: str = None,
    segment_epochs: int = 0,
    crosswalk_filter: bool = False,
    pix_per_meter: float = None,
    eval_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run trajectory prediction comparison experiment."""
    logger.info("=" * 60)
    logger.info("Experiment 1: Trajectory Prediction Comparison")
    logger.info("=" * 60)

    if skip_models is None:
        skip_models = set()
    epochs = epochs_override or (5 if quick else 50)
    batch_size = 64
    lr = 1e-3

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=trajectory_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=trajectory_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=trajectory_collate_fn)

    # Build crosswalk polygons if filtering enabled
    crosswalk_polygons = None
    if crosswalk_filter and config:
        cw_polys = {}
        for key in ("intersection_A", "intersection_B"):
            cfg = config.get(key, {})
            cw = cfg.get("crosswalk_roi", None)
            if cw and len(cw) >= 4:
                poly = [(float(p[0]), float(p[1])) for p in cw]
                # Map intersection_A → "timing", intersection_B → "numbered"
                pattern = "timing" if key == "intersection_A" else "numbered"
                cw_polys[pattern] = poly
        if cw_polys:
            crosswalk_polygons = cw_polys
            logger.info(f"Crosswalk filter: {cw_polys}")

    results = {}

    # --- Baseline 1: Social-LSTM (official wrapper) ---
    if "social-lstm" not in skip_models:
        logger.info("\nTraining Social-LSTM (official)...")
        model = SocialLSTMOfficial(obs_len=8, pred_len=12).to(device)
        train_model(model, train_loader, val_loader, epochs, lr, device, "trajectory")
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["Social-LSTM"] = metrics
        logger.info(f"  Social-LSTM: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}")

    # --- Baseline 2: STGCNN (single-pedestrian, no social graph) ---
    if "stgcnn" not in skip_models:
        logger.info("\nTraining STGCNN...")
        model = SocialSTGCNNOfficial(obs_len=8, pred_len=12).to(device)
        train_model(model, train_loader, val_loader, epochs, lr, device, "trajectory")
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["STGCNN"] = metrics
        logger.info(f"  STGCNN: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}")

    # --- Baseline 3: FlowChain (vanilla) ---
    if "flowchain" not in skip_models:
        if eval_only:
            logger.info("\nLoading FlowChain checkpoint for eval-only...")
            model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(device)
            model.load_state_dict(torch.load("checkpoints/flowchain_best.pt", map_location=device))
            metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        else:
            logger.info("\nTraining FlowChain (vanilla)...")
            model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(device)
            train_model(model, train_loader, val_loader, epochs, lr, device, "trajectory")
            torch.save(model.state_dict(), "checkpoints/flowchain_best.pt")
            metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["FlowChain"] = metrics
        logger.info(f"  FlowChain: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}, NLL={metrics['NLL']:.4f}")

    # --- Baseline 4: Transformer (Trajectory-Transformer, official) ---
    if "transformer" not in skip_models:
        logger.info("\nTraining Transformer (official Trajectory-Transformer)...")
        model = TransformerBaseline(
            obs_len=8, pred_len=12,
            d_model=128, d_ff=512, heads=4, layers=3, dropout=0.1,
        ).to(device)
        # Fit delta normalization on training set (mirrors official code)
        model.fit_normalization(train_loader, device, norm=[3840.0, 2160.0])
        train_model(model, train_loader, val_loader, epochs, lr, device, "trajectory")
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["Transformer"] = metrics
        logger.info(f"  Transformer: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}")

    # --- Baseline 5: Vanilla RNN Seq2Seq (uestc-db official architecture) ---
    if "rnn" not in skip_models:
        logger.info("\nTraining Vanilla RNN Seq2Seq...")
        model = RNNBaseline(obs_len=8, pred_len=12, hidden_dim=128, dropout=0.5).to(device)
        train_model(model, train_loader, val_loader, epochs, lr, device, "trajectory")
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["RNN"] = metrics
        logger.info(f"  RNN: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}")

    # --- OurMethod: full TrafficPerceptionModel (Stage 2, scene data) ---
    if "ourmethod" not in skip_models:
        logger.info("\nTraining Our Method (TrafficPerceptionModel)...")
        model = TrafficPerceptionModel(config, stage=2).to(device)
        train_scene_loader = DataLoader(
            train_set_scene, batch_size=16, shuffle=True,
            collate_fn=trajectory_collate_fn, num_workers=0,
        )
        train_result = train_perception_model(
            model, train_scene_loader, epochs, lr, device,
            checkpoint_path=resume_from or "checkpoints/perception_segment.pt",
            segment_epochs=segment_epochs,
        )
        if train_result.get("status") == "segment_complete":
            # Mid-segment: exit cleanly, user re-runs with --resume
            logger.info(f"Segment done. Next: epoch {train_result['next_epoch']}/{epochs}")
            logger.info("Evaluation deferred — run final segment to evaluate")
            return {}
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results["OurMethod"] = metrics
        logger.info(f"  OurMethod: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}, NLL={metrics['NLL']:.4f}")

    # Export
    export_trajectory_results_csv(results, "trajectory_prediction_results.csv")
    return results


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting point-in-polygon test. polygon: list of (x,y) pairs or 2×N array."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _get_crosswalk_mask(
    obs_px: torch.Tensor,        # (B, obs_len, 2) in pixels
    target_px: torch.Tensor,     # (B, pred_len, 2) in pixels
    video_names: list,           # [B] video name strings
    crosswalk_polygons: dict,    # {intersection_key: [(x,y), ...]}
) -> torch.Tensor:
    """
    Return a boolean mask (B,) indicating which samples in the batch
    have at least one frame inside the crosswalk polygon.
    """
    B = len(video_names)
    mask = torch.zeros(B, dtype=torch.bool)

    for b in range(B):
        vname = video_names[b]
        key = "timing" if "timing_" in vname else "numbered"
        polygon = crosswalk_polygons.get(key)
        if polygon is None:
            continue

        full_traj = torch.cat([obs_px[b], target_px[b]], dim=0)  # (20, 2)
        for t in range(full_traj.shape[0]):
            x, y = float(full_traj[t, 0]), float(full_traj[t, 1])
            if _point_in_polygon(x, y, polygon):
                mask[b] = True
                break

    return mask


@torch.no_grad()
def evaluate_trajectory(
    model, loader, device,
    num_eval_samples: int = 100,
    crosswalk_polygons: dict = None,
    pix_per_meter: float = None,
) -> dict:
    """
    Evaluate trajectory prediction model on test set.

    Uses best-of-100 with median aggregation (consistent with earlier results).

    Parameters
    ----------
    crosswalk_polygons : dict or None
        If provided, filter to only evaluate trajectories that enter the
        crosswalk polygon. Keys: "timing" (intersection A) and "numbered"
        (intersection B).
    pix_per_meter : float or None
        If provided, also report ADE/FDE in metres.
    """
    model.eval()
    norm_tensor = NORM.detach().to(device)
    all_ade, all_fde, all_nll = [], [], []
    total_filtered, total_kept = 0, 0

    for batch in loader:
        obs = batch["obs_trajectory"].to(device) / NORM.to(device)
        target = batch["target_trajectory"].to(device) / NORM.to(device)
        video_names = batch.get("video", [])

        # ---- Crosswalk filter ----
        if crosswalk_polygons is not None and len(video_names) > 0:
            obs_px = batch["obs_trajectory"]  # original pixels
            target_px = batch["target_trajectory"]
            cw_mask = _get_crosswalk_mask(
                obs_px, target_px, video_names, crosswalk_polygons,
            )
            total_filtered += int((~cw_mask).sum())
            total_kept += int(cw_mask.sum())

            if cw_mask.sum() == 0:
                continue  # no crossing samples in this batch

            # Filter batch
            obs = obs[cw_mask]
            target = target[cw_mask]
            # Also filter video_names for NLL path
            video_names_filtered = [v for i, v in enumerate(video_names) if cw_mask[i]]
        else:
            video_names_filtered = video_names

        pred = model(obs_trajectory=obs, num_samples=num_eval_samples)

        # ---- Best-of-N selection for ADE/FDE ----
        if "samples" in pred and pred["samples"].dim() >= 4:
            samples = pred["samples"].clamp(0.0, 1.0)
            N_samples = samples.shape[0]

            samples_px = samples * norm_tensor
            target_px = target * norm_tensor

            diff = samples_px - target_px.unsqueeze(0)
            l2_per_step = torch.sqrt((diff ** 2).sum(dim=-1))
            ade_per_sample = l2_per_step.mean(dim=-1)
            fde_per_sample = l2_per_step[:, :, -1]

            best_idx = ade_per_sample.argmin(dim=0)
            best_ade = ade_per_sample.gather(0, best_idx.unsqueeze(0)).squeeze(0)
            best_fde = fde_per_sample.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        elif "mean" in pred:
            mean_px = pred["mean"] * norm_tensor
            target_px = target * norm_tensor
            diff = mean_px - target_px
            l2 = torch.sqrt((diff ** 2).sum(dim=-1))
            best_ade = l2.mean(dim=-1)
            best_fde = l2[:, -1]
        else:
            continue

        all_ade.append(best_ade.cpu())
        all_fde.append(best_fde.cpu())

        # NLL
        if "log_probs" in pred:
            lp = pred["log_probs"]
            if lp.dim() == 2:
                best_lp = lp.gather(0, best_idx.unsqueeze(0).to(lp.device)).squeeze(0)
                all_nll.append(-best_lp.cpu())
            else:
                all_nll.append(-lp.cpu())

    ade_px = torch.cat(all_ade) if all_ade else torch.tensor([float('nan')])
    fde_px = torch.cat(all_fde) if all_fde else torch.tensor([float('nan')])
    nll_val = torch.cat(all_nll).mean() if all_nll else torch.tensor(float('inf'))

    result = {
        "ADE": float(ade_px.median()),
        "FDE": float(fde_px.median()),
        "NLL": float(nll_val),
    }

    if crosswalk_polygons is not None:
        result["ADE_px"] = result["ADE"]
        result["FDE_px"] = result["FDE"]
        result["n_total"] = total_filtered + total_kept
        result["n_crosswalk"] = total_kept
        if pix_per_meter is not None and pix_per_meter > 0:
            result["ADE_m"] = round(result["ADE"] / pix_per_meter, 4)
            result["FDE_m"] = round(result["FDE"] / pix_per_meter, 4)

    return result


# ======================================================================
# Experiment 2: Red-Light Violation Prediction
# ======================================================================

def run_classification_experiment(
    train_set, val_set, test_set,
    config: dict, device: str, quick: bool = False,
    epochs_override: int = None,
    skip_models: set = None,
) -> Dict[str, Dict[str, float]]:
    """
    Red-light violation prediction comparison.

    Models:
        STRR       — GCN + inner-product edges + GRU → binary classifier
        STGCNN-MC  — STGCNN trajectory + Gaussian MC + geometric check
        OurMethod  — Perception pipeline → trajectory → geometric violation check
    """
    if skip_models is None:
        skip_models = set()
    logger.info("=" * 60)
    logger.info("Experiment 2: Red-Light Violation Prediction")
    logger.info("=" * 60)

    epochs = epochs_override or (5 if quick else 50)
    lr = 1e-3

    # Use scene-enabled datasets (passed from main, already split + labeled)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=trajectory_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=trajectory_collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=trajectory_collate_fn)

    results = {}

    # ================================================================
    # Baseline: STRR (GCN + inner-product edge weights + GRU → classifier)
    # ================================================================
    if "strr" not in skip_models:
        logger.info("\nTraining STRR...")
        model = STRROfficial(obs_len=8, pred_len=12).to(device)
        train_classifier_with_scene(model, train_loader, val_loader, epochs, lr, device)
        metrics = evaluate_classifier_with_scene(model, test_loader, device)
        results["STRR"] = metrics
        logger.info(f"  STRR: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}")

    # ================================================================
    # Baseline: STGCNN + Gaussian MC Violation Checker
    # ================================================================
    if "stgcnn-mc" not in skip_models:
        logger.info("\n=== STGCNN + Gaussian MC Classification ===")

        # Step 1: Train STGCNN trajectory model (standard MSE)
        logger.info("Training STGCNN trajectory model...")
        stgcnn_model = SocialSTGCNNOfficial(obs_len=8, pred_len=12).to(device)
        # Use larger batch size for trajectory training (doesn't need scene data)
        stgcnn_train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                         collate_fn=trajectory_collate_fn)
        stgcnn_val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                                       collate_fn=trajectory_collate_fn)
        train_model(stgcnn_model, stgcnn_train_loader, stgcnn_val_loader, 10, lr, device, "trajectory")

        # Step 2: Build violation checker from config
        checker = _build_violation_checker(config)
        logger.info(f"  Violation checker: stop_line={checker.stop_line is not None}, "
                     f"junction={checker.junction is not None}")

        # Step 3: MC classification
        mc_classifier = GaussianMCViolationClassifier(
            trajectory_model=stgcnn_model,
            violation_checker=checker,
        ).to(device)

        metrics = evaluate_gaussian_mc_classifier(mc_classifier, test_loader, device)
        results["STGCNN-MC"] = metrics
        logger.info(f"  STGCNN-MC: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}")

    # ================================================================
    # OurMethod — full perception pipeline + geometric violation check
    # ================================================================
    if "ourmethod" not in skip_models:
        logger.info("\nEvaluating OurMethod (Stage 3)...")

        our_model = TrafficPerceptionModel(config, stage=3, ablation=None).to(device)
        n_params = sum(p.numel() for p in our_model.parameters())
        logger.info(f"  Parameters: {n_params:,}")

        # Try to load pre-trained stage-2 checkpoint
        ckpt_path = "checkpoints/ablation_fullmodel.pt"
        if os.path.exists(ckpt_path):
            logger.info(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model_state = our_model.state_dict()
            filtered = {k: v for k, v in ckpt.get("model_state", ckpt).items()
                        if k in model_state and v.shape == model_state[k].shape}
            our_model.load_state_dict(filtered, strict=False)
            logger.info(f"  Loaded {len(filtered)}/{len(model_state)} params")
        else:
            logger.warning(f"  No checkpoint found at {ckpt_path} — using random weights")

        metrics = evaluate_ourmethod_violation(our_model, test_loader, device)
        results["OurMethod"] = metrics
        logger.info(f"  OurMethod: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}")

    export_classification_results_csv(results, "red_light_prediction_results.csv")
    return results


# ======================================================================
# Experiment 2 helpers
# ======================================================================

def train_classifier_with_scene(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> dict:
    """Train a scene-aware classifier (STRR) with BCE loss on violation labels.

    Uses pos_weight to handle severe class imbalance (~4.6% violations).
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class imbalance: ~4.6% positive → pos_weight ≈ 20.6
    POS_WEIGHT = torch.tensor([20.0], device=device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        n_pos = 0
        n_total = 0
        for batch in tqdm(train_loader, desc=f"STRR-E{epoch}", leave=False):
            obs = batch["obs_trajectory"].to(device) / NORM.to(device)
            scene_list = batch.get("scene_list", [None])
            scene_data = scene_list[0] if scene_list else None  # batch_size=1

            # Get violation label
            label_val = batch.get("is_violation")
            if isinstance(label_val, torch.Tensor):
                label = label_val.float().to(device)
            elif isinstance(label_val, (list, np.ndarray)):
                label = torch.tensor(label_val[0] if len(label_val) > 0 else 0.0, dtype=torch.float, device=device)
            else:
                label = torch.tensor([0.0], dtype=torch.float, device=device)

            if label.dim() == 0:
                label = label.unsqueeze(0)

            n_pos += int(label.sum().item())
            n_total += label.numel()

            # Skip batches with no positive samples (optional: comment out to train on all)
            # Can help with convergence early on, but pos_weight should handle it

            optimizer.zero_grad()

            logits = model(obs_trajectory=obs, scene_data=scene_data)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs_trajectory"].to(device) / NORM.to(device)
                scene_list = batch.get("scene_list", [None])
                scene_data = scene_list[0] if scene_list else None

                label_val = batch.get("is_violation")
                if isinstance(label_val, torch.Tensor):
                    label = label_val.float().to(device)
                elif isinstance(label_val, (list, np.ndarray)):
                    label = torch.tensor(label_val[0] if len(label_val) > 0 else 0.0, dtype=torch.float, device=device)
                else:
                    label = torch.tensor([0.0], dtype=torch.float, device=device)

                if label.dim() == 0:
                    label = label.unsqueeze(0)

                logits = model(obs_trajectory=obs, scene_data=scene_data)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                val_loss += nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)(logits, label).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        pos_rate = n_pos / max(n_total, 1)
        logger.info(f"  STRR E{epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, pos_rate={pos_rate:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return {"best_val_loss": best_val_loss}


# ======================================================================
# Gaussian MC Classifier helpers
# ======================================================================

def _build_violation_checker(config: dict):
    """Build RedLightViolationChecker from config intersections."""
    from src.classification.red_light_classifier import (
        RedLightViolationChecker, StopLine, JunctionRegion,
    )

    # Try to get geometry from first available intersection
    stop_line = None
    junction = None

    for key in ("intersection_A", "intersection_B"):
        cfg = config.get(key, {})
        sl = cfg.get("stop_line", None)
        cw = cfg.get("crosswalk_roi", None)

        if sl and len(sl) >= 4:
            # Flat format: [x1, y1, x2, y2]
            stop_line = StopLine(float(sl[0]), float(sl[1]), float(sl[2]), float(sl[3]))
        if cw and len(cw) >= 3:
            # Could be [[x,y],...] or flat [x1,y1,x2,y2,...]
            if isinstance(cw[0], (list, tuple)):
                junction = JunctionRegion([(float(p[0]), float(p[1])) for p in cw])
            else:
                # Flat: treat as polygon vertices [x1,y1, x2,y2, x3,y3, x4,y4]
                pts = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
                junction = JunctionRegion(pts) if len(pts) >= 3 else None
        if stop_line or junction:
            break

    return RedLightViolationChecker(stop_line=stop_line, junction=junction)


@torch.no_grad()
def evaluate_gaussian_mc_classifier(model, loader, device, num_samples=100) -> dict:
    """Evaluate Gaussian MC violation classifier."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for batch in tqdm(loader, desc="MC-Eval", leave=False):
        obs = batch["obs_trajectory"].to(device) / NORM.to(device)

        result = model(obs, num_samples=num_samples)

        prob = result["violation_probability"]
        pred = result["is_violation"]
        if prob.dim() == 0:
            prob = prob.unsqueeze(0)
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)

        label_val = batch.get("is_violation")
        if isinstance(label_val, torch.Tensor):
            label = label_val.float()
        else:
            label = torch.tensor([float(label_val[0]) if isinstance(label_val, list) and len(label_val) > 0 else 0.0], device=device)

        all_preds.append(pred.cpu())
        all_probs.append(prob.cpu())
        all_labels.append(label.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    return compute_classification_metrics(y_true, y_pred, y_prob)


@torch.no_grad()
def evaluate_classifier_with_scene(model, loader, device) -> dict:
    """Evaluate scene-aware classifier (STRR)."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for batch in loader:
        obs = batch["obs_trajectory"].to(device) / NORM.to(device)
        scene_list = batch.get("scene_list", [None])
        scene_data = scene_list[0] if scene_list else None  # batch_size=1

        logits = model(obs_trajectory=obs, scene_data=scene_data)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        label_val = batch.get("is_violation")
        if isinstance(label_val, torch.Tensor):
            label = label_val.float()
        elif isinstance(label_val, (list, np.ndarray)):
            label = torch.tensor(label_val, dtype=torch.float)
        else:
            label = torch.zeros(probs.shape[0])
        if label.dim() == 0:
            label = label.unsqueeze(0)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(label)

    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    return compute_classification_metrics(y_true, y_pred, y_prob)


@torch.no_grad()
def evaluate_ourmethod_violation(model, loader, device) -> dict:
    """
    Evaluate OurMethod stage-3 violation prediction.

    Runs full perception pipeline → trajectory prediction → geometric violation check.
    No BCE training needed — the violation check is rule-based (MC + geometry).
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for batch in tqdm(loader, desc="OurMethod-Eval", leave=False):
        obs = batch["obs_trajectory"].to(device) / NORM.to(device)
        scene_list = batch.get("scene_list", [None])

        # batch_size=1, each sample gets its own scene
        for b in range(obs.shape[0]):
            model.reset_state()
            scene_data = scene_list[b] if isinstance(scene_list, list) and b < len(scene_list) else None

            pred = model(
                obs_trajectory=obs[b:b+1],
                scene_data=scene_data,
                num_samples=20,
            )

            viol_prob = pred.get("violation_probability")
            if viol_prob is not None:
                if viol_prob.dim() > 0:
                    viol_prob = viol_prob.mean()
                prob = viol_prob.item()
                pred_binary = 1 if prob > 0.5 else 0
            else:
                prob = 0.0
                pred_binary = 0

            all_probs.append(prob)
            all_preds.append(pred_binary)

            label = batch.get("is_violation", [0])
            if isinstance(label, torch.Tensor):
                label = label[b:b+1].item() if label.numel() > b else label.item() if label.numel() > 0 else 0
            elif isinstance(label, (list, np.ndarray)):
                label = label[b] if b < len(label) else 0
            else:
                label = 0
            all_labels.append(int(label))

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    return compute_classification_metrics(y_true, y_pred, y_prob)


# ======================================================================
# Experiment 3: Ablation Study
# ======================================================================

ABLATION_VARIANTS = {
    "FullModel":    None,            # 完整 OurMethod
    "NoGraph":      "no_graph",      # 去感知图GAT → 简单编码
    "NoMemory":     "no_memory",      # 去三支记忆 → 直接投影
    "NoCogContext":  "no_cogcontext", # GRU不含认知状态c
    "NoFlowChain":  "no_flowchain",  # 去FlowChain → MLP
    "NoChange":     "no_change",     # 去变化检测+衰减
}


def run_ablation_experiment(
    train_set, val_set, test_set,
    train_set_scene=None, val_set_scene=None, test_set_scene=None,
    config: dict = None, device: str = "cuda", quick: bool = False,
    epochs_override: int = None,
    resume_from: str = None,
    segment_epochs: int = 0,
    variant: str = None,
) -> Dict[str, Dict[str, float]]:
    """Run ablation study using TrafficPerceptionModel with ablation flags.

    Each variant starts from the full OurMethod pipeline and disables
    exactly one component via the `ablation` parameter. All variants
    use scene data and are trained with train_perception_model().

    Parameters
    ----------
    variant : str or None
        If set, only run this specific ablation variant.
    """
    # Filter to single variant if requested
    variants_to_run = ABLATION_VARIANTS
    if variant is not None:
        if variant not in ABLATION_VARIANTS:
            logger.error(f"Unknown variant: {variant}. Choose from: {list(ABLATION_VARIANTS.keys())}")
            return {}
        variants_to_run = {variant: ABLATION_VARIANTS[variant]}

    n_variants = len(variants_to_run)
    logger.info("=" * 60)
    logger.info(f"Experiment 3: Ablation Study (Real OurMethod) — {n_variants} variant(s)")
    logger.info("=" * 60)

    epochs = epochs_override or (5 if quick else 50)
    lr = 1e-3

    # Test loader (trajectory-only for evaluation)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False,
                             collate_fn=trajectory_collate_fn)

    results = {}

    for variant_name, ablation_flag in variants_to_run.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Ablation: {variant_name} (flag={ablation_flag})")
        logger.info(f"{'='*40}")

        # Create model with ablation flag
        model = TrafficPerceptionModel(config, stage=2, ablation=ablation_flag).to(device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

        # Use scene-enabled data for variants with perception modules
        needs_scene = ablation_flag not in ("no_graph", "no_memory") or True
        # Actually all variants benefit from scene data (even no_graph/no_memory
        # use scene data for node positions etc.)
        train_data = train_set_scene if train_set_scene is not None else train_set
        val_data = val_set_scene if val_set_scene is not None else val_set

        train_scene_loader = DataLoader(
            train_data, batch_size=16, shuffle=True,
            collate_fn=trajectory_collate_fn, num_workers=0,
        )

        # Train with perception model trainer
        ckpt_name = f"checkpoints/ablation_{variant_name.lower()}.pt"
        train_result = train_perception_model(
            model, train_scene_loader, epochs, lr, device,
            checkpoint_path=resume_from or ckpt_name,
            segment_epochs=segment_epochs,
        )

        if train_result.get("status") == "segment_complete":
            logger.info(f"  Segment done at epoch {train_result['next_epoch']}/{epochs}")
            logger.info(f"  Resume with: --resume {ckpt_name} --segment {segment_epochs}")
            continue

        # Evaluate
        metrics = evaluate_trajectory(model, test_loader, device, crosswalk_polygons=crosswalk_polygons, pix_per_meter=pix_per_meter)
        results[variant_name] = metrics
        logger.info(
            f"  {variant_name}: ADE={metrics['ADE']:.4f}, "
            f"FDE={metrics['FDE']:.4f}, NLL={metrics['NLL']:.4f}"
        )

        # Free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Export
    if results:
        export_trajectory_results_csv(results, "ablation_study_results.csv")

        # Compute delta vs FullModel
        if "FullModel" in results:
            logger.info("\n--- Ablation Impact (Δ vs FullModel) ---")
            full = results["FullModel"]
            for name, metrics in results.items():
                if name == "FullModel":
                    continue
                d_ade = metrics["ADE"] - full["ADE"]
                d_fde = metrics["FDE"] - full["FDE"]
                d_nll = metrics["NLL"] - full["NLL"]
                logger.info(
                    f"  {name:15s}  ΔADE={d_ade:+.2f}  "
                    f"ΔFDE={d_fde:+.2f}  ΔNLL={d_nll:+.2f}"
                )

    return results


# ======================================================================
# Data loading with date split
# ======================================================================

def load_split_datasets(
    processed_dir: str,
    label_dir: str = "labels",
    obs_len: int = 8,
    pred_len: int = 12,
    quick: bool = False,
    filter_candidates: bool = False,
    crosswalk_roi: list = None,
    stop_line: list = None,
    junction_roi: list = None,
):
    """Load trajectory and scene datasets, split by date.

    Parameters
    ----------
    filter_candidates : bool
        If True, filter scene subsets to only crossing-candidate pedestrians
        (GT enters junction OR heading 80-90° to stop line).
        Requires: crosswalk_roi, stop_line, junction_roi.
    """
    # Trajectory-only dataset (for baselines)
    ds = TrajectoryDataset(
        data_dir=processed_dir,
        obs_len=obs_len, pred_len=pred_len,
        stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="trajectory_only",
    )
    logger.info(f"Trajectory-only: {len(ds)} samples")

    # Scene-enabled dataset (for OurMethod)
    ds_scene = TrajectoryDataset(
        data_dir=processed_dir,
        label_dir=label_dir,
        obs_len=obs_len, pred_len=pred_len,
        stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="with_scene",
        max_scene_samples=3000 if quick else 10000,
    )
    scene_indices = ds_scene.with_scene_subset()
    logger.info(f"Scene-enabled: {len(ds_scene)} total, {len(scene_indices)} with scene data")

    # Split by date
    def _filter_by_date(dataset, dates, sample_list=None):
        indices = []
        src = sample_list if sample_list is not None else range(len(dataset.samples))
        for i in src:
            try:
                s = dataset.samples[i]
            except IndexError:
                continue
            video = s.get("video", "")
            if any(d.replace("_", "") in video for d in dates):
                indices.append(i)
        return indices

    train_idx = _filter_by_date(ds, TRAIN_DATES)
    val_idx = _filter_by_date(ds, VAL_DATES)
    test_idx = _filter_by_date(ds, TEST_DATES)

    train_scene_idx = _filter_by_date(ds_scene, TRAIN_DATES, scene_indices)
    val_scene_idx = _filter_by_date(ds_scene, VAL_DATES, scene_indices)
    test_scene_idx = _filter_by_date(ds_scene, TEST_DATES, scene_indices)

    logger.info(f"Split trajectory: T={len(train_idx)} V={len(val_idx)} Te={len(test_idx)}")
    logger.info(f"Split scene:      T={len(train_scene_idx)} V={len(val_scene_idx)} Te={len(test_scene_idx)}")

    if quick:
        train_idx = train_idx[:500]; val_idx = val_idx[:100]; test_idx = test_idx[:100]
        train_scene_idx = train_scene_idx[:200]; val_scene_idx = val_scene_idx[:50]; test_scene_idx = test_scene_idx[:50]

    # Cap
    max_t = 100000; max_v = 10000; max_te = 10000
    train_idx = train_idx[:max_t]; val_idx = val_idx[:max_v]; test_idx = test_idx[:max_te]

    # ── Optional: filter scene subsets to crossing candidates only ──
    if filter_candidates:
        from data.dataset import is_crossing_candidate as _icc

        def _filter_scene(indices, use_future_gt, name):
            kept = []
            for idx in indices:
                s = ds_scene.samples[idx]
                obs = s["obs_positions"]
                tgt = s.get("target_positions") if use_future_gt else None
                if _icc(obs, tgt, crosswalk_roi, stop_line, junction_roi):
                    kept.append(idx)
            logger.info(f"  {name} filtered: {len(kept)}/{len(indices)} kept "
                        f"({100*len(kept)/max(1,len(indices)):.1f}%)")
            return kept

        train_scene_idx = _filter_scene(train_scene_idx, True, "train")
        val_scene_idx = _filter_scene(val_scene_idx, True, "val")
        test_scene_idx = _filter_scene(test_scene_idx, True, "test")

    return (
        Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx),
        Subset(ds_scene, train_scene_idx), Subset(ds_scene, val_scene_idx), Subset(ds_scene, test_scene_idx),
    )


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="运行对比实验和消融实验")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default="data/processed/trajectories/")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--exp", default="all",
                        choices=["all", "trajectory", "classification", "ablation"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (少量数据 + 少量epoch)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip", default="", help="跳过指定模型 (逗号分隔, 如 social-lstm,stgcnn)")
    parser.add_argument("--segment", type=int, default=0,
                        help="分段训练: 每N个epoch保存checkpoint并退出 (0=不分段)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从checkpoint恢复训练")
    parser.add_argument("--crosswalk-filter", action="store_true",
                        help="只评估穿越斑马线的轨迹")
    parser.add_argument("--pix-per-meter", type=float, default=None,
                        help="像素/米 换算比例，用于输出以米为单位的ADE/FDE")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已保存的checkpoint，不训练")
    parser.add_argument("--variant", type=str, default=None,
                        help="消融实验: 只跑指定变体 (FullModel/NoGraph/NoMemory/NoCogContext/NoFlowChain/NoChange)")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"
    # pix_per_meter: CLI arg takes priority, else config
    if args.pix_per_meter is None:
        args.pix_per_meter = config.get("video", {}).get("pix_per_meter", None)
    skip_models = set(m.strip().lower() for m in args.skip.split(",") if m.strip())
    if skip_models:
        logger.info(f"Skipping models: {skip_models}")
    logger.info(f"Device: {device}")

    # Load data
    train_set, val_set, test_set, \
        train_scene, val_scene, test_scene = load_split_datasets(
        args.processed_dir, label_dir=args.label_dir, quick=args.quick,
    )

    # Run experiments
    all_results = {}

    if args.exp in ("all", "trajectory"):
        traj_results = run_trajectory_experiment(
            train_set, val_set, test_set,
            train_scene, val_scene, test_scene,
            config, device, args.quick,
            epochs_override=args.epochs,
            skip_models=skip_models,
            resume_from=args.resume,
            segment_epochs=args.segment,
            crosswalk_filter=args.crosswalk_filter,
            pix_per_meter=args.pix_per_meter,
            eval_only=args.eval_only,
        )
        all_results["trajectory"] = traj_results

    if args.exp in ("all", "classification"):
        cls_results = run_classification_experiment(
            train_scene, val_scene, test_scene, config, device, args.quick,
            epochs_override=args.epochs,
            skip_models=skip_models,
        )
        all_results["classification"] = cls_results

    if args.exp in ("all", "ablation"):
        abl_results = run_ablation_experiment(
            train_set, val_set, test_set,
            train_scene, val_scene, test_scene,
            config, device, args.quick,
            epochs_override=args.epochs,
            resume_from=args.resume,
            segment_epochs=args.segment,
            variant=args.variant,
        )
        all_results["ablation"] = abl_results

    # Save all results
    with open("experiment_results.json", "w") as f:
        # Convert to serializable
        serializable = {}
        for exp_name, exp_data in all_results.items():
            serializable[exp_name] = {
                method: {k: float(v) for k, v in metrics.items()}
                for method, metrics in exp_data.items()
            }
        json.dump(serializable, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("All experiments complete!")
    logger.info("Results saved to:")
    logger.info("  trajectory_prediction_results.csv")
    logger.info("  red_light_prediction_results.csv")
    logger.info("  ablation_study_results.csv")
    logger.info("  experiment_results.json")

    # Print summary table
    for exp_name, exp_data in all_results.items():
        print(f"\n--- {exp_name} ---")
        for method, metrics in exp_data.items():
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {method:20s} {metric_str}")


if __name__ == "__main__":
    main()
