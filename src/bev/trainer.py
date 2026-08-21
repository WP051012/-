"""
Training / validation loop for MonocularBEV.

Implements the research training pipeline: AMP (mixed precision), gradient
accumulation, gradient clipping, warmup + cosine LR, NaN/Inf guards, and a
"prevent trivial solution" activation check. Memory is controlled by AMP /
grad-accum / grad-checkpointing — never by lowering the BEV resolution.
"""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import compute_losses, resolve_loss_cfg
from .metrics import compute_bev_metrics, activation_stats

logger = logging.getLogger(__name__)


# -- AMP helpers (torch.amp preferred; torch.cuda.amp fallback for old torch) --

def _use_amp(config: dict, device) -> bool:
    return bool(config.get("use_amp", False)) and device.type == "cuda"


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, config: dict, steps_per_epoch: int, epochs: int):
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(config.get("warmup_epochs", 0)) * steps_per_epoch)

    def lr_lambda(step: int):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BEVTrainer:
    def __init__(self, model: nn.Module, config: dict, device):
        self.model = model
        self.config = config
        self.device = device
        self.temporal = bool(config.get("data", {}).get("bev", {}).get("temporal", False))
        self.loss_cfg = resolve_loss_cfg(config)

    # -- forward + loss -----------------------------------------------------

    def _to_device(self, batch: dict, key: str):
        v = batch.get(key)
        return v.to(self.device) if isinstance(v, torch.Tensor) else v

    def train_step(self, batch: dict) -> dict:
        model = self.model
        image = batch["image"].to(self.device)
        out = model(image, return_cycle=True)

        if self.temporal:
            with torch.no_grad():
                out_prev = model(batch["image_prev"].to(self.device))
                batch["pred_bev_prev"] = out_prev["pred_bev"].detach()

        batch_dev = {
            "pseudo_bev": batch["pseudo_bev"].to(self.device),
            "camera_mask": batch["camera_mask"].to(self.device),
        }
        if self.temporal:
            batch_dev["pseudo_bev_prev"] = batch["pseudo_bev_prev"].to(self.device)
            batch_dev["pred_bev_prev"] = batch["pred_bev_prev"]

        return compute_losses(out, batch_dev, self.loss_cfg, model=model)

    # -- training epoch -----------------------------------------------------

    def train_epoch(self, loader, optimizer, scaler, grad_accum, grad_clip, scheduler):
        model = self.model
        model.train()
        use_amp = _use_amp(self.config, self.device)

        total_loss = 0.0
        n = 0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            with _autocast(use_amp):
                losses = self.train_step(batch)
                loss = losses["total"] / max(1, grad_accum)

            scaler.scale(loss).backward()

            is_last = (i == len(loader) - 1)
            if (i + 1) % grad_accum == 0 or is_last:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if not torch.isfinite(losses["total"]):
                raise RuntimeError("NaN/Inf loss detected during training — aborting")

            total_loss += losses["total"].detach().item()
            n += 1

        return total_loss / max(1, n)

    # -- validation ---------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader, resolution=None):
        model = self.model
        model.eval()
        use_amp = _use_amp(self.config, self.device)

        agg = {}
        n = 0
        act_sum = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        for batch in loader:
            image = batch["image"].to(self.device)
            with _autocast(use_amp):
                out = model(image)
            pred = out["pred_bev"]
            pseudo = batch["pseudo_bev"].to(self.device)

            m = compute_bev_metrics(pred, pseudo, resolution)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + (v if v == v else 0.0)  # NaN → skip
            stats = activation_stats(pred)
            for k, v in stats.items():
                act_sum[k] += v
            n += 1

        if n == 0:
            return {"loss": float("nan")}, {}

        metrics = {k: v / n for k, v in agg.items()}
        act_stats = {k: v / n for k, v in act_sum.items()}
        return metrics, act_stats

    # -- fit ----------------------------------------------------------------

    def fit(self, train_loader, val_loader, epochs: int, resolution=None):
        config = self.config
        optimizer = build_optimizer(self.model, config)
        scheduler = build_scheduler(optimizer, config, len(train_loader), epochs)
        scaler = _grad_scaler(_use_amp(config, self.device))
        grad_accum = max(1, int(config.get("grad_accum_steps", 1)))
        grad_clip = float(config.get("gradient_clip", 0.0))

        best = float("inf")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(
                train_loader, optimizer, scaler, grad_accum, grad_clip, scheduler
            )
            metrics, act_stats = self.validate(val_loader, resolution)
            dt = time.time() - t0

            logger.info(
                f"[epoch {epoch:3d}/{epochs}] train_loss={train_loss:.4f} "
                f"val_mse={metrics.get('mse', float('nan')):.4f} "
                f"val_pos_err={metrics.get('pos_err_cells', float('nan')):.3f} cells "
                f"({dt:.1f}s)"
            )

            # Prevent trivial solutions: warn if the predicted heatmap collapses.
            if act_stats.get("std", 0.0) < 1e-3:
                logger.warning(
                    "BEV activation std is near zero — model may be collapsing to a "
                    "trivial (constant) prediction. Check loss weights / pseudo-BEV "
                    "sparsity."
                )

            if metrics.get("mse", float("inf")) < best:
                best = metrics["mse"]
                if config.get("save_best", True):
                    self.save(config.get("checkpoint_dir", "checkpoints/bev/"),
                              tag="best")

        return best

    # -- checkpoint ---------------------------------------------------------

    def save(self, checkpoint_dir, tag: str = "last"):
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{tag}.pt"
        torch.save({"model_state": self.model.state_dict(),
                    "config": self.config}, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded checkpoint from {path}")
