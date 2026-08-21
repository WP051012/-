"""
BEV evaluation metrics — *pseudo*-supervised (pred vs pseudo_bev, never GT).

Because we have no BEV ground truth, all metrics measure agreement with the
homography+detection derived ``pseudo_bev``. They are diagnostic, not an oracle:

    mse            : heatmap mean-squared error
    pos_err_cells  : soft-argmax peak localisation error (grid cells)
    pos_err_meters : the above × resolution
    peak_hit@k     : fraction of active channels whose argmax lands within k cells
    active         : number of channels with a present object

Also exposes activation statistics for the "prevent trivial solution" check.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _argmax_2d(heat):
    """(B, C, H, W) → (row, col) argmax per channel, as float tensors (B, C)."""
    B, C, H, W = heat.shape
    flat = heat.reshape(B, C, -1)
    idx = flat.argmax(dim=-1)
    return (idx // W).float(), (idx % W).float()


def activation_stats(pred_bev):
    """Mean/std/min/max of a [0,1] BEV heatmap (batch-averaged)."""
    return {
        "mean": float(pred_bev.mean()),
        "std": float(pred_bev.std()),
        "min": float(pred_bev.min()),
        "max": float(pred_bev.max()),
    }


def compute_bev_metrics(pred_bev, pseudo_bev, resolution=None):
    """Compare predicted BEV heatmap against pseudo-BEV heatmap.

    Parameters
    ----------
    pred_bev : (B, C, H, W) tensor in [0, 1].
    pseudo_bev : (B, C, H, W) tensor in [0, 1].
    resolution : float | None — meters per cell (to report position error in m).

    Returns
    -------
    dict of scalar metrics (NaN where undefined).
    """
    mse = float(F.mse_loss(pred_bev, pseudo_bev))

    mass = pseudo_bev.sum(dim=(2, 3))          # (B, C)
    mask = mass > 1e-3
    active = int(mask.sum().item())

    pr, pc = _argmax_2d(pred_bev)
    tr, tc = _argmax_2d(pseudo_bev)
    dist = torch.sqrt((pr - tr) ** 2 + (pc - tc) ** 2)   # (B, C) cells

    if active > 0:
        pos_err_cells = float(dist[mask].mean())
        peak_hit3 = float((dist[mask] <= 3).float().mean())
    else:
        pos_err_cells = float("nan")
        peak_hit3 = float("nan")

    out = {
        "mse": mse,
        "pos_err_cells": pos_err_cells,
        "peak_hit@3": peak_hit3,
        "active": active,
    }
    if resolution is not None and not math.isnan(pos_err_cells):
        out["pos_err_meters"] = pos_err_cells * float(resolution)
    return out
