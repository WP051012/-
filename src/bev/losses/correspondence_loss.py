"""
L_corr — object-position correspondence between pred and pseudo BEV.

We derive a differentiable object position from each heatmap via soft-argmax
(expected cell), then minimise ||p_pred - p_pseudo||_2 (normalised grid coords).
Frames/channels with no object are masked out so empty scenes contribute no
signal rather than a spurious 0/0 position.
"""

from __future__ import annotations

import torch


def soft_expected_position(heat):
    """Soft-argmax expected (x, y) position per channel.

    Parameters
    ----------
    heat : (B, C, H, W) non-negative heatmap.

    Returns
    -------
    (x_norm, y_norm, mass) : each (B, C); x_norm, y_norm ∈ [0, 1].
    """
    B, C, H, W = heat.shape
    ys = torch.linspace(0.0, 1.0, H, device=heat.device)
    xs = torch.linspace(0.0, 1.0, W, device=heat.device)
    mass = heat.sum(dim=(2, 3))
    y_hat = (heat * ys.view(1, 1, H, 1)).sum(dim=(2, 3))
    x_hat = (heat * xs.view(1, 1, 1, W)).sum(dim=(2, 3))
    denom = mass.clamp_min(1e-6)
    return x_hat / denom, y_hat / denom, mass


def correspondence_loss(pred_bev_prob, pseudo_bev):
    """Object-position L2 correspondence, masked to present objects."""
    xp, yp, _ = soft_expected_position(pred_bev_prob)
    xg, yg, mg = soft_expected_position(pseudo_bev)
    mask = (mg > 1e-3).float()                     # (B, C)
    err = (xp - xg) ** 2 + (yp - yg) ** 2          # (B, C)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)
