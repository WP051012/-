"""
L_pseudo — the weak-supervision loss between predicted BEV and *pseudo*-BEV.

The target is the homography+detection derived ``pseudo_bev`` (never GT). A
Gaussian-center heatmap is sparse, so plain BCE is heavily class-imbalanced;
we provide focal loss and BCE+Dice / focal+Dice variants and expose the choice
through config (``loss.pseudo_mode``).

The predicted input is *logits* (decoder output); the target is a [0,1]
heatmap. sigmoid is applied inside.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss_with_logits(logits, target, alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss on logits. target ∈ [0, 1]."""
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)          # prob of the correct class
    focal_weight = (1.0 - pt) ** gamma
    if alpha is not None:
        at = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = focal_weight * at
    return (ce * focal_weight).mean()


def dice_loss(prob, target, eps: float = 1e-6):
    """Soft Dice loss. prob, target ∈ [0, 1]."""
    num = 2.0 * (prob * target).sum() + eps
    den = prob.sum() + target.sum() + eps
    return 1.0 - num / den


def pseudo_bev_loss(pred_logits, pseudo_bev, mode: str = "focal",
                    alpha: float = 0.25, gamma: float = 2.0,
                    dice_weight: float = 1.0):
    """Weak-supervision loss on the BEV heatmap.

    Parameters
    ----------
    pred_logits : (B, C, H, W) decoder logits.
    pseudo_bev : (B, C, H, W) pseudo heatmap in [0, 1].
    mode : "bce" | "focal" | "dice" | "bce_dice" | "focal_dice".
    """
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(pred_logits, pseudo_bev)
    if mode == "focal":
        return focal_loss_with_logits(pred_logits, pseudo_bev, alpha, gamma)
    if mode == "dice":
        return dice_loss(torch.sigmoid(pred_logits), pseudo_bev)
    if mode == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(pred_logits, pseudo_bev)
        return bce + dice_weight * dice_loss(torch.sigmoid(pred_logits), pseudo_bev)
    if mode == "focal_dice":
        f = focal_loss_with_logits(pred_logits, pseudo_bev, alpha, gamma)
        return f + dice_weight * dice_loss(torch.sigmoid(pred_logits), pseudo_bev)
    raise ValueError(f"unknown pseudo_bev loss mode: {mode}")
