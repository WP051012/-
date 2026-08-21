"""
Cycle-consistency losses.

Two distinct cycles (kept separate per spec):

    L_cvp_cycle : feature-level cycle inside CVP
                  ||F_cam - F_cam_reconstructed||_1  (CVP forward → backward)

    L_cycle     : camera ⇄ BEV mask cycle (Yan-style)
                  camera  →(H)→  BEV  →(H⁻¹)→  camera
                  BCE + λ_dice·Dice(warped_back(pred_bev), pseudo_camera_mask)

The camera mask used here is the *pseudo* camera mask rasterised from detector
boxes — it is not ground truth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pseudo_bev_loss import dice_loss


def cvp_cycle_loss(F_cam, F_cam_rec):
    """Feature-level cycle: L1 between camera feature and its reconstruction."""
    return (F_cam - F_cam_rec).abs().mean()


def camera_bev_cycle_loss(M_cam, M_cam_rec, dice_weight: float = 1.0):
    """Camera ⇄ BEV cycle on [0,1] masks.

    Parameters
    ----------
    M_cam : (B, C, mask_h, mask_w) pseudo camera mask.
    M_cam_rec : (B, C, mask_h, mask_w) pred_bev warped back to camera.
    """
    bce = F.binary_cross_entropy(M_cam_rec, M_cam)
    return bce + dice_weight * dice_loss(M_cam_rec, M_cam)
