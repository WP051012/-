"""
Full loss assembly.

    L = λ_pseudo·L_pseudo + λ_cvp·L_cvp_cycle + λ_cycle·L_cycle
        + λ_corr·L_corr + λ_temporal·L_temporal

All weights come from the config ``loss:`` block. A term is only computed when
its weight is > 0 *and* the required batch fields are present, so the same
function drives the 3 modes and ablations A0–A5 via config flags only (no code
branches to delete).
"""

from __future__ import annotations

import torch

from .pseudo_bev_loss import pseudo_bev_loss
from .cycle_loss import cvp_cycle_loss, camera_bev_cycle_loss
from .correspondence_loss import correspondence_loss
from .temporal_loss import temporal_velocity_loss


def compute_losses(outputs: dict, batch: dict, loss_cfg: dict, model=None) -> dict:
    """Compute all configured losses.

    Parameters
    ----------
    outputs : dict — model forward() result with keys
        ``pred_logits``, ``pred_bev``, ``F_cam``, ``F_cam_rec``, (``pred_cam``).
    batch : dict — with (optional) keys
        ``pseudo_bev``, ``camera_mask``,
        ``pseudo_bev_prev``, ``pred_bev_prev`` (for temporal).
    loss_cfg : dict — ``loss:`` config block with weights + modes.
    model : MonocularBEV | None — needed to warp pred_bev back to camera for
        the cycle loss if ``pred_cam`` was not precomputed in forward.

    Returns
    -------
    dict of per-term tensors plus ``total`` (a scalar tensor, always present).
    """
    losses = {}
    terms = []

    pred_logits = outputs["pred_logits"]
    pred_bev = outputs["pred_bev"]
    F_cam = outputs["F_cam"]
    F_cam_rec = outputs["F_cam_rec"]

    zero = pred_logits.sum() * 0.0  # scalar tensor tied to the graph

    # -- L_pseudo ----------------------------------------------------------
    w = float(loss_cfg.get("pseudo_weight", 0.0))
    if w > 0 and "pseudo_bev" in batch:
        l = pseudo_bev_loss(
            pred_logits, batch["pseudo_bev"],
            mode=loss_cfg.get("pseudo_mode", "focal"),
            alpha=float(loss_cfg.get("focal_alpha", 0.25)),
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        )
        losses["L_pseudo"] = l
        terms.append(w * l)

    # -- L_cvp_cycle -------------------------------------------------------
    w = float(loss_cfg.get("cvp_cycle_weight", 0.0))
    if w > 0:
        l = cvp_cycle_loss(F_cam, F_cam_rec)
        losses["L_cvp_cycle"] = l
        terms.append(w * l)

    # -- L_cycle -----------------------------------------------------------
    w = float(loss_cfg.get("cycle_weight", 0.0))
    if w > 0 and "camera_mask" in batch:
        pred_cam = outputs.get("pred_cam")
        if pred_cam is None and model is not None and model.camera_bev_proj is not None:
            pred_cam = model.camera_bev_proj.bev_to_camera(pred_bev)
        if pred_cam is not None:
            l = camera_bev_cycle_loss(
                batch["camera_mask"], pred_cam,
                dice_weight=float(loss_cfg.get("cycle_dice_weight", 1.0)),
            )
            losses["L_cycle"] = l
            terms.append(w * l)

    # -- L_corr ------------------------------------------------------------
    w = float(loss_cfg.get("corr_weight", 0.0))
    if w > 0 and "pseudo_bev" in batch:
        l = correspondence_loss(pred_bev, batch["pseudo_bev"])
        losses["L_corr"] = l
        terms.append(w * l)

    # -- L_temporal --------------------------------------------------------
    w = float(loss_cfg.get("temporal_weight", 0.0))
    if w > 0 and "pseudo_bev_prev" in batch and "pred_bev_prev" in batch:
        l = temporal_velocity_loss(
            pred_bev, batch["pred_bev_prev"],
            batch["pseudo_bev"], batch["pseudo_bev_prev"],
        )
        losses["L_temporal"] = l
        terms.append(w * l)

    losses["total"] = sum(terms) if terms else zero
    return losses
