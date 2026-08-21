"""
Loss modules for the monocular-BEV model.

    pseudo_bev_loss.py    : L_pseudo (focal / BCE / Dice on pseudo heatmap)
    cycle_loss.py         : L_cvp_cycle (feature cycle) + L_cycle (camera⇄BEV)
    correspondence_loss.py: L_corr (soft-argmax object-position L2)
    temporal_loss.py      : L_temporal (velocity consistency)
    aggregate.py          : compute_losses — weighted sum from config
"""

from .pseudo_bev_loss import (
    pseudo_bev_loss,
    focal_loss_with_logits,
    dice_loss,
)
from .cycle_loss import cvp_cycle_loss, camera_bev_cycle_loss
from .correspondence_loss import correspondence_loss, soft_expected_position
from .temporal_loss import temporal_velocity_loss, temporal_acceleration_loss
from .aggregate import compute_losses
from .presets import (
    resolve_loss_cfg,
    mode_uses_network,
    PROPOSED_LOSS,
    ABLATION_DISABLED,
)

__all__ = [
    "pseudo_bev_loss",
    "focal_loss_with_logits",
    "dice_loss",
    "cvp_cycle_loss",
    "camera_bev_cycle_loss",
    "correspondence_loss",
    "soft_expected_position",
    "temporal_velocity_loss",
    "temporal_acceleration_loss",
    "compute_losses",
    "resolve_loss_cfg",
    "mode_uses_network",
    "PROPOSED_LOSS",
    "ABLATION_DISABLED",
]
