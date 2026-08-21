"""
L_temporal — temporal consistency across consecutive frames.

Following the spec's velocity formulation:

        v_t = p_t - p_{t-1}
        L_velocity = ||v_pred - v_pseudo||_2

where p are object positions (soft-argmax expected cell). Only channels where
the object is present in *both* frames contribute, so a disappearing/appearing
object does not inject a spurious velocity.

The training loop must supply the previous frame's prediction and pseudo-BEV;
see ``scripts/train_bev.py`` for the consecutive-frame sampling.
"""

from __future__ import annotations

from .correspondence_loss import soft_expected_position


def temporal_velocity_loss(pred_bev_t, pred_bev_tm1, pseudo_bev_t, pseudo_bev_tm1):
    """Velocity-consistency loss between frame t and t-1.

    Parameters
    ----------
    pred_bev_t / pred_bev_tm1 : (B, C, H, W) predicted BEV heatmaps.
    pseudo_bev_t / pseudo_bev_tm1 : (B, C, H, W) pseudo BEV heatmaps.
    """
    xp_t, yp_t, _ = soft_expected_position(pred_bev_t)
    xp_p, yp_p, _ = soft_expected_position(pred_bev_tm1)
    xg_t, yg_t, mg_t = soft_expected_position(pseudo_bev_t)
    xg_p, yg_p, mg_p = soft_expected_position(pseudo_bev_tm1)

    v_pred_x = xp_t - xp_p
    v_pred_y = yp_t - yp_p
    v_pseu_x = xg_t - xg_p
    v_pseu_y = yg_t - yg_p

    mask = ((mg_t > 1e-3) & (mg_p > 1e-3)).float()
    err = (v_pred_x - v_pseu_x) ** 2 + (v_pred_y - v_pseu_y) ** 2
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def temporal_acceleration_loss(pred_positions, pseudo_positions):
    """Second-order smoothness across a sequence of expected positions.

    Parameters
    ----------
    pred_positions / pseudo_positions : list of (B, C, 2) positions over time.
        Each element is a stack of (x_norm, y_norm) per channel.
    """
    if len(pred_positions) < 3:
        return pred_positions[0].new_zeros(())
    total = pred_positions[0].new_zeros(())
    n = 0
    for i in range(1, len(pred_positions) - 1):
        a_pred = pred_positions[i + 1] - 2 * pred_positions[i] + pred_positions[i - 1]
        a_pseu = pseudo_positions[i + 1] - 2 * pseudo_positions[i] + pseudo_positions[i - 1]
        total = total + ((a_pred - a_pseu) ** 2).mean()
        n += 1
    return total / max(n, 1)
