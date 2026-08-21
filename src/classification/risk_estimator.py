"""
Continuous risk-based red-light violation estimator.

Replaces hard geometric 0/1 judgment with:
  R_space   = sigmoid(-α · d_min)  — spatial risk (continuous)
  R_light   = traffic signal risk   — red/yellow/green
  R_interact = vehicle interaction  — distance-based (simplified)

  Risk(Y_i) = R_space × R_light × R_interact

  P_cross = Σ p(Y_i) × Risk(Y_i)    — probability-weighted integral

Then threshold search on validation set for optimal F1.

References:
    Paper Section 10 revision — continuous risk reasoning
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Geometric helpers
# ======================================================================

def _signed_dist_to_line(x: float, y: float, sl) -> float:
    """Signed distance from (x,y) to stop_line [x1,y1,x2,y2]. Positive = junction side."""
    A = float(sl[1]) - float(sl[3])
    B = float(sl[2]) - float(sl[0])
    C = float(sl[0]) * float(sl[3]) - float(sl[2]) * float(sl[1])
    return (A * x + B * y + C) / math.sqrt(A**2 + B**2 + 1e-8)


def _min_dist_to_polygon(x: float, y: float, polygon) -> float:
    """Minimum Euclidean distance from (x,y) to polygon vertices."""
    best = float("inf")
    for px, py in polygon:
        d = math.sqrt((x - px) ** 2 + (y - py) ** 2)
        if d < best:
            best = d
    return best


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside


# ======================================================================
# Continuous Risk Estimator
# ======================================================================

class ContinuousRiskEstimator(nn.Module):
    """
    Estimates P(red-light violation) via probability-weighted continuous risk.

    For each sampled trajectory Y_i with probability p_i:
        Risk(Y_i) = R_space × R_light × R_interact
        P_cross = Σ p_i × Risk(Y_i)

    Parameters
    ----------
    stop_line : [x1, y1, x2, y2] or None
    crosswalk_roi : [(x,y), ...] polygon vertices or None
    alpha : float — sigmoid steepness for spatial risk (higher = sharper)
    use_interaction : bool — enable vehicle interaction risk
    """

    def __init__(
        self,
        stop_line=None,
        crosswalk_roi=None,
        alpha: float = 0.02,
        use_interaction: bool = False,
        use_motion: bool = True,
    ):
        super().__init__()
        self.stop_line = stop_line
        self.crosswalk_roi = crosswalk_roi
        self.alpha = alpha
        self.use_interaction = use_interaction
        self.use_motion = use_motion

        # Threshold (can be overridden after search)
        self.register_buffer("threshold", torch.tensor(0.5))

        # Debug storage
        self.debug_risks: List[Dict] = []
        self.debug_probs: List[float] = []

    # ------------------------------------------------------------------
    # Risk components
    # ------------------------------------------------------------------

    def _spatial_risk(self, trajectory: np.ndarray) -> float:
        """
        R_space for a single trajectory.

        Uses sigmoid(-alpha * d_abs) where d_abs = minimum absolute distance
        to the stop_line. Closer → higher risk.

        Also boosts risk if any point crosses to junction side (d > 0).
        """
        if self.stop_line is None:
            return 0.0

        d_abs = float("inf")
        crossed = False
        for i in range(trajectory.shape[0]):
            d = _signed_dist_to_line(trajectory[i, 0], trajectory[i, 1], self.stop_line)
            d_abs = min(d_abs, abs(d))
            if d > 0:  # crossed to junction side
                crossed = True

        # sigmoid(-alpha * distance): closer → higher risk
        risk = 1.0 / (1.0 + math.exp(self.alpha * d_abs))

        # Boost if crossed the line
        if crossed:
            risk = min(1.0, risk + 0.3)

        return risk

    def _light_risk(self, light_state: str = "unknown") -> float:
        """R_light from traffic signal state. Always 1.0 (no light data available)."""
        return 1.0  # was 0.3 for unknown — constant multiplier, no info

    def _motion_risk(self, obs_trajectory: Optional[np.ndarray] = None) -> float:
        """
        R_motion as multiplicative modulation centered at 1.0.

        Accelerating → modulation > 1.0 (boosts spatial risk).
        Decelerating → modulation < 1.0 (reduces spatial risk).
        Constant speed → modulation ≈ 1.0 (neutral).

        Returns 1.0 if no obs data available.
        """
        if obs_trajectory is None or obs_trajectory.shape[0] < 3:
            return 1.0

        vel = np.diff(obs_trajectory, axis=0)          # (T-1, 2)
        speed = np.sqrt((vel ** 2).sum(axis=-1))        # (T-1,)

        if len(speed) < 3:
            return 1.0

        # Use all available speeds for more stable trend estimation
        x = np.arange(len(speed))
        slope = np.polyfit(x, speed, 1)[0]
        mean_speed = speed.mean() + 1e-8
        trend = slope / mean_speed  # positive = accelerating, negative = decelerating

        # sigmoid(8.0 * trend): accelerating → >0.5, decelerating → <0.5
        raw = 1.0 / (1.0 + math.exp(-8.0 * trend))
        # Modulate around 1.0: raw∈[0,1] → modulation∈[0.25, 1.75]
        modulation = 0.25 + 1.5 * raw
        return modulation

    def _interaction_risk(self, trajectory: np.ndarray,
                          scene_data: Optional[dict] = None) -> float:
        """
        R_interact from nearby vehicles. Simplified version:
        min distance to any vehicle bbox → sigmoid(-alpha * d_veh).

        Returns max risk over all frames.
        """
        if scene_data is None or not self.use_interaction:
            return 1.0  # no penalty

        bboxes = scene_data.get("bboxes")    # (obs_len, N, 4) or (N, 4)
        class_names = scene_data.get("class_names", [])

        if bboxes is None:
            return 1.0

        # Flatten to per-frame lists
        vehicle_classes = {"car", "bus", "truck", "bicycle", "motorcycle"}
        max_risk = 0.0

        for t in range(trajectory.shape[0]):
            tx, ty = trajectory[t, 0], trajectory[t, 1]
            for n in range(bboxes.shape[1] if bboxes.ndim >= 3 else bboxes.shape[0]):
                if bboxes.ndim >= 3:
                    bb = bboxes[min(t, bboxes.shape[0] - 1), n]
                else:
                    bb = bboxes[n]

                # Get class name
                cn = ""
                if isinstance(class_names, list) and len(class_names) > 0:
                    if isinstance(class_names[0], list):
                        row = class_names[min(t, len(class_names) - 1)]
                        cn = row[n] if n < len(row) else ""
                    elif n < len(class_names):
                        cn = class_names[n]

                if cn not in vehicle_classes:
                    continue

                cx = (float(bb[0]) + float(bb[2])) / 2
                cy = (float(bb[1]) + float(bb[3])) / 2
                d_veh = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                risk = 1.0 / (1.0 + math.exp(-0.01 * (50.0 - d_veh)))
                if risk > max_risk:
                    max_risk = risk

        return min(1.0, max_risk)

    # ------------------------------------------------------------------
    # Full risk for one trajectory
    # ------------------------------------------------------------------

    def _trajectory_risk(
        self,
        trajectory: np.ndarray,       # (T, 2)
        light_state: str = "unknown",
        scene_data: Optional[dict] = None,
        obs_trajectory: Optional[np.ndarray] = None,
    ) -> float:
        """Risk(Y_i) = R_space × [1 + proximity × (R_motion - 1)] × R_interact."""
        r = self._spatial_risk(trajectory)
        if self.use_motion:
            motion_mod = self._motion_risk(obs_trajectory)
            # Proximity weight: full modulation only when close to stop line
            # R_space ≥ 0.3 → weight=1.0; R_space < 0.3 → linear ramp
            proximity_weight = min(1.0, r / 0.3)
            effective_mod = 1.0 + (motion_mod - 1.0) * proximity_weight
            r *= effective_mod
        if self.use_interaction:
            r *= self._interaction_risk(trajectory, scene_data)
        return r

    # ------------------------------------------------------------------
    # Probability-weighted estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        trajectory_samples: Tensor,    # (N, B, T, 2) or (N, T, 2)
        log_probs: Optional[Tensor] = None,  # (N, B) or (N,)
        light_state: str = "unknown",
        scene_data: Optional[dict] = None,
        norm: Optional[Tensor] = None,  # for un-normalising coords
        obs_trajectory: Optional[np.ndarray] = None,  # (T, 2) pixel coords
    ) -> Tuple[Tensor, dict]:
        """
        Estimate P_cross via probability-weighted continuous risk.

        Returns (prob, stats_dict).
        """
        # Handle shapes
        if trajectory_samples.dim() == 4:
            N, B = trajectory_samples.shape[:2]
            probs = []
            for b in range(B):
                p, _ = self._estimate_single(
                    trajectory_samples[:, b], log_probs[:, b] if log_probs is not None else None,
                    light_state, scene_data, norm, obs_trajectory,
                )
                probs.append(p)
            return torch.stack(probs), {}
        else:
            return self._estimate_single(trajectory_samples, log_probs, light_state,
                                         scene_data, norm, obs_trajectory)

    def _estimate_single(self, samples, log_probs, light_state, scene_data, norm,
                         obs_trajectory=None):
        """Estimate for one batch element."""
        N = samples.shape[0]
        samples_np = samples.detach().cpu().numpy()

        # Un-normalise if norm provided
        if norm is not None:
            norm_np = norm.detach().cpu().numpy()
            samples_np = samples_np * norm_np

        # Convert log_probs → probabilities (softmax over samples)
        if log_probs is not None:
            lp = log_probs.detach().cpu()
            lp = lp - lp.max()  # numerical stability
            probs_np = torch.softmax(lp, dim=0).numpy()
        else:
            probs_np = np.ones(N) / N  # uniform

        # Compute risk per sample
        risks = []
        for n in range(N):
            r = self._trajectory_risk(samples_np[n], light_state, scene_data, obs_trajectory)
            risks.append(r)

        risks_np = np.array(risks)
        p_cross = float(np.sum(probs_np * risks_np))

        # Stats
        stats = {
            "mean_risk": float(np.mean(risks_np)),
            "max_risk": float(np.max(risks_np)),
            "weighted_risk": p_cross,
            "n_high_risk": int(np.sum(risks_np > 0.3)),
            "risks": risks_np.tolist(),
            "probs": probs_np.tolist(),
        }

        # Store debug info
        self.debug_risks.append(stats)
        self.debug_probs.append(p_cross)

        return torch.tensor(p_cross, device=samples.device), stats

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, trajectory_samples, log_probs=None, light_state="unknown",
                 scene_data=None, norm=None, obs_trajectory=None):
        """Binary classification with current threshold."""
        prob, stats = self.estimate(trajectory_samples, log_probs, light_state,
                                    scene_data, norm, obs_trajectory)
        pred = (prob >= self.threshold).long()
        return pred, prob, stats

    def reset_debug(self):
        """Clear debug buffers between evaluations."""
        self.debug_risks = []
        self.debug_probs = []

    def get_debug_summary(self) -> dict:
        """Return summary of debug info for analysis."""
        if not self.debug_probs:
            return {}
        arr = np.array(self.debug_probs)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "nonzero_frac": float(np.mean(arr > 1e-6)),
            "high_frac": float(np.mean(arr > 0.1)),
        }


# ======================================================================
# Threshold search
# ======================================================================

def search_threshold(
    estimator: ContinuousRiskEstimator,
    model,
    val_loader,
    device: str,
    norm: Optional[Tensor] = None,
    num_samples: int = 100,
) -> Tuple[float, float]:
    """
    Search best threshold on validation set for maximum F1.

    Returns (best_threshold, best_f1).
    """
    estimator.reset_debug()
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs_trajectory"].to(device) / norm.to(device) if norm is not None else batch["obs_trajectory"].to(device)

            pred = model(obs_trajectory=obs, num_samples=num_samples)
            samples = pred.get("samples")  # (N, B, T, 2)
            log_probs = pred.get("log_probs", None)

            if samples is None:
                continue

            # Compute per-batch P_cross (without classification — raw prob)
            for b in range(samples.shape[1]):
                prob, _ = estimator.estimate(
                    samples[:, b], log_probs[:, b] if log_probs is not None else None,
                    norm=norm,
                )
                all_probs.append(float(prob.item()))
                label = batch.get("is_violation")
                if isinstance(label, torch.Tensor):
                    all_labels.append(float(label[b].item() if label.numel() > b else label.item()))
                else:
                    all_labels.append(0.0)

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    pos_mask = labels == 1

    if pos_mask.sum() == 0:
        return 0.5, 0.0

    best_thresh, best_f1 = 0.5, 0.0
    for th in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= th).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((labels == 1) & (preds == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    return best_thresh, best_f1
