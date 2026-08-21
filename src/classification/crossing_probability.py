"""
Crossing Probability Estimator — Stage 1 of the two-stage risk framework.

Converts FlowChain trajectory distribution samples into P_cross:
    P_cross = fraction of sampled trajectories that enter the crossing region.

This is a PURE GEOMETRIC computation — no trainable parameters, no
environmental constraints. It answers the question:

    "What is the probability this pedestrian will enter the road?"

The environmental constraints (traffic light, vehicles) are applied in
Stage 2 by the RiskRegressionHead.

References:
    Proposal: FlowChain-based Probabilistic Pedestrian Risk Estimation
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ======================================================================
# Point-in-polygon (ray-casting)
# ======================================================================

def point_in_polygon(x: float, y: float, polygon) -> bool:
    """
    Ray-casting test: is (x, y) inside the polygon?

    Parameters
    ----------
    x, y : float — pixel coordinates
    polygon : list of (x, y) tuples, or np.ndarray of shape (V, 2)

    Returns
    -------
    True if point is inside or on the edge.
    """
    if hasattr(polygon, 'shape'):
        # numpy array
        verts = [(float(polygon[i, 0]), float(polygon[i, 1]))
                 for i in range(polygon.shape[0])]
    else:
        verts = [(float(p[0]), float(p[1])) for p in polygon]

    n = len(verts)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside


# ======================================================================
# Crossing Probability Estimator
# ======================================================================

class CrossingProbabilityEstimator:
    """
    Computes P_cross from trajectory distribution samples.

    For N Monte Carlo trajectory samples, checks each sample for whether
    ANY frame enters the crossing_region polygon, then:

        P_cross = count(samples that enter) / N

    This is a stateless geometric computation with no trainable parameters.

    Parameters
    ----------
    crossing_region : list of (x, y) — polygon vertices in pixel coords.
                       Typically the crosswalk_roi or junction_roi from config.
    """

    def __init__(self, crossing_region=None):
        self.crossing_region = crossing_region

    def set_region(self, crossing_region):
        """Update the crossing region polygon."""
        self.crossing_region = crossing_region

    # ------------------------------------------------------------------
    # Single trajectory check
    # ------------------------------------------------------------------

    def trajectory_enters_region(self, trajectory: np.ndarray) -> bool:
        """
        Check if ANY frame of a single trajectory enters the crossing region.

        Parameters
        ----------
        trajectory : (T, 2) ndarray — predicted positions in pixel coords

        Returns
        -------
        True if any frame's (x, y) falls inside the polygon.
        """
        if self.crossing_region is None:
            return False

        for i in range(trajectory.shape[0]):
            x, y = float(trajectory[i, 0]), float(trajectory[i, 1])
            if point_in_polygon(x, y, self.crossing_region):
                return True
        return False

    # ------------------------------------------------------------------
    # P_cross from Monte Carlo samples
    # ------------------------------------------------------------------

    def compute_p_cross(
        self,
        trajectory_samples: Tensor,       # (N, T, 2) or (N, B, T, 2)
        log_probs: Optional[Tensor] = None,  # (N,) or (N, B) — not used in hard count
    ) -> Tensor:
        """
        Compute P_cross from trajectory samples.

        Parameters
        ----------
        trajectory_samples : Tensor
            FlowChain Monte Carlo samples. Shape (N, T, 2) for single sample,
            or (N, B, T, 2) for batched.
        log_probs : Tensor, optional
            Sample log-probabilities (not used in hard count, kept for
            interface compatibility with future probability-weighted versions).

        Returns
        -------
        p_cross : Tensor — scalar or (B,) shaped, values in [0, 1]
        """
        if self.crossing_region is None:
            if trajectory_samples.dim() == 4:
                B = trajectory_samples.shape[1]
                return torch.zeros(B, device=trajectory_samples.device)
            return torch.tensor(0.0, device=trajectory_samples.device)

        # Handle batch dimension
        if trajectory_samples.dim() == 4:
            # (N, B, T, 2)
            N, B = trajectory_samples.shape[:2]
            samples_np = trajectory_samples.detach().cpu().numpy()
            p_cross = torch.zeros(B, device=trajectory_samples.device)
            for b in range(B):
                count = 0
                for n in range(N):
                    if self.trajectory_enters_region(samples_np[n, b]):
                        count += 1
                p_cross[b] = count / N
            return p_cross
        else:
            # (N, T, 2)
            N = trajectory_samples.shape[0]
            samples_np = trajectory_samples.detach().cpu().numpy()
            count = 0
            for n in range(N):
                if self.trajectory_enters_region(samples_np[n]):
                    count += 1
            return torch.tensor(count / N, device=trajectory_samples.device)

    # ------------------------------------------------------------------
    # Per-sample detail (for debugging / analysis)
    # ------------------------------------------------------------------

    def compute_details(
        self,
        trajectory_samples: Tensor,
    ) -> dict:
        """
        Compute P_cross with per-sample entry flags.

        Returns
        -------
        dict with:
            "p_cross": float
            "n_samples": int
            "n_entered": int
            "entered": list of bool per sample
        """
        if self.crossing_region is None:
            N = trajectory_samples.shape[0]
            return {"p_cross": 0.0, "n_samples": N, "n_entered": 0,
                    "entered": [False] * N}

        samples_np = trajectory_samples.detach().cpu().numpy()
        if samples_np.ndim == 4:
            samples_np = samples_np[:, 0, :, :]  # take first batch

        N = samples_np.shape[0]
        entered = [self.trajectory_enters_region(samples_np[n]) for n in range(N)]
        n_entered = sum(entered)

        return {
            "p_cross": n_entered / N,
            "n_samples": N,
            "n_entered": n_entered,
            "entered": entered,
        }


# ======================================================================
# Signal factor computation
# ======================================================================

def compute_signal_factor(
    traffic_light_states: List[str],
    obs_len: int = 8,
) -> float:
    """
    Compute Signal_factor from per-frame traffic light states.

    Signal_factor ∈ {0.0, 1.0}:
        1.0 = red light during observation (pedestrian crossing would be a violation)
        0.0 = green light (pedestrian crossing is legal)

    Uses the LAST frame's state as the decision signal (what the pedestrian
    sees at the moment of decision). If any frame in the observation window
    is red, the signal is treated as red (conservative).

    Parameters
    ----------
    traffic_light_states : list of str
        Per-frame states: 'red', 'green', 'yellow', 'off', 'unknown'.
    obs_len : int
        Number of observation frames.

    Returns
    -------
    signal_factor : float — 1.0 for red, 0.0 for green, 0.5 for unknown.
    """
    if not traffic_light_states:
        return 0.5  # unknown → neutral

    # Use last frame as the decision signal
    last_state = traffic_light_states[-1] if len(traffic_light_states) > 0 else "unknown"

    if last_state == "red":
        return 1.0
    elif last_state == "green":
        return 0.0
    elif last_state == "yellow":
        return 0.7  # yellow → elevated risk but not full red
    else:
        return 0.5  # unknown/off → neutral
