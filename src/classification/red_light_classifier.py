"""
Red-light running classification from predicted trajectory distributions.

Converts FlowChain trajectory distribution samples into a binary
red-light-running prediction via:
    1. Monte Carlo trajectory sampling
    2. Per-sample violation checking against stop-line / junction criteria
    3. Violation probability aggregation
    4. Threshold-based binary classification

References:
    Paper Section 10: 闯红灯概率转换
    Chinese Road Traffic Safety Law, Article 38 & 90
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Geometric primitives
# ======================================================================

@dataclass
class StopLine:
    """A stop-line in the image / world plane."""
    x1: float
    y1: float
    x2: float
    y2: float

    def line_params(self) -> Tuple[float, float, float]:
        """Returns (A, B, C) for line equation Ax + By + C = 0."""
        A = self.y1 - self.y2
        B = self.x2 - self.x1
        C = self.x1 * self.y2 - self.x2 * self.y1
        return A, B, C

    def signed_distance(self, x: float, y: float) -> float:
        """Signed distance from point (x, y) to the stop line.
        Positive = beyond the line (in junction direction)."""
        A, B, C = self.line_params()
        return (A * x + B * y + C) / math.sqrt(A**2 + B**2)


@dataclass
class JunctionRegion:
    """
    Junction (intersection) region.
    Defined as a polygon or a bounding box beyond the stop line.
    """
    # Polygon vertices (x, y) defining the junction area.
    polygon: List[Tuple[float, float]]  # [(x1,y1), (x2,y2), ...]

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


# ======================================================================
# Red-Light Violation Checker
# ======================================================================

class RedLightViolationChecker:
    """
    Checks whether a given trajectory constitutes a red-light violation.

    Violation criteria (configurable):
        1. Traffic light is RED during the crossing
        2. Target crosses the STOP LINE while light is red
        3. Target ENTERS THE JUNCTION while light is red

    Parameters
    ----------
    stop_line : StopLine, optional
    junction : JunctionRegion, optional
    red_light_frames : tuple (start, end)
        Frame indices where the light is red.
    violation_distance_threshold : float
        Min distance past stop line to count as "crossed".
    min_junction_frames : int
        Min frames inside junction to count as "entered".
    """

    def __init__(
        self,
        stop_line: Optional[StopLine] = None,
        junction: Optional[JunctionRegion] = None,
        red_light_frames: Optional[Tuple[int, int]] = None,
        violation_distance_threshold: float = 2.0,
        min_junction_frames: int = 2,
    ):
        self.stop_line = stop_line
        self.junction = junction
        self.red_light_frames = red_light_frames  # (start, end) inclusive
        self.violation_dist_threshold = violation_distance_threshold
        self.min_junction_frames = min_junction_frames

    def is_red_light(self, frame_idx: int) -> bool:
        """Check if frame_idx occurs during red-light phase."""
        if self.red_light_frames is None:
            return True  # assume red if unknown
        start, end = self.red_light_frames
        return start <= frame_idx <= end

    def check_trajectory(
        self,
        trajectory: np.ndarray,       # (T, 2)  positions over time
        frame_indices: np.ndarray,    # (T,)    frame IDs
    ) -> Tuple[bool, float, Dict]:
        """
        Check a single trajectory for red-light violation.

        Returns
        -------
        violated : bool
        confidence : float
            Violation confidence ∈ [0, 1].
        details : dict
            Diagnostic info (crossed_line, entered_junction, etc.).
        """
        crossed_line = False
        entered_junction = False
        junction_frames = 0

        for i in range(len(trajectory)):
            x, y = trajectory[i]
            fi = int(frame_indices[i])

            if not self.is_red_light(fi):
                continue

            # Check stop line crossing
            if self.stop_line is not None and not crossed_line:
                dist = self.stop_line.signed_distance(x, y)
                if dist > self.violation_dist_threshold:
                    crossed_line = True

            # Check junction entry
            if self.junction is not None and not entered_junction:
                if self.junction.contains(x, y):
                    junction_frames += 1
                    if junction_frames >= self.min_junction_frames:
                        entered_junction = True

        # --- Violation logic ---
        if self.stop_line is not None and self.junction is not None:
            violated = crossed_line and entered_junction
        elif self.stop_line is not None:
            violated = crossed_line
        elif self.junction is not None:
            violated = entered_junction
        else:
            violated = False

        # Confidence: proportional to how far past the line
        confidence = 0.0
        if violated and self.stop_line is not None:
            max_dist = max(
                self.stop_line.signed_distance(x, y)
                for x, y in trajectory
            )
            confidence = min(1.0, max_dist / (self.violation_dist_threshold * 3))
        elif violated:
            confidence = 0.7  # default for junction-only

        details = {
            "crossed_stop_line": crossed_line,
            "entered_junction": entered_junction,
            "junction_frames": junction_frames,
            "max_distance_past_line": (
                max(self.stop_line.signed_distance(x, y) for x, y in trajectory)
                if self.stop_line else 0.0
            ),
        }

        return violated, confidence, details


# ======================================================================
# Monte Carlo Red-Light Probability Estimator
# ======================================================================

class RedLightProbabilityEstimator(nn.Module):
    """
    Estimates P(red-light violation) from trajectory distribution samples.

    For N Monte Carlo samples from FlowChain:
        P(violation) = (1/N) Σ 1[violation_check(sample_i)]

    The result is then thresholded to produce a binary prediction.

    Parameters
    ----------
    violation_checker : RedLightViolationChecker
    threshold : float
        Probability threshold for binary classification.
    """

    def __init__(
        self,
        violation_checker: Optional[RedLightViolationChecker] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.violation_checker = violation_checker or RedLightViolationChecker()
        self.threshold = threshold

        # Trainable threshold (sigmoid-bounded)
        self._logit_threshold = nn.Parameter(
            torch.tensor(math.log(threshold / (1 - threshold)))
        )

    @property
    def current_threshold(self) -> float:
        """Trainable threshold value."""
        return float(torch.sigmoid(self._logit_threshold))

    # ------------------------------------------------------------------
    # Monte Carlo estimation
    # ------------------------------------------------------------------

    def estimate_probability(
        self,
        trajectory_samples: torch.Tensor,    # (N, pred_len, 2)  or  (N, B, pred_len, 2)
        frame_indices: Optional[np.ndarray] = None,  # (pred_len,)  future frame IDs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate violation probability from trajectory samples.

        Parameters
        ----------
        trajectory_samples : Tensor
            Monte Carlo trajectory samples from FlowChain.
        frame_indices : np.ndarray, optional
            Frame indices for each prediction step.

        Returns
        -------
        prob : Tensor (scalar or (B,))
            Estimated violation probability.
        stats : dict
            Per-sample violation statistics.
        """
        if trajectory_samples.dim() == 4:
            # (N, B, pred_len, 2) → iterate batch
            N, B = trajectory_samples.shape[:2]
            probs = []
            for b in range(B):
                samples_b = trajectory_samples[:, b]  # (N, pred_len, 2)
                p, _ = self._estimate_single(samples_b, frame_indices)
                probs.append(p)
            prob = torch.stack(probs)
        else:
            prob, stats = self._estimate_single(trajectory_samples, frame_indices)
            return prob, stats

        return prob, {}

    def _estimate_single(
        self,
        samples: torch.Tensor,          # (N, pred_len, 2)
        frame_indices: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Estimate for a single batch element."""
        N = samples.size(0)
        samples_np = samples.detach().cpu().numpy()

        if frame_indices is None:
            frame_indices = np.arange(samples.size(1))

        violations = []
        confidences = []
        for n in range(N):
            violated, conf, _ = self.violation_checker.check_trajectory(
                samples_np[n], frame_indices,
            )
            violations.append(float(violated))
            confidences.append(conf)

        prob = torch.tensor(sum(violations) / N, device=samples.device)

        stats = {
            "num_violations": sum(violations),
            "num_samples": N,
            "mean_confidence": np.mean(confidences),
        }

        return prob, stats

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------

    def classify(
        self,
        trajectory_samples: torch.Tensor,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binary classification: will this trajectory violate the red light?

        Returns
        -------
        prediction : Tensor
            0 = no violation, 1 = violation.
        probability : Tensor
            Violation probability ∈ [0, 1].
        """
        prob, stats = self.estimate_probability(trajectory_samples, frame_indices)
        pred = (prob >= self.current_threshold).long()
        return pred, prob

    # ------------------------------------------------------------------
    # Training (threshold optimization)
    # ------------------------------------------------------------------

    def forward(
        self,
        trajectory_samples: torch.Tensor,   # (N, B, pred_len, 2)
        labels: Optional[torch.Tensor] = None,  # (B,)  ground-truth violations
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns
        -------
        dict with "probability", "prediction", and (if labels given) "loss".
        """
        prob, _ = self.estimate_probability(trajectory_samples)

        threshold = torch.sigmoid(self._logit_threshold)
        pred = (prob >= threshold).long()

        result = {
            "probability": prob,
            "prediction": pred,
            "threshold": threshold,
        }

        if labels is not None:
            # Binary cross-entropy with learnable threshold
            # Loss encourages: prob > threshold when label=1, prob < threshold when label=0
            labels_f = labels.float()
            loss = F.binary_cross_entropy(prob, labels_f)
            result["loss"] = loss

        return result


# ======================================================================
# Counterfactual simulation module (Paper Section 9)
# ======================================================================

class CounterfactualSimulator:
    """
    Counterfactual "what-if" simulation for traffic scenarios.

    Modifies traffic conditions and re-runs the prediction to see
    how the red-light violation probability changes.

    Example modifications:
        - Change vehicle speed
        - Change traffic light state (red → green)
        - Add / remove traffic participants

    References:
        Paper Section 9: 反事实模拟

    Parameters
    ----------
    predictor : callable
        Function that takes (obs_trajectory, perception_c) → samples.
    classifier : RedLightProbabilityEstimator
    """

    def __init__(
        self,
        predictor,
        classifier: RedLightProbabilityEstimator,
    ):
        self.predictor = predictor
        self.classifier = classifier

    def simulate(
        self,
        obs_trajectory: torch.Tensor,
        perception_c: torch.Tensor,
        modifications: List[Dict],
    ) -> List[Dict]:
        """
        Run counterfactual simulations.

        Parameters
        ----------
        obs_trajectory : Tensor
            Original observation trajectory.
        perception_c : Tensor
            Original perception vector.
        modifications : list of dict
            Each dict specifies modifications, e.g.:
                {"type": "traffic_light", "new_state": "green"}
                {"type": "vehicle_speed", "agent_idx": 2, "scale": 0.5}

        Returns
        -------
        list of dict with per-modification results.
        """
        results = []

        # Baseline
        base_pred = self.predictor(obs_trajectory, perception_c)
        base_prob, _ = self.classifier.estimate_probability(base_pred["samples"])
        results.append({
            "modification": "baseline",
            "violation_prob": float(base_prob),
        })

        for mod in modifications:
            c_modified = self._apply_modification(perception_c, mod)
            pred_mod = self.predictor(obs_trajectory, c_modified)
            prob_mod, _ = self.classifier.estimate_probability(pred_mod["samples"])
            results.append({
                "modification": mod,
                "violation_prob": float(prob_mod),
                "delta": float(prob_mod - base_prob),
            })

        return results

    @staticmethod
    def _apply_modification(
        c: torch.Tensor,
        modification: Dict,
    ) -> torch.Tensor:
        """
        Apply a modification to the perception vector.

        This is a simplified placeholder — real implementation would
        modify the perception graph before encoding to c.
        """
        c_modified = c.clone()
        mod_type = modification.get("type", "")

        if mod_type == "traffic_light":
            # Flip traffic light component in perception vector
            new_state = modification.get("new_state", "green")
            # Simplified: add perturbation in the environmental dimension
            noise_scale = 1.0 if new_state == "green" else -1.0
            c_modified = c_modified + noise_scale * 0.1 * torch.randn_like(c_modified)

        elif mod_type == "vehicle_speed":
            scale = modification.get("scale", 1.0)
            c_modified = c_modified * scale

        elif mod_type == "remove_agent":
            agent_idx = modification.get("agent_idx", 0)
            # Simplified masking
            c_modified = c_modified * 0.9

        return c_modified
