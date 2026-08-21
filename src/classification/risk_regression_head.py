"""
Risk Regression Head — Stage 2 of the two-stage risk estimation framework.

Architecture (per external AI proposal):
    Stage 1 (no training): FlowChain → P_cross via polygon check + Signal_factor
    Stage 2 (trainable):   Risk = MLP(P_cross, Vehicle_features) × Signal_factor

Key design decisions:
    1. FlowChain stays FROZEN — trajectory prediction quality preserved
    2. Risk head is a small MLP with SmoothL1 regression loss
    3. SmoothL1 gives continuous gradient signals to ALL samples (vs sparse BCE
       which only gives strong gradients near the decision boundary)
    4. Signal_factor gates the output: green light → Risk → 0
       (legal crossing ≠ violation, regardless of trajectory)

References:
    Proposal: FlowChain-based Probabilistic Pedestrian Risk Estimation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Reuse environment feature extraction from agent_centric_risk
from .agent_centric_risk import extract_environment_features
from .crossing_probability import compute_signal_factor


# ======================================================================
# Risk Regression Head
# ======================================================================

class RiskRegressionHead(nn.Module):
    """
    Trainable regression head for Stage 2 risk estimation.

    Input:  (P_cross, vehicle_features)  — Stage 1 outputs
    Output: Risk ∈ [0, 1]                — gated by Signal_factor

    Trained with SmoothL1 loss against binary violation labels.

    Parameters
    ----------
    env_dim : int — dimension of vehicle environment features (default 8)
    hidden_dim : int — hidden layer size
    dropout : float — dropout rate for regularization
    use_signal_gate : bool — multiply output by Signal_factor (default True)
    """

    def __init__(
        self,
        env_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        use_signal_gate: bool = True,
    ):
        super().__init__()
        self.env_dim = env_dim
        self.use_signal_gate = use_signal_gate

        # Input: P_cross(1) + env_feat(env_dim) = 1 + env_dim
        input_dim = 1 + env_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Per-feature projections for interpretability
        self.p_cross_proj = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
        )
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, 8),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Binary classification threshold
        self.register_buffer("threshold", torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        p_cross: Tensor,              # (B,) or scalar — Stage 1 crossing probability
        env_feat: Tensor,             # (B, env_dim) — vehicle environment features
        signal_factor: Optional[Tensor] = None,  # (B,) or scalar — red=1, green=0
    ) -> Dict[str, Tensor]:
        """
        Compute violation risk.

        Parameters
        ----------
        p_cross : Tensor — crossing probability from Stage 1, values in [0, 1]
        env_feat : Tensor — pre-extracted vehicle environment features
        signal_factor : Tensor, optional — traffic light signal factor

        Returns
        -------
        dict with:
            "risk": Tensor — final risk after signal gating, values in [0, 1]
            "risk_raw": Tensor — MLP output before gating
            "p_cross": Tensor — input P_cross (for logging)
        """
        # Ensure batch dimension
        if p_cross.dim() == 0:
            p_cross = p_cross.unsqueeze(0)
        if env_feat.dim() == 1:
            env_feat = env_feat.unsqueeze(0)

        # Per-feature projections → fused risk
        p_proj = self.p_cross_proj(p_cross.unsqueeze(-1))   # (B, 8)
        e_proj = self.env_proj(env_feat)                     # (B, 8)
        fused = torch.cat([p_proj, e_proj], dim=-1)         # (B, 16)
        risk_raw = self.fusion(fused).squeeze(-1)            # (B,)

        # Signal gate: green light → risk ≈ 0
        if self.use_signal_gate and signal_factor is not None:
            if signal_factor.dim() == 0:
                signal_factor = signal_factor.unsqueeze(0)
            risk = risk_raw * signal_factor
        else:
            risk = risk_raw

        return {
            "risk": risk,
            "risk_raw": risk_raw,
            "p_cross": p_cross,
        }

    # ------------------------------------------------------------------
    # Simplified forward (single MLP, no per-feature projections)
    # ------------------------------------------------------------------

    def forward_simple(
        self,
        p_cross: Tensor,
        env_feat: Tensor,
        signal_factor: Optional[Tensor] = None,
    ) -> Tensor:
        """Simplified forward: single MLP, returns risk scalar/vector directly."""
        if p_cross.dim() == 0:
            p_cross = p_cross.unsqueeze(0)
        if env_feat.dim() == 1:
            env_feat = env_feat.unsqueeze(0)

        x = torch.cat([p_cross.unsqueeze(-1), env_feat], dim=-1)
        risk_raw = self.mlp(x).squeeze(-1)

        if self.use_signal_gate and signal_factor is not None:
            if signal_factor.dim() == 0:
                signal_factor = signal_factor.unsqueeze(0)
            return risk_raw * signal_factor
        return risk_raw

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        risk: Tensor,
        labels: Tensor,
        beta: float = 1.0,
    ) -> Tensor:
        """
        SmoothL1 regression loss.

        SmoothL1(pred, target) gives:
        - L1 gradient for large errors (|pred - target| >= beta)
        - L2 gradient for small errors (|pred - target| < beta)

        This gives continuous gradient signals to ALL samples, addressing
        the extreme class imbalance (28/1010 = 2.8% positive).

        Parameters
        ----------
        risk : Tensor — predicted risk values in [0, 1]
        labels : Tensor — binary violation labels (0 or 1)
        beta : float — SmoothL1 transition point (default 1.0)

        Returns
        -------
        loss : scalar Tensor
        """
        # Ensure consistent shapes
        risk_v = risk.view(-1)
        labels_v = labels.float().view(-1)
        return F.smooth_l1_loss(risk_v, labels_v, beta=beta)

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def set_threshold(self, th: float):
        """Set classification threshold."""
        self.threshold.fill_(th)

    def classify(self, p_cross, env_feat, signal_factor=None):
        """Binary classification at current threshold."""
        result = self.forward(p_cross, env_feat, signal_factor)
        pred = (result["risk"] >= self.threshold).long()
        return pred, result["risk"], result


# ======================================================================
# Training utilities
# ======================================================================

def extract_vehicle_features_from_scene(
    scene_data: dict,
    norm: Tensor,
    target_idx: int = 0,
    junction_roi: Optional[list] = None,
    crosswalk_roi: Optional[list] = None,
) -> np.ndarray:
    """
    Extract vehicle environment features from scene_data for Stage 2 input.

    Thin wrapper around extract_environment_features for use in the
    two-stage training loop.

    Returns: np.ndarray of shape (8,) — same format as agent_centric_risk
    """
    if scene_data is None:
        return np.zeros(8, dtype=np.float32)

    return extract_environment_features(
        scene_data=scene_data,
        norm=norm,
        target_idx=target_idx,
        junction_roi=junction_roi,
        crosswalk_roi=crosswalk_roi,
    )


def prepare_stage2_features(
    p_cross: Tensor,
    traffic_light_states: List[str],
    scene_data: Optional[dict],
    norm: Tensor,
    device: torch.device,
    junction_roi: Optional[list] = None,
    crosswalk_roi: Optional[list] = None,
) -> Dict[str, Tensor]:
    """
    Prepare all Stage 2 input features from Stage 1 outputs + scene data.

    This is the main entry point for the two-stage pipeline:
        1. Compute P_cross via CrossingProbabilityEstimator (Stage 1)
        2. Call this function to get features
        3. Forward through RiskRegressionHead (Stage 2)

    Parameters
    ----------
    p_cross : Tensor — scalar or (B,) crossing probability from Stage 1
    traffic_light_states : list of str — per-frame traffic light states
    scene_data : dict — scene graph data with positions, class_names
    norm : Tensor — (2,) [W, H] normalization
    device : torch.device
    junction_roi : list, optional — junction polygon for vehicle filtering
    crosswalk_roi : list, optional — crosswalk polygon

    Returns
    -------
    dict with:
        "p_cross": Tensor
        "signal_factor": Tensor
        "env_feat": Tensor (B, 8)
    """
    # Signal factor
    signal_val = compute_signal_factor(traffic_light_states)
    signal_factor = torch.tensor(signal_val, device=device, dtype=torch.float32)

    # Vehicle features
    env_np = extract_vehicle_features_from_scene(
        scene_data, norm, junction_roi=junction_roi, crosswalk_roi=crosswalk_roi
    )
    env_feat = torch.from_numpy(env_np).float().to(device)

    return {
        "p_cross": p_cross,
        "signal_factor": signal_factor,
        "env_feat": env_feat,
    }


# ======================================================================
# Threshold search for RiskRegressionHead
# ======================================================================

def search_threshold_regression(
    head: RiskRegressionHead,
    all_features: List[dict],    # list of {"p_cross", "signal_factor", "env_feat"}
    all_labels: List[float],
) -> Tuple[float, float]:
    """
    Search best classification threshold for the regression head.

    Since the head outputs continuous risk values (not probabilities with BCE
    calibration), we sweep thresholds on validation data to find the best F1.

    Parameters
    ----------
    head : RiskRegressionHead
    all_features : list of dicts from prepare_stage2_features
    all_labels : list of float (0 or 1)

    Returns
    -------
    (best_threshold, best_f1)
    """
    head.eval()
    device = next(head.parameters()).device

    risks = []
    labels = np.array(all_labels)

    with torch.no_grad():
        for feat in all_features:
            p_cross = feat["p_cross"].to(device)
            env_feat = feat["env_feat"].to(device)
            signal_factor = feat["signal_factor"].to(device)

            result = head.forward_simple(p_cross, env_feat, signal_factor)
            risks.append(float(result.item()))

    risks = np.array(risks)
    pos_mask = labels == 1

    if pos_mask.sum() == 0:
        return 0.5, 0.0

    best_thresh, best_f1 = 0.5, 0.0
    for th in np.arange(0.01, 1.0, 0.01):
        preds = (risks >= th).astype(int)
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
