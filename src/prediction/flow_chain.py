"""
FlowChain Predictor — wrapper around the official TransformerFlowChain.

This module provides the standard interface used by the rest of the project
(baselines, perception_model, train.py, inference.py).

Backed by the official FlowChain (ICCV 2023) implementation in
flow_chain_official.py (Transformer encoder-decoder + RealNVP with
MADE-style LinearMaskedCoupling).

Interface (maintained for backward compatibility):
    FlowChainPredictor(obs_len, pred_len, ...)
        .forward(obs_trajectory, perception_c, num_samples) → dict

    flow_chain_nll_loss(pred, target) → Tensor (joint NLL+MSE)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .flow_chain_official import (
    TransformerFlowChain,
    joint_nll_mse_loss,
    transformer_flow_nll_loss,
    RealNVP,
    Flow,
    MeanScaler,
    NOPScaler,
    LinearMaskedCoupling,
    BatchNorm,
    FlowSequential,
)


# ======================================================================
# FlowChainPredictor — thin wrapper
# ======================================================================

class FlowChainPredictor(nn.Module):
    """
    FlowChain-based trajectory predictor with traffic perception conditioning.

    Wraps TransformerFlowChain with our standard (B, T, 2) interface.

    Parameters
    ----------
    obs_len : int
    pred_len : int
    trajectory_dim : int    Per-frame coord dim (2 for x,y).
    hidden_dim : int        Transformer d_model.
    condition_dim : int     Traffic perception vector dimension.
    num_flows : int         Number of RealNVP coupling blocks.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        trajectory_dim: int = 2,
        hidden_dim: int = 64,
        condition_dim: int = 256,
        num_flows: int = 3,
        use_adapter: bool = True,
        cond_inject: str = "encoder",
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim

        # Map our param names to TransformerFlowChain params
        self.model = TransformerFlowChain(
            obs_len=obs_len,
            pred_len=pred_len,
            feature_dim=trajectory_dim,
            d_model=hidden_dim,
            num_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward_scale=4,
            dropout_rate=0.1,
            n_blocks=num_flows,
            n_hidden=2,
            flow_hidden_size=hidden_dim * 2,  # 2 × d_model
            conditioning_length=16,
            cond_label_size=condition_dim,
            scaling=True,
            use_adapter=use_adapter,
            cond_inject=cond_inject,
        )

    def encode_history(
        self,
        obs_trajectory: Tensor,    # (B, obs_len, 2)
    ) -> Tuple[Tensor, Tensor]:
        """Encode observation trajectory. Returns (enc_out, scale)."""
        return self.model._encode(obs_trajectory, cond=None)

    def fuse_condition(
        self,
        history_code: Tensor,   # (B, obs_len, d_model) — actually encoder output
        c: Tensor,              # (B, condition_dim)
    ) -> Tensor:
        """Fuse history encoding with perception vector. (Compatibility stub.)"""
        return c  # perception is injected at encoder input, not here

    def forward(
        self,
        obs_trajectory: Tensor,       # (..., obs_len, 2)
        perception_c: Tensor,         # (..., condition_dim)
        num_samples: int = 20,
        prompts: Optional[Tensor] = None,  # (..., num_prompts, d_model) — prefix
    ) -> Dict[str, Tensor]:
        """
        Full forward pass: encode + autoregressive decode.

        Returns
        -------
        dict with:
            "samples":   (N, ..., pred_len, 2)  sampled trajectories
            "log_probs": (N, ...)               log-probabilities
            "mean":      (..., pred_len, 2)      mean prediction
            "std":       (..., pred_len, 2)      per-step std
        """
        # Handle extra batch dims
        orig_shape = obs_trajectory.shape
        if obs_trajectory.dim() == 2:
            obs_trajectory = obs_trajectory.unsqueeze(0)  # (1, T, 2)
        if perception_c.dim() == 1:
            perception_c = perception_c.unsqueeze(0)  # (1, D)

        # Always pass perception_c to maintain gradient flow for training.
        # When zeros: functionally identical to unconditional (encoder pads zeros),
        # but gradients can flow back through the condition path.
        cond = perception_c

        result = self.model.forward(
            obs=obs_trajectory,
            cond=cond,
            prompts=prompts,
            num_samples=num_samples,
        )
        # result: {"samples": (B, N, pred, 2), "log_probs": (B, N), "mean": (B, pred, 2), "std": (B, pred, 2)}

        # Reorder to match our old interface: (N, B, ...)
        result["samples"] = result["samples"].permute(1, 0, 2, 3)  # (N, B, pred, 2)
        result["log_probs"] = result["log_probs"].permute(1, 0)      # (N, B)

        return result

    def log_prob(
        self,
        obs_trajectory: Tensor,       # (..., obs_len, 2)
        target: Tensor,               # (..., pred_len, 2)
        perception_c: Tensor,         # (..., condition_dim)
        prompts: Optional[Tensor] = None,  # (..., num_prompts, d_model)
    ) -> Tensor:
        """Teacher-forced log-probability of target trajectory (training loss)."""
        if obs_trajectory.dim() == 2:
            obs_trajectory = obs_trajectory.unsqueeze(0)
        if perception_c.dim() == 1:
            perception_c = perception_c.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        cond = perception_c  # Always pass for gradient flow
        return self.model.log_prob(obs=obs_trajectory, target=target, cond=cond, prompts=prompts)

    def reinitialize_and_predict(
        self,
        obs_trajectory: Tensor,
        perception_c_new: Tensor,
        num_samples: int = 20,
    ) -> Dict[str, Tensor]:
        """Re-run prediction after perception state change. (Compatibility stub.)"""
        return self.forward(obs_trajectory, perception_c_new, num_samples)


# ======================================================================
# Loss functions
# ======================================================================

def flow_chain_nll_loss(
    pred: Dict[str, Tensor],
    target: Tensor,            # (..., pred_len, 2)
    mse_weight: float = 1.0,
) -> Tensor:
    """
    Joint NLL + MSE loss.

    - NLL: uses best-of-N sample (lowest NLL) for distribution calibration.
    - MSE: directly penalizes mean prediction position error.

    Parameters
    ----------
    pred : dict      Output of FlowChainPredictor.forward().
    target : Tensor  Ground-truth future trajectory.
    mse_weight : float  Weight for MSE term (default 1.0).

    Returns
    -------
    Tensor (scalar) — mean loss over batch.
    """
    # Handle shape: ensure (B, pred_len, 2)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    B = target.shape[0]
    device = target.device

    # NLL: best-of-N log_prob
    log_probs = pred["log_probs"]  # (N, B)
    if log_probs.dim() == 1:
        log_probs = log_probs.unsqueeze(1)  # (N, 1)
    # For each batch element, pick the best sample
    best_log_prob = log_probs.max(dim=0)[0]  # (B,)
    nll = -best_log_prob.mean()

    # MSE: mean prediction vs target
    mean_pred = pred["mean"]  # (B, pred_len, 2)
    if mean_pred.dim() == 2:
        mean_pred = mean_pred.unsqueeze(0)
    mse = ((mean_pred - target) ** 2).mean()

    return nll + mse_weight * mse


# ======================================================================
# Legacy class aliases for backward compatibility
# ======================================================================

# FiLMAffineCoupling / FiLMConditionProjector / ConditionalRealNVP are
# no longer used (replaced by official RealNVP + Transformer).  Provide
# dummy references so old serialized checkpoints don't break on import.

class _DeprecatedMixin:
    """Base for deprecated classes — warns on init."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. "
            f"Use flow_chain_official.TransformerFlowChain instead.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__()


class FiLMAffineCoupling(_DeprecatedMixin, nn.Module):
    """Deprecated — use LinearMaskedCoupling from flow_chain_official."""
    def forward(self, x, c=None, reverse=False):
        return x, torch.zeros(x.shape[:-1], device=x.device)


class FiLMConditionProjector(_DeprecatedMixin, nn.Module):
    """Deprecated — conditioning is now handled by TransformerFlowChain."""
    def forward(self, c):
        return [c] * 3


class ConditionalRealNVP(_DeprecatedMixin, nn.Module):
    """Deprecated — use RealNVP from flow_chain_official."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, z, c, reverse=False):
        return z, torch.zeros(z.shape[:-1], device=z.device)
    def sample(self, c, num_samples=20):
        return torch.zeros(num_samples, *c.shape[:-1], 24, device=c.device), \
               torch.zeros(num_samples, *c.shape[:-1], device=c.device)
    def log_prob(self, y, c):
        return torch.zeros(y.shape[:-1], device=y.device)
