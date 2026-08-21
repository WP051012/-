"""
Conditional FlowChain — FlowChain + traffic/geometry/scene/goal context.

First-version architecture (beat trajectory-only FlowChain on this dataset).
NO meta-learning, NO ModulationNet, NO domain adaptation.  The FlowChain core
is reused *unchanged* (same checkpoint-compatible `FlowChainPredictor`), and a
256-dim context vector is fed through the existing `perception_c` interface.

    signal (B,8,5)  ──SignalEncoder────> signal_ctx (B,64)
    geom   (B,8,6)  ──GeometryEncoder──> geom_ctx   (B,64)
    scene  (B,64)   ──SceneEncoder─────> scene_ctx  (B,64)
                                                |
    base_ctx = cat([signal_ctx, geom_ctx, scene_ctx])        (B,192)
        |── GoalHead ──> goal (B,2) ──GoalEmbed──> goal_emb (B,64)
        `── context = cat([signal_ctx, geom_ctx, scene_ctx, goal_emb])  (B,256)
                |── IntentHead ──────> (B,2)   [WAIT, CROSS]
                `── CrossingTimeHead ─> (B,13)  [cross at t=1..12, NO_CROSS]

    context (B,256) ──> Linear(256→16) ──> flow native conditioning (dist_args)
                         (cond_inject="flow": encoder stays trajectory-only)

Ablation is done by toggling `use_signal` / `use_geom` / `use_scene` /
`use_goal`; a disabled branch contributes zeros, keeping the context at a
constant 256 dims so the flow backbone always sees the same input width.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .context_encoders import (
    SignalEncoder,
    GeometryEncoder,
    SceneEncoder,
    GoalHead,
    GoalEmbed,
    IntentHead,
    CrossingTimeHead,
)
from .prediction.flow_chain import FlowChainPredictor, flow_chain_nll_loss


class ConditionalFlowChain(nn.Module):
    """FlowChain conditioned on signal + geometry + scene + goal context."""

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        d_model: int = 64,
        nvp_num_blocks: int = 3,
        condition_dim: int = 256,
        # Branch toggles (for stepwise ablation)
        use_signal: bool = True,
        use_geom: bool = True,
        use_scene: bool = True,
        use_goal: bool = True,
        use_intent: bool = True,
        use_crossing: bool = True,
        condition_flow: bool = True,
        condition_norm: bool = False,
        condition_gate: bool = False,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.use_signal = use_signal
        self.use_geom = use_geom
        self.use_scene = use_scene
        self.use_goal = use_goal
        self.use_intent = use_intent
        self.use_crossing = use_crossing
        # When False, the context is used ONLY for the auxiliary heads; the flow
        # sees a zero condition (keeping the zero-condition fine-tuned backbone
        # at its baseline quality instead of corrupting it through the stale
        # `encoder_input` condition columns).
        self.condition_flow = condition_flow
        # Trajectory-conditioning safeguards: the stale `encoder_input`
        # condition columns were fine-tuned with ZERO input, so a raw non-zero
        # context saturates them (1800px). LayerNorm bounds the context scale,
        # and a learnable gate initialized to 0 ramps the condition in from the
        # known 28px baseline instead of injecting it full-strength at step 0.
        self.condition_norm = condition_norm
        self.condition_gate = condition_gate
        self.cond_ln = nn.LayerNorm(condition_dim) if condition_norm else None
        self.cond_scale = nn.Parameter(torch.zeros(1)) if condition_gate else None

        # Context encoders (always constructed so ablation toggles don't change
        # state_dict keys; disabled branches are simply zeroed at forward time).
        self.signal_encoder = SignalEncoder(in_dim=5, hidden=32, out_dim=64)
        self.geometry_encoder = GeometryEncoder(in_dim=6, hidden=32, out_dim=64)
        self.scene_encoder = SceneEncoder(in_dim=64, out_dim=64)
        self.goal_head = GoalHead(in_dim=192, hidden=128, out_dim=2)
        self.goal_embed = GoalEmbed(in_dim=2, out_dim=64)
        self.intent_head = IntentHead(in_dim=256, hidden=128, out_dim=2)
        self.crossing_time_head = CrossingTimeHead(in_dim=256, hidden=128, out_dim=13)

        # FlowChain backbone — MUST match the baseline checkpoint
        # (FlowChainBase: d_model=64, num_flows=3, condition_dim=256).
        self.predictor = FlowChainPredictor(
            obs_len=obs_len,
            pred_len=pred_len,
            trajectory_dim=2,
            hidden_dim=d_model,
            condition_dim=condition_dim,
            num_flows=nvp_num_blocks,
            cond_inject="flow",
        )

    def build_context(
        self,
        signal: Optional[Tensor],   # (B, obs_len, 5)
        geom: Optional[Tensor],     # (B, obs_len, 6)
        scene: Optional[Tensor],    # (B, 64)
    ) -> Dict[str, Tensor]:
        """Build the 256-dim context + auxiliary predictions."""
        B = None
        for t in (signal, geom, scene):
            if t is not None:
                B = t.shape[0]
                break
        device = next(self.parameters()).device
        if B is None:
            raise ValueError("at least one of signal/geom/scene must be provided")

        def _zeros(size: int) -> Tensor:
            return torch.zeros(B, size, device=device)

        signal_ctx = self.signal_encoder(signal) if (self.use_signal and signal is not None) else _zeros(64)
        geom_ctx = self.geometry_encoder(geom) if (self.use_geom and geom is not None) else _zeros(64)
        scene_ctx = self.scene_encoder(scene) if (self.use_scene and scene is not None) else _zeros(64)

        base_ctx = torch.cat([signal_ctx, geom_ctx, scene_ctx], dim=-1)  # (B, 192)

        goal = self.goal_head(base_ctx)                                   # (B, 2)
        goal_emb = self.goal_embed(goal) if self.use_goal else _zeros(64)

        context = torch.cat([signal_ctx, geom_ctx, scene_ctx, goal_emb], dim=-1)  # (B, 256)

        aux: Dict[str, Tensor] = {}
        if self.use_goal:
            aux["goal"] = goal
        if self.use_intent:
            aux["intent"] = self.intent_head(context)          # (B, 2)
        if self.use_crossing:
            aux["crossing"] = self.crossing_time_head(context) # (B, 13)

        return {"context": context, "aux": aux}

    def _flow_cond(self, context: Tensor) -> Tensor:
        """Condition fed to the flow. Zero when condition_flow=False.

        When conditioning the trajectory, bound the context with LayerNorm and
        ramp it in through a zero-initialized gate so the flow starts from its
        zero-condition baseline and the condition grows in gently.
        """
        if not self.condition_flow:
            return torch.zeros_like(context)
        if self.cond_ln is not None:
            context = self.cond_ln(context)
        if self.cond_scale is not None:
            context = context * self.cond_scale
        return context

    def forward(
        self,
        obs_trajectory: Tensor,     # (B, obs_len, 2)
        signal: Optional[Tensor] = None,
        geom: Optional[Tensor] = None,
        scene: Optional[Tensor] = None,
        num_samples: int = 20,
    ):
        """Return (pred_dict, aux_dict).

        pred_dict matches FlowChainPredictor.forward(): samples (N,B,pred,2),
        log_probs (N,B), mean (B,pred,2), std (B,pred,2).
        """
        built = self.build_context(signal, geom, scene)
        pred = self.predictor.forward(
            obs_trajectory=obs_trajectory,
            perception_c=self._flow_cond(built["context"]),
            num_samples=num_samples,
        )
        return pred, built["aux"]

    def log_prob(
        self,
        obs_trajectory: Tensor,     # (B, obs_len, 2)
        target: Tensor,             # (B, pred_len, 2)
        signal: Optional[Tensor] = None,
        geom: Optional[Tensor] = None,
        scene: Optional[Tensor] = None,
    ) -> Tensor:
        """Teacher-forced NLL of the target trajectory (batch,)."""
        built = self.build_context(signal, geom, scene)
        return self.predictor.log_prob(
            obs_trajectory=obs_trajectory,
            target=target,
            perception_c=self._flow_cond(built["context"]),
        )


# ======================================================================
# Combined loss
# ======================================================================

def conditional_flow_loss(
    pred: Dict[str, Tensor],
    target: Tensor,
    aux: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    mse_weight: float = 1.0,
    w_goal: float = 1.0,
    w_intent: float = 1.0,
    w_crossing: float = 1.0,
):
    """Joint flow NLL+MSE plus auxiliary goal/intent/crossing losses.

    Returns (total: Tensor, components: Dict[str, Tensor]).
    """
    flow_loss = flow_chain_nll_loss(pred, target, mse_weight=mse_weight)
    components = {"flow": flow_loss}
    total = flow_loss

    if aux is not None and labels is not None:
        if "goal" in aux and "goal_label" in labels:
            gl = F.mse_loss(aux["goal"], labels["goal_label"])
            components["goal"] = gl
            total = total + w_goal * gl
        if "intent" in aux and "intent_label" in labels:
            il = F.cross_entropy(aux["intent"], labels["intent_label"].long())
            components["intent"] = il
            total = total + w_intent * il
        if "crossing" in aux and "crossing_label" in labels:
            # crossing_label is 1..pred_len, or pred_len+1 = NO_CROSS.
            # Shift to 0..pred_len class indices.
            cl = F.cross_entropy(aux["crossing"], labels["crossing_label"].long() - 1)
            components["crossing"] = cl
            total = total + w_crossing * cl

    return total, components
