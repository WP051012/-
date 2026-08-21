"""
Context encoders + auxiliary heads for the conditional FlowChain.

First-version architecture (traffic/geometry-aware + goal-conditioned,
NO meta-learning / no ModulationNet / no domain adaptation):

    signal (B,8,5) --SignalEncoder----> signal_ctx (B,64)
    geom   (B,8,6) --GeometryEncoder--> geom_ctx   (B,64)
    scene  (B,64)  --SceneEncoder-----> scene_ctx  (B,64)
                                             |
    base_ctx = cat([signal_ctx, geom_ctx, scene_ctx])          (B,192)
        |-- GoalHead ----------> goal (B,2) --GoalEmbed--> goal_emb (B,64)
        `-- context = cat([signal_ctx, geom_ctx, scene_ctx, goal_emb])  (B,256)
                |-- IntentHead ------> (B,2)   [WAIT, CROSS]
                `-- CrossingTimeHead -> (B,13)  [crossing at t=1..12, NO_CROSS]

The 256-dim context is fed to FlowChainPredictor as `perception_c`, reusing the
existing `cond_label_size=256` interface (zero change to FlowChain core).
"""

import torch
import torch.nn as nn
from torch import Tensor


class SignalEncoder(nn.Module):
    """8-frame traffic-signal one-hot (B,8,5) → (B,64) context."""

    def __init__(self, in_dim: int = 5, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, signal: Tensor) -> Tensor:
        # signal: (B, 8, 5)
        _, h = self.gru(signal)          # h: (1, B, hidden)
        return self.proj(h.squeeze(0))   # (B, 64)


class GeometryEncoder(nn.Module):
    """8-frame geometry features (B,8,6) → (B,64) context."""

    def __init__(self, in_dim: int = 6, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, geom: Tensor) -> Tensor:
        # geom: (B, 8, 6)
        _, h = self.gru(geom)
        return self.proj(h.squeeze(0))   # (B, 64)


class SceneEncoder(nn.Module):
    """GAT scene embedding (B,64) → (B,64). Lightweight linear projection."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, scene: Tensor) -> Tensor:
        return self.proj(scene)          # (B, 64)


class GoalHead(nn.Module):
    """base context (B,192) → goal point (B,2) in normalized [0,1] coords."""

    def __init__(self, in_dim: int = 192, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, base_ctx: Tensor) -> Tensor:
        return self.net(base_ctx)        # (B, 2)


class GoalEmbed(nn.Module):
    """goal point (B,2) → (B,64) embedding, injected back into context."""

    def __init__(self, in_dim: int = 2, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, goal: Tensor) -> Tensor:
        return self.proj(goal)           # (B, 64)


class IntentHead(nn.Module):
    """context (B,256) → (B,2) logits [WAIT, CROSS]."""

    def __init__(self, in_dim: int = 256, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)         # (B, 2)


class CrossingTimeHead(nn.Module):
    """context (B,256) → (B,13) logits: cross at t=1..12, or NO_CROSS."""

    def __init__(self, in_dim: int = 256, hidden: int = 128, out_dim: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)         # (B, 13)
