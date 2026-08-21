"""
BEV decoder — BEV feature → per-class object heatmap logits.

Output spatial size matches the pseudo-BEV grid exactly so that
``pred_bev.shape == pseudo_bev.shape`` (an assertion enforced during training).
The decoder emits *logits*; the loss applies sigmoid + BCE (or focal), and
visualisation applies ``sigmoid`` to recover [0,1] heatmaps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDecoder(nn.Module):
    def __init__(self, bev_dim: int, num_classes: int,
                 target_h: int, target_w: int, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.target_h, self.target_w = target_h, target_w

        self.blocks = nn.Sequential(
            nn.Conv2d(bev_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, F_bev: torch.Tensor) -> torch.Tensor:
        """F_bev (B, bev_dim, h, w) → logits (B, num_classes, target_h, target_w)."""
        x = self.blocks(F_bev)
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = F.interpolate(x, size=(self.target_h, self.target_w),
                              mode="bilinear", align_corners=False)
        return self.head(x)
