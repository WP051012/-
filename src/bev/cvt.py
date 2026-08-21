"""
CVT — Cross-View Transformer.

Establishes cross-view correspondence between the camera feature and the BEV
feature with bidirectional cross-attention (NOT a concat + conv):

    BEV query  ──attend──▶  Camera keys/values   (BEV searches for visual evidence)
    Camera query ─attend──▶ BEV keys/values      (reverse attention, kept per spec)

Spatial maps are flattened into token sequences with learned 2D positional
embeddings so the transformer knows where each token lives in its view.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossViewLayer(nn.Module):
    """One bidirectional cross-view transformer layer."""

    def __init__(self, bev_dim: int, cam_dim: int, num_heads: int = 4,
                 ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.bev_dim = bev_dim
        self.cam_dim = cam_dim

        # BEV queries camera (visual evidence lookup).
        self.bev_cross = nn.MultiheadAttention(
            bev_dim, num_heads, batch_first=True, kdim=cam_dim, vdim=cam_dim
        )
        self.bev_self = nn.MultiheadAttention(bev_dim, num_heads, batch_first=True)

        # Camera queries BEV (reverse attention).
        self.cam_cross = nn.MultiheadAttention(
            cam_dim, num_heads, batch_first=True, kdim=bev_dim, vdim=bev_dim
        )
        self.cam_self = nn.MultiheadAttention(cam_dim, num_heads, batch_first=True)

        self.bev_ff = nn.Sequential(
            nn.Linear(bev_dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, bev_dim)
        )
        self.cam_ff = nn.Sequential(
            nn.Linear(cam_dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, cam_dim)
        )

        self.bev_norm1 = nn.LayerNorm(bev_dim)
        self.bev_norm2 = nn.LayerNorm(bev_dim)
        self.bev_norm3 = nn.LayerNorm(bev_dim)
        self.cam_norm1 = nn.LayerNorm(cam_dim)
        self.cam_norm2 = nn.LayerNorm(cam_dim)
        self.cam_norm3 = nn.LayerNorm(cam_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bev_tokens: torch.Tensor, cam_tokens: torch.Tensor):
        # --- BEV branch: cross-attend camera, then self-attend, then FFN ---
        b = bev_tokens + self.dropout(
            self.bev_cross(self.bev_norm1(bev_tokens), cam_tokens, cam_tokens)[0]
        )
        b = b + self.dropout(self.bev_self(self.bev_norm2(b), b, b)[0])
        b = b + self.dropout(self.bev_ff(self.bev_norm3(b)))

        # --- Camera branch: cross-attend BEV, then self-attend, then FFN ---
        c = cam_tokens + self.dropout(
            self.cam_cross(self.cam_norm1(cam_tokens), bev_tokens, bev_tokens)[0]
        )
        c = c + self.dropout(self.cam_self(self.cam_norm2(c), c, c)[0])
        c = c + self.dropout(self.cam_ff(self.cam_norm3(c)))

        return b, c


class CrossViewTransformer(nn.Module):
    """Stacked bidirectional cross-view transformer.

    Parameters
    ----------
    bev_dim, cam_dim : int — token dims of BEV and camera features.
    num_heads, num_layers, ff_dim, dropout : standard transformer hyperparams.
    """

    def __init__(self, bev_dim: int, cam_dim: int, num_heads: int = 4,
                 num_layers: int = 1, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.bev_dim = bev_dim
        self.cam_dim = cam_dim
        self.layers = nn.ModuleList([
            CrossViewLayer(bev_dim, cam_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, F_bev: torch.Tensor, F_cam: torch.Tensor):
        """F_bev (B,C,bev_h,bev_w), F_cam (B,C,cam_h,cam_w) → refined F_bev.

        Returns the refined BEV feature map as the primary output (the refined
        camera tokens are computed for the reverse-attention path but the BEV
        map is what the decoder consumes).
        """
        B = F_bev.shape[0]
        bev_h, bev_w = F_bev.shape[-2], F_bev.shape[-1]
        cam_h, cam_w = F_cam.shape[-2], F_cam.shape[-1]

        bev_tokens = F_bev.flatten(2).transpose(1, 2)     # (B, bev_h*bev_w, bev_dim)
        cam_tokens = F_cam.flatten(2).transpose(1, 2)     # (B, cam_h*cam_w, cam_dim)

        for layer in self.layers:
            bev_tokens, cam_tokens = layer(bev_tokens, cam_tokens)

        F_bev_out = bev_tokens.transpose(1, 2).reshape(B, -1, bev_h, bev_w)
        F_cam_out = cam_tokens.transpose(1, 2).reshape(B, -1, cam_h, cam_w)
        return F_bev_out, F_cam_out
