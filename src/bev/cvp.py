"""
CVP — Cycled View Projection.

Projects the camera feature map into a BEV feature map and cycles back to the
camera view, yielding a cycle-consistency signal (L_cvp_cycle) that keeps the
view projection reliable without BEV ground truth.

The projection is a *learned* dense view transform, NOT a Linear(F) over the
flattened map and NOT a plain reshape:

    forward:  conv →  P_h (image height → BEV longitudinal)  →  P_w (image width → BEV lateral)
    backward: conv →  Q_h (BEV longitudinal → image height)  →  Q_w (BEV lateral → image width)

P_h / P_w are learned matrices that map the camera's vertical/horizontal axes
onto the BEV's longitudinal/lateral axes, mirroring the physical geometry of a
front-facing road camera.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _nearest_proj(src: int, dst: int) -> torch.Tensor:
    """Initialise a (src, dst) projection matrix as nearest-neighbour sampling.

    Each destination column selects the nearest source index, so the module
    starts as a plain (non-degenerate) up/down-sampling and learns the true
    view correspondence from supervision.
    """
    M = torch.zeros(src, dst, dtype=torch.float32)
    for j in range(dst):
        i = int(round(j * (src - 1) / max(dst - 1, 1))) if dst > 1 else 0
        M[i, j] = 1.0
    return M


class CycledViewProjection(nn.Module):
    """Camera feature ⇄ BEV feature with a cycle path.

    Parameters
    ----------
    cam_dim, bev_dim : int — camera / BEV feature channels.
    cam_h, cam_w : int — camera feature spatial size (encoder output).
    bev_h, bev_w : int — BEV feature spatial size (before decoder).
    hidden_dim : int — shared hidden channels for the projection convs.
    """

    def __init__(self, cam_dim: int, bev_dim: int,
                 cam_h: int, cam_w: int, bev_h: int, bev_w: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.cam_h, self.cam_w = cam_h, cam_w
        self.bev_h, self.bev_w = bev_h, bev_w

        # Forward projector: camera → BEV.
        self.cam_proj = nn.Sequential(
            nn.Conv2d(cam_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.P_h = nn.Parameter(_nearest_proj(cam_h, bev_h))   # vertical → longitudinal
        self.P_w = nn.Parameter(_nearest_proj(cam_w, bev_w))   # horizontal → lateral
        self.to_bev = nn.Conv2d(hidden_dim, bev_dim, 1)

        # Backward projector: BEV → camera.
        self.bev_proj = nn.Sequential(
            nn.Conv2d(bev_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.Q_h = nn.Parameter(_nearest_proj(bev_h, cam_h))   # longitudinal → vertical
        self.Q_w = nn.Parameter(_nearest_proj(bev_w, cam_w))   # lateral → horizontal
        self.to_cam = nn.Conv2d(hidden_dim, cam_dim, 1)

    def forward_projection(self, F_cam: torch.Tensor) -> torch.Tensor:
        """F_cam (B, cam_dim, cam_h, cam_w) → F_bev_init (B, bev_dim, bev_h, bev_w)."""
        x = self.cam_proj(F_cam)                                    # (B, D, cam_h, cam_w)
        x = torch.einsum("bdhw,hl->bdlw", x, self.P_h)              # height → bev_h
        x = torch.einsum("bdlw,wm->bdlm", x, self.P_w)              # width  → bev_w
        return self.to_bev(x)                                       # (B, bev_dim, bev_h, bev_w)

    def backward_projection(self, F_bev: torch.Tensor) -> torch.Tensor:
        """F_bev (B, bev_dim, bev_h, bev_w) → F_cam_rec (B, cam_dim, cam_h, cam_w)."""
        x = self.bev_proj(F_bev)                                    # (B, D, bev_h, bev_w)
        x = torch.einsum("bdhw,hl->bdlw", x, self.Q_h)              # bev_h → cam_h
        x = torch.einsum("bdlw,wm->bdlm", x, self.Q_w)              # bev_w → cam_w
        return self.to_cam(x)                                       # (B, cam_dim, cam_h, cam_w)

    def forward(self, F_cam: torch.Tensor):
        """Returns (F_bev_init, F_cam_reconstructed) for the cycle loss."""
        F_bev = self.forward_projection(F_cam)
        F_cam_rec = self.backward_projection(F_bev)
        return F_bev, F_cam_rec
