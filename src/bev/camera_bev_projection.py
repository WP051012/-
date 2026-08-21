"""
Differentiable camera ⇄ BEV warping via the homography.

This is the geometry bridge that makes the Yan-style *cross-view cycle* loss
differentiable. Instead of a learned/black-box warp, we use the calibrated
homography H to precompute two ``grid_sample`` sampling grids:

    camera → BEV   : for each BEV cell, where to sample in the camera mask
    BEV → camera   : for each camera-mask pixel, where to sample in the BEV map

Both grids are fixed for a stationary camera, registered as non-trainable
buffers, and reused every forward pass. The camera mask is a *pseudo* mask
rasterised from detector boxes (never GT), so the cycle loss compares

    warped-back(pred_bev)  vs  pseudo_camera_mask

which keeps the BEV decoder consistent with the camera view without any BEV
ground truth.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.homography import Homography
from ..geometry.coordinate import BEVGrid
from .pseudo_bev import BEV_CLASSES, class_name_to_channel


# ---------------------------------------------------------------------------
# Pseudo camera mask rasterisation
# ---------------------------------------------------------------------------

def _get(det, key, default=None):
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


def rasterize_camera_mask(
    detections,
    mask_h: int,
    mask_w: int,
    img_h: int,
    img_w: int,
    classes: Sequence[str] = BEV_CLASSES,
) -> np.ndarray:
    """Rasterise detector boxes into a per-class *pseudo* camera mask.

    Parameters
    ----------
    detections : iterable with ``bbox`` (x1,y1,x2,y2 in full-res pixels) and
        ``class_name``.
    mask_h, mask_w : camera-mask resolution (rows, cols).
    img_h, img_w : original image resolution the boxes are defined in.
    classes : BEV channel order.

    Returns
    -------
    (C, mask_h, mask_w) float32 mask with 1.0 inside boxes, 0 elsewhere.
    """
    C = len(classes)
    mask = np.zeros((C, mask_h, mask_w), dtype=np.float32)
    sx = mask_w / max(img_w, 1)
    sy = mask_h / max(img_h, 1)

    for det in detections:
        ch = class_name_to_channel(_get(det, "class_name", "unknown"))
        if ch is None:
            continue
        bbox = _get(det, "bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        mx1 = int(round(x1 * sx)); mx2 = int(round(x2 * sx))
        my1 = int(round(y1 * sy)); my2 = int(round(y2 * sy))
        mx1 = max(0, min(mask_w - 1, mx1))
        mx2 = max(0, min(mask_w - 1, mx2))
        my1 = max(0, min(mask_h - 1, my1))
        my2 = max(0, min(mask_h - 1, my2))
        if mx2 <= mx1 or my2 <= my1:
            continue
        mask[ch, my1:my2, mx1:mx2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Differentiable projection module
# ---------------------------------------------------------------------------

class CameraBEVProjection(nn.Module):
    """Homography-guided, differentiable warp between camera and BEV grids.

    Parameters
    ----------
    homography : Homography — maps full-res image pixels → ground meters.
    grid : BEVGrid — the target BEV lattice.
    mask_h, mask_w : camera-mask (cycle) resolution.
    img_h, img_w : original image resolution the homography is calibrated on.
    """

    def __init__(self, homography: Homography, grid: BEVGrid,
                 mask_h: int, mask_w: int, img_h: int, img_w: int):
        super().__init__()
        self.H = homography
        self.grid = grid
        self.mask_h = int(mask_h)
        self.mask_w = int(mask_w)
        self.img_h = int(img_h)
        self.img_w = int(img_w)

        c2b, b2c = self._build_grids()
        self.register_buffer("cam_to_bev_grid", c2b)   # (1, H_bev, W_bev, 2)
        self.register_buffer("bev_to_cam_grid", b2c)   # (1, mask_h, mask_w, 2)

    # -- grid construction ---------------------------------------------------

    @staticmethod
    def _norm(u, size):
        """Continuous pixel *center* coordinate → grid_sample normalised coord.

        align_corners=False convention: x_norm ∈ [-1,1], pixel center c sits at
        x_norm = 2*(c + 0.5)/size - 1.
        """
        return 2.0 * (u + 0.5) / size - 1.0

    def _build_grids(self):
        H = self.grid.height
        W = self.grid.width

        # --- camera → BEV grid -------------------------------------------------
        # Output grid is indexed by BEV cell (row i, col j); value is the
        # normalised coordinate in the *camera mask* to sample from.
        ys = np.arange(H)
        xs = np.arange(W)
        X = self.grid.x_min + (xs + 0.5) * self.grid.resolution
        Y = self.grid.y_min + (ys + 0.5) * self.grid.resolution
        XX, YY = np.meshgrid(X, Y)                        # (H, W)
        ground = np.stack([XX.ravel(), YY.ravel()], axis=1)
        uv = self.H.ground_to_pixel(ground)               # full-res (u, v)
        u = uv[:, 0] * self.mask_w / self.img_w           # → mask coords
        v = uv[:, 1] * self.mask_h / self.img_h
        nx = self._norm(u, self.mask_w)
        ny = self._norm(v, self.mask_h)
        c2b = np.stack([nx, ny], axis=1).reshape(H, W, 2).astype(np.float32)
        cam_to_bev_grid = torch.from_numpy(c2b).unsqueeze(0)

        # --- BEV → camera grid -------------------------------------------------
        # Output grid is indexed by camera-mask pixel (row i, col j); value is
        # the normalised coordinate in the BEV map to sample from.
        ys = np.arange(self.mask_h)
        xs = np.arange(self.mask_w)
        XX, YY = np.meshgrid(xs, ys)                      # (mask_h, mask_w), XX=col
        u_full = (XX + 0.5) * self.img_w / self.mask_w    # mask → full-res pixel
        v_full = (YY + 0.5) * self.img_h / self.mask_h
        px = np.stack([u_full.ravel(), v_full.ravel()], axis=1)
        ground = self.H.pixel_to_ground(px)               # (X, Y) meters
        bev_x = (ground[:, 0] - self.grid.x_min) / self.grid.resolution
        bev_y = (ground[:, 1] - self.grid.y_min) / self.grid.resolution
        nx = self._norm(bev_x, W)
        ny = self._norm(bev_y, H)
        b2c = np.stack([nx, ny], axis=1).reshape(self.mask_h, self.mask_w, 2).astype(np.float32)
        bev_to_cam_grid = torch.from_numpy(b2c).unsqueeze(0)

        return cam_to_bev_grid, bev_to_cam_grid

    # -- warping -------------------------------------------------------------

    def _expand(self, grid: torch.Tensor, B: int, device):
        return grid.expand(B, -1, -1, -1).to(device)

    def camera_to_bev(self, cam_mask: torch.Tensor) -> torch.Tensor:
        """(B, C, mask_h, mask_w) camera mask → (B, C, H_bev, W_bev)."""
        grid = self._expand(self.cam_to_bev_grid, cam_mask.shape[0], cam_mask.device)
        return F.grid_sample(cam_mask, grid, mode="bilinear",
                             align_corners=False, padding_mode="zeros")

    def bev_to_camera(self, bev: torch.Tensor) -> torch.Tensor:
        """(B, C, H_bev, W_bev) BEV map → (B, C, mask_h, mask_w) camera map."""
        grid = self._expand(self.bev_to_cam_grid, bev.shape[0], bev.device)
        return F.grid_sample(bev, grid, mode="bilinear",
                             align_corners=False, padding_mode="zeros")
