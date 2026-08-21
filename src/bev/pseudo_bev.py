"""
Pseudo-BEV generation from detections + homography.

This is the *geometry teacher* of the model. For each detected object we take
its bottom-center (ground-contact) point, push it through the homography onto
the ground plane, and rasterise a Gaussian heatmap onto the BEV grid.

IMPORTANT — naming contract:
    The result is ``pseudo_bev``, NEVER ``gt_bev``. We do NOT have BEV ground
    truth. Homography output is a weak, geometry-derived supervision signal
    that carries calibration + detection error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..geometry.homography import Homography
from ..geometry.coordinate import BEVGrid

# BEV object channels (order is fixed; C = len(BEV_CLASSES)).
BEV_CLASSES = ["pedestrian", "vehicle"]

# Detector class names that fold into the single "vehicle" BEV channel.
VEHICLE_CLASS_NAMES = {"bicycle", "motorcycle", "car", "bus", "truck"}


def class_name_to_channel(cls_name: str) -> Optional[int]:
    """Map a detector class name → BEV channel index (or None to skip)."""
    if cls_name == "pedestrian":
        return 0
    if cls_name in VEHICLE_CLASS_NAMES:
        return 1
    return None


def bottom_center_from_bbox(bbox) -> tuple:
    """(x1, y1, x2, y2) → bottom-center contact point ((x1+x2)/2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def gaussian_2d_kernel(sigma: float) -> np.ndarray:
    """2D Gaussian kernel of radius ceil(3σ), normalised to peak value 1."""
    sigma = float(sigma)
    r = max(1, int(math.ceil(3 * sigma)))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    gx = np.exp(-(ax ** 2) / (2 * sigma * sigma))
    kernel = np.outer(gx, gx)
    kernel /= kernel.max() + 1e-12
    return kernel.astype(np.float32)


def gaussian_heatmap(
    cells: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    """Rasterise (M, 2) integer cells [(bev_x, bev_y)] into a Gaussian heatmap.

    Parameters
    ----------
    cells : (M, 2) int array, column = bev_x (lateral), row = bev_y (longitudinal).
    height, width : BEV grid (H, W).
    sigma : float — Gaussian std in grid cells.

    Returns
    -------
    (height, width) float32 heatmap.
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    if cells.shape[0] == 0:
        return heatmap

    kernel = gaussian_2d_kernel(sigma)
    r = kernel.shape[0] // 2
    for cx, cy in cells:
        cx, cy = int(round(cx)), int(round(cy))
        # clip the kernel footprint to the grid
        x0 = max(0, cx - r)
        x1 = min(width, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(height, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        kx0 = x0 - (cx - r)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (cy - r)
        ky1 = ky0 + (y1 - y0)
        heatmap[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]
    # Cap at 1.0 (overlapping objects shouldn't exceed a peak)
    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap


@dataclass
class PseudoBEV:
    """One frame's pseudo-BEV supervision bundle.

    Attributes
    ----------
    heatmap : (C, H_bev, W_bev) float32 object-center Gaussian heatmaps.
    positions : (M, 2) float32 ground-plane (X, Y) meters per object.
    cells : (M, 2) int64 BEV grid cells per object (bev_x, bev_y).
    channels : (M,) int64 BEV channel per object.
    ids : (M,) object track ids (or -1).
    class_names : list[str] detector class name per object.
    """

    heatmap: np.ndarray
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    channels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    class_names: List[str] = field(default_factory=list)


def _get(det, key, default=None):
    """Fetch a field from a dict-like or attribute-carrying detection."""
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


def generate_pseudo_bev(
    detections,
    homography: Homography,
    grid: BEVGrid,
    sigma: float = 1.5,
    classes: Sequence[str] = BEV_CLASSES,
) -> PseudoBEV:
    """Convert detections → pseudo-BEV object heatmap.

    Parameters
    ----------
    detections : iterable of detections, each with ``bbox`` (x1,y1,x2,y2) and
        ``class_name`` (and optional ``track_id``). Accepts both dicts and
        objects with attributes (e.g. ``DetectionResult``).
    homography : Homography — maps image pixels → ground meters.
    grid : BEVGrid — target BEV lattice.
    sigma : float — Gaussian heatmap std (grid cells).
    classes : Sequence[str] — channel order (default BEV_CLASSES).

    Returns
    -------
    PseudoBEV
    """
    C = len(classes)
    heatmaps = np.zeros((C, grid.height, grid.width), dtype=np.float32)

    positions = []
    cells = []
    channels = []
    ids = []
    class_names = []

    for det in detections:
        cls_name = _get(det, "class_name", "unknown")
        ch = class_name_to_channel(cls_name)
        if ch is None:
            continue
        bbox = _get(det, "bbox")
        if bbox is None:
            continue
        u, v = bottom_center_from_bbox(bbox)

        (X, Y) = homography.pixel_to_ground(np.array([[u, v]]))[0]
        bev_x = (X - grid.x_min) / grid.resolution
        bev_y = (Y - grid.y_min) / grid.resolution

        # Drop objects projected outside the BEV grid.
        if not (0 <= bev_x < grid.width and 0 <= bev_y < grid.height):
            continue

        positions.append((X, Y))
        cells.append((bev_x, bev_y))
        channels.append(ch)
        ids.append(_get(det, "track_id", -1))
        class_names.append(cls_name)

    if cells:
        cells_arr = np.asarray(cells, dtype=np.float64)
        int_cells = np.round(cells_arr).astype(np.int64)
        for ch in range(C):
            sel = np.asarray(channels) == ch
            if sel.any():
                heatmaps[ch] = gaussian_heatmap(
                    int_cells[sel], grid.height, grid.width, sigma
                )

    return PseudoBEV(
        heatmap=heatmaps,
        positions=np.asarray(positions, dtype=np.float32).reshape(-1, 2),
        cells=np.asarray(cells, dtype=np.float64).reshape(-1, 2),
        channels=np.asarray(channels, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        class_names=class_names,
    )
