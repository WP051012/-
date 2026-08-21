"""
BEV grid coordinate helpers.

Ground plane is expressed in meters with origin defined by the homography.
The BEV grid is a fixed rectangular lattice over that ground plane:

    bev_x = (X - x_min) / resolution    # column index (lateral)
    bev_y = (Y - y_min) / resolution    # row    index (longitudinal)

This mirrors the spec's `bev: {x_min, x_max, y_min, y_max, resolution}` config
block and keeps all geometry out of the model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BEVGrid:
    """A rectangular BEV lattice over the ground plane (meters).

    Parameters
    ----------
    x_min, x_max : float
        Lateral extent (meters). X = lateral axis.
    y_min, y_max : float
        Longitudinal extent (meters). Y = longitudinal (depth) axis.
    resolution : float
        Meters per grid cell (== meters_per_pixel).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @property
    def width(self) -> int:
        """Number of grid columns (lateral)."""
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        """Number of grid rows (longitudinal)."""
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self):
        """(H_bev, W_bev) == (height, width)."""
        return self.height, self.width

    def validate(self) -> None:
        assert self.x_max > self.x_min, "BEVGrid: x_max must exceed x_min"
        assert self.y_max > self.y_min, "BEVGrid: y_max must exceed y_min"
        assert self.resolution > 0, "BEVGrid: resolution must be positive"
        assert self.width > 0 and self.height > 0, "BEVGrid: zero-size grid"

    @staticmethod
    def from_config(cfg: dict) -> "BEVGrid":
        g = BEVGrid(
            x_min=float(cfg["x_min"]),
            x_max=float(cfg["x_max"]),
            y_min=float(cfg["y_min"]),
            y_max=float(cfg["y_max"]),
            resolution=float(cfg["resolution"]),
        )
        g.validate()
        return g


def ground_to_bev(X, Y, grid: BEVGrid) -> np.ndarray:
    """Ground-plane meters (X, Y) → fractional BEV grid indices.

    Returns an (N, 2) float array of (bev_x, bev_y) where bev_x is the column
    (lateral) and bev_y the row (longitudinal). Points outside the grid are
    NOT clipped here — callers decide whether to drop or clamp.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    bev_x = (X - grid.x_min) / grid.resolution
    bev_y = (Y - grid.y_min) / grid.resolution
    return np.stack([bev_x, bev_y], axis=-1)


def bev_to_ground(bev_x, bev_y, grid: BEVGrid) -> np.ndarray:
    """BEV grid indices (bev_x, bev_y) → ground-plane meters (X, Y)."""
    bev_x = np.asarray(bev_x, dtype=np.float64)
    bev_y = np.asarray(bev_y, dtype=np.float64)
    X = grid.x_min + bev_x * grid.resolution
    Y = grid.y_min + bev_y * grid.resolution
    return np.stack([X, Y], axis=-1)


def points_to_bev_grid(points_ground, grid: BEVGrid, keep_in_bounds: bool = True):
    """Map ground points to integer BEV cells.

    Parameters
    ----------
    points_ground : (N, 2) array of (X, Y) meters.
    grid : BEVGrid
    keep_in_bounds : bool
        If True, drop points that fall outside [x_min,x_max]×[y_min,y_max].

    Returns
    -------
    (M, 2) int array of (bev_x, bev_y) column/row cells.
    """
    frac = ground_to_bev(points_ground[:, 0], points_ground[:, 1], grid)
    cells = np.floor(frac).astype(np.int64)
    if keep_in_bounds:
        valid = (
            (frac[:, 0] >= 0) & (frac[:, 0] < grid.width) &
            (frac[:, 1] >= 0) & (frac[:, 1] < grid.height)
        )
        cells = cells[valid]
    return cells
