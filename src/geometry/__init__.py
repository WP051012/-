"""
Geometry module — camera↔ground-plane (BEV) coordinate transforms.

Submodules
----------
homography.py  : image-pixel ⇄ ground-plane (meters) homography H
coordinate.py  : ground-plane meters ⇄ BEV grid index helpers

Conventions (single fixed monocular camera, no stereo / no second view):

    Image pixel : (u, v)     u = column (x-axis), v = row (y-axis), origin top-left.
    Ground plane: (X, Y)     meters. X = lateral, Y = longitudinal (depth).
    BEV grid    : (bev_x, bev_y)  integer cells, column/row of a [C, H_bev, W_bev] map.

Projective map (as required by the spec):

        s · [X, Y, 1]^T = H · [u, v, 1]^T

i.e. H maps image pixels FORWARD onto the ground plane; H^{-1} maps back.
"""

from .homography import (
    Homography,
    compute_homography,
    load_homography,
    homography_from_config,
)
from .coordinate import (
    BEVGrid,
    ground_to_bev,
    bev_to_ground,
    points_to_bev_grid,
)

__all__ = [
    "Homography",
    "compute_homography",
    "load_homography",
    "homography_from_config",
    "BEVGrid",
    "ground_to_bev",
    "bev_to_ground",
    "points_to_bev_grid",
]
