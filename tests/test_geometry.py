"""
STEP 2 test — homography + BEV grid coordinate transforms.

Run:  py tests/test_geometry.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.homography import Homography, compute_homography, homography_from_config
from src.geometry.coordinate import BEVGrid, ground_to_bev, bev_to_ground


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def test_planar_scale_roundtrip():
    print("== planar_scale homography round-trip ==")
    mpp = 1.0 / 47.5  # pix_per_meter = 47.5
    H = Homography.from_planar_scale(mpp, origin_u=1920.0, origin_v=1000.0)
    _assert(H.validate_homography(), "H is valid")

    pts = np.array([[1920.0, 1000.0], [1920 + 475, 1000], [1500, 1500]])
    ground = H.pixel_to_ground(pts)
    back = H.ground_to_pixel(ground)
    err = np.max(np.abs(back - pts))
    _assert(err < 1e-4, f"pixel→ground→pixel round-trip (max err {err:.2e})")

    # sanity: a point 475 px right of origin should be ~10 m lateral
    X, Y = ground[1]
    _assert(abs(X - 10.0) < 1e-3, f"lateral scale: X={X:.4f} ≈ 10 m")
    print(f"  ground[1] = ({X:.3f}, {Y:.3f})")


def test_dlt_homography():
    print("== DLT homography from correspondences ==")
    # Ground-truth H (image → ground) then invert to make correspondences.
    H_true = Homography.from_planar_scale(0.02, origin_u=100.0, origin_v=50.0)
    uv = np.random.RandomState(0).uniform(0, 500, size=(8, 2))
    XY = H_true.pixel_to_ground(uv)
    H_est = compute_homography(uv, XY, method="dlt")
    _assert(np.isfinite(H_est).all(), "DLT H finite")
    # Reprojection error should be tiny (exact correspondences)
    Hc = Homography(H_est)
    err = np.max(np.abs(Hc.pixel_to_ground(uv) - XY))
    _assert(err < 1e-5, f"DLT reconstruction (max err {err:.2e})")


def test_bev_grid():
    print("== BEV grid coordinate mapping ==")
    grid = BEVGrid(x_min=-20, x_max=20, y_min=0, y_max=60, resolution=0.2)
    grid.validate()
    _assert(grid.shape == (300, 200), f"grid shape (H,W)={grid.shape} == (300,200)")

    X, Y = np.array([0.0]), np.array([30.0])
    bev = ground_to_bev(X, Y, grid)
    _assert(abs(bev[0, 0] - 100) < 1e-6, f"bev_x={bev[0,0]:.2f} == 100")
    _assert(abs(bev[0, 1] - 150) < 1e-6, f"bev_y={bev[0,1]:.2f} == 150")

    back = bev_to_ground(bev[:, 0], bev[:, 1], grid)
    _assert(np.allclose(back[0], [0.0, 30.0]), f"bev→ground inverse {back[0]}")


def test_config_construction():
    print("== homography_from_config ==")
    cfg = {"enabled": True, "mode": "planar_scale", "pix_per_meter": 47.5}
    H = homography_from_config(cfg)
    _assert(H is not None and H.validate_homography(), "config planar_scale builds valid H")

    cfg2 = {"enabled": False}
    _assert(homography_from_config(cfg2) is None, "disabled homography returns None")


if __name__ == "__main__":
    test_planar_scale_roundtrip()
    test_dlt_homography()
    test_bev_grid()
    test_config_construction()
    print("\nALL GEOMETRY TESTS PASSED")
