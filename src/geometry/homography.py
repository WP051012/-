"""
Homography — image pixel ⇄ ground-plane (meters) projective transform.

Implements the projective map

        s · [X, Y, 1]^T = H · [u, v, 1]^T

where (u, v) are image pixels and (X, Y) are ground-plane meters.

The homography is a *geometry teacher* for the BEV model — it is NEVER treated
as ground truth. Outputs derived from it must be named ``pseudo_bev`` (or
``pseudo_*``), never ``gt_bev``.

Provides
--------
compute_homography      : DLT (+ optional RANSAC) from point correspondences
Homography              : H holder with forward/backward transforms + validation
load_homography         : load H from .npy / .json / config
homography_from_config  : build a Homography from a ``homography:`` config block

A ``planar_scale`` fallback is provided so the full pipeline is runnable before
real calibration: it assumes a fronto-parallel ground plane with a single
``meters_per_pixel`` scale. Real calibration should use
``tools/estimate_homography.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# DLT solver
# ---------------------------------------------------------------------------

def _normalize_points(pts: np.ndarray):
    """Hartley normalisation: translate to centroid, scale so mean |p| = sqrt2."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return T


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform: solve H s.t. dst ~ H @ src.

    Uses Hartley-normalised DLT with an SVD null-space solve (robust, no
    h22=1 assumption). src, dst : (N, 2). Returns (3, 3) H, H[2,2] normalised to 1.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.ndim == 2 and dst.ndim == 2, "src/dst must be (N, 2)"
    assert src.shape[0] == dst.shape[0], "src/dst length mismatch"
    N = src.shape[0]
    assert N >= 4, "homography needs at least 4 correspondences"

    # Normalise both point sets for conditioning.
    T_src = _normalize_points(src)
    T_dst = _normalize_points(dst)

    def _apply_T(T, pts):
        hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        out = hom @ T.T
        return out[:, :2] / out[:, 2:3]

    s = _apply_T(T_src, src)
    d = _apply_T(T_dst, dst)
    u, v = s[:, 0], s[:, 1]
    X, Y = d[:, 0], d[:, 1]

    # Homogeneous DLT system A h = 0, h = [h1; h2; h3] (9 unknowns).
    #   X*(h3·p) - h1·p = 0  →  [-u,-v,-1, 0,0,0, X*u, X*v, X]
    #   Y*(h3·p) - h2·p = 0  →  [0,0,0, -u,-v,-1, Y*u, Y*v, Y]
    A = np.zeros((2 * N, 9), dtype=np.float64)
    A[0::2, 0] = -u; A[0::2, 1] = -v; A[0::2, 2] = -1.0
    A[0::2, 6] = X * u; A[0::2, 7] = X * v; A[0::2, 8] = X
    A[1::2, 3] = -u; A[1::2, 4] = -v; A[1::2, 5] = -1.0
    A[1::2, 6] = Y * u; A[1::2, 7] = Y * v; A[1::2, 8] = Y

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]                       # smallest-singular-value right vector
    H_norm = h.reshape(3, 3)

    # Denormalise: H = T_dst^{-1} @ H_norm @ T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def compute_homography(
    src_pts: Sequence[Sequence[float]],
    dst_pts: Sequence[Sequence[float]],
    method: str = "dlt",
) -> np.ndarray:
    """Compute homography mapping src_pts → dst_pts.

    Parameters
    ----------
    src_pts : (N, 2) image pixel points (u, v).
    dst_pts : (N, 2) ground-plane points (X, Y) meters.
    method : str
        "dlt"     — pure numpy least-squares DLT.
        "ransac"  — OpenCV RANSAC (more robust to outliers; requires cv2).

    Returns
    -------
    (3, 3) homography H with H @ [u,v,1] ∝ [X,Y,1].
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    assert src.shape == dst.shape, "src/dst shapes differ"
    assert src.shape[0] >= 4, "need >= 4 correspondences"

    if method == "ransac":
        try:
            import cv2
        except ImportError as e:  # pragma: no cover
            raise ImportError("cv2 required for ransac homography") from e
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("RANSAC homography failed to find a model")
        return H.astype(np.float64)

    return _dlt(src, dst)


# ---------------------------------------------------------------------------
# Homography container
# ---------------------------------------------------------------------------

class Homography:
    """Homography H mapping image pixels → ground-plane meters.

    ``pixel_to_ground`` / ``ground_to_pixel`` accept a single (2,) point, an
    (N, 2) array, or a list of 2-tuples.
    """

    def __init__(self, H: np.ndarray):
        H = np.asarray(H, dtype=np.float64)
        assert H.shape == (3, 3), f"H must be (3,3), got {H.shape}"
        assert np.isfinite(H).all(), "Homography contains NaN/Inf"
        # Normalise so H[2,2] != 0 and det(H) != 0 (non-degenerate).
        assert abs(np.linalg.det(H)) > 1e-12, "Homography is singular"
        self.H = H
        self.H_inv = np.linalg.inv(H)

    # -- transforms ---------------------------------------------------------

    @staticmethod
    def _as_points(pts) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, 2)
        assert a.ndim == 2 and a.shape[1] == 2, "points must be (N,2) or (2,)"
        return a

    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Projective transform of (N,2) points under H. Returns (N,2)."""
        N = pts.shape[0]
        hom = np.concatenate([pts, np.ones((N, 1))], axis=1)          # (N,3)
        proj = hom @ H.T                                              # (N,3)
        w = proj[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)                     # avoid div0
        out = proj[:, :2] / w
        return out

    def pixel_to_ground(self, pts) -> np.ndarray:
        """Image pixels (u, v) → ground-plane meters (X, Y)."""
        return self._apply(self.H, self._as_points(pts))

    def ground_to_pixel(self, pts) -> np.ndarray:
        """Ground-plane meters (X, Y) → image pixels (u, v)."""
        return self._apply(self.H_inv, self._as_points(pts))

    def transform_points(self, pts, inverse: bool = False) -> np.ndarray:
        """Generic point transform: forward (pixel→ground) or inverse."""
        H = self.H_inv if inverse else self.H
        return self._apply(H, self._as_points(pts))

    # -- validation ---------------------------------------------------------

    def validate_homography(self, tol: float = 1e-6) -> bool:
        """Checks: finite, square (3x3), non-singular, invertible round-trip."""
        H = self.H
        if not np.isfinite(H).all():
            return False
        if H.shape != (3, 3):
            return False
        if abs(np.linalg.det(H)) < tol:
            return False
        # Round-trip identity sanity: H @ H_inv ≈ I
        I = H @ self.H_inv
        if not np.allclose(I, np.eye(3), atol=tol):
            return False
        return True

    def round_trip_error(self, pts) -> float:
        """Mean reprojection error of pixel→ground→pixel (meters, on ground)."""
        pts = self._as_points(pts)
        ground = self.pixel_to_ground(pts)
        back = self.ground_to_pixel(ground)
        return float(np.mean(np.linalg.norm(back - pts, axis=1)))

    def inverse(self) -> "Homography":
        return Homography(self.H_inv)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_planar_scale(
        cls,
        meters_per_pixel: float,
        origin_u: float = 0.0,
        origin_v: float = 0.0,
    ) -> "Homography":
        """Fronto-parallel scale fallback (no perspective).

        Ground origin (X=0, Y=0) sits at image pixel (origin_u, origin_v); a
        single ``meters_per_pixel`` scale maps pixels to meters. This is a
        *placeholder* to keep the pipeline runnable before real calibration —
        NOT a substitute for a calibrated H.
        """
        assert meters_per_pixel > 0, "meters_per_pixel must be positive"
        H = np.array([
            [meters_per_pixel, 0.0, -meters_per_pixel * origin_u],
            [0.0, meters_per_pixel, -meters_per_pixel * origin_v],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return cls(H)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"matrix": self.H.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Homography":
        H = np.asarray(d["matrix"], dtype=np.float64).reshape(3, 3)
        return cls(H)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npy":
            np.save(path, self.H)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            # default: text matrix
            np.savetxt(path, self.H)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Homography":
        path = Path(path)
        if path.suffix == ".npy":
            return cls(np.load(path))
        if path.suffix == ".json":
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        return cls(np.loadtxt(path))


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------

def homography_from_config(cfg: dict) -> Optional[Homography]:
    """Build a Homography from a ``homography:`` config block, or None if disabled.

    Supported keys (all optional):
        enabled: bool
        mode: "matrix" | "planar_scale"
        matrix_path: str           (mode=matrix)
        meters_per_pixel: float    (mode=planar_scale)
        origin_u, origin_v: float  (mode=planar_scale)
        image_width, image_height, bev_width, bev_height : informational
    """
    if cfg is None:
        return None
    if not cfg.get("enabled", True):
        return None

    mode = cfg.get("mode", "planar_scale")
    if mode == "matrix":
        mp = cfg.get("matrix_path")
        if not mp:
            raise ValueError("homography.mode=matrix requires matrix_path")
        return Homography.load(mp)

    mpp = cfg.get("meters_per_pixel", cfg.get("pix_per_meter", 47.5))
    if mpp is None or mpp <= 0:
        raise ValueError("planar_scale homography needs a positive meters_per_pixel")

    # A planar_scale map is really pixels_per_meter, so meters_per_pixel = 1 / ppm.
    # Accept both: if the key is named 'pix_per_meter', interpret as ppm and invert.
    if "meters_per_pixel" in cfg:
        mpp = cfg["meters_per_pixel"]
    else:
        mpp = 1.0 / cfg.get("pix_per_meter", 47.5)

    origin_u = cfg.get("origin_u", 0.0)
    origin_v = cfg.get("origin_v", 0.0)
    return Homography.from_planar_scale(mpp, origin_u, origin_v)


def load_homography(cfg: dict) -> Optional[Homography]:
    """Convenience alias for homography_from_config."""
    return homography_from_config(cfg)
