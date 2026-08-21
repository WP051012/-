"""
STEP 3 test — detection → bottom-center → homography → pseudo-BEV.

Run:  py tests/test_pseudo_bev.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.homography import Homography
from src.geometry.coordinate import BEVGrid
from src.bev.label_reader import LabelReader
from src.bev.pseudo_bev import (
    generate_pseudo_bev,
    bottom_center_from_bbox,
    class_name_to_channel,
    BEV_CLASSES,
)

LABEL_DIR = Path("D:/Red-Light视频数据/labels")
VIS_DIR = Path("visualization")
VIS_DIR.mkdir(exist_ok=True)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def test_bottom_center():
    print("== bottom_center ==")
    _assert(bottom_center_from_bbox((10, 20, 30, 50)) == (20.0, 50.0),
            "bottom_center = (x_center, y2)")
    _assert(class_name_to_channel("pedestrian") == 0, "pedestrian → ch0")
    _assert(class_name_to_channel("car") == 1, "car → ch1 (vehicle)")
    _assert(class_name_to_channel("traffic_light") is None, "traffic_light skipped")


def test_label_reader():
    print("== label reader (real data) ==")
    label_files = sorted(LABEL_DIR.glob("*.txt"))
    _assert(len(label_files) > 0, f"found {len(label_files)} label files")
    reader = LabelReader(3840, 2160)
    frames = reader.load(label_files[0])
    _assert(len(frames) > 0, f"label file has {len(frames)} frames")

    # pick a frame with detections
    frame_id, dets = next(iter(frames.items()))
    _assert(all("bbox" in d and "class_name" in d for d in dets),
            "detections have bbox + class_name")
    # bbox in pixel coords within [0, W]x[0, H]
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        _assert(0 <= x1 < x2 <= 3840 and 0 <= y1 < y2 <= 2160,
                f"bbox within image: {d['bbox']}")
    print(f"  frame {frame_id}: {len(dets)} detections, "
          f"classes={sorted(set(d['class_name'] for d in dets))}")


def test_generate_pseudo_bev():
    print("== pseudo-BEV generation (real data) ==")
    # Use a fronto-parallel scale homography (placeholder, not calibrated).
    mpp = 1.0 / 47.5
    H = Homography.from_planar_scale(mpp, origin_u=0.0, origin_v=0.0)
    grid = BEVGrid(x_min=0.0, x_max=80.0, y_min=0.0, y_max=48.0, resolution=0.4)
    grid.validate()

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    reader = LabelReader(3840, 2160)
    frames = reader.load(label_files[0])
    # accumulate detections from the first few frames to guarantee objects
    dets = []
    for fid in list(frames.keys())[:10]:
        dets.extend(frames[fid])

    res = generate_pseudo_bev(dets, H, grid, sigma=1.5)
    _assert(res.heatmap.shape == (2, grid.height, grid.width),
            f"heatmap shape {res.heatmap.shape} == (2, H, W)")

    n_objects = res.positions.shape[0]
    _assert(n_objects > 0, f"{n_objects} objects projected into BEV")

    for ch in range(2):
        act = res.heatmap[ch].max()
        print(f"  {BEV_CLASSES[ch]} heatmap max={act:.3f}, "
              f"n_cells={int((res.channels == ch).sum())}")

    # positions must be finite and within grid extent
    _assert(np.isfinite(res.positions).all(), "all object positions finite")
    _assert(res.positions[:, 0].min() >= grid.x_min - 1e-3, "X within grid")
    _assert(res.positions[:, 1].min() >= grid.y_min - 1e-3, "Y within grid")

    # at least one channel should be active
    _assert(res.heatmap.sum() > 0.0, "heatmap has nonzero activation")


def test_visualization():
    print("== save pseudo-BEV visualization ==")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mpp = 1.0 / 47.5
    H = Homography.from_planar_scale(mpp, 0.0, 0.0)
    grid = BEVGrid(x_min=0.0, x_max=80.0, y_min=0.0, y_max=48.0, resolution=0.4)
    reader = LabelReader(3840, 2160)
    label_files = sorted(LABEL_DIR.glob("*.txt"))
    frames = reader.load(label_files[0])
    dets = []
    for fid in list(frames.keys())[:20]:
        dets.extend(frames[fid])
    res = generate_pseudo_bev(dets, H, grid, sigma=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ch, ax in enumerate(axes):
        im = ax.imshow(res.heatmap[ch], origin="lower", cmap="hot",
                       extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
        ax.set_title(f"pseudo-BEV: {BEV_CLASSES[ch]}")
        ax.set_xlabel("X (lateral, m)"); ax.set_ylabel("Y (longitudinal, m)")
        fig.colorbar(im, ax=ax)
    out = VIS_DIR / "pseudo_bev_smoke.png"
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)
    _assert(out.exists(), f"saved {out}")


if __name__ == "__main__":
    test_bottom_center()
    test_label_reader()
    test_generate_pseudo_bev()
    test_visualization()
    print("\nALL PSEUDO-BEV TESTS PASSED")
