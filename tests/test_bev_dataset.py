"""
STEP 11-12 test — BEV dataset, presets, and config-driven build.

Run:  py tests/test_bev_dataset.py

Covers:
  * resolve_loss_cfg (modes + ablations A0-A5)
  * BEVDataset on a synthetic video + label file (image/pseudo_bev/camera_mask)
  * temporal sampling (previous frame)
  * build_geometry + build_model + a full forward on a dataset sample
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.homography import Homography
from src.geometry.coordinate import BEVGrid
from src.bev.losses import resolve_loss_cfg
from src.bev.build import build_geometry, build_model
from data.bev_dataset import BEVDataset, bev_collate_fn


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def test_presets():
    print("== loss presets (modes + ablations) ==")
    proposed = resolve_loss_cfg({"mode": "proposed", "ablation": "a0"})
    _assert(all(proposed[k] == 1.0 for k in [
        "pseudo_weight", "cvp_cycle_weight", "cycle_weight", "corr_weight", "temporal_weight"]),
        "proposed: all five terms active")

    yang = resolve_loss_cfg({"mode": "yang", "ablation": "a0"})
    _assert(yang["pseudo_weight"] == 1.0, "yang: pseudo on")
    _assert(all(yang[k] == 0.0 for k in [
        "cvp_cycle_weight", "cycle_weight", "corr_weight", "temporal_weight"]),
        "yang: all cycle/corr/temporal off")

    a1 = resolve_loss_cfg({"mode": "proposed", "ablation": "a1"})
    _assert(a1["cycle_weight"] == 0.0 and a1["cvp_cycle_weight"] == 1.0,
            "a1: only camera<->BEV cycle disabled")

    a5 = resolve_loss_cfg({"mode": "proposed", "ablation": "a5"})
    _assert(a5["pseudo_weight"] == 0.0 and a5["cycle_weight"] == 1.0,
            "a5: only pseudo disabled")


def _make_synthetic(tmp):
    """Create a tiny video + Ultralytics label file. Returns (video_dir, label_dir)."""
    import cv2
    video_dir = Path(tmp) / "videos"
    label_dir = Path(tmp) / "labels"
    video_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    w, h, n = 320, 180, 10
    video_path = None
    for ext, fourcc in ((".avi", "MJPG"), (".mp4", "mp4v")):
        p = video_dir / f"test_video{ext}"
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), 10, (w, h))
        if writer.isOpened():
            video_path = p
            break
        writer.release()
    assert video_path is not None, "no OpenCV video codec available"

    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (128 + i * 2, 99), (192 + i * 2, 153), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    # label: one car per frame, moving right, plus a pedestrian on frame 1
    with open(label_dir / "test_video.txt", "w") as f:
        for fid in range(1, n + 1):
            f.write(f"### Frame: blurred_{fid}.txt ###\n")
            f.write(f"2 0.5 0.7 0.2 0.3 1\n")      # car
            if fid == 1:
                f.write(f"0 0.3 0.8 0.1 0.2 2\n")  # pedestrian
    return str(video_dir), str(label_dir)


def test_dataset(tmp):
    print("== BEVDataset (image / pseudo_bev / camera_mask) ==")
    video_dir, label_dir = _make_synthetic(tmp)
    H = Homography.from_planar_scale(meters_per_pixel=0.05, origin_u=160.0, origin_v=180.0)
    grid = BEVGrid(x_min=-10.0, x_max=10.0, y_min=-20.0, y_max=0.0, resolution=0.25)

    ds = BEVDataset(
        video_dir=video_dir, label_dir=label_dir,
        video_names=["test_video"], homography=H, grid=grid,
        input_h=96, input_w=128, img_h=180, img_w=320,
        mask_h=96, mask_w=128, sigma=1.5, temporal=False,
    )
    _assert(len(ds) == 10, f"10 frames, got {len(ds)}")

    s = ds[0]
    _assert(s["image"].shape == (3, 96, 128), f"image {tuple(s['image'].shape)}")
    _assert(s["pseudo_bev"].shape == (2, 80, 80), f"pseudo_bev {tuple(s['pseudo_bev'].shape)}")
    _assert(s["camera_mask"].shape == (2, 96, 128), f"camera_mask {tuple(s['camera_mask'].shape)}")
    _assert(torch.isfinite(s["image"]).all(), "image finite")
    # car present → vehicle channel has mass
    _assert(s["pseudo_bev"][1].sum() > 0, "vehicle channel has pseudo-BEV mass")

    batch = bev_collate_fn([ds[0], ds[1]])
    _assert(batch["image"].shape == (2, 3, 96, 128), f"collated image {tuple(batch['image'].shape)}")


def test_dataset_temporal(tmp):
    print("== temporal sampling ==")
    video_dir, label_dir = _make_synthetic(tmp)
    H = Homography.from_planar_scale(meters_per_pixel=0.05, origin_u=160.0, origin_v=180.0)
    grid = BEVGrid(x_min=-10.0, x_max=10.0, y_min=-20.0, y_max=0.0, resolution=0.25)

    ds = BEVDataset(
        video_dir=video_dir, label_dir=label_dir,
        video_names=["test_video"], homography=H, grid=grid,
        input_h=96, input_w=128, img_h=180, img_w=320,
        mask_h=96, mask_w=128, sigma=1.5, temporal=True,
    )
    _assert(len(ds) == 9, f"temporal: 9 frames (needs prev), got {len(ds)}")
    s = ds[0]
    _assert(s["image_prev"].shape == (3, 96, 128), "image_prev present")
    _assert(s["pseudo_bev_prev"].shape == (2, 80, 80), "pseudo_bev_prev present")


def test_build_model(tmp):
    print("== config-driven build + forward ==")
    video_dir, label_dir = _make_synthetic(tmp)
    config = {
        "mode": "proposed", "ablation": "a0",
        "homography": {"enabled": True, "mode": "planar_scale",
                       "meters_per_pixel": 0.05, "origin_u": 160.0, "origin_v": 180.0},
        "bev": {"x_min": -10.0, "x_max": 10.0, "y_min": -20.0, "y_max": 0.0, "resolution": 0.25},
        "model": {"backbone": "resnet18", "pretrained": False, "feature_dim": 64,
                  "input_h": 96, "input_w": 128, "num_classes": 2,
                  "bev_feature_h": 20, "bev_feature_w": 20,
                  "cvp_hidden_dim": 64, "cvt_num_heads": 4, "cvt_num_layers": 1,
                  "cvt_ff_dim": 128, "cvt_dropout": 0.0, "decoder_hidden_dim": 64},
        "data": {"bev": {"video_dir": video_dir, "label_dir": label_dir,
                         "img_h": 180, "img_w": 320,
                         "input_h": 96, "input_w": 128,
                         "mask_h": 96, "mask_w": 128,
                         "sigma": 1.5, "normalize": True, "temporal": False}},
        "loss": {},
        "training": {"batch_size": 2, "num_workers": 0},
    }
    H, grid = build_geometry(config)
    _assert(grid.shape == (80, 80), f"grid shape {grid.shape}")

    model = build_model(config, H, grid)
    _assert(model is not None, "model built")

    ds = BEVDataset(video_dir=video_dir, label_dir=label_dir,
                    video_names=["test_video"], homography=H, grid=grid,
                    input_h=96, input_w=128, img_h=180, img_w=320,
                    mask_h=96, mask_w=128, sigma=1.5, temporal=False)
    batch = bev_collate_fn([ds[0], ds[1]])
    out = model(batch["image"], return_cycle=True)
    _assert(out["pred_bev"].shape == (2, 2, 80, 80), f"pred_bev {tuple(out['pred_bev'].shape)}")
    _assert(out["pred_cam"].shape == (2, 2, 96, 128), f"pred_cam {tuple(out['pred_cam'].shape)}")


if __name__ == "__main__":
    test_presets()
    with tempfile.TemporaryDirectory() as tmp:
        test_dataset(tmp)
        test_dataset_temporal(tmp)
        test_build_model(tmp)
    print("\nALL BEV DATASET / PRESET / BUILD TESTS PASSED")
