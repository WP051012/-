"""
STEP 5-10 test — losses, camera⇄BEV cycle projection, and full model assembly.

Run:  py tests/test_losses.py

Covers:
  * rasterize_camera_mask            (pseudo camera mask)
  * CameraBEVProjection              (homography-guided differentiable warp)
  * camera → BEV → camera round-trip sanity
  * all five losses                  (pseudo / cvp_cycle / cycle / corr / temporal)
  * MonocularBEV end-to-end forward + compute_losses + backward (no NaN/Inf)
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.homography import Homography
from src.geometry.coordinate import BEVGrid
from src.bev.pseudo_bev import generate_pseudo_bev
from src.bev.camera_bev_projection import CameraBEVProjection, rasterize_camera_mask
from src.bev.monocular_bev import build_monocular_bev
from src.bev.losses import compute_losses


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def _make_setup():
    # Fixed front-facing camera: pixel (320, 600) is the ground origin, 0.05 m/px.
    H = Homography.from_planar_scale(meters_per_pixel=0.05, origin_u=320.0, origin_v=600.0)
    grid = BEVGrid(x_min=-10.0, x_max=10.0, y_min=0.0, y_max=20.0, resolution=0.25)
    # full-res image 720x1280; cycle mask at half resolution 360x640.
    return H, grid, dict(mask_h=360, mask_w=640, img_h=720, img_w=1280)


def _detections(car_u, car_v):
    """car bottom-center at (car_u, car_v); plus a fixed pedestrian."""
    return [
        {"bbox": [car_u - 40, car_v - 80, car_u + 40, car_v], "class_name": "car",
         "track_id": 1},
        {"bbox": [360, 660, 440, 760], "class_name": "pedestrian", "track_id": 2},
    ]


def test_camera_mask():
    print("== pseudo camera mask ==")
    H, grid, sizes = _make_setup()
    dets = _detections(320, 640)
    mask = rasterize_camera_mask(dets, **sizes)
    _assert(mask.shape == (2, 360, 640), f"mask shape {mask.shape}")
    _assert(np.isfinite(mask).all() and mask.max() <= 1.0, "mask in [0,1]")
    # car channel has a filled box somewhere in the middle
    _assert(mask[1].sum() > 100, "car box rasterised (vehicle channel 1)")
    _assert(mask[0].sum() > 100, "pedestrian box rasterised (channel 0)")


def test_projection_round_trip():
    print("== camera <-> BEV differentiable projection ==")
    H, grid, sizes = _make_setup()
    dets = _detections(320, 640)
    pb = generate_pseudo_bev(dets, H, grid, sigma=1.5)
    proj = CameraBEVProjection(H, grid, **sizes)

    cam_mask = torch.from_numpy(
        rasterize_camera_mask(dets, **sizes)
    ).unsqueeze(0).requires_grad_(True)  # (1,2,360,640)
    bev = torch.from_numpy(pb.heatmap).unsqueeze(0).requires_grad_(True)  # (1,2,80,80)

    b2c = proj.bev_to_camera(bev)        # (1,2,360,640)
    c2b = proj.camera_to_bev(cam_mask)   # (1,2,80,80)
    _assert(b2c.shape == (1, 2, 360, 640), f"bev_to_camera {tuple(b2c.shape)}")
    _assert(c2b.shape == (1, 2, 80, 80), f"camera_to_bev {tuple(c2b.shape)}")
    _assert(torch.isfinite(b2c).all() and torch.isfinite(c2b).all(), "warps finite")

    # round trip: BEV -> camera -> BEV should keep the car peak near its cell
    back = proj.camera_to_bev(b2c)[0, 1]          # (80,80) vehicle channel
    peak = torch.argmax(back).item()
    cy, cx = peak // 80, peak % 80
    _assert(30 <= cx <= 50 and 0 <= cy <= 20, f"car peak stays near (40,8), got ({cx},{cy})")

    # warp is differentiable: gradients flow back to the input tensors
    (b2c.mean() + c2b.mean()).backward()
    _assert(bev.grad is not None and torch.isfinite(bev.grad).all(),
            "bev_to_camera gradient finite")
    _assert(cam_mask.grad is not None and torch.isfinite(cam_mask.grad).all(),
            "camera_to_bev gradient finite")


def test_losses_and_model():
    print("== full model + all losses + backward ==")
    H, grid, sizes = _make_setup()

    cfg = {
        "model": {
            "backbone": "resnet18", "pretrained": False,
            "feature_dim": 64, "input_h": 96, "input_w": 128,
            "num_classes": 2,
            "bev_feature_h": 20, "bev_feature_w": 20,
            "cvp_hidden_dim": 64,
            "cvt_num_heads": 4, "cvt_num_layers": 1,
            "cvt_ff_dim": 128, "cvt_dropout": 0.0,
            "decoder_hidden_dim": 64,
        },
        "bev": {"bev_h": 80, "bev_w": 80},
    }
    model = build_monocular_bev(cfg, H, grid, **sizes)

    # t-1 (car moved left) and t (car at origin)
    pb_prev = generate_pseudo_bev(_detections(280, 640), H, grid, sigma=1.5)
    pb_t = generate_pseudo_bev(_detections(320, 640), H, grid, sigma=1.5)

    batch = {
        "pseudo_bev": torch.from_numpy(pb_t.heatmap).unsqueeze(0),
        "camera_mask": torch.from_numpy(rasterize_camera_mask(_detections(320, 640), **sizes)).unsqueeze(0),
        "pseudo_bev_prev": torch.from_numpy(pb_prev.heatmap).unsqueeze(0),
    }

    img = torch.randn(1, 3, 96, 128)
    img_prev = torch.randn(1, 3, 96, 128)
    out = model(img, return_cycle=True)
    out_prev = model(img_prev)
    batch["pred_bev_prev"] = out_prev["pred_bev"]

    _assert(out["pred_bev"].shape == (1, 2, 80, 80), f"pred_bev {tuple(out['pred_bev'].shape)}")
    _assert(out["pred_cam"].shape == (1, 2, 360, 640), f"pred_cam {tuple(out['pred_cam'].shape)}")
    _assert(out["pred_bev"].min() >= 0 and out["pred_bev"].max() <= 1, "pred_bev in [0,1]")

    loss_cfg = {
        "pseudo_weight": 1.0, "pseudo_mode": "focal",
        "cvp_cycle_weight": 1.0,
        "cycle_weight": 1.0, "cycle_dice_weight": 1.0,
        "corr_weight": 1.0,
        "temporal_weight": 1.0,
    }
    losses = compute_losses(out, batch, loss_cfg, model=model)
    for name in ["L_pseudo", "L_cvp_cycle", "L_cycle", "L_corr", "L_temporal"]:
        _assert(name in losses, f"{name} present")
        _assert(torch.isfinite(losses[name]), f"{name} finite (={float(losses[name]):.4f})")

    losses["total"].backward()
    # every trainable param got a finite grad
    grads_ok = all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    _assert(grads_ok, "loss.backward() -> all gradients finite (no NaN/Inf)")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    _assert(has_grad, "some parameter received a non-zero gradient")


if __name__ == "__main__":
    test_camera_mask()
    test_projection_round_trip()
    test_losses_and_model()
    print("\nALL LOSS / CYCLE / MODEL TESTS PASSED")
