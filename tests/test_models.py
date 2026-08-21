"""
STEP 4 test — Encoder / CVP / CVT / BEV Decoder shape + forward checks.

Run:  py tests/test_models.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bev.encoder import ResNetEncoder
from src.bev.cvp import CycledViewProjection
from src.bev.cvt import CrossViewTransformer
from src.bev.bev_decoder import BEVDecoder


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def test_encoder():
    print("== encoder ==")
    enc = ResNetEncoder("resnet18", pretrained=False, out_channels=256)
    x = torch.randn(2, 3, 480, 640)
    f = enc(x)
    _assert(f.shape == (2, 256, 15, 20), f"F_cam shape {tuple(f.shape)} == (2,256,15,20)")
    _assert(torch.isfinite(f).all(), "F_cam finite")


def test_cvp():
    print("== CVP (cycled view projection) ==")
    cvp = CycledViewProjection(256, 256, cam_h=15, cam_w=20, bev_h=30, bev_w=50)
    F_cam = torch.randn(2, 256, 15, 20)
    F_bev, F_cam_rec = cvp(F_cam)
    _assert(F_bev.shape == (2, 256, 30, 50), f"F_bev shape {tuple(F_bev.shape)}")
    _assert(F_cam_rec.shape == (2, 256, 15, 20), f"F_cam_rec shape {tuple(F_cam_rec.shape)}")
    _assert(torch.isfinite(F_bev).all() and torch.isfinite(F_cam_rec).all(),
            "CVP outputs finite")
    # cycle loss path is differentiable
    loss = (F_cam - F_cam_rec).abs().mean()
    loss.backward()
    _assert(cvp.P_h.grad is not None and torch.isfinite(cvp.P_h.grad).all(),
            "CVP cycle loss backprops to projection matrices")


def test_cvt():
    print("== CVT (cross-view transformer) ==")
    cvt = CrossViewTransformer(bev_dim=256, cam_dim=256, num_heads=4, num_layers=1)
    F_bev = torch.randn(2, 256, 30, 50)
    F_cam = torch.randn(2, 256, 15, 20)
    out_bev, out_cam = cvt(F_bev, F_cam)
    _assert(out_bev.shape == F_bev.shape, f"refined BEV shape {tuple(out_bev.shape)}")
    _assert(out_cam.shape == F_cam.shape, f"refined CAM shape {tuple(out_cam.shape)}")
    _assert(torch.isfinite(out_bev).all(), "CVT output finite")
    out_bev.mean().backward()
    _assert(torch.isfinite(list(cvt.parameters())[0].grad).all(), "CVT backprops finite")


def test_decoder():
    print("== BEV decoder ==")
    dec = BEVDecoder(bev_dim=256, num_classes=2, target_h=120, target_w=200)
    F_bev = torch.randn(2, 256, 30, 50)
    logits = dec(F_bev)
    _assert(logits.shape == (2, 2, 120, 200), f"logits shape {tuple(logits.shape)}")
    _assert(torch.isfinite(logits).all(), "logits finite")
    # logits -> sigmoid heatmap matches pseudo-BEV shape (C,H,W)
    heat = torch.sigmoid(logits)
    _assert(0.0 <= heat.min() and heat.max() <= 1.0, "sigmoid in [0,1]")


def test_end_to_end_shapes():
    print("== end-to-end shape chain ==")
    enc = ResNetEncoder("resnet18", pretrained=False, out_channels=256)
    cvp = CycledViewProjection(256, 256, cam_h=15, cam_w=20, bev_h=30, bev_w=50)
    cvt = CrossViewTransformer(256, 256, num_heads=4, num_layers=1)
    dec = BEVDecoder(256, 2, target_h=120, target_w=200)

    x = torch.randn(2, 3, 480, 640)
    F_cam = enc(x)
    F_bev, F_cam_rec = cvp(F_cam)
    F_bev_ref, F_cam_ref = cvt(F_bev, F_cam)
    pred = dec(F_bev_ref)
    _assert(pred.shape == (2, 2, 120, 200), f"final pred {tuple(pred.shape)}")
    print("  chain: (3,480,640) → F_cam(256,15,20) → F_bev(256,30,50) → pred(2,120,200)")


if __name__ == "__main__":
    test_encoder()
    test_cvp()
    test_cvt()
    test_decoder()
    test_end_to_end_shapes()
    print("\nALL MODEL SHAPE/FORWARD TESTS PASSED")
