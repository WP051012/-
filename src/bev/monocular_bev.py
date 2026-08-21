"""
MonocularBEV — full model assembly.

Pipeline:

    image ──▶ ResNetEncoder ──▶ F_cam
                                  │
                                  ▼
                        CycledViewProjection (CVP) ──▶ F_bev_init, F_cam_rec
                                  │                        └── L_cvp_cycle
                                  ▼
                     CrossViewTransformer (CVT) ──▶ F_bev_ref, F_cam_ref
                                  │
                                  ▼
                           BEVDecoder ──▶ logits ──▶ pred_bev (sigmoid)
                                                      │
                          CameraBEVProjection (H⁻¹) ◀─┘  └── L_cycle (camera⇄BEV)

Modes and ablations (A0–A5) are realised purely by which losses are switched on
in config — the network graph is shared, nothing is deleted.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .encoder import ResNetEncoder, build_encoder
from .cvp import CycledViewProjection
from .cvt import CrossViewTransformer
from .bev_decoder import BEVDecoder
from .camera_bev_projection import CameraBEVProjection


class MonocularBEV(nn.Module):
    def __init__(self, encoder: nn.Module, cvp: nn.Module, cvt: nn.Module,
                 decoder: nn.Module, camera_bev_proj: Optional[CameraBEVProjection] = None):
        super().__init__()
        self.encoder = encoder
        self.cvp = cvp
        self.cvt = cvt
        self.decoder = decoder
        self.camera_bev_proj = camera_bev_proj
        self.num_classes = decoder.num_classes

    def forward(self, image: torch.Tensor, return_cycle: bool = False) -> dict:
        """image (B, 3, H, W) → dict of intermediate + final predictions.

        Keys
        ----
        pred_logits : (B, C, H_bev, W_bev) decoder logits.
        pred_bev    : (B, C, H_bev, W_bev) sigmoid heatmap in [0, 1].
        F_cam       : (B, D, cam_h, cam_w) encoder feature.
        F_bev       : (B, D, bev_h, bev_w) CVP forward projection.
        F_cam_rec   : (B, D, cam_h, cam_w) CVP cycle reconstruction.
        F_bev_ref   : (B, D, bev_h, bev_w) CVT-refined BEV feature.
        F_cam_ref   : (B, D, cam_h, cam_w) CVT-refined camera feature.
        pred_cam    : (B, C, mask_h, mask_w) pred_bev warped back to camera
                      (only when ``return_cycle`` and a projection is attached).
        """
        F_cam = self.encoder(image)
        F_bev, F_cam_rec = self.cvp(F_cam)
        F_bev_ref, F_cam_ref = self.cvt(F_bev, F_cam)
        logits = self.decoder(F_bev_ref)

        out = {
            "pred_logits": logits,
            "pred_bev": torch.sigmoid(logits),
            "F_cam": F_cam,
            "F_bev": F_bev,
            "F_cam_rec": F_cam_rec,
            "F_bev_ref": F_bev_ref,
            "F_cam_ref": F_cam_ref,
        }
        if return_cycle and self.camera_bev_proj is not None:
            out["pred_cam"] = self.camera_bev_proj.bev_to_camera(out["pred_bev"])
        return out

    @torch.no_grad()
    def predict_bev(self, image: torch.Tensor) -> torch.Tensor:
        """Inference: return the (B, C, H_bev, W_bev) sigmoid heatmap only."""
        return self.forward(image)["pred_bev"]


def build_monocular_bev(cfg: dict, homography=None, grid=None,
                        mask_h: Optional[int] = None,
                        mask_w: Optional[int] = None,
                        img_h: Optional[int] = None,
                        img_w: Optional[int] = None) -> MonocularBEV:
    """Assemble a MonocularBEV from a config dict.

    Reads the ``model:`` and ``bev:`` config blocks. The homography, grid and
    image sizes can be passed explicitly or resolved from config (``bev:``,
    ``homography:`` blocks). The cycle projection is attached only when a
    homography + grid are available.
    """
    m = cfg.get("model", cfg)
    bev_cfg = cfg.get("bev", {})

    feature_dim = int(m.get("feature_dim", 256))
    input_h = int(m.get("input_h", 480))
    input_w = int(m.get("input_w", 640))
    num_classes = int(m.get("num_classes", 2))

    encoder = build_encoder(cfg)

    # Camera feature spatial size (ResNet stride 32).
    cam_h = input_h // encoder.stride
    cam_w = input_w // encoder.stride

    # BEV feature spatial size (before decoder; intermediate CVP/CVT grid).
    if grid is not None:
        tgt_h, tgt_w = grid.height, grid.width
    else:
        tgt_h = int(bev_cfg.get("bev_h", 120))
        tgt_w = int(bev_cfg.get("bev_w", 200))
    bev_h = int(m.get("bev_feature_h", tgt_h // 4))
    bev_w = int(m.get("bev_feature_w", tgt_w // 4))

    cvp = CycledViewProjection(
        cam_dim=feature_dim, bev_dim=feature_dim,
        cam_h=cam_h, cam_w=cam_w, bev_h=bev_h, bev_w=bev_w,
        hidden_dim=int(m.get("cvp_hidden_dim", 256)),
    )
    cvt = CrossViewTransformer(
        bev_dim=feature_dim, cam_dim=feature_dim,
        num_heads=int(m.get("cvt_num_heads", 4)),
        num_layers=int(m.get("cvt_num_layers", 1)),
        ff_dim=int(m.get("cvt_ff_dim", 512)),
        dropout=float(m.get("cvt_dropout", 0.1)),
    )
    decoder = BEVDecoder(
        bev_dim=feature_dim, num_classes=num_classes,
        target_h=tgt_h, target_w=tgt_w,
        hidden_dim=int(m.get("decoder_hidden_dim", 128)),
    )

    camera_bev_proj = None
    if homography is not None and grid is not None:
        if mask_h is None:
            mask_h = int(cfg.get("mask_h", input_h))
        if mask_w is None:
            mask_w = int(cfg.get("mask_w", input_w))
        if img_h is None:
            img_h = int(cfg.get("img_h", input_h))
        if img_w is None:
            img_w = int(cfg.get("img_w", input_w))
        camera_bev_proj = CameraBEVProjection(
            homography, grid, mask_h=mask_h, mask_w=mask_w, img_h=img_h, img_w=img_w
        )

    return MonocularBEV(encoder, cvp, cvt, decoder, camera_bev_proj)
