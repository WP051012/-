"""
Camera encoder — ResNet backbone producing the camera feature map F_cam.

Reuses torchvision ResNet (not a hand-written CNN). Output spatial size is
``(H_in // stride, W_in // stride)`` with stride 32; an optional 1×1 conv
projects the backbone channels to ``out_channels`` (kept equal to the BEV
feature dim so CVP/CVT share a common token dimension).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

try:
    from torchvision.models import get_model_weights  # noqa: F401
    _HAS_WEIGHTS_API = True
except ImportError:  # older torchvision
    _HAS_WEIGHTS_API = False

RESNET_OUT_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}

# Weight enum name per backbone (torchvision >= 0.13).
_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
}


class ResNetEncoder(nn.Module):
    """ResNet feature encoder for camera images.

    Parameters
    ----------
    backbone : str — one of resnet18/34/50/101.
    pretrained : bool — load ImageNet weights (default True).
    out_channels : int | None — output channels; None keeps backbone channels.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 out_channels: int | None = None):
        super().__init__()
        assert backbone in RESNET_OUT_CHANNELS, f"unknown backbone {backbone}"
        base = self._make_backbone(backbone, pretrained)
        # Drop avgpool + fc, keep conv stem .. layer4.
        self.features = nn.Sequential(*list(base.children())[:-2])

        in_ch = RESNET_OUT_CHANNELS[backbone]
        self.out_channels = out_channels or in_ch
        self.proj = (
            nn.Conv2d(in_ch, self.out_channels, 1)
            if self.out_channels != in_ch else nn.Identity()
        )
        self.stride = 32

    @staticmethod
    def _make_backbone(backbone: str, pretrained: bool):
        if _HAS_WEIGHTS_API:
            weights_cls = getattr(models, _WEIGHT_ENUMS[backbone], None)
            weights = weights_cls.DEFAULT if (pretrained and weights_cls) else None
            return getattr(models, backbone)(weights=weights)
        return getattr(models, backbone)(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → F_cam: (B, out_channels, H//32, W//32)."""
        f = self.features(x)
        return self.proj(f)


def build_encoder(cfg: dict) -> ResNetEncoder:
    """Build a ResNetEncoder from a ``model:`` config block."""
    m = cfg.get("model", cfg)
    return ResNetEncoder(
        backbone=m.get("backbone", "resnet18"),
        pretrained=m.get("pretrained", True),
        out_channels=m.get("encoder_out_channels", m.get("feature_dim", 256)),
    )
