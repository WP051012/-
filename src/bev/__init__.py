"""
Monocular Camera → BEV package.

Geometry-guided weakly-supervised monocular BEV, built on the Yang-style
CVP/CVT backbone with Yan-style cross-view cycle consistency and temporal
consistency. Homography + detection produce *pseudo*-BEV supervision only —
never ground truth.

Submodules
----------
pseudo_bev.py       : detection bottom-center → homography → pseudo-BEV heatmap
label_reader.py     : read Ultralytics tracking label .txt files (reuses the
                      existing detection+tracking output — no re-detection)
encoder.py          : ResNet camera encoder
cvp.py              : Cycled View Projection (forward/backward/cycle)
cvt.py              : Cross-View Transformer (BEV-query cross attention)
bev_decoder.py      : BEV feature → object/semantic heatmaps
camera_bev_projection.py : differentiable BEV⇄camera warping via homography grid
monocular_bev.py    : full model assembling the above
"""

from .pseudo_bev import (
    BEV_CLASSES,
    VEHICLE_CLASS_NAMES,
    class_name_to_channel,
    bottom_center_from_bbox,
    gaussian_heatmap,
    generate_pseudo_bev,
)
from .label_reader import LabelReader, read_frame_detections
from .camera_bev_projection import CameraBEVProjection, rasterize_camera_mask
from .encoder import ResNetEncoder, build_encoder
from .cvp import CycledViewProjection
from .cvt import CrossViewTransformer
from .bev_decoder import BEVDecoder
from .monocular_bev import MonocularBEV, build_monocular_bev

__all__ = [
    "BEV_CLASSES",
    "VEHICLE_CLASS_NAMES",
    "class_name_to_channel",
    "bottom_center_from_bbox",
    "gaussian_heatmap",
    "generate_pseudo_bev",
    "LabelReader",
    "read_frame_detections",
    "CameraBEVProjection",
    "rasterize_camera_mask",
    "ResNetEncoder",
    "build_encoder",
    "CycledViewProjection",
    "CrossViewTransformer",
    "BEVDecoder",
    "MonocularBEV",
    "build_monocular_bev",
]
