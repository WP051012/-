"""
YOLO-based object detector for traffic scenes.

Supports YOLOv8/YOLO11 via ultralytics, with class mapping from
COCO categories to traffic-specific classes (pedestrian, vehicle,
traffic light, traffic sign, etc.).

References:
    STRR: Spatiotemporal Relationship Reasoning for Pedestrian Intent Prediction
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default class mapping: COCO → traffic scene categories
# ---------------------------------------------------------------------------
DEFAULT_CLASS_MAPPING = {
    # COCO classes → traffic scene labels
    0:  "pedestrian",       # person
    1:  "bicycle",          # bicycle
    2:  "car",              # car
    3:  "motorcycle",       # motorcycle
    5:  "bus",              # bus
    7:  "truck",            # truck
    9:  "traffic_light",    # traffic light (COCO class 9)
    # Custom classes (require fine-tuned model for additional types)
    100: "traffic_sign",    # custom
    101: "lane_line",       # custom (if detectable)
}

# Super-category grouping
SUPER_CATEGORIES = {
    "vehicle":        {"bicycle", "motorcycle", "car", "bus", "truck"},
    "person":         {"pedestrian"},
    "infrastructure": {"traffic_light", "traffic_sign", "lane_line"},
}

# Reverse mapping: label → super-category
LABEL_TO_SUPER = {}
for super_cat, labels in SUPER_CATEGORIES.items():
    for lbl in labels:
        LABEL_TO_SUPER[lbl] = super_cat


class DetectionResult:
    """Single-frame detection result for one object."""

    __slots__ = (
        "bbox", "class_id", "class_name", "confidence",
        "track_id", "feature",
    )

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],   # (x1, y1, x2, y2)
        class_id: int,
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None,
        feature: Optional[np.ndarray] = None,       # appearance feature vector
    ):
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.track_id = track_id
        self.feature = feature

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding-box center (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground-contact point: (x_center, y_bottom) = ((x1+x2)/2, y2).

        This is the approximate contact point of the object with the road
        plane and is the default projection point for homography → BEV
        (NOT the box center).
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def super_category(self) -> str:
        return LABEL_TO_SUPER.get(self.class_name, "unknown")

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "center": list(self.center),
            "super_category": self.super_category,
        }

    def __repr__(self) -> str:
        tid = f" tid={self.track_id}" if self.track_id is not None else ""
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox}{tid})"
        )


class YOLODetector:
    """
    YOLO-based object detector for traffic scenes.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights (.pt) or model name (e.g. "yolov8n.pt").
    class_mapping : dict, optional
        Mapping from YOLO class IDs to traffic class labels.
        Defaults to DEFAULT_CLASS_MAPPING.
    conf_threshold : float
        Confidence threshold for detections.
    iou_threshold : float
        IoU threshold for NMS.
    img_size : int or tuple
        Inference image size.
    device : str
        Device string ("cuda", "cpu", "cuda:0", etc.).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        class_mapping: Optional[Dict[int, str]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "cuda",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Resolve device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        # Load model
        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Class mapping
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        # Build inverse: name → id
        self.name_to_id = {v: k for k, v in self.class_mapping.items()}

        # Extract class names from model if using default COCO model
        self._model_class_names = self._load_model_class_names()

    def _load_model_class_names(self) -> Dict[int, str]:
        """Retrieve class names from the loaded YOLO model."""
        try:
            return self.model.names
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, np.ndarray],
        classes: Optional[List[str]] = None,
        return_features: bool = False,
    ) -> List[DetectionResult]:
        """
        Run detection on a single image.

        Parameters
        ----------
        image : str or np.ndarray
            Image path or BGR numpy array.
        classes : list of str, optional
            If given, only return detections of these traffic class names
            (e.g. ["pedestrian", "car"]).
        return_features : bool
            If True, extract appearance features for each detection.

        Returns
        -------
        list of DetectionResult
        """
        # Determine target class IDs for YOLO filtering
        target_ids = None
        if classes is not None:
            target_ids = [
                cid for cid, cname in self.class_mapping.items()
                if cname in classes
            ]

        # YOLO inference
        results = self.model(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_ids,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # Map to traffic class name
                cls_name = self.class_mapping.get(cls_id, self._model_class_names.get(cls_id, "unknown"))

                # Skip if not in requested classes
                if classes is not None and cls_name not in classes:
                    continue

                bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

                # Extract appearance feature (optional)
                feature = None
                if return_features and hasattr(self.model.model, 'model'):
                    feature = self._extract_feature(image, bbox)

                detections.append(DetectionResult(
                    bbox=bbox,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    feature=feature,
                ))

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        classes: Optional[List[str]] = None,
    ) -> List[List[DetectionResult]]:
        """Run detection on a batch of images."""
        return [self.detect(img, classes=classes) for img in images]

    # ------------------------------------------------------------------
    # Feature extraction (appearance embedding via backbone)
    # ------------------------------------------------------------------

    def _extract_feature(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Extract appearance feature for a bounding-box region using the
        YOLO backbone (ResNet-like feature extractor).

        Returns a 1-D feature vector or None if extraction fails.
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Resize to model input size
            crop_resized = cv2.resize(crop_rgb, (self.img_size, self.img_size))
            # Normalise and convert to tensor
            tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            # Forward through backbone only
            with torch.no_grad():
                # Access model backbone — works for ultralytics>=8
                model = self.model.model
                if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
                    # YOLOv8/v11 structure: model.model[0:10] is backbone
                    backbone = model.model[:10]
                    feat = tensor
                    for layer in backbone:
                        feat = layer(feat)
                    # Global average pooling
                    feature = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()
                    return feature

            return None
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def traffic_classes(self) -> List[str]:
        """All recognised traffic class names."""
        return list(set(self.class_mapping.values()))

    def filter_by_super_category(
        self,
        detections: List[DetectionResult],
        super_category: str,
    ) -> List[DetectionResult]:
        """Filter detections by super-category."""
        return [d for d in detections if d.super_category == super_category]

    def split_by_category(
        self,
        detections: List[DetectionResult],
    ) -> Dict[str, List[DetectionResult]]:
        """Group detections by class name."""
        groups: Dict[str, List[DetectionResult]] = {}
        for d in detections:
            groups.setdefault(d.class_name, []).append(d)
        return groups


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_detector(config: dict) -> YOLODetector:
    """Create a YOLODetector from a configuration dictionary."""
    det_cfg = config.get("detection", {})

    model_path = det_cfg.get("fine_tuned_model") or det_cfg.get("model_name", "yolov8n.pt")

    return YOLODetector(
        model_path=model_path,
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        img_size=det_cfg.get("img_size", 640),
        device=det_cfg.get("device", "cuda"),
    )
