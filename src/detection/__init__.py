"""
Detection & Tracking Module
----------------------------
YOLO-based object detection + ByteTrack multi-object tracking
for traffic scene understanding.

Classes:
    YOLODetector      — object detector
    ByteTrackWrapper  — multi-object tracker
    TrajectoryManager — trajectory storage & queries
    TrackedObject     — per-frame tracked detection
    Trajectory        — full object trajectory
"""

from .detector import YOLODetector, DetectionResult, create_detector
from .tracker import (
    ByteTrackWrapper,
    TrajectoryManager,
    TrackedObject,
    Trajectory,
)
