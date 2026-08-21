"""
Multi-object tracker integrating ByteTrack with YOLO detections.

Uses boxmot for ByteTrack/BotSort/StrongSort, producing temporally
consistent track IDs for trajectory extraction.

References:
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box (ECCV 2022)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """A single tracked object at one frame."""
    track_id: int
    class_name: str
    class_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_id: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Trajectory:
    """
    Complete trajectory of one tracked object over multiple frames.

    Attributes
    ----------
    track_id : int
    class_name : str
    frames : list of int
        Frame indices.
    positions : np.ndarray (T, 2)
        World / pixel centre positions (cx, cy) at each frame.
    bboxes : np.ndarray (T, 4)
        Bounding boxes at each frame.
    confidences : np.ndarray (T,)
    """
    track_id: int
    class_name: str
    frames: List[int] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def add(self, obj: TrackedObject) -> None:
        self.frames.append(obj.frame_id)
        self.positions.append(obj.center)
        self.bboxes.append(obj.bbox)
        self.confidences.append(obj.confidence)

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> int:
        """Duration in frames (last - first)."""
        if not self.frames:
            return 0
        return self.frames[-1] - self.frames[0] + 1

    def to_numpy(self):
        """Return numpy arrays for positions, bboxes, confidences."""
        return {
            "frames": np.array(self.frames, dtype=np.int32),
            "positions": np.array(self.positions, dtype=np.float32),
            "bboxes": np.array(self.bboxes, dtype=np.float32),
            "confidences": np.array(self.confidences, dtype=np.float32),
        }

    def __repr__(self) -> str:
        return (
            f"Trajectory(tid={self.track_id}, class={self.class_name}, "
            f"len={self.length}, frames=[{self.frames[0]}...{self.frames[-1]}])"
        )


# ---------------------------------------------------------------------------
# ByteTrack wrapper
# ---------------------------------------------------------------------------

class ByteTrackWrapper:
    """
    Wrapper around boxmot's ByteTrack for traffic scene tracking.

    Parameters
    ----------
    track_buffer : int
        Frames to keep a track alive without detection.
    track_thresh : float
        Detection confidence threshold for high-confidence detections.
    match_thresh : float
        IoU threshold for matching.
    frame_rate : int
        Video frame rate (for time-based decay).
    """

    def __init__(
        self,
        track_buffer: int = 30,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self._tracker = None  # lazy init on first frame

    def _init_tracker(self):
        """Lazy-initialise the ByteTrack tracker."""
        try:
            from boxmot import ByteTrack
            self._tracker = ByteTrack(
                track_buffer=self.track_buffer,
                track_thresh=self.track_thresh,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        except ImportError:
            logger.warning(
                "boxmot not installed. Falling back to SimpleTracker. "
                "Install with: pip install boxmot"
            )
            self._tracker = _SimpleTracker(
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
            )

    def update(
        self,
        detections: np.ndarray,     # (N, 6)  [x1, y1, x2, y2, conf, cls]
        frame: np.ndarray,          # (H, W, 3)  BGR image
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : np.ndarray (N, 6)
            Columns: [x1, y1, x2, y2, confidence, class_id].
        frame : np.ndarray
            Current video frame.

        Returns
        -------
        np.ndarray (M, 7)
            Columns: [x1, y1, x2, y2, track_id, conf, class_id].
        """
        if self._tracker is None:
            self._init_tracker()

        if len(detections) == 0:
            return np.empty((0, 7))

        return self._tracker.update(detections, frame)


# ---------------------------------------------------------------------------
# Simple IoU-based fallback tracker (when boxmot is unavailable)
# ---------------------------------------------------------------------------

class _SimpleTracker:
    """Minimal IoU-based tracker as fallback for ByteTrack."""

    def __init__(self, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.next_id = 0
        self.active_tracks: Dict[int, np.ndarray] = {}   # tid → bbox
        self.lost_tracks: Dict[int, Tuple[np.ndarray, int]] = {}  # tid → (bbox, age)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Returns (M, 7): [x1, y1, x2, y2, tid, conf, cls].
        """
        if len(self.active_tracks) == 0:
            # First frame — initialise all as new tracks
            result = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.active_tracks[tid] = det[:4]
                result.append([*det[:4], tid, det[4], det[5]])
            return np.array(result)

        # Compute IoU matrix
        active_ids = list(self.active_tracks.keys())
        active_boxes = np.array([self.active_tracks[tid] for tid in active_ids])
        det_boxes = detections[:, :4]

        iou_matrix = self._iou_batch(det_boxes, active_boxes)

        matched_det: set = set()
        matched_track: set = set()
        result = []

        # Greedy matching
        for di in range(len(detections)):
            best_iou = self.match_thresh
            best_ti = -1
            for ti in range(len(active_ids)):
                if ti in matched_track:
                    continue
                iou = iou_matrix[di, ti]
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti
            if best_ti >= 0:
                tid = active_ids[best_ti]
                self.active_tracks[tid] = detections[di, :4]
                matched_det.add(di)
                matched_track.add(best_ti)
                result.append([*detections[di, :4], tid, detections[di, 4], detections[di, 5]])

        # Assign new IDs to unmatched detections
        for di in range(len(detections)):
            if di not in matched_det:
                tid = self.next_id
                self.next_id += 1
                self.active_tracks[tid] = detections[di, :4]
                result.append([*detections[di, :4], tid, detections[di, 4], detections[di, 5]])

        # Remove lost tracks
        for ti in range(len(active_ids)):
            if ti not in matched_track:
                tid = active_ids[ti]
                self.lost_tracks[tid] = (self.active_tracks[tid], 0)
                del self.active_tracks[tid]

        # Age lost tracks
        expired = []
        for tid, (bbox, age) in list(self.lost_tracks.items()):
            age += 1
            if age > self.track_buffer:
                expired.append(tid)
            else:
                self.lost_tracks[tid] = (bbox, age)
        for tid in expired:
            del self.lost_tracks[tid]

        return np.array(result) if result else np.empty((0, 7))

    @staticmethod
    def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU between two sets of boxes (xyxy)."""
        xa1, ya1, xa2, ya2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        xb1, yb1, xb2, yb2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)

        # Intersection
        inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
        inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
        inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
        inter_y2 = np.minimum(ya2[:, None], yb2[None, :])

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = area_a[:, None] + area_b[None, :] - inter_area
        return inter_area / np.maximum(union_area, 1e-6)


# ---------------------------------------------------------------------------
# Trajectory Manager
# ---------------------------------------------------------------------------

class TrajectoryManager:
    """
    Manages trajectories across frames — collects per-frame TrackedObjects,
    builds full trajectories, and supports trajectory queries.

    Parameters
    ----------
    min_length : int
        Minimum trajectory length (frames) to retain.
    """

    def __init__(self, min_length: int = 15):
        self.min_length = min_length
        self._trajectories: Dict[int, Trajectory] = {}
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Frame-by-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        tracked_objects: List[TrackedObject],
    ) -> None:
        """
        Register one frame of tracking results.

        Parameters
        ----------
        frame_id : int
            Current frame index.
        tracked_objects : list of TrackedObject
            All tracked objects in this frame.
        """
        self._frame_count += 1

        active_ids = set()
        for obj in tracked_objects:
            active_ids.add(obj.track_id)
            if obj.track_id not in self._trajectories:
                self._trajectories[obj.track_id] = Trajectory(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                )
            self._trajectories[obj.track_id].add(obj)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """Get trajectory by track ID."""
        return self._trajectories.get(track_id)

    def get_active_trajectories(
        self,
        current_frame: int,
        max_age: int = 30,
    ) -> List[Trajectory]:
        """
        Return trajectories active within `max_age` of `current_frame`.
        """
        active = []
        for traj in self._trajectories.values():
            if not traj.frames:
                continue
            if current_frame - traj.frames[-1] <= max_age:
                active.append(traj)
        return active

    def get_trajectories_by_class(self, class_name: str) -> List[Trajectory]:
        """Filter trajectories by class name."""
        return [t for t in self._trajectories.values() if t.class_name == class_name]

    def get_valid_trajectories(self) -> List[Trajectory]:
        """Return trajectories meeting the minimum-length criterion."""
        return [t for t in self._trajectories.values() if t.length >= self.min_length]

    def get_all(self) -> Dict[int, Trajectory]:
        return self._trajectories

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_numpy(self, track_id: Optional[int] = None) -> dict:
        """Export trajectory data as numpy arrays."""
        if track_id is not None:
            traj = self._trajectories.get(track_id)
            return traj.to_numpy() if traj else {}
        return {tid: t.to_numpy() for tid, t in self._trajectories.items()}

    def to_dataframe(self):
        """Export all trajectories as a pandas DataFrame."""
        import pandas as pd
        rows = []
        for tid, traj in self._trajectories.items():
            for i, frame in enumerate(traj.frames):
                rows.append({
                    "track_id": tid,
                    "class_name": traj.class_name,
                    "frame_id": frame,
                    "x": traj.positions[i][0],
                    "y": traj.positions[i][1],
                    "x1": traj.bboxes[i][0],
                    "y1": traj.bboxes[i][1],
                    "x2": traj.bboxes[i][2],
                    "y2": traj.bboxes[i][3],
                    "confidence": traj.confidences[i],
                })
        return pd.DataFrame(rows)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    @property
    def total_detections(self) -> int:
        return sum(t.length for t in self._trajectories.values())
