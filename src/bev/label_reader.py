"""
Reader for Ultralytics tracking label .txt files.

The project already runs YOLO detection + ByteTrack tracking offline; the
output lives in ``D:/Red-Light视频数据/labels/{video_stem}.txt`` (one file per
video). Each line is

    class_id  xc  yc  w  h  track_id

with xc/yc/w/h normalised to [0,1]. Frames are delimited by

    ### Frame: <blurred_...>_<frame>.txt ###

This module reads that existing output — it does NOT re-run detection, so it
reuses (rather than replaces) the project's detection/tracking interface.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# COCO class ids → traffic class names (matches the project's own mapping,
# extended with truck=7 which the detector also emits).
COCO_TO_NAME = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
}

_FRAME_RE = re.compile(r"_(\d+)\.txt")


class LabelReader:
    """Parse a per-video Ultralytics label .txt into per-frame detections.

    Parameters
    ----------
    img_width, img_height : float
        Full-resolution frame size used to denormalise bboxes to pixels.
    """

    def __init__(self, img_width: float = 3840.0, img_height: float = 2160.0):
        self.img_width = float(img_width)
        self.img_height = float(img_height)

    def load(self, label_path) -> Dict[int, List[dict]]:
        """Parse a label file.

        Returns
        -------
        dict: frame_id (1-based int) → list of detection dicts, each with keys
            class_id, class_name, bbox (x1,y1,x2,y2 pixels), track_id.
        """
        label_path = Path(label_path)
        frames: Dict[int, List[dict]] = defaultdict(list)
        current_frame: Optional[int] = None

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("### Frame:"):
                    m = _FRAME_RE.search(line)
                    if m:
                        current_frame = int(m.group(1))
                    continue

                parts = line.split()
                if len(parts) < 6 or current_frame is None:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = (float(p) for p in parts[1:5])
                    track_id = int(parts[5])
                except (ValueError, IndexError):
                    continue

                x1 = (xc - w / 2.0) * self.img_width
                y1 = (yc - h / 2.0) * self.img_height
                x2 = (xc + w / 2.0) * self.img_width
                y2 = (yc + h / 2.0) * self.img_height

                frames[current_frame].append({
                    "class_id": cls_id,
                    "class_name": COCO_TO_NAME.get(cls_id, f"cls_{cls_id}"),
                    "bbox": (x1, y1, x2, y2),
                    "track_id": track_id,
                })

        return dict(frames)


def read_frame_detections(
    label_path,
    frame_id: int,
    img_width: float = 3840.0,
    img_height: float = 2160.0,
) -> List[dict]:
    """Read detections for a single frame id from a label file."""
    reader = LabelReader(img_width, img_height)
    frames = reader.load(label_path)
    return frames.get(int(frame_id), [])
