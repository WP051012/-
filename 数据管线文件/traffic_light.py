"""
Traffic light state detection via HSV colour segmentation.

Given a bounding-box ROI of a traffic light in the image, this module
determines the current light colour (red / green / yellow / off) using
HSV thresholding.

References:
    Paper Section 3.2: 交通灯状态识别
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


class LightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"


@dataclass
class TrafficLightROI:
    """A traffic light region of interest."""
    x1: int
    y1: int
    x2: int
    y2: int
    direction: str = "vertical"   # "vertical" (红灯在上) or "horizontal" (红灯在左)
    light_count: int = 3           # number of light bulbs (typically 3: red/yellow/green)


# ---------------------------------------------------------------
# HSV colour ranges for traffic light bulbs
# ---------------------------------------------------------------

# Red wraps around in HSV (0-10 and 170-180)
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])

# Green
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])

# Yellow
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])


# ---------------------------------------------------------------
# Detector
# ---------------------------------------------------------------

class TrafficLightDetector:
    """
    Detect traffic light state from an ROI in the image.

    For a vertical 3-bulb traffic light, the ROI is divided into
    top / middle / bottom thirds. For each third, the ratio of
    red / yellow / green pixels determines the active colour.

    Parameters
    ----------
    brightness_threshold : float
        Minimum mean pixel brightness in a bulb ROI to consider it "on".
    saturation_ratio : float
        Minimum ratio of coloured pixels to ROI area.
    """

    def __init__(
        self,
        brightness_threshold: float = 50.0,
        saturation_ratio: float = 0.05,
    ):
        self.brightness_threshold = brightness_threshold
        self.saturation_ratio = saturation_ratio

    # ------------------------------------------------------------------
    # Single-ROI detection
    # ------------------------------------------------------------------

    def detect_roi(
        self,
        image: np.ndarray,        # BGR image (H, W, 3)
        roi: TrafficLightROI,
    ) -> LightState:
        """
        Detect the current state of a traffic light from its ROI.

        For a 3-bulb vertical light, the ROI is split into 3 equal
        vertical segments. The segment with the most saturated pixels
        of a matching colour determines the state.

        Returns
        -------
        LightState
        """
        x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2

        # Clamp to image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return LightState.OFF

        roi_img = image[y1:y2, x1:x2]
        return self._detect_from_crop(roi_img, roi)

    def _detect_from_crop(
        self,
        crop: np.ndarray,          # BGR image of traffic light ROI
        roi: TrafficLightROI,
    ) -> LightState:
        """Analyse cropped traffic light image."""
        roi_h, roi_w = crop.shape[:2]

        if roi.direction == "vertical" and roi.light_count == 3:
            # Split vertically into 3 segments
            bulb_h = roi_h // 3
            bulbs = [
                crop[0:bulb_h, :],                          # top    (usually red)
                crop[bulb_h:2 * bulb_h, :],                 # middle (yellow)
                crop[2 * bulb_h:roi_h, :],                  # bottom (green)
            ]
        elif roi.direction == "horizontal" and roi.light_count == 3:
            bulb_w = roi_w // 3
            bulbs = [
                crop[:, 0:bulb_w],                          # left
                crop[:, bulb_w:2 * bulb_w],                 # middle
                crop[:, 2 * bulb_w:roi_w],                  # right
            ]
        else:
            # Single bulb or unknown layout — treat whole ROI
            return self._classify_bulb(crop)

        # Score each bulb
        scores = [self._score_bulb(b) for b in bulbs]

        # Find the brightest colour match
        best_idx = -1
        best_score = -1
        for i, s in enumerate(scores):
            for colour, score in s.items():
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_score < self.saturation_ratio:
            return LightState.OFF

        # Map bulb position to expected colour
        if roi.direction == "vertical":
            expected = [LightState.RED, LightState.YELLOW, LightState.GREEN]
        else:
            expected = [LightState.RED, LightState.YELLOW, LightState.GREEN]

        if best_idx < len(expected):
            # Verify the detected colour is plausible for this bulb position
            bulb_scores = scores[best_idx]
            expected_colour = expected[best_idx]
            if bulb_scores.get(expected_colour, 0) > 0:
                return expected_colour
            # Fallback: return the dominant colour
            dominant = max(bulb_scores, key=bulb_scores.get)
            return dominant

        return LightState.OFF

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_bulb(self, bulb: np.ndarray) -> dict:
        """
        Score a single bulb for red / yellow / green presence.

        Returns dict: {LightState.RED: score, LightState.YELLOW: score, ...}
        """
        if bulb.size == 0:
            return {LightState.RED: 0.0, LightState.YELLOW: 0.0,
                    LightState.GREEN: 0.0}

        hsv = cv2.cvtColor(bulb, cv2.COLOR_BGR2HSV)
        total_pixels = bulb.shape[0] * bulb.shape[1]

        # Check brightness (V channel mean)
        mean_brightness = hsv[:, :, 2].mean()
        if mean_brightness < self.brightness_threshold:
            return {LightState.RED: 0.0, LightState.YELLOW: 0.0,
                    LightState.GREEN: 0.0, LightState.OFF: 1.0}

        # Red (two ranges)
        mask_r1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask_r2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        red_ratio = (mask_r1.sum() + mask_r2.sum()) / total_pixels

        # Yellow
        mask_y = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        yellow_ratio = mask_y.sum() / total_pixels

        # Green
        mask_g = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        green_ratio = mask_g.sum() / total_pixels

        return {
            LightState.RED: red_ratio,
            LightState.YELLOW: yellow_ratio,
            LightState.GREEN: green_ratio,
        }

    def _classify_bulb(self, bulb: np.ndarray) -> LightState:
        """Classify a single bulb (no segmentation)."""
        scores = self._score_bulb(bulb)
        dominant = max(scores, key=scores.get)
        if scores[dominant] < self.saturation_ratio:
            return LightState.OFF
        return dominant

    # ------------------------------------------------------------------
    # Batch detection on multiple ROIs
    # ------------------------------------------------------------------

    def detect_all(
        self,
        image: np.ndarray,
        rois: List[TrafficLightROI],
    ) -> List[LightState]:
        """Detect states for all ROIs in a frame."""
        return [self.detect_roi(image, roi) for roi in rois]

    def detect_intersection_state(
        self,
        image: np.ndarray,
        rois: List[TrafficLightROI],
    ) -> LightState:
        """
        Detect the overall traffic light state for an intersection.

        For intersections with multiple traffic lights facing the same
        direction, returns the most restrictive state (red > yellow > green).

        Returns
        -------
        LightState
            RED if any signal is red, else YELLOW if any is yellow, else GREEN.
        """
        states = self.detect_all(image, rois)
        if any(s == LightState.RED for s in states):
            return LightState.RED
        if any(s == LightState.YELLOW for s in states):
            return LightState.YELLOW
        if any(s == LightState.GREEN for s in states):
            return LightState.GREEN
        return LightState.OFF


# ---------------------------------------------------------------
# Traffic light ROI discovery from cls=9 cluster centres
# ---------------------------------------------------------------

def discover_traffic_light_rois(
    cls9_positions: np.ndarray,       # (N, 2)  [xc, yc] normalised
    img_width: int = 3840,
    img_height: int = 2160,
    n_clusters: int = 5,
    roi_size: float = 60.0,            # half-size of square ROI in pixels
) -> List[TrafficLightROI]:
    """
    Discover traffic light ROIs from clustered cls=9 detections.

    Uses simple K-Means on normalised (xc, yc) of cls=9 detections
    to find cluster centres, then creates ROIs around each centre.

    Parameters
    ----------
    cls9_positions : np.ndarray (N, 2)
        Normalised (xc, yc) coordinates of cls=9 detections.
    img_width, img_height : int
        Image dimensions in pixels.
    n_clusters : int
        Number of traffic light clusters to find.
    roi_size : float
        Half-size (in pixels) of the square ROI around each cluster centre.

    Returns
    -------
    list of TrafficLightROI
    """
    if len(cls9_positions) < n_clusters:
        n_clusters = max(1, len(cls9_positions) // 100)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(cls9_positions)
    centres = kmeans.cluster_centers_  # (n_clusters, 2) normalised

    rois = []
    for cx, cy in centres:
        px_x = int(cx * img_width)
        px_y = int(cy * img_height)
        half = int(roi_size)
        rois.append(TrafficLightROI(
            x1=max(0, px_x - half),
            y1=max(0, px_y - half),
            x2=min(img_width, px_x + half),
            y2=min(img_height, px_y + half),
            direction="vertical",
            light_count=3,
        ))

    return rois


# ---------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python traffic_light.py <image_path> [x1 y1 x2 y2]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Cannot read image: {sys.argv[1]}")
        sys.exit(1)

    detector = TrafficLightDetector()

    if len(sys.argv) >= 6:
        x1, y1, x2, y2 = map(int, sys.argv[2:6])
        roi = TrafficLightROI(x1=x1, y1=y1, x2=x2, y2=y2)
        state = detector.detect_roi(img, roi)
        print(f"Traffic light state: {state.value}")
    else:
        # Try the whole image centre-top region
        h, w = img.shape[:2]
        roi = TrafficLightROI(
            x1=w // 2 - 60, y1=10,
            x2=w // 2 + 60, y2=130,
        )
        state = detector.detect_roi(img, roi)
        print(f"Default ROI state: {state.value}")
