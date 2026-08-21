#!/usr/bin/env python3
"""
生成闯红灯真值标签（论文版）
===========================
根据轨迹数据 + 交通灯状态 + 路口标注，自动判定每个行人轨迹
是否包含闯红灯行为。

判定规则（三个硬性客观条件，缺一不可）：
    1. 信号灯条件：交通灯为红灯 (overall_state == "red")
    2. 空间区域条件：行人进入斑马线四边形区域（点多边形判定）
    3. 时序同步条件：同一帧同时满足 ①+②

补充约束（车辆抑制）：
    若斑马线附近有正在行驶的车辆，不标注为闯红灯。
    （车辆存在会抑制过街意图）

输入:
    data/processed/trajectories/{video}/trajectories.npz
    data/processed/trajectories/{video}/traffic_lights.csv
    configs/default.yaml  (路口标注: stop_line, crosswalk_roi)

输出:
    data/processed/trajectories/{video}/violation_labels.csv
        columns: track_key, class_name, is_violation,
                 violation_frame, vehicle_suppressed, confidence

用法:
    python scripts/generate_violation_labels.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import StopLine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Crosswalk polygon helper
# ======================================================================

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting point-in-polygon test.

    Parameters
    ----------
    x, y : float — point coordinates
    polygon : list of (x, y) — vertices in order (must be closed implicitly)

    Returns
    -------
    True if point is inside or on the edge of the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check if edge crosses the horizontal ray from (x, y)
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygon_from_stop_line(
    x1: float, y1: float, x2: float, y2: float,
    extend_pixels: float = 80.0,
) -> List[Tuple[float, float]]:
    """
    Derive a crosswalk quadrilateral from a stop line segment.

    The stop line defines one edge; the polygon extends `extend_pixels`
    perpendicularly (roughly downward/into the junction in image coords).

    Returns 4 vertices of the crosswalk polygon.
    """
    # Direction perpendicular to stop line (rotate 90° CW ≈ downward in image)
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        length = 1.0
    # Perpendicular unit vector (downward in image: +y direction from stop line)
    px = -dy / length
    py = dx / length
    # Ensure it points downward (positive y)
    if py < 0:
        px, py = -px, -py

    ex = px * extend_pixels
    ey = py * extend_pixels

    return [
        (x1, y1),
        (x2, y2),
        (x2 + ex, y2 + ey),
        (x1 + ex, y1 + ey),
    ]


def load_traffic_light_states(tl_path: str) -> Optional[pd.DataFrame]:
    """Load traffic light CSV if exists."""
    path = Path(tl_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def is_red_light(frame_id: int, tl_df: Optional[pd.DataFrame]) -> bool:
    """Check if given frame has red light (nearest-frame lookup)."""
    if tl_df is None or len(tl_df) == 0:
        return False
    # Nearest-frame lookup: TL data may be subsampled (e.g. every 30 frames)
    idx = (tl_df["frame_id"] - frame_id).abs().idxmin()
    return tl_df.iloc[idx]["overall_state"] == "red"


# ======================================================================
# Vehicle suppression
# ======================================================================

def build_vehicle_near_crosswalk_frames(
    traj_data: dict,
    crosswalk_polygon: List[Tuple[float, float]],
    proximity_pixels: float = 60.0,
) -> Set[int]:
    """
    Find all frames where any vehicle is near the crosswalk.

    "Near" = vehicle center position within `proximity_pixels` of the
    crosswalk polygon, OR vehicle bounding box overlaps the polygon.

    Parameters
    ----------
    traj_data : np.load(..., allow_pickle=True) result
    crosswalk_polygon : list of (x, y) vertices

    Returns
    -------
    Set of frame indices where vehicles suppress violation labeling.
    """
    vehicle_classes = {"car", "bus", "truck", "bicycle", "motorcycle"}
    suppressed_frames: Set[int] = set()

    for key in traj_data.files:
        try:
            traj = traj_data[key].item()
        except AttributeError:
            continue

        class_name = traj.get("class_name", "")
        if isinstance(class_name, np.ndarray):
            class_name = str(class_name[0]) if len(class_name) > 0 else ""
        if class_name not in vehicle_classes:
            continue

        positions = traj.get("positions")
        frames = traj.get("frames")
        if positions is None or frames is None:
            continue

        for i in range(len(positions)):
            x, y = float(positions[i][0]), float(positions[i][1])

            # Check if vehicle position is inside or near the crosswalk
            if point_in_polygon(x, y, crosswalk_polygon):
                suppressed_frames.add(int(frames[i]))
                continue

            # Also check proximity: distance to nearest polygon edge
            if _distance_to_polygon(x, y, crosswalk_polygon) < proximity_pixels:
                suppressed_frames.add(int(frames[i]))

    return suppressed_frames


def _distance_to_polygon(
    x: float, y: float, polygon: List[Tuple[float, float]]
) -> float:
    """Minimum distance from point to polygon edges."""
    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dist = _point_to_segment_distance(x, y, x1, y1, x2, y2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2))

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))


# ======================================================================
# Core violation check
# ======================================================================

def check_violation(
    positions: np.ndarray,                   # (T, 2) trajectory positions (pixel x, y)
    frames: np.ndarray,                      # (T,) frame indices
    stop_line: StopLine,
    crosswalk_polygon: List[Tuple[float, float]],
    tl_df: Optional[pd.DataFrame],
    vehicle_suppressed_frames: Set[int],
    dist_threshold: float = 10.0,            # pixels past stop line
    min_crosswalk_frames: int = 3,
) -> Tuple[bool, Optional[int], float, bool]:
    """
    Check if a trajectory violates red light (per paper criteria).

    Returns (is_violation, first_violation_frame, confidence, vehicle_suppressed).
    """
    max_dist = 0.0
    crosswalk_frames = 0
    entered_crosswalk = False
    first_violation_frame = None
    was_vehicle_suppressed = False

    for i in range(len(positions)):
        x, y = float(positions[i][0]), float(positions[i][1])
        fi = int(frames[i])

        # Condition 1: red light
        if not is_red_light(fi, tl_df):
            continue

        # Distance past stop line (for confidence, not a hard criterion)
        dist = stop_line.signed_distance(x, y)
        if dist > max_dist:
            max_dist = dist

        # Crossed stop line?
        if dist > dist_threshold and first_violation_frame is None:
            first_violation_frame = fi

        # Condition 2: in crosswalk polygon
        if point_in_polygon(x, y, crosswalk_polygon):
            crosswalk_frames += 1

            # Vehicle suppression check
            if fi in vehicle_suppressed_frames:
                was_vehicle_suppressed = True
                continue  # this frame doesn't count

            if crosswalk_frames >= min_crosswalk_frames:
                entered_crosswalk = True

    # Condition 3: all three conditions satisfied
    if was_vehicle_suppressed:
        # If ANY frame was vehicle-suppressed, the pedestrian was near a
        # vehicle — suppress the entire violation (per paper, vehicle
        # presence inhibits crossing intent)
        violated = False
    else:
        violated = (first_violation_frame is not None) and entered_crosswalk

    confidence = min(1.0, max_dist / (dist_threshold * 5))

    return violated, first_violation_frame, confidence, was_vehicle_suppressed


# ======================================================================
# Per-video processing
# ======================================================================

def process_video_trajectories(
    traj_dir: str,
    stop_line: StopLine,
    crosswalk_polygon: List[Tuple[float, float]],
    vehicle_proximity: float = 60.0,
) -> pd.DataFrame:
    """
    Generate violation labels for all trajectories in one video.
    """
    traj_path = Path(traj_dir) / "trajectories.npz"
    tl_path = Path(traj_dir) / "traffic_lights.csv"

    if not traj_path.exists():
        logger.warning(f"No trajectories.npz in {traj_dir}")
        return pd.DataFrame()

    data = np.load(traj_path, allow_pickle=True)
    tl_df = load_traffic_light_states(str(tl_path))

    # Pre-compute vehicle-suppressed frames
    vehicle_suppressed_frames = build_vehicle_near_crosswalk_frames(
        data, crosswalk_polygon, proximity_pixels=vehicle_proximity,
    )

    rows = []
    for key in data.files:
        try:
            traj = data[key].item()
        except AttributeError:
            continue

        class_name = traj.get("class_name", "unknown")
        if isinstance(class_name, np.ndarray):
            class_name = str(class_name[0]) if len(class_name) > 0 else "unknown"

        if class_name not in ("pedestrian",):
            continue

        positions = traj.get("positions")
        frames = traj.get("frames")
        if positions is None or frames is None:
            continue
        if positions.shape[0] < 15:
            continue

        violated, viol_frame, conf, veh_supp = check_violation(
            positions=positions,
            frames=frames,
            stop_line=stop_line,
            crosswalk_polygon=crosswalk_polygon,
            tl_df=tl_df,
            vehicle_suppressed_frames=vehicle_suppressed_frames,
        )

        rows.append({
            "track_key": key,
            "class_name": class_name,
            "is_violation": violated,
            "violation_frame": viol_frame if viol_frame is not None else -1,
            "vehicle_suppressed": veh_supp,
            "confidence": conf,
            "trajectory_length": positions.shape[0],
        })

    df = pd.DataFrame(rows)
    csv_path = Path(traj_dir) / "violation_labels.csv"
    df.to_csv(csv_path, index=False)

    n_viol = df["is_violation"].sum() if len(df) > 0 else 0
    n_total = len(df)
    n_veh_supp = df["vehicle_suppressed"].sum() if len(df) > 0 and "vehicle_suppressed" in df.columns else 0
    if n_total > 0:
        logger.info(f"  {Path(traj_dir).name}: {n_viol}/{n_total} violations "
                    f"({n_viol/n_total:.1%}), vehicle-suppressed: {n_veh_supp}")

    return df


# ======================================================================
# Config parsing
# ======================================================================

def parse_intersection_config(
    config: dict, video_name: str,
) -> Tuple[StopLine, List[Tuple[float, float]]]:
    """Parse intersection annotation from config for a given video."""
    if "timing_" in video_name:
        cfg = config.get("intersection_A", {})
    else:
        cfg = config.get("intersection_B", {})

    # Stop line
    sl = cfg.get("stop_line", [0, 0, 0, 0])
    stop_line = StopLine(x1=sl[0], y1=sl[1], x2=sl[2], y2=sl[3])

    # Crosswalk polygon: use explicit config if provided, else derive from stop line
    cw = cfg.get("crosswalk_roi", None)
    if cw and len(cw) >= 4:
        # Explicit polygon: list of [x, y] or (x, y) pairs
        crosswalk_polygon = [(float(p[0]), float(p[1])) for p in cw]
    else:
        # Derive from stop line
        crosswalk_polygon = polygon_from_stop_line(
            float(sl[0]), float(sl[1]), float(sl[2]), float(sl[3]),
        )

    return stop_line, crosswalk_polygon


# ======================================================================
# Batch runner
# ======================================================================

def run_batch(processed_dir: str, config: dict) -> dict:
    """Run violation label generation on all videos."""
    proc = Path(processed_dir)
    video_dirs = [
        d for d in sorted(proc.iterdir())
        if d.is_dir() and (d / "trajectories.npz").exists()
    ]

    # Only process videos that have traffic_light data
    video_dirs_with_tl = [
        d for d in video_dirs if (d / "traffic_lights.csv").exists()
    ]
    skipped_no_tl = len(video_dirs) - len(video_dirs_with_tl)

    vehicle_proximity = config.get("red_light", {}).get(
        "vehicle_proximity_pixels", 60.0,
    )

    logger.info(f"Found {len(video_dirs_with_tl)} videos with TL data "
                f"(skipping {skipped_no_tl} without TL)")

    total_violations = 0
    total_trajectories = 0
    total_veh_suppressed = 0
    processed = 0
    skipped = 0

    # Pre-count stats from already-processed videos
    logger.info("Scanning already-processed videos...")
    for vdir in tqdm(video_dirs_with_tl, desc="Scanning existing"):
        vl_path = vdir / "violation_labels.csv"
        if vl_path.exists() and vl_path.stat().st_size > 0:
            try:
                df = pd.read_csv(vl_path)
                if len(df) > 0 and "is_violation" in df.columns:
                    total_violations += int(df["is_violation"].sum())
                    total_trajectories += len(df)
                    if "vehicle_suppressed" in df.columns:
                        total_veh_suppressed += int(df["vehicle_suppressed"].sum())
                    skipped += 1
            except Exception:
                # Corrupt CSV — will be regenerated
                vl_path.unlink(missing_ok=True)

    logger.info(f"Already processed: {skipped}, remaining: {len(video_dirs_with_tl) - skipped}")

    for vdir in tqdm(video_dirs_with_tl, desc="Generating labels"):
        video_name = vdir.name
        # Skip if already processed (resume support)
        if (vdir / "violation_labels.csv").exists():
            continue
        try:
            stop_line, crosswalk_poly = parse_intersection_config(config, video_name)
            df = process_video_trajectories(
                str(vdir), stop_line, crosswalk_poly, vehicle_proximity,
            )
            if len(df) > 0:
                total_violations += int(df["is_violation"].sum())
                total_trajectories += len(df)
                if "vehicle_suppressed" in df.columns:
                    total_veh_suppressed += int(df["vehicle_suppressed"].sum())
            processed += 1
        except Exception as e:
            logger.error(f"Failed {video_name}: {e}")

    summary = {
        "total_videos": len(video_dirs_with_tl),
        "processed": processed,
        "skipped_no_tl_data": skipped_no_tl,
        "total_trajectories": int(total_trajectories),
        "total_violations": int(total_violations),
        "violation_rate": round(float(total_violations / max(total_trajectories, 1)), 4),
        "vehicle_suppressed_trajectories": int(total_veh_suppressed),
        "criteria": {
            "must_be_red_light": True,
            "must_enter_crosswalk": True,
            "vehicle_near_crosswalk_suppresses": True,
        },
    }

    summary_path = Path(processed_dir) / "violation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    return summary


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="生成闯红灯真值标签（论文版）")
    parser.add_argument("--processed-dir", default="data/processed/trajectories/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--single", default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.single:
        vdir = Path(args.processed_dir) / args.single
        stop_line, crosswalk_poly = parse_intersection_config(config, args.single)
        process_video_trajectories(str(vdir), stop_line, crosswalk_poly)
    else:
        run_batch(args.processed_dir, config)


if __name__ == "__main__":
    main()
