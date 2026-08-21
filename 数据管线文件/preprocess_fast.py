#!/usr/bin/env python3
"""
快速预处理 — 直接使用已有标签，跳过YOLO检测
==============================================
数据集已包含每个视频的 YOLO 跟踪标签 (labels/ 目录),
格式: cls xc yc w h track_id (每帧一个段落)。

本脚本直接从标签构建轨迹 + 提取交通灯位置，
跳过耗时的 YOLO 推理，处理速度快 100x。

输出 (每个视频目录下):
    trajectories.npz      — 轨迹数据
    traffic_lights.npz    — 交通灯位置/ROI (可选)

用法:
    python scripts/preprocess_fast.py --all
    python scripts/preprocess_fast.py --start-date 2026_01_21 --end-date 2026_01_21
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.traffic_light import (
    TrafficLightROI,
    discover_traffic_light_rois,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class ID → name mapping (from label README + COCO)
CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    9: "traffic_light",
}


# ======================================================================
# Label Parser
# ======================================================================

def parse_label_file(label_path: str) -> Dict[int, List[dict]]:
    """
    Parse one Ultralytics tracking label file.

    File format:
        ### Frame: blurred_<video>_<frame_id>.txt ###
        cls xc yc w h track_id
        ...

    Returns
    -------
    dict: frame_id → list of {cls, xc, yc, w, h, track_id}
    """
    frames: Dict[int, List[dict]] = defaultdict(list)

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        current_frame = None
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("### Frame:"):
                # Parse frame ID: "### Frame: ..._<frame_id>.txt ###"
                m = re.search(r'_(\d+)\.txt', line)
                if m:
                    current_frame = int(m.group(1))
                continue

            parts = line.split()
            if len(parts) < 6 or current_frame is None:
                continue

            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                track_id = int(parts[5])
            except (ValueError, IndexError):
                continue

            frames[current_frame].append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                "xc": xc, "yc": yc, "w": w, "h": h,
                "track_id": track_id,
            })

    return dict(frames)


# ======================================================================
# Trajectory Builder
# ======================================================================

def build_trajectories(
    all_frames: Dict[int, List[dict]],
    min_length: int = 15,
    img_width: int = 3840,
    img_height: int = 2160,
) -> dict:
    """
    Build per-track trajectories from per-frame detection data.

    Returns
    -------
    dict: track_key → {class_name, frames[], positions[], bboxes[], confidences[]}
    """
    # Collect all detections per track
    tracks: Dict[str, dict] = defaultdict(lambda: {
        "class_name": None,
        "frames": [],
        "positions": [],
        "bboxes": [],
    })

    for frame_id in sorted(all_frames.keys()):
        for det in all_frames[frame_id]:
            tid = det["track_id"]
            cls_name = det["class_name"]
            key = f"{cls_name}_{tid}"

            # Convert normalized → pixel
            xc_px = det["xc"] * img_width
            yc_px = det["yc"] * img_height
            w_px = det["w"] * img_width
            h_px = det["h"] * img_height
            x1 = xc_px - w_px / 2
            y1 = yc_px - h_px / 2
            x2 = xc_px + w_px / 2
            y2 = yc_px + h_px / 2

            tracks[key]["class_name"] = cls_name
            tracks[key]["frames"].append(frame_id)
            tracks[key]["positions"].append([xc_px, yc_px])
            tracks[key]["bboxes"].append([x1, y1, x2, y2])

    # Filter by min length and convert to arrays
    result = {}
    for key, data in tracks.items():
        if len(data["frames"]) >= min_length:
            result[key] = {
                "class_name": data["class_name"],
                "frames": np.array(data["frames"], dtype=np.int32),
                "positions": np.array(data["positions"], dtype=np.float32),
                "bboxes": np.array(data["bboxes"], dtype=np.float32),
                "confidences": np.ones(len(data["frames"]), dtype=np.float32),
            }

    return result


# ======================================================================
# Main — process one label file
# ======================================================================

def process_label_file(
    label_path: str,
    output_dir: str,
    img_width: int = 3840,
    img_height: int = 2160,
    discover_traffic_lights: bool = True,
) -> dict:
    """
    Process a single label file → trajectories + traffic light ROIs.

    Parameters
    ----------
    label_path : str
        Path to .txt label file.
    output_dir : str
        Output directory for this video's processed data.
    img_width, img_height : int
        Video dimensions.
    discover_traffic_lights : bool
        Discover traffic light ROIs from cls=9 clusters.

    Returns
    -------
    dict with stats.
    """
    label_name = Path(label_path).stem
    video_out = Path(output_dir) / label_name
    video_out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # 1. Parse label file
    all_frames = parse_label_file(label_path)

    if not all_frames:
        logger.warning(f"No frames found in {label_path}")
        return {"label": label_name, "error": "no frames"}

    n_frames = len(all_frames)
    n_detections = sum(len(v) for v in all_frames.values())

    # 2. Build trajectories
    trajectories = build_trajectories(all_frames, img_width=img_width, img_height=img_height)

    # 3. Collect cls=9 positions for traffic light discovery
    cls9_positions = []
    for frame_id, dets in all_frames.items():
        for d in dets:
            if d["class_id"] == 9:
                cls9_positions.append((d["xc"], d["yc"]))

    tl_rois = None
    if discover_traffic_lights and len(cls9_positions) >= 10:
        tl_rois = discover_traffic_light_rois(
            np.array(cls9_positions),
            img_width=img_width, img_height=img_height,
            n_clusters=5,
        )

    # 4. Save
    traj_packed = {}
    for key, data in trajectories.items():
        traj_packed[key] = data

    np.savez_compressed(video_out / "trajectories.npz", **traj_packed)

    # 5. Traffic light ROIs
    if tl_rois:
        roi_data = [
            {"x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
             "direction": r.direction}
            for r in tl_rois
        ]
        with open(video_out / "traffic_light_rois.json", "w") as f:
            json.dump(roi_data, f, indent=2)

    # 6. Class stats
    class_counts = defaultdict(int)
    for data in trajectories.values():
        class_counts[data["class_name"]] += 1

    elapsed = time.time() - t_start

    stats = {
        "label": label_name,
        "frames": n_frames,
        "detections": n_detections,
        "trajectories": {k: v for k, v in sorted(class_counts.items())},
        "total_trajectories": len(trajectories),
        "traffic_light_rois": len(tl_rois) if tl_rois else 0,
        "elapsed_sec": elapsed,
    }

    # Save metadata
    with open(video_out / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


# ======================================================================
# Batch processing
# ======================================================================

def discover_label_files(
    label_dir: str,
    video_dates: Optional[Set[str]] = None,
) -> List[str]:
    """
    Find all label .txt files, optionally filtered by video date patterns.
    """
    label_path = Path(label_dir)
    files = sorted(label_path.glob("*.txt"))
    # Filter out README and other non-label files
    files = [f for f in files if f.name != "README.md"]
    return [str(f) for f in files]


def match_label_to_video_date(label_name: str) -> Optional[str]:
    """Extract date from label filename."""
    m = re.search(r'_(\d{8})_', label_name)
    if m:
        date_str = m.group(1)  # "20260121"
        return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
    return None


def run_batch(
    label_dir: str,
    output_dir: str,
    config: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Process all label files.

    Parameters
    ----------
    label_dir : str
        Path to labels/ directory.
    output_dir : str
        Output directory for processed trajectories.
    config : dict
    start_date, end_date : str, optional
        Filter by date.
    """
    video_cfg = config.get("video", {})
    img_w = video_cfg.get("width", 3840)
    img_h = video_cfg.get("height", 2160)

    label_files = discover_label_files(label_dir)

    # Filter by date
    if start_date or end_date:
        filtered = []
        for f in label_files:
            date = match_label_to_video_date(Path(f).name)
            if date:
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                filtered.append(f)
        label_files = filtered

    logger.info(f"Processing {len(label_files)} label files")
    logger.info(f"Image size: {img_w}x{img_h}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_stats = []
    total_trajs = 0

    pbar = tqdm(label_files, desc="处理标签", unit="file")
    for label_path in pbar:
        name = Path(label_path).stem
        try:
            stats = process_label_file(
                label_path=label_path,
                output_dir=output_dir,
                img_width=img_w,
                img_height=img_h,
            )
            all_stats.append(stats)
            total_trajs += stats.get("total_trajectories", 0)
            pbar.set_postfix({
                "traj": stats.get("total_trajectories", 0),
                "cls9": stats.get("traffic_light_rois", 0),
            })
        except Exception as e:
            logger.error(f"Failed: {name}: {e}")

    # Summary
    total_time = sum(s.get("elapsed_sec", 0) for s in all_stats)
    summary = {
        "total_labels": len(label_files),
        "total_trajectories": total_trajs,
        "total_elapsed_sec": total_time,
        "output_dir": output_dir,
    }

    summary_path = Path(output_dir) / "preprocess_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nDone! {len(all_stats)} labels processed, "
                f"{total_trajs} trajectories, {total_time:.1f}s")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Summary: {summary_path}")

    return summary


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="快速预处理 — 从标签直接构建轨迹"
    )
    parser.add_argument("--label-dir", default="D:/Red-Light视频数据/labels/",
                        help="标签目录")
    parser.add_argument("--output", default="data/processed/trajectories/",
                        help="输出目录")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--all", action="store_true", help="处理所有标签")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--single", default=None, help="处理单个标签文件")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.single:
        stats = process_label_file(
            args.single, args.output,
            img_width=config.get("video", {}).get("width", 3840),
            img_height=config.get("video", {}).get("height", 2160),
        )
        print(json.dumps(stats, indent=2))
        return

    run_batch(
        label_dir=args.label_dir,
        output_dir=args.output,
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
