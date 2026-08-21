#!/usr/bin/env python3
"""
提取视频中交通灯状态 — 云端运行版本
=====================================
从预处理得到的交通灯ROI位置，遍历视频帧做HSV颜色识别，
生成每帧的交通灯状态。

输入:
    data/processed/trajectories/{video}/traffic_light_rois.json  — 交通灯ROI坐标
    D:/Red-Light视频数据/{date}/{video}.mp4                       — 原始视频

输出:
    data/processed/trajectories/{video}/traffic_lights.csv        — 逐帧交通灯状态
        columns: frame_id, overall_state, per_roi_states...

用法:
    python scripts/extract_traffic_lights.py --all
    python scripts/extract_traffic_lights.py --start-date 2026_01_21 --end-date 2026_01_22
    python scripts/extract_traffic_lights.py --skip-frames 5   # 每5帧采样以加速
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.traffic_light import (
    TrafficLightDetector,
    TrafficLightROI,
    LightState,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_traffic_light_rois(roi_path: str) -> List[TrafficLightROI]:
    """Load traffic light ROIs from JSON file."""
    with open(roi_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    return [
        TrafficLightROI(
            x1=int(r["x1"]), y1=int(r["y1"]),
            x2=int(r["x2"]), y2=int(r["y2"]),
            direction=r.get("direction", "vertical"),
            light_count=r.get("light_count", 3),
        )
        for r in data
    ]


def find_video_path(video_name: str, data_root: str) -> Optional[str]:
    """Find the MP4 file matching a video name across date folders."""
    base = Path(data_root)
    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("2026_"):
            continue
        candidate = date_dir / f"{video_name}.mp4"
        if candidate.exists():
            return str(candidate)
    return None


def extract_video_traffic_lights(
    video_path: str,
    output_dir: str,
    tl_rois: List[TrafficLightROI],
    skip_frames: int = 5,
    max_frames: int = -1,
) -> pd.DataFrame:
    """
    Extract traffic light states for all frames of a video.

    Returns DataFrame with per-frame states.
    """
    detector = TrafficLightDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    records = []
    prev_state = None

    for fi in tqdm(range(0, total, skip_frames), desc="Extracting TL", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect per-ROI states
        per_roi = detector.detect_all(frame, tl_rois)
        overall = detector.detect_intersection_state(frame, tl_rois)

        record = {
            "frame_id": fi,
            "overall_state": overall.value,
        }
        for i, s in enumerate(per_roi):
            record[f"roi_{i}"] = s.value

        records.append(record)
        prev_state = overall

    cap.release()

    df = pd.DataFrame(records)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "traffic_lights.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    state_counts = df["overall_state"].value_counts().to_dict()
    logger.info(f"  Saved {len(df)} frames to {csv_path}")
    logger.info(f"  States: {state_counts}")

    return df


def run_batch(
    processed_dir: str,
    data_root: str,
    skip_frames: int = 5,
) -> dict:
    """
    Process all videos that have traffic light ROI files.

    Parameters
    ----------
    processed_dir : str
        Path to data/processed/trajectories/
    data_root : str
        Path to raw video data root.
    skip_frames : int
        Sample every N frames.
    """
    proc = Path(processed_dir)
    video_dirs = [
        d for d in sorted(proc.iterdir())
        if d.is_dir() and (d / "trajectories.npz").exists()
    ]

    roi_files = [
        d for d in video_dirs
        if (d / "traffic_light_rois.json").exists()
    ]

    logger.info(f"Found {len(roi_files)} videos with traffic light ROIs "
                f"(out of {len(video_dirs)} total)")

    processed = 0
    for vdir in tqdm(roi_files, desc="Processing videos"):
        video_name = vdir.name
        roi_path = vdir / "traffic_light_rois.json"

        # Find video
        video_path = find_video_path(video_name, data_root)
        if video_path is None:
            logger.warning(f"Video not found: {video_name}")
            continue

        # Load ROIs
        try:
            tl_rois = load_traffic_light_rois(str(roi_path))
        except Exception as e:
            logger.warning(f"Failed to load ROIs for {video_name}: {e}")
            continue

        if not tl_rois:
            continue

        # Skip if already processed
        if (vdir / "traffic_lights.csv").exists():
            processed += 1
            continue

        # Extract
        try:
            extract_video_traffic_lights(
                video_path=str(video_path),
                output_dir=str(vdir),
                tl_rois=tl_rois,
                skip_frames=skip_frames,
            )
            processed += 1
        except Exception as e:
            logger.error(f"Failed {video_name}: {e}")

    return {"total_with_rois": len(roi_files), "processed": processed}


def main():
    parser = argparse.ArgumentParser(description="提取视频中交通灯状态")
    parser.add_argument("--processed-dir", default="data/processed/trajectories/")
    parser.add_argument("--data-root", default="D:/Red-Light视频数据/")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=5)
    parser.add_argument("--single", default=None, help="处理单个视频目录名")

    args = parser.parse_args()

    if args.single:
        proc = Path(args.processed_dir) / args.single
        roi_path = proc / "traffic_light_rois.json"
        tl_rois = load_traffic_light_rois(str(roi_path)) if roi_path.exists() else []
        if not tl_rois:
            logger.error(f"No ROIs found for {args.single}")
            return
        video_path = find_video_path(args.single, args.data_root)
        if not video_path:
            logger.error(f"Video not found: {args.single}")
            return
        extract_video_traffic_lights(video_path, str(proc), tl_rois, args.skip_frames)
    else:
        run_batch(args.processed_dir, args.data_root, args.skip_frames)


if __name__ == "__main__":
    main()
