#!/usr/bin/env python3
"""
离线预处理脚本 — 从视频提取轨迹、交通灯状态和感知数据
========================================================
处理所有 953 个视频，生成训练用的轨迹数据和感知标注。

输出 (每个视频):
    {output_dir}/{video_name}/
        trajectories.csv        — 所有目标的轨迹 (track_id, frame, x, y, bbox, cls)
        trajectories.npz        — numpy 格式轨迹
        traffic_lights.npz      — 每帧交通灯状态
        perception/             — (可选) 逐帧感知图数据
        metadata.json           — 视频元信息

用法:
    # 处理所有视频
    python scripts/preprocess.py --config configs/default.yaml

    # 处理指定日期范围
    python scripts/preprocess.py --start-date 2026_01_21 --end-date 2026_01_22

    # 处理单个视频
    python scripts/preprocess.py --video path/to/video.mp4

    # 仅提取轨迹 + 交通灯(不构建感知图)
    python scripts/preprocess.py --mode track_and_trafficlight
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import (
    YOLODetector,
    ByteTrackWrapper,
    TrajectoryManager,
    TrackedObject,
    DetectionResult,
)
from utils.traffic_light import (
    TrafficLightDetector,
    TrafficLightROI,
    LightState,
    discover_traffic_light_rois,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# YOLO classes to track
TRACK_CLASSES = ["pedestrian", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light"]
# Classes for trajectory extraction (exclude static infrastructure)
TRAJECTORY_CLASSES = ["pedestrian", "bicycle", "motorcycle", "car", "bus", "truck"]


# ======================================================================
# Main preprocessing function
# ======================================================================

def preprocess_video(
    video_path: str,
    output_dir: str,
    config: dict,
    detector: YOLODetector,
    tracker: ByteTrackWrapper,
    tl_detector: Optional[TrafficLightDetector] = None,
    tl_rois: Optional[List[TrafficLightROI]] = None,
    skip_frames: int = 1,
    max_frames: int = -1,
) -> dict:
    """
    Process a single video: detect → track → extract trajectories →
    detect traffic light states.

    Parameters
    ----------
    video_path : str
    output_dir : str
    config : dict
    detector : YOLODetector
    tracker : ByteTrackWrapper
    tl_detector : TrafficLightDetector, optional
    tl_rois : list of TrafficLightROI, optional
        Pre-defined traffic light ROIs. If None, will be discovered
        from cls=9 detections.
    skip_frames : int
        Process every N frames (1 = every frame).
    max_frames : int
        Maximum frames to process (-1 = all).

    Returns
    -------
    dict with summary statistics.
    """
    trk_cfg = config.get("tracking", {})
    video_cfg = config.get("video", {})
    img_w = video_cfg.get("width", 3840)
    img_h = video_cfg.get("height", 2160)
    fps = video_cfg.get("fps", 25)

    video_name = Path(video_path).stem
    video_out = Path(output_dir) / video_name
    video_out.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Trajectory manager
    traj_manager = TrajectoryManager(
        min_length=trk_cfg.get("min_trajectory_length", 15),
    )

    # Collect cls=9 positions for traffic light ROI discovery
    cls9_positions: List[Tuple[float, float]] = []
    tl_states: List[dict] = []      # per-frame traffic light states
    auto_discover_rois = tl_rois is None

    # Statistics
    stats = {
        "video": video_name,
        "total_frames": total_frames,
        "processed_frames": 0,
        "total_detections": 0,
        "total_tracks": 0,
        "traffic_light_changes": 0,
        "elapsed_sec": 0.0,
    }

    t_start = time.time()

    # --- Frame loop ---
    pbar = tqdm(range(0, total_frames, skip_frames),
                desc=f"预处理 {video_name[:50]}", unit="frame")

    prev_tl_state = None
    for fi, frame_id in enumerate(pbar):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO Detection
        detections: List[DetectionResult] = detector.detect(frame)

        # 2. Collect cls=9 positions for traffic light discovery
        for d in detections:
            if d.class_name == "traffic_light":
                xc, yc = d.center
                cls9_positions.append((xc / img_w, yc / img_h))

        # 3. ByteTrack tracking
        det_np = _detections_to_numpy(detections)
        tracked_np = tracker.update(det_np, frame)

        # 4. Register tracked objects
        tracked_objs = _numpy_to_tracked_objects(
            tracked_np, detector.class_mapping,
        )
        for obj in tracked_objs:
            obj.frame_id = frame_id
        traj_manager.update(frame_id, tracked_objs)

        # 5. Traffic light state detection (if ROIs known)
        if tl_detector is not None and tl_rois:
            states = tl_detector.detect_all(frame, tl_rois)
            overall = tl_detector.detect_intersection_state(frame, tl_rois)
            tl_info = {
                "frame_id": frame_id,
                "per_roi": [s.value for s in states],
                "overall": overall.value,
            }
            tl_states.append(tl_info)

            if prev_tl_state is not None and overall != prev_tl_state:
                stats["traffic_light_changes"] += 1
            prev_tl_state = overall

        stats["processed_frames"] += 1
        stats["total_detections"] += len(detections)
        stats["total_tracks"] += len(tracked_objs)

        pbar.set_postfix({
            "det": len(detections),
            "trk": len(tracked_objs),
            "traj": traj_manager.num_trajectories,
        })

    cap.release()
    stats["elapsed_sec"] = time.time() - t_start

    # --- Post-processing ---

    # 6. Discover traffic light ROIs (if not pre-defined)
    if auto_discover_rois and len(cls9_positions) >= 10:
        logger.info(f"Discovering traffic light ROIs from {len(cls9_positions)} cls=9 detections")
        tl_rois = discover_traffic_light_rois(
            np.array(cls9_positions),
            img_width=img_w, img_height=img_h,
            n_clusters=5,
        )
        # Save discovered ROIs
        roi_data = [
            {"x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
             "direction": r.direction}
            for r in tl_rois
        ]
        with open(video_out / "traffic_light_rois.json", "w") as f:
            json.dump(roi_data, f, indent=2)

    # 7. Save trajectories
    valid_trajs = traj_manager.get_valid_trajectories()
    class_breakdown = defaultdict(int)
    for t in valid_trajs:
        class_breakdown[t.class_name] += 1
    stats["trajectories"] = {k: v for k, v in sorted(class_breakdown.items())}
    stats["total_trajectories"] = len(valid_trajs)

    df = traj_manager.to_dataframe()
    df.to_csv(video_out / "trajectories.csv", index=False)
    np.savez_compressed(video_out / "trajectories.npz",
                        **traj_manager.export_numpy())

    # 8. Save traffic light states
    if tl_states:
        tl_df = pd.DataFrame(tl_states)
        tl_df.to_csv(video_out / "traffic_lights.csv", index=False)
        np.savez_compressed(
            video_out / "traffic_lights.npz",
            states=np.array([s["overall"] for s in tl_states], dtype=str),
            frame_ids=np.array([s["frame_id"] for s in tl_states], dtype=np.int32),
        )

    # 9. Save metadata
    with open(video_out / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


# ======================================================================
# Batch processing
# ======================================================================

def discover_videos(
    data_root: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """
    Discover all MP4 videos in the dataset directory structure.

    Directory layout:
        data_root/
            2026_01_15/  →  *.mp4
            2026_01_21/  →  *.mp4
            ...

    Returns sorted list of absolute video paths.
    """
    data_path = Path(data_root)
    videos = []

    for date_dir in sorted(data_path.iterdir()):
        if not date_dir.is_dir():
            continue
        date_name = date_dir.name
        if not date_name.startswith("2026_"):
            continue

        if start_date and date_name < start_date:
            continue
        if end_date and date_name > end_date:
            continue

        for vid_file in sorted(date_dir.glob("*.mp4")):
            videos.append(str(vid_file))

    return videos


def run_batch_preprocess(
    config: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    single_video: Optional[str] = None,
    mode: str = "full",
    skip_frames: int = 1,
) -> dict:
    """
    Run preprocessing on all videos or a single video.

    Parameters
    ----------
    config : dict
    start_date, end_date : str, optional
        Filter videos by date folder (e.g. "2026_01_21").
    single_video : str, optional
        Process only this video.
    mode : str
        "full"        — trajectories + traffic lights + perception data
        "track_and_trafficlight" — trajectories + traffic lights only
        "trajectories_only" — trajectories only
    skip_frames : int
        Process every N frames.

    Returns
    -------
    dict with overall summary.
    """
    data_cfg = config.get("data", {})
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    video_cfg = config.get("video", {})

    output_dir = data_cfg.get("trajectory_dir", "data/processed/trajectories/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Initialise modules ---
    device = det_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")

    detector = YOLODetector(
        model_path=det_cfg.get("finetuned_model") or det_cfg.get("model_name", "yolov8n.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        img_size=det_cfg.get("img_size", 640),
        device=device,
    )

    tracker = ByteTrackWrapper(
        track_buffer=trk_cfg.get("track_buffer", 30),
        track_thresh=trk_cfg.get("track_thresh", 0.5),
        match_thresh=trk_cfg.get("match_thresh", 0.8),
        frame_rate=video_cfg.get("fps", 25),
    )

    tl_detector = None
    if mode in ("full", "track_and_trafficlight"):
        tl_detector = TrafficLightDetector()

    # --- Discover videos ---
    if single_video:
        videos = [single_video]
    else:
        data_root = data_cfg.get("video_dir", data_cfg.get("raw_video_dir", ""))
        videos = discover_videos(data_root, start_date, end_date)

    logger.info(f"Found {len(videos)} videos to process")
    logger.info(f"Mode: {mode}, Device: {device}, Skip frames: {skip_frames}")

    # --- Process ---
    all_stats = []
    for i, vid_path in enumerate(videos):
        logger.info(f"[{i+1}/{len(videos)}] Processing: {Path(vid_path).name}")
        try:
            stats = preprocess_video(
                video_path=vid_path,
                output_dir=output_dir,
                config=config,
                detector=detector,
                tracker=tracker,
                tl_detector=tl_detector,
                skip_frames=skip_frames,
            )
            all_stats.append(stats)
            logger.info(f"  Done: {stats['total_frames']} frames, "
                        f"{stats['total_trajectories']} trajectories, "
                        f"{stats['elapsed_sec']:.1f}s")
        except Exception as e:
            logger.error(f"Failed to process {vid_path}: {e}", exc_info=True)
            all_stats.append({"video": vid_path, "error": str(e)})

    # --- Summary ---
    total_trajs = sum(s.get("total_trajectories", 0) for s in all_stats)
    total_frames = sum(s.get("processed_frames", 0) for s in all_stats)
    total_time = sum(s.get("elapsed_sec", 0) for s in all_stats)
    errors = [s for s in all_stats if "error" in s]

    summary = {
        "total_videos": len(videos),
        "processed_successfully": len(all_stats) - len(errors),
        "total_frames": total_frames,
        "total_trajectories": total_trajs,
        "total_elapsed_hours": total_time / 3600,
        "errors": len(errors),
    }

    # Save summary
    summary_path = Path(output_dir) / "preprocess_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Videos: {summary['total_videos']}")
    logger.info(f"  Frames: {summary['total_frames']:,}")
    logger.info(f"  Trajectories: {summary['total_trajectories']:,}")
    logger.info(f"  Time: {summary['total_elapsed_hours']:.1f} hours")
    logger.info(f"  Errors: {summary['errors']}")
    logger.info(f"  Summary saved to: {summary_path}")

    return summary


# ======================================================================
# Helpers
# ======================================================================

def _detections_to_numpy(detections: List[DetectionResult]) -> np.ndarray:
    """Convert DetectionResult list → (N, 6) [x1, y1, x2, y2, conf, cls]."""
    arr = []
    for d in detections:
        arr.append([*d.bbox, d.confidence, d.class_id])
    return np.array(arr, dtype=np.float32) if arr else np.empty((0, 6))


def _numpy_to_tracked_objects(
    tracked: np.ndarray,          # (M, 7) [x1,y1,x2,y2,track_id,conf,cls]
    class_mapping: dict,
) -> List[TrackedObject]:
    objs = []
    for row in tracked:
        x1, y1, x2, y2, tid, conf, cls_id = row
        cls_id = int(cls_id)
        tid = int(tid)
        cls_name = class_mapping.get(cls_id, f"cls_{cls_id}")
        objs.append(TrackedObject(
            track_id=tid,
            class_name=cls_name,
            class_id=cls_id,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=float(conf),
            frame_id=0,
        ))
    return objs


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="离线预处理 — 从视频提取轨迹与交通灯状态"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--video", default=None, help="处理单个视频")
    parser.add_argument("--start-date", default=None, help="起始日期 (e.g. 2026_01_21)")
    parser.add_argument("--end-date", default=None, help="结束日期")
    parser.add_argument("--mode", default="full",
                        choices=["full", "track_and_trafficlight", "trajectories_only"])
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="跳帧数 (1=每帧, 5=每5帧)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计视频数,不实际处理")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Dry run
    if args.dry_run:
        data_root = config.get("data", {}).get("video_dir",
                      config.get("data", {}).get("raw_video_dir", ""))
        videos = discover_videos(data_root, args.start_date, args.end_date)
        print(f"Would process {len(videos)} videos:")
        for v in videos[:10]:
            print(f"  {v}")
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
        return

    # Run
    run_batch_preprocess(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        single_video=args.video,
        mode=args.mode,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    main()
