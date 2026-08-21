#!/usr/bin/env python3
"""
检测+追踪主流程脚本
====================
从视频中提取交通参与者轨迹，为后续图构建和轨迹预测提供结构化数据。

用法:
    python scripts/detect_track.py --video path/to/video.mp4 --output path/to/output/
    python scripts/detect_track.py --video path/to/video.mp4 --config configs/default.yaml --visualize
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import (
    YOLODetector,
    ByteTrackWrapper,
    TrajectoryManager,
    TrackedObject,
    DetectionResult,
    create_detector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detections_to_numpy(detections: list) -> np.ndarray:
    """Convert DetectionResult list → numpy (N, 6) for tracker input."""
    arr = []
    for d in detections:
        arr.append([*d.bbox, d.confidence, d.class_id])
    return np.array(arr, dtype=np.float32) if arr else np.empty((0, 6))


def numpy_to_tracked_objects(
    tracked: np.ndarray,
    class_id_to_name: dict,
) -> list:
    """
    Convert tracker output (M, 7) → list of TrackedObject.

    tracked columns: [x1, y1, x2, y2, track_id, conf, class_id]
    """
    objs = []
    for row in tracked:
        x1, y1, x2, y2, tid, conf, cls_id = row
        cls_id = int(cls_id)
        tid = int(tid)
        cls_name = class_id_to_name.get(cls_id, f"class_{cls_id}")
        objs.append(TrackedObject(
            track_id=tid,
            class_name=cls_name,
            class_id=cls_id,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=float(conf),
            frame_id=0,  # filled in by TrajectoryManager
        ))
    return objs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_detection_tracking(
    video_path: str,
    output_dir: str,
    config: dict,
    visualize: bool = False,
    save_video: bool = False,
    start_frame: int = 0,
    end_frame: int = -1,
) -> TrajectoryManager:
    """
    Run full detection + tracking pipeline on a video.

    Parameters
    ----------
    video_path : str
    output_dir : str
    config : dict
    visualize : bool
    save_video : bool
    start_frame : int
    end_frame : int

    Returns
    -------
    TrajectoryManager
    """
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialise modules ---
    detector = create_detector(config)
    tracker = ByteTrackWrapper(
        track_buffer=trk_cfg.get("track_buffer", 30),
        track_thresh=trk_cfg.get("track_thresh", 0.5),
        match_thresh=trk_cfg.get("match_thresh", 0.8),
    )
    traj_manager = TrajectoryManager(
        min_length=trk_cfg.get("min_trajectory_length", 15),
    )

    # Reverse class mapping for tracker output
    id_to_name = {v: k for k, v in detector.class_mapping.items()}
    # Also include raw model class names
    for cid, cname in detector._model_class_names.items():
        if cid not in id_to_name:
            id_to_name[cid] = cname

    # --- Video I/O ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames

    logger.info(f"Video: {video_path}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    logger.info(f"Processing frames {start_frame}–{end_frame}")

    # Video writer (optional)
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video_path = out_dir / f"{Path(video_path).stem}_tracked.mp4"
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        logger.info(f"Writing output video to: {out_video_path}")

    # --- Frame loop ---
    pbar = tqdm(range(start_frame, end_frame), desc="检测追踪", unit="frame")

    for frame_id in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detection
        detections: list = detector.detect(frame)

        # 2. Tracking
        det_np = detections_to_numpy(detections)
        # Map class IDs through traffic class mapping
        cls_map = detector.class_mapping
        for i in range(len(det_np)):
            raw_cls = int(det_np[i, 5])
            det_np[i, 5] = raw_cls  # keep original for consistency

        tracked_np = tracker.update(det_np, frame)

        # 3. Register with trajectory manager
        tracked_objs = numpy_to_tracked_objects(tracked_np, id_to_name)
        # Override frame_id
        for obj in tracked_objs:
            obj.frame_id = frame_id
        traj_manager.update(frame_id, tracked_objs)

        # 4. Visualize
        if visualize or save_video:
            vis_frame = draw_detections(frame, tracked_objs)
            if visualize:
                cv2.imshow("Detection & Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer:
                writer.write(vis_frame)

        # Update progress
        pbar.set_postfix({
            "det": len(detections),
            "trk": len(tracked_objs),
            "traj": traj_manager.num_trajectories,
        })

    # --- Cleanup ---
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # --- Report ---
    valid_trajs = traj_manager.get_valid_trajectories()
    logger.info(f"Total trajectories: {traj_manager.num_trajectories}")
    logger.info(f"Valid trajectories (>= {traj_manager.min_length} frames): {len(valid_trajs)}")

    # Per-class breakdown
    class_counts = {}
    for t in valid_trajs:
        class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1
    for cls, count in sorted(class_counts.items()):
        logger.info(f"  {cls}: {count} trajectories")

    # --- Save trajectories ---
    df = traj_manager.to_dataframe()
    csv_path = out_dir / "trajectories.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Trajectories saved to {csv_path}")

    npz_path = out_dir / "trajectories.npz"
    np.savez_compressed(npz_path, **traj_manager.export_numpy())
    logger.info(f"Numpy trajectories saved to {npz_path}")

    return traj_manager


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Class colours (BGR)
CLASS_COLORS = {
    "pedestrian":     (0, 255, 0),     # green
    "bicycle":        (255, 255, 0),   # cyan
    "motorcycle":     (255, 255, 0),   # cyan
    "car":            (255, 0, 0),     # blue
    "bus":            (0, 0, 255),     # red
    "truck":          (0, 0, 200),     # dark red
    "traffic_light":  (0, 255, 255),   # yellow
    "traffic_sign":   (255, 0, 255),   # magenta
    "lane_line":      (128, 128, 128), # gray
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(
    frame: np.ndarray,
    tracked_objs: list,
    show_trail: bool = True,
    trail_length: int = 20,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and track IDs on frame.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3)
    tracked_objs : list of TrackedObject
    show_trail : bool
        Draw trajectory trail.
    trail_length : int
        Number of past positions to show.

    Returns
    -------
    np.ndarray
    """
    vis = frame.copy()
    for obj in tracked_objs:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"[{obj.track_id}] {obj.class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="检测+追踪 — 从视频提取交通参与者轨迹")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default="output/trajectories/", help="输出目录")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--visualize", action="store_true", help="实时可视化")
    parser.add_argument("--save-video", action="store_true", help="保存标注视频")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    run_detection_tracking(
        video_path=args.video,
        output_dir=args.output,
        config=config,
        visualize=args.visualize,
        save_video=args.save_video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
