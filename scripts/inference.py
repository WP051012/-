#!/usr/bin/env python3
"""
端到端闯红灯预测推理
====================
完整 pipeline: 视频 → YOLO检测+ByteTrack追踪 → 交通灯HSV识别
→ 感知图构建 → 感知记忆 → FlowChain轨迹预测 → 闯红灯概率分类

用法:
    # 基本推理
    python scripts/inference.py --video path/to/video.mp4 --config configs/default.yaml

    # 带可视化和路口标注
    python scripts/inference.py --video path/to/video.mp4 \
        --stop-line 1000,1800,2800,1800 \
        --junction-roi 1000,1800,2800,2100 \
        --visualize

    # 使用预训练权重
    python scripts/inference.py --video path/to/video.mp4 \
        --checkpoint checkpoints/stage3_best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import YOLODetector, ByteTrackWrapper, TrajectoryManager, TrackedObject
from src.graph import (
    TrafficPerceptionGraph,
    PerceptionGraphBuilder,
    SceneGraph,
)
from src.memory import TrafficPerceptionMemory, DecayController
from src.prediction import (
    PerceptionGRU,
    PerceptionContextEncoder,
    PerceptionChangeDetector,
    FlowChainPredictor,
)
from src.classification import (
    StopLine,
    JunctionRegion,
    RedLightViolationChecker,
    RedLightProbabilityEstimator,
)
from utils.traffic_light import (
    TrafficLightDetector,
    TrafficLightROI,
    LightState,
    discover_traffic_light_rois,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Model building
# ======================================================================

def build_models(config: dict, device: str = "cuda") -> dict:
    """Build all model components from config."""
    graph_cfg = config.get("graph", {})
    memory_cfg = config.get("memory", {})
    gru_cfg = config.get("perception_gru", {})
    flow_cfg = config.get("flow_chain", {})

    models = {}

    models["perception_graph"] = TrafficPerceptionGraph(
        node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
        gat_hidden_dim=graph_cfg.get("gat_hidden_dim", 64),
        gat_out_dim=graph_cfg.get("gat_hidden_dim", 128),
        gat_heads=graph_cfg.get("gat_heads", 4),
    )

    models["perception_memory"] = TrafficPerceptionMemory(
        node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
        behavioral_dim=memory_cfg.get("behavioral_dim", 128),
        environmental_dim=memory_cfg.get("environmental_dim", 128),
        interactive_dim=memory_cfg.get("interactive_dim", 128),
        fusion_dim=memory_cfg.get("fusion_dim", 256),
    )

    models["flow_chain"] = FlowChainPredictor(
        obs_len=flow_cfg.get("obs_len", 8),
        pred_len=flow_cfg.get("pred_len", 12),
        trajectory_dim=flow_cfg.get("trajectory_dim", 2),
        hidden_dim=flow_cfg.get("d_model", 64),
        condition_dim=flow_cfg.get("condition_dim", 256),
        num_flows=flow_cfg.get("nvp_num_blocks", 3),
    )

    models["change_detector"] = PerceptionChangeDetector()

    for name, model in models.items():
        models[name] = model.to(device).eval()

    return models


def load_checkpoint(models: dict, ckpt_path: str, device: str = "cuda") -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_state = ckpt.get("model", ckpt)
    for name, model in models.items():
        state = {k.replace(f"{name}.", ""): v
                 for k, v in model_state.items()
                 if k.startswith(f"{name}.")}
        if state:
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights for {name}")


# ======================================================================
# Observation buffer
# ======================================================================

class ObsBuffer:
    def __init__(self, length: int = 8):
        self.length = length
        self.buf: List[np.ndarray] = []

    def add(self, pos: np.ndarray) -> None:
        self.buf.append(pos)
        if len(self.buf) > self.length:
            self.buf.pop(0)

    @property
    def full(self) -> bool:
        return len(self.buf) >= self.length

    def tensor(self) -> torch.Tensor:
        arr = np.stack(self.buf[-self.length:])
        return torch.tensor(arr, dtype=torch.float32)

    def reset(self) -> None:
        self.buf.clear()


# ======================================================================
# Main inference
# ======================================================================

def run_inference(
    video_path: str,
    config: dict,
    models: Optional[dict] = None,
    checkpoint: Optional[str] = None,
    stop_line: Optional[StopLine] = None,
    junction: Optional[JunctionRegion] = None,
    tl_rois: Optional[List[TrafficLightROI]] = None,
    device: str = "cuda",
    visualize: bool = False,
    output_dir: Optional[str] = None,
    skip_frames: int = 1,
) -> List[dict]:
    """
    Run full inference pipeline on a video.

    Returns list of per-frame prediction results.
    """
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    flow_cfg = config.get("flow_chain", {})
    video_cfg = config.get("video", {})

    obs_len = flow_cfg.get("obs_len", 8)
    pred_len = flow_cfg.get("pred_len", 12)
    num_samples = flow_cfg.get("num_samples", 20)
    img_w = video_cfg.get("width", 3840)
    img_h = video_cfg.get("height", 2160)
    fps = video_cfg.get("fps", 25)

    # --- Models ---
    if models is None:
        models = build_models(config, device)
    if checkpoint:
        load_checkpoint(models, checkpoint, device)

    # --- Detection & Tracking ---
    detector = YOLODetector(
        model_path=det_cfg.get("model_name", "yolov8n.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        device=device,
    )
    tracker = ByteTrackWrapper(
        track_buffer=trk_cfg.get("track_buffer", 30),
        frame_rate=fps,
    )
    traj_manager = TrajectoryManager()
    graph_builder = PerceptionGraphBuilder()

    # --- Traffic Light Detector ---
    tl_detector = TrafficLightDetector()
    discover_rois = tl_rois is None

    # --- Violation Checker & Classifier ---
    violation_checker = RedLightViolationChecker(
        stop_line=stop_line, junction=junction,
    )
    classifier = RedLightProbabilityEstimator(violation_checker)

    # --- State ---
    obs_buffer = ObsBuffer(obs_len)
    results: List[dict] = []
    cls9_positions: List[tuple] = []   # for ROI discovery
    prev_tl_state: Optional[LightState] = None
    perception_c = torch.zeros(flow_cfg.get("condition_dim", 256), device=device)

    # --- Video ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(range(0, total_frames, skip_frames), desc="Inference", unit="frame")

    for frame_id in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        # ============================================================
        # Step 1: Detection
        # ============================================================
        detections = detector.detect(frame)

        # Collect cls=9 positions
        for d in detections:
            if d.class_name == "traffic_light":
                cls9_positions.append((d.center[0] / img_w, d.center[1] / img_h))

        # ============================================================
        # Step 2: Tracking
        # ============================================================
        det_np = np.array(
            [[*d.bbox, d.confidence, d.class_id] for d in detections],
            dtype=np.float32,
        ) if detections else np.empty((0, 6))

        tracked_np = tracker.update(det_np, frame)

        tracked_objs = []
        for row in tracked_np:
            x1, y1, x2, y2, tid, conf, cls_id = row
            cls_name = detector.class_mapping.get(int(cls_id), f"cls_{int(cls_id)}")
            tracked_objs.append(TrackedObject(
                track_id=int(tid), class_name=cls_name, class_id=int(cls_id),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf), frame_id=frame_id,
            ))
        traj_manager.update(frame_id, tracked_objs)

        # ============================================================
        # Step 3: Find target pedestrian
        # ============================================================
        target_obj = next(
            (o for o in tracked_objs if o.class_name == "pedestrian"), None,
        )

        if target_obj is None:
            results.append({"frame": frame_id, "status": "no_target"})
            obs_buffer.reset()
            continue

        obs_buffer.add(np.array(target_obj.center, dtype=np.float32))

        # ============================================================
        # Step 4: Traffic light state detection
        # ============================================================
        tl_state = None
        if tl_rois and not discover_rois:
            tl_state = tl_detector.detect_intersection_state(frame, tl_rois)
        elif discover_rois and len(cls9_positions) >= 200:
            # Discover ROIs after enough data
            tl_rois = discover_traffic_light_rois(
                np.array(cls9_positions), img_width=img_w, img_height=img_h,
                n_clusters=5,
            )
            logger.info(f"Discovered {len(tl_rois)} traffic light ROIs")
            discover_rois = False

        # Detect change
        tl_changed = (tl_state is not None and prev_tl_state is not None
                      and tl_state != prev_tl_state)
        prev_tl_state = tl_state

        # ============================================================
        # Step 5: Perception graph (if enough data)
        # ============================================================
        if len(tracked_objs) >= 2:
            N = len(tracked_objs)
            bboxes_t = torch.tensor([o.bbox for o in tracked_objs], device=device)
            positions_t = torch.tensor([o.center for o in tracked_objs], device=device)
            class_names = [o.class_name for o in tracked_objs]

            # Compute velocities
            velocities_t = torch.zeros(N, 2, device=device)
            for i, obj in enumerate(tracked_objs):
                traj = traj_manager.get_trajectory(obj.track_id)
                if traj and traj.length >= 2:
                    p1, p2 = np.array(traj.positions[-2]), np.array(traj.positions[-1])
                    velocities_t[i] = torch.tensor(
                        (p2 - p1) * fps, device=device,
                    )

            target_idx = next(
                i for i, o in enumerate(tracked_objs)
                if o.track_id == target_obj.track_id
            )

            with torch.no_grad():
                node_emb, target_emb = models["perception_graph"](
                    bboxes=bboxes_t, class_names=class_names,
                    positions=positions_t, velocities=velocities_t,
                    target_idx=target_idx,
                )

                # Perception memory
                _, node_types, _ = graph_builder.build(
                    positions_t.cpu().numpy(), class_names, target_idx,
                )
                c_vec, _ = models["perception_memory"].from_graph_output(
                    node_embeddings=node_emb, node_types=node_types,
                    target_idx=target_idx,
                )
                perception_c = c_vec

        # ============================================================
        # Step 6: Change detection
        # ============================================================
        current_ids = {o.track_id for o in tracked_objs}
        events = models["change_detector"].detect_structural(
            frame_id=frame_id, current_node_ids=current_ids,
            traffic_light_state=(
                {"red": 0, "yellow": 1, "green": 2, "off": -1}.get(
                    tl_state.value if tl_state else "off", -1,
                )
            ),
            agent_count=len(tracked_objs),
        )

        reinit = len(events) > 0 or tl_changed
        if reinit:
            obs_buffer.reset()

        # ============================================================
        # Step 7: FlowChain prediction
        # ============================================================
        result = {"frame": frame_id, "status": "buffering"}
        result["traffic_light"] = tl_state.value if tl_state else "unknown"

        if obs_buffer.full:
            obs_t = obs_buffer.tensor().unsqueeze(0).to(device)
            c_t = perception_c.unsqueeze(0)

            with torch.no_grad():
                pred = models["flow_chain"](
                    obs_trajectory=obs_t, perception_c=c_t,
                    num_samples=num_samples,
                )

            # ============================================================
            # Step 8: Red-light classification
            # ============================================================
            samples = pred["samples"].squeeze(1)
            is_viol, viol_prob = classifier.classify(samples)

            result["status"] = "predicted"
            result["target_id"] = target_obj.track_id
            result["violation_probability"] = float(viol_prob)
            result["is_violation"] = bool(is_viol)
            result["pred_mean"] = pred["mean"].squeeze(0).cpu().numpy()
            result["pred_std"] = pred["std"].squeeze(0).cpu().numpy()
            result["perception_c_norm"] = float(perception_c.norm())

            if events:
                result["change_events"] = [e.change_type.value for e in events]

        results.append(result)

        # ============================================================
        # Visualization
        # ============================================================
        if visualize:
            _draw(frame, result, obs_buffer, tracked_objs, tl_state)

        pbar.set_postfix({
            "det": len(detections), "trk": len(tracked_objs),
            "tl": tl_state.value if tl_state else "?",
            "risk": f"{result.get('violation_probability', 0):.2f}",
        })

    cap.release()
    cv2.destroyAllWindows()

    # --- Save ---
    if output_dir:
        out_path = Path(output_dir) / f"{Path(video_path).stem}_predictions.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays for JSON
        serializable = []
        for r in results:
            r2 = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    r2[k] = v.tolist()
                else:
                    r2[k] = v
            serializable.append(r2)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    return results


# ======================================================================
# Visualization
# ======================================================================

def _draw(frame, result, obs_buf, trk_objs, tl_state):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Traffic light state indicator
    if tl_state:
        colours = {LightState.RED: (0, 0, 255), LightState.YELLOW: (0, 255, 255),
                   LightState.GREEN: (0, 255, 0), LightState.OFF: (128, 128, 128)}
        cv2.circle(vis, (40, 40), 20, colours.get(tl_state, (128, 128, 128)), -1)
        cv2.putText(vis, tl_state.value.upper(), (70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Observed trajectory
    if obs_buf.buf:
        pts = [tuple(p.astype(int)) for p in obs_buf.buf]
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], (0, 255, 0), 2)

    # Predicted trajectory
    if "pred_mean" in result:
        pred = result["pred_mean"]
        last = obs_buf.buf[-1] if obs_buf.buf else np.array([w // 2, h // 2])
        for i in range(1, len(pred)):
            p1 = (int(last[0] + pred[i - 1][0]), int(last[1] + pred[i - 1][1]))
            p2 = (int(last[0] + pred[i][0]), int(last[1] + pred[i][1]))
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # Risk indicator
    prob = result.get("violation_probability", 0)
    is_viol = result.get("is_violation", False)
    color = (0, 0, 255) if is_viol else (0, 255, 0)
    label = f"Red-Light Risk: {prob:.2f} {'!!' if is_viol else 'OK'}"
    cv2.putText(vis, label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Red-Light Prediction", vis)
    cv2.waitKey(1)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="闯红灯预测推理")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="模型权重")
    parser.add_argument("--stop-line", default=None,
                        help="停止线: x1,y1,x2,y2")
    parser.add_argument("--junction-roi", default=None,
                        help="路口区域: x1,y1,x2,y2")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip", type=int, default=1, help="跳帧间隔")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Stop line
    stop_line = None
    if args.stop_line:
        x1, y1, x2, y2 = map(float, args.stop_line.split(","))
        stop_line = StopLine(x1=x1, y1=y1, x2=x2, y2=y2)

    # Junction ROI
    junction = None
    if args.junction_roi:
        x1, y1, x2, y2 = map(float, args.junction_roi.split(","))
        jx, jy = (x1 + x2) / 2, (y1 + y2) / 2
        junction = JunctionRegion(polygon=[
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ])
        _ = jx, jy  # unused

    models = build_models(config, args.device)
    if args.checkpoint:
        load_checkpoint(models, args.checkpoint, args.device)

    results = run_inference(
        video_path=args.video,
        config=config,
        models=models,
        stop_line=stop_line,
        junction=junction,
        device=args.device,
        visualize=args.visualize,
        output_dir=args.output,
        skip_frames=args.skip,
    )

    # Summary
    viol_frames = [r for r in results if r.get("is_violation")]
    print(f"\n{'='*50}")
    print(f"Frames processed: {len(results)}")
    print(f"Violations detected: {len(viol_frames)}")
    if viol_frames:
        probs = [r.get("violation_probability", 0) for r in viol_frames]
        print(f"Max risk: {max(probs):.3f}")


if __name__ == "__main__":
    main()
