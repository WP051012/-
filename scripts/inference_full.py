#!/usr/bin/env python3
"""
端到端闯红灯预测推理脚本
=========================
完整 pipeline:
    视频 → 检测追踪 → 交通感知图 → 感知记忆 → 感知GRU
    → 状态变化检测 → FlowChain轨迹预测 → 闯红灯分类

用法:
    python scripts/inference_full.py --video path/to/video.mp4 --config configs/default.yaml
    python scripts/inference_full.py --video path/to/video.mp4 --stop-line 100,200,300,200 --visualize
"""

import argparse
import logging
import pickle
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
    NodeFeatureEncoder,
    PerceptionGraphBuilder,
    TrafficPerceptionGraph,
    SceneGraph,
    SpatialEncoder,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Model loading helpers
# ======================================================================

def build_model(config: dict, device: str = "cuda") -> dict:
    """Build all model components from config."""
    graph_cfg = config.get("graph", {})
    memory_cfg = config.get("memory", {})
    gru_cfg = config.get("perception_gru", {})
    flow_cfg = config.get("flow_chain", {})

    models = {}

    # Perception graph
    models["perception_graph"] = TrafficPerceptionGraph(
        node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
        gat_hidden_dim=graph_cfg.get("gat_hidden_dim", 64),
        gat_out_dim=graph_cfg.get("gat_hidden_dim", 128),
        gat_heads=graph_cfg.get("gat_heads", 4),
    )

    # Scene graph
    models["scene_graph"] = SceneGraph(
        in_dim=graph_cfg.get("gat_hidden_dim", 128),
        hidden_dim=64,
        heads=graph_cfg.get("gat_heads", 4),
    )

    # Perception memory
    models["perception_memory"] = TrafficPerceptionMemory(
        node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
        behavioral_dim=memory_cfg.get("behavioral_dim", 128),
        environmental_dim=memory_cfg.get("environmental_dim", 128),
        interactive_dim=memory_cfg.get("interactive_dim", 128),
        fusion_dim=memory_cfg.get("fusion_dim", 256),
    )

    # Decay controller
    models["decay_controller"] = DecayController(
        memory_names=("behavioral", "environmental", "interactive"),
        memory_dim=memory_cfg.get("behavioral_dim", 128),
        decay_rate=memory_cfg.get("decay_rate", 0.01),
    )

    # Perception GRU
    models["perception_gru"] = PerceptionGRU(
        input_dim=flow_cfg.get("trajectory_dim", 2),
        hidden_dim=gru_cfg.get("hidden_dim", 256),
        behavioral_dim=memory_cfg.get("behavioral_dim", 128),
        environmental_dim=memory_cfg.get("environmental_dim", 128),
        interactive_dim=memory_cfg.get("interactive_dim", 128),
    )

    # Perception context encoder
    models["context_encoder"] = PerceptionContextEncoder(
        behavioral_dim=memory_cfg.get("behavioral_dim", 128),
        environmental_dim=memory_cfg.get("environmental_dim", 128),
        interactive_dim=memory_cfg.get("interactive_dim", 128),
        context_dim=flow_cfg.get("condition_dim", 256),
    )

    # FlowChain
    models["flow_chain"] = FlowChainPredictor(
        obs_len=flow_cfg.get("obs_len", 8),
        pred_len=flow_cfg.get("pred_len", 12),
        trajectory_dim=flow_cfg.get("trajectory_dim", 2),
        hidden_dim=flow_cfg.get("hidden_dim", 256),
        condition_dim=flow_cfg.get("condition_dim", 256),
        num_flows=flow_cfg.get("num_flows", 3),
    )

    # Change detector
    models["change_detector"] = PerceptionChangeDetector()

    # Move to device
    for name, model in models.items():
        models[name] = model.to(device)
        model.eval()

    return models


def load_checkpoint(models: dict, checkpoint_path: str, device: str = "cuda") -> None:
    """Load model weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for name, model in models.items():
        if name in ckpt:
            model.load_state_dict(ckpt[name])
            logger.info(f"Loaded weights for {name}")


# ======================================================================
# Observation buffer
# ======================================================================

class ObservationBuffer:
    """Sliding-window buffer for observation trajectories."""

    def __init__(self, obs_len: int = 8, traj_dim: int = 2):
        self.obs_len = obs_len
        self.traj_dim = traj_dim
        self._buffer: List[np.ndarray] = []  # list of (2,) arrays

    def add(self, position: np.ndarray) -> None:
        self._buffer.append(position)
        if len(self._buffer) > self.obs_len:
            self._buffer.pop(0)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.obs_len

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(np.stack(self._buffer[-self.obs_len:]),
                            dtype=torch.float32)  # (obs_len, 2)

    def reset(self) -> None:
        self._buffer.clear()


# ======================================================================
# Main inference function
# ======================================================================

def run_inference(
    video_path: str,
    config: dict,
    models: Optional[dict] = None,
    checkpoint: Optional[str] = None,
    stop_line: Optional[StopLine] = None,
    junction: Optional[JunctionRegion] = None,
    device: str = "cuda",
    visualize: bool = False,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run full end-to-end inference on a video.

    Returns
    -------
    list of dict
        Per-frame prediction results.
    """
    # --- Config ---
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    flow_cfg = config.get("flow_chain", {})
    obs_len = flow_cfg.get("obs_len", 8)
    pred_len = flow_cfg.get("pred_len", 12)
    num_samples = flow_cfg.get("num_samples", 20)

    # --- Models ---
    if models is None:
        models = build_model(config, device)
    if checkpoint:
        load_checkpoint(models, checkpoint, device)

    # --- Detection / Tracking ---
    detector = YOLODetector(
        model_path=det_cfg.get("model_name", "yolov8n.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        device=device,
    )
    tracker = ByteTrackWrapper(
        track_buffer=trk_cfg.get("track_buffer", 30),
    )
    traj_manager = TrajectoryManager()

    # --- Classifier ---
    violation_checker = RedLightViolationChecker(
        stop_line=stop_line,
        junction=junction,
    )
    classifier = RedLightProbabilityEstimator(
        violation_checker=violation_checker,
        threshold=config.get("red_light", {}).get("violation_threshold", 0.5),
    )

    # --- Buffers ---
    obs_buffer = ObservationBuffer(obs_len)
    memory_state: Dict[str, torch.Tensor] = {}
    gru_hidden: Optional[torch.Tensor] = None
    frame_count = 0
    results: List[Dict] = []

    # --- Video ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    pbar = tqdm(range(total_frames), desc="闯红灯预测推理", unit="frame")

    for frame_id in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ---------------------------------------------------------------
        # Step 1: Detection + Tracking
        # ---------------------------------------------------------------
        detections = detector.detect(frame)
        det_np = np.array([[*d.bbox, d.confidence, d.class_id]
                           for d in detections], dtype=np.float32) if detections \
                  else np.empty((0, 6))
        tracked_np = tracker.update(det_np, frame)

        tracked_objs = []
        for row in tracked_np:
            x1, y1, x2, y2, tid, conf, cls_id = row
            cls_name = detector.class_mapping.get(int(cls_id), f"cls_{int(cls_id)}")
            tracked_objs.append(TrackedObject(
                track_id=int(tid),
                class_name=cls_name,
                class_id=int(cls_id),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                frame_id=frame_id,
            ))
        traj_manager.update(frame_id, tracked_objs)

        # ---------------------------------------------------------------
        # Step 2: Find target pedestrian
        # ---------------------------------------------------------------
        target_obj = None
        for obj in tracked_objs:
            if obj.class_name == "pedestrian":
                target_obj = obj
                break

        if target_obj is None:
            results.append({"frame": frame_id, "status": "no_target"})
            obs_buffer.reset()
            continue

        # Update observation buffer with target position as (cx, cy) in pixels
        obs_buffer.add(np.array(target_obj.center, dtype=np.float32))

        if not obs_buffer.is_full:
            results.append({"frame": frame_id, "status": "buffering"})
            continue

        # ---------------------------------------------------------------
        # Step 3: Build perception graph for this frame
        # ---------------------------------------------------------------
        N = len(tracked_objs)
        bboxes = torch.tensor([o.bbox for o in tracked_objs], device=device)
        positions = torch.tensor([o.center for o in tracked_objs], device=device)
        class_names = [o.class_name for o in tracked_objs]

        # Compute velocities from trajectory history
        velocities = torch.zeros(N, 2, device=device)
        for i, obj in enumerate(tracked_objs):
            traj = traj_manager.get_trajectory(obj.track_id)
            if traj and traj.length >= 2:
                p1 = np.array(traj.positions[-2])
                p2 = np.array(traj.positions[-1])
                velocities[i] = torch.tensor(p2 - p1, device=device) * fps

        # Find target index
        target_idx = next(
            i for i, obj in enumerate(tracked_objs)
            if obj.track_id == target_obj.track_id
        )

        # Forward through perception graph
        with torch.no_grad():
            node_embeddings, target_emb = models["perception_graph"](
                bboxes=bboxes,
                class_names=class_names,
                positions=positions,
                velocities=velocities,
                target_idx=target_idx,
            )

        # ---------------------------------------------------------------
        # Step 4: Perception memory
        # ---------------------------------------------------------------
        # Classify node types
        graph_builder = PerceptionGraphBuilder()
        _, node_types, _ = graph_builder.build(
            positions=positions.cpu().numpy(),
            class_names=class_names,
            target_idx=target_idx,
        )

        with torch.no_grad():
            c, memory_info = models["perception_memory"].from_graph_output(
                node_embeddings=node_embeddings,
                node_types=node_types,
                target_idx=target_idx,
            )

        # ---------------------------------------------------------------
        # Step 5: Change detection
        # ---------------------------------------------------------------
        current_ids = {obj.track_id for obj in tracked_objs}
        struct_events = models["change_detector"].detect_structural(
            frame_id=frame_id,
            current_node_ids=current_ids,
            agent_count=len(tracked_objs),
        )

        # Check if re-initialisation needed
        reinit_needed = len(struct_events) > 0

        if reinit_needed:
            models["decay_controller"].reset()
            gru_hidden = None
            obs_buffer.reset()
            for event in struct_events:
                logger.debug(f"Change detected: {event.detail}")

        # ---------------------------------------------------------------
        # Step 6: FlowChain prediction
        # ---------------------------------------------------------------
        if obs_buffer.is_full:
            obs_tensor = obs_buffer.to_tensor().unsqueeze(0).to(device)  # (1, obs_len, 2)
            c_tensor = c.unsqueeze(0)  # (1, fusion_dim)

            with torch.no_grad():
                prediction = models["flow_chain"](
                    obs_trajectory=obs_tensor,
                    perception_c=c_tensor,
                    num_samples=num_samples,
                )

            # -----------------------------------------------------------
            # Step 7: Red-light classification
            # -----------------------------------------------------------
            samples = prediction["samples"].squeeze(1)  # (N, pred_len, 2)
            is_violation, viol_prob = classifier.classify(samples)

            results.append({
                "frame": frame_id,
                "status": "predicted",
                "target_id": target_obj.track_id,
                "violation_probability": float(viol_prob),
                "is_violation": bool(is_violation),
                "predicted_trajectory_mean": prediction["mean"].squeeze(0).cpu().numpy(),
                "predicted_trajectory_std": prediction["std"].squeeze(0).cpu().numpy(),
                "perception_vector": c.cpu().numpy(),
                "memory_confidence": memory_info.get("confidence_weights", None),
                "change_events": [e.change_type.value for e in struct_events],
            })
        else:
            results.append({"frame": frame_id, "status": "buffering"})

        # ---------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------
        if visualize and "predicted_trajectory_mean" in results[-1]:
            vis = _draw_prediction(
                frame, results[-1], obs_buffer, tracked_objs,
            )
            cv2.imshow("Red-Light Prediction", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pbar.set_postfix({
            "det": len(detections),
            "trk": len(tracked_objs),
            "state": results[-1].get("status", "?"),
        })

    cap.release()
    cv2.destroyAllWindows()

    # --- Save results ---
    if output_path:
        out_file = Path(output_path) / "predictions.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {out_file}")

    return results


# ======================================================================
# Visualization helper
# ======================================================================

def _draw_prediction(frame, result, obs_buffer, tracked_objs):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw observed trajectory (green)
    obs_positions = obs_buffer._buffer
    for i in range(1, len(obs_positions)):
        p1 = tuple(obs_positions[i-1].astype(int))
        p2 = tuple(obs_positions[i].astype(int))
        cv2.line(vis, p1, p2, (0, 255, 0), 2)

    # Draw predicted trajectory mean (blue)
    if "predicted_trajectory_mean" in result:
        pred = result["predicted_trajectory_mean"]
        # Convert from relative to absolute by offsetting from last obs
        last_obs = obs_positions[-1] if obs_positions else np.array([w/2, h/2])
        for i in range(1, len(pred)):
            p1 = tuple((last_obs + pred[i-1]).astype(int))
            p2 = tuple((last_obs + pred[i]).astype(int))
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # Draw violation probability
    prob = result.get("violation_probability", 0)
    is_viol = result.get("is_violation", False)
    color = (0, 0, 255) if is_viol else (0, 255, 0)
    label = f"Red-Light Risk: {prob:.2f} {'⚠️' if is_viol else '✓'}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return vis


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="端到端闯红灯预测推理")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--checkpoint", default=None, help="模型权重路径")
    parser.add_argument("--stop-line", default=None,
                        help="停止线坐标: x1,y1,x2,y2")
    parser.add_argument("--output", default="output/results/", help="输出目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="FlowChain蒙特卡洛采样数")

    args = parser.parse_args()

    # Config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override num_samples
    config.setdefault("flow_chain", {})["num_samples"] = args.num_samples

    # Stop line
    stop_line = None
    if args.stop_line:
        x1, y1, x2, y2 = map(float, args.stop_line.split(","))
        stop_line = StopLine(x1=x1, y1=y1, x2=x2, y2=y2)

    # Build models
    models = build_model(config, args.device)
    if args.checkpoint:
        load_checkpoint(models, args.checkpoint, args.device)

    # Run
    results = run_inference(
        video_path=args.video,
        config=config,
        models=models,
        stop_line=stop_line,
        device=args.device,
        visualize=args.visualize,
        output_path=args.output,
    )

    # Summary
    violations = [r for r in results if r.get("is_violation")]
    print(f"\n{'='*50}")
    print(f"Total frames processed: {len(results)}")
    print(f"Frames with violation detected: {len(violations)}")
    if violations:
        print(f"Max violation probability: {max(r.get('violation_probability', 0) for r in violations):.3f}")


if __name__ == "__main__":
    main()
