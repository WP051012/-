"""
Trajectory dataset with optional scene context for perception graph.

Two modes:
    "trajectory_only" — only trajectory coords (fast, for baselines)
    "with_scene"      — also loads per-frame scene objects from label files
                        (for TrafficPerceptionModel)

Data sources:
    data/processed/trajectories/{video}/trajectories.npz
    labels/{video}.txt  (Ultralytics tracking format)
"""

import csv as _csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "pedestrian", 1: "bicycle", 2: "car",
               3: "motorcycle", 5: "bus", 9: "traffic_light"}


# ======================================================================
# Dataset
# ======================================================================

class TrajectoryDataset(Dataset):
    """
    Parameters
    ----------
    data_dir : str        — path to preprocessed trajectories
    label_dir : str       — path to raw label .txt files (only for mode="with_scene")
    obs_len, pred_len, stride, min_trajectory_len : int
    target_classes : list — only include these classes
    mode : str            — "trajectory_only" or "with_scene"
    img_width, img_height : int
    max_scene_samples : int — cap on samples with scene data (to limit RAM)
    """

    def __init__(
        self,
        data_dir: str,
        label_dir: Optional[str] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        stride: int = 4,
        min_trajectory_len: int = 20,
        target_classes: Optional[List[str]] = None,
        mode: str = "trajectory_only",
        img_width: float = 3840.0,
        img_height: float = 2160.0,
        max_scene_samples: int = 50000,
        max_samples: int = 0,
        domain_label_map: Optional[Dict[str, int]] = None,
        precomputed_dir: Optional[str] = None,
        condition_map: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        force_scene: bool = False,
        crossing_region: Optional[List[Tuple[float, float]]] = None,
        junction_roi: Optional[List[Tuple[float, float]]] = None,
        crosswalk_roi: Optional[List[Tuple[float, float]]] = None,
        stop_line: Optional[List[float]] = None,
        return_context: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.seq_len = obs_len + pred_len
        self.mode = mode
        self.target_classes = target_classes or ["pedestrian"]
        self.img_width = img_width
        self.img_height = img_height
        self.domain_label_map = domain_label_map or {}
        self.max_samples = max_samples
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self._precomputed_cache: Dict[str, dict] = {}
        self.force_scene = force_scene
        self.crossing_region = crossing_region
        self.junction_roi = junction_roi
        self.crosswalk_roi = crosswalk_roi
        self.stop_line = stop_line
        self.return_context = return_context

        # Build samples
        self.samples = self._build_all_samples(min_trajectory_len)

        # Apply max_samples cap (shuffled for representative subset)
        if self.max_samples > 0 and len(self.samples) > self.max_samples:
            import random
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]

        logger.info(f"TrajectoryDataset: {len(self.samples)} samples (mode={mode})")

        # Attach GAT condition embeddings to samples
        if condition_map is not None:
            self._attach_conditions(condition_map)

        # Scene cache: video_name → {frame_id: [{cls, xc, yc, w, h, tid}, ...]}
        self._scene_cache: Dict[str, Dict[int, list]] = {}
        self._loaded_videos: set = set()

        # Traffic light cache: video_name → pd.DataFrame with columns [frame_id, overall_state, ...]
        self._tl_cache: Dict[str, tuple] = {}

        # Window-level violation labels (crossing in pred window while red)
        if self.crossing_region is not None and self.samples:
            self._compute_window_violation_labels()

        # If with_scene mode, preload scene data for samples (capped)
        if mode == "with_scene" and self.label_dir and self.samples:
            self._preload_scene_data(max_scene_samples)

    def _attach_conditions(self, condition_map: Dict[str, Dict[str, torch.Tensor]]):
        """Attach GAT condition embeddings to samples by (video, track_id__obs_start)."""
        cond_dim = None
        n_matched = 0
        n_fallback = 0
        for sample in self.samples:
            video = sample["video"]
            key = f"{sample['track_id']}__{sample['obs_start']}"
            emb = condition_map.get(video, {}).get(key)
            if emb is not None:
                sample["cond_embedding"] = emb.clone().detach()
                if cond_dim is None:
                    cond_dim = emb.shape[-1]
                n_matched += 1
            else:
                # Fallback: zeros (should be rare since precompute had 100% match)
                if cond_dim is None:
                    cond_dim = 64  # default
                sample["cond_embedding"] = torch.zeros(cond_dim)
                n_fallback += 1
        match_pct = 100 * n_matched / max(len(self.samples), 1)
        logger.info(f"  GAT conditions: {n_matched} matched, {n_fallback} fallback "
                    f"({match_pct:.1f}%), dim={cond_dim}")

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _load_violation_map(self) -> dict:
        """Build {(video_name, track_key): is_violation} from all violation_labels.csv."""
        viol_map = {}
        for csv_path in self.data_dir.rglob("violation_labels.csv"):
            video_name = csv_path.parent.name
            try:
                with open(csv_path, "r") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        key = (video_name, row.get("track_key", ""))
                        val = row.get("is_violation", "0")
                        # Handle both "True"/"False" strings and 0/1 ints
                        if isinstance(val, str):
                            viol_map[key] = val.strip().lower() in ("true", "1")
                        else:
                            viol_map[key] = bool(int(val))
            except Exception:
                continue
        return viol_map

    def _build_all_samples(self, min_len: int) -> List[dict]:
        samples = []
        npz_files = list(self.data_dir.rglob("trajectories.npz"))

        # Pre-load violation labels for all videos
        violation_map = self._load_violation_map()
        n_viol_labeled = 0

        for npz_path in npz_files:
            video_name = npz_path.parent.name
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue

            for tid_str in data.files:
                try:
                    traj_data = data[tid_str].item()
                except AttributeError:
                    continue

                positions = traj_data.get("positions")
                if positions is None:
                    continue
                T = positions.shape[0]
                if T < min_len:
                    continue

                class_name = traj_data.get("class_name", "unknown")
                if isinstance(class_name, np.ndarray):
                    class_name = str(class_name[0]) if len(class_name) > 0 else "unknown"
                if self.target_classes and class_name not in self.target_classes:
                    continue

                track_id = int(tid_str.split("_")[-1]) if "_" in tid_str else 0
                frames = traj_data.get("frames")

                # Look up violation label
                is_viol = violation_map.get((video_name, tid_str), False)

                for start in range(0, T - self.seq_len, self.stride):
                    obs_end = start + self.obs_len
                    pred_end = obs_end + self.pred_len
                    if pred_end > T:
                        break

                    # Domain ID lookup (for prompt generator / meta-learning)
                    domain_id = self.domain_label_map.get(video_name, -1)

                    sample = {
                        "video": video_name,
                        "track_id": track_id,
                        "class_name": class_name,
                        "obs_start": int(start),
                        "obs_positions": positions[start:obs_end].astype(np.float32),
                        "target_positions": positions[obs_end:pred_end].astype(np.float32),
                        "is_violation": is_viol,
                        "domain_id": domain_id,
                    }

                    if is_viol:
                        n_viol_labeled += 1

                    if frames is not None:
                        sample["obs_frames"] = frames[start:obs_end].astype(np.int32)
                        sample["target_frames"] = frames[obs_end:pred_end].astype(np.int32)
                    else:
                        sample["obs_frames"] = np.arange(start, obs_end, dtype=np.int32)
                        sample["target_frames"] = np.arange(obs_end, pred_end, dtype=np.int32)

                    samples.append(sample)

        logger.info(f"  Violation-labeled samples: {n_viol_labeled}/{len(samples)}")
        return samples

    # ------------------------------------------------------------------
    # Scene data preloading
    # ------------------------------------------------------------------

    def _preload_scene_data(self, max_samples: int):
        """Preload scene context data for a subset of samples."""
        import random
        random.seed(42)

        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        indices = indices[:max_samples]

        # Group by video for efficient label loading
        video_groups = defaultdict(list)
        for i in indices:
            v = self.samples[i]["video"]
            video_groups[v].append(i)

        loaded = 0
        for video_name, sample_idxs in video_groups.items():
            if loaded >= max_samples:
                break
            self._ensure_scene_loaded(video_name)
            for idx in sample_idxs:
                self.samples[idx]["has_scene"] = True
                loaded += 1
                if loaded >= max_samples:
                    break

        logger.info(f"Preloaded scene data for {loaded} samples")

    def _ensure_scene_loaded(self, video_name: str):
        """Load scene data for a video into cache (if not already)."""
        if video_name in self._loaded_videos:
            return

        label_path = self.label_dir / f"{video_name}.txt"
        if not label_path.exists():
            self._scene_cache[video_name] = {}
            self._loaded_videos.add(video_name)
            return

        frames = defaultdict(list)
        current_frame = None

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("### Frame:"):
                    m = re.search(r'_(\d+)\.txt', line)
                    if m:
                        current_frame = int(m.group(1))
                    continue
                parts = line.split()
                if len(parts) < 6 or current_frame is None:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    track_id = int(parts[5])
                except (ValueError, IndexError):
                    continue

                frames[current_frame].append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                    "xc": xc, "yc": yc, "w": w, "h": h,
                    "track_id": track_id,
                })

        self._scene_cache[video_name] = dict(frames)
        self._loaded_videos.add(video_name)

    def _load_traffic_lights(self, video_name: str):
        """Load traffic_lights.csv for a video into cache.

        Returns (frame_ids, states) — two aligned, sorted numpy arrays for
        O(log n) nearest-frame lookup — or None if the video has no CSV.
        """
        if video_name in self._tl_cache:
            return self._tl_cache[video_name]

        tl_path = self.data_dir / video_name / "traffic_lights.csv"
        if not tl_path.exists():
            self._tl_cache[video_name] = None
            return None

        try:
            df = pd.read_csv(tl_path)
            frame_ids = df["frame_id"].to_numpy(dtype=np.int64)
            states = df["overall_state"].astype(str).to_numpy()
            order = np.argsort(frame_ids, kind="stable")
            self._tl_cache[video_name] = (frame_ids[order], states[order])
            return self._tl_cache[video_name]
        except Exception:
            self._tl_cache[video_name] = None
            return None

    @staticmethod
    def _get_traffic_light_states(tl, query_frames) -> List[str]:
        """Nearest-frame traffic light states for one or more frames (vectorized).

        tl = (frame_ids, states) sorted arrays from _load_traffic_lights.
        Returns a list of states, one per query frame, each in
        {'red','green','yellow','off','unknown'}.
        """
        qs = list(query_frames)
        if tl is None or len(tl[0]) == 0:
            return ["unknown"] * len(qs)
        frame_ids, states = tl
        q = np.asarray(qs, dtype=np.int64)
        idx = np.searchsorted(frame_ids, q, side="left")
        n = len(frame_ids)
        right = np.clip(idx, 0, n - 1)
        left = np.clip(idx - 1, 0, n - 1)
        d_left = np.abs(frame_ids[left] - q)
        d_right = np.abs(frame_ids[right] - q)
        nearest = np.where(d_left <= d_right, left, right)
        return [str(states[i]) for i in nearest]

    def _compute_window_violation_labels(self):
        """Compute per-window violation label, aligned with the prediction task.

        A window is a violation iff the pedestrian's ground-truth target
        trajectory enters the crossing region for >=3 frames WHILE the light is
        red during those frames. This mirrors check_violation but is scoped to
        the prediction window instead of the whole track, so it matches the
        gate (P_cross x is_red) which predicts crossing in the next 12 frames.
        """
        n_pos = 0
        for sample in self.samples:
            target_pos = sample.get("target_positions")
            target_frames = sample.get("target_frames")
            if target_pos is None or target_frames is None:
                sample["is_violation_window"] = False
                continue
            tl = self._load_traffic_lights(sample["video"])
            states = self._get_traffic_light_states(tl, target_frames)
            cross_frames = 0
            viol = False
            for i in range(target_pos.shape[0]):
                if states[i] != "red":
                    continue
                x = float(target_pos[i, 0]); y = float(target_pos[i, 1])
                if _point_in_polygon_px(x, y, self.crossing_region):
                    cross_frames += 1
                    if cross_frames >= 3:
                        viol = True
                        break
            sample["is_violation_window"] = viol
            if viol:
                n_pos += 1
        logger.info(f"  Window-level violations: {n_pos}/{len(self.samples)} "
                    f"({100*n_pos/max(1,len(self.samples)):.1f}%)")

    def _get_scene_data(self, video_name: str, obs_frames: np.ndarray,
                        target_frames: Optional[np.ndarray] = None):
        """Extract per-frame scene data for a sample's observation (and prediction) window."""
        self._ensure_scene_loaded(video_name)
        frames_data = self._scene_cache.get(video_name, {})

        # Load traffic light data for this video
        tl = self._load_traffic_lights(video_name)

        all_bboxes = []
        all_class_names = []
        all_positions = []
        all_track_ids = []

        for fi in obs_frames:
            dets = frames_data.get(int(fi), [])
            bboxes = np.array([
                [d["xc"] * self.img_width - d["w"] * self.img_width / 2,
                 d["yc"] * self.img_height - d["h"] * self.img_height / 2,
                 d["xc"] * self.img_width + d["w"] * self.img_width / 2,
                 d["yc"] * self.img_height + d["h"] * self.img_height / 2]
                for d in dets
            ], dtype=np.float32) if dets else np.zeros((0, 4), dtype=np.float32)

            positions = np.array([
                [d["xc"] * self.img_width, d["yc"] * self.img_height]
                for d in dets
            ], dtype=np.float32) if dets else np.zeros((0, 2), dtype=np.float32)

            class_names = [d["class_name"] for d in dets]
            track_ids = [d["track_id"] for d in dets]

            all_bboxes.append(bboxes)
            all_class_names.append(class_names)
            all_positions.append(positions)
            all_track_ids.append(track_ids)

        # Traffic light states for observation + prediction windows (vectorized)
        tl_states = self._get_traffic_light_states(tl, obs_frames)
        pred_tl_states = (self._get_traffic_light_states(tl, target_frames)
                          if target_frames is not None else [])

        # Pad to same N per frame for batching (max N per frame)
        max_N = max(len(b) for b in all_bboxes) if all_bboxes else 0
        if max_N == 0:
            max_N = 1  # at least 1 dummy

        padded_bboxes = np.zeros((self.obs_len, max_N, 4), dtype=np.float32)
        padded_positions = np.zeros((self.obs_len, max_N, 2), dtype=np.float32)
        mask = np.zeros((self.obs_len, max_N), dtype=bool)

        for t in range(self.obs_len):
            n = len(all_bboxes[t])
            if n > 0:
                padded_bboxes[t, :n] = all_bboxes[t]
                padded_positions[t, :n] = all_positions[t]
                mask[t, :n] = True

        return {
            "bboxes": torch.tensor(padded_bboxes),
            "positions": torch.tensor(padded_positions),
            "class_names": all_class_names,
            "track_ids": all_track_ids,
            "mask": torch.tensor(mask),
            "traffic_light_states": tl_states,  # list of str, length obs_len
            "pred_traffic_light_states": pred_tl_states,  # list of str, length pred_len
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_geom_feat(self, obs_positions: np.ndarray) -> np.ndarray:
        """(obs_len, 2) px coords -> (obs_len, 6) geometry features.

        Columns: [d_stop_line, d_crosswalk, inside_crosswalk, inside_junction,
                  dir_x_to_crosswalk, dir_y_to_crosswalk].
        Distances are normalized by the image diagonal; direction is a unit
        vector. A column is 0 when the corresponding polygon is not provided.
        """
        T = obs_positions.shape[0]
        feat = np.zeros((T, 6), dtype=np.float32)
        diag = float(np.sqrt(self.img_width ** 2 + self.img_height ** 2)) + 1e-8

        cross_centroid = None
        if self.crosswalk_roi is not None and len(self.crosswalk_roi) >= 1:
            pts = np.asarray(self.crosswalk_roi, dtype=np.float32)
            cross_centroid = pts.mean(axis=0)

        for t in range(T):
            x = float(obs_positions[t, 0])
            y = float(obs_positions[t, 1])

            if self.stop_line is not None and len(self.stop_line) >= 4:
                feat[t, 0] = _point_to_segment_dist(x, y, self.stop_line) / diag
            if self.crosswalk_roi is not None and len(self.crosswalk_roi) >= 1:
                feat[t, 1] = _min_dist_to_polygon(x, y, self.crosswalk_roi) / diag
            if self.crosswalk_roi is not None and len(self.crosswalk_roi) >= 3:
                feat[t, 2] = 1.0 if _point_in_polygon_px(x, y, self.crosswalk_roi) else 0.0
            if self.junction_roi is not None and len(self.junction_roi) >= 3:
                feat[t, 3] = 1.0 if _point_in_polygon_px(x, y, self.junction_roi) else 0.0
            if cross_centroid is not None:
                dx = float(cross_centroid[0]) - x
                dy = float(cross_centroid[1]) - y
                nrm = float(np.sqrt(dx * dx + dy * dy)) + 1e-8
                feat[t, 4] = dx / nrm
                feat[t, 5] = dy / nrm

        return feat

    def _compute_aux_labels(self, target_positions: np.ndarray) -> dict:
        """Auxiliary labels from FUTURE ground-truth (teacher supervision only).

        These are NOT available at inference time — they are returned solely to
        supervise the Goal/Intent/CrossingTime heads during training.

        Returns
        -------
        goal_label    (2,)   normalized final position [0,1]
        intent_label  (1,)   1.0 if any future frame enters the region else 0.0
        crossing_label (1,)  first entering frame (1..pred_len), or pred_len+1 = NO_CROSS
        """
        T = target_positions.shape[0]
        norm = np.array([self.img_width, self.img_height], dtype=np.float32)
        goal = target_positions[-1] / norm

        region = self.junction_roi if self.junction_roi is not None else self.crosswalk_roi
        intent = 0.0
        crossing = T + 1  # NO_CROSS default
        if region is not None and len(region) >= 3:
            for t in range(T):
                x = float(target_positions[t, 0])
                y = float(target_positions[t, 1])
                if _point_in_polygon_px(x, y, region):
                    intent = 1.0
                    crossing = t + 1  # 1-indexed frame
                    break

        return {
            "goal_label": torch.tensor(goal, dtype=torch.float32),
            "intent_label": torch.tensor(intent, dtype=torch.float32),
            "crossing_label": torch.tensor(crossing, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        obs = torch.tensor(sample["obs_positions"], dtype=torch.float32)
        target = torch.tensor(sample["target_positions"], dtype=torch.float32)

        result = {
            "obs_trajectory": obs,
            "target_trajectory": target,
            "track_id": sample["track_id"],
            "video": sample["video"],
            "class_name": sample["class_name"],
            "is_violation": sample.get("is_violation", False),
            "is_violation_window": sample.get("is_violation_window", sample.get("is_violation", False)),
            "domain_id": sample.get("domain_id", -1),
            "img_width": self.img_width,
            "img_height": self.img_height,
        }

        # Scene data for perception graph
        if self.mode == "with_scene" and (self.force_scene or sample.get("has_scene")):
            scene = self._get_scene_data(sample["video"], sample["obs_frames"],
                                         sample.get("target_frames"))
            result["scene"] = scene
            result["scene"]["target_id"] = sample["track_id"]
        elif self.precomputed_dir is not None:
            # Load from precomputed .npz files
            scene = self._load_precomputed_scene(sample["video"], sample["obs_frames"])
            if scene is not None:
                result["scene"] = scene

        # GAT condition embedding (attached by _attach_conditions)
        if "cond_embedding" in sample:
            result["cond_embedding"] = sample["cond_embedding"]

        # Context features for conditional FlowChain (signal + geometry).
        # Signal uses the OBSERVATION window only (never the future), so no
        # label leakage. Geometry is derived dynamically from obs positions.
        if self.return_context:
            tl = self._load_traffic_lights(sample["video"])
            tl_states = self._get_traffic_light_states(tl, sample["obs_frames"])
            result["signal"] = signal_states_to_one_hot(tl_states)  # (obs_len, 5)
            result["geom_feat"] = torch.tensor(
                self._compute_geom_feat(sample["obs_positions"]), dtype=torch.float32
            )  # (obs_len, 6)
            result.update(self._compute_aux_labels(sample["target_positions"]))

        return result

    def _load_precomputed_scene(self, video_name: str, obs_frames: np.ndarray) -> Optional[dict]:
        """Load scene data from precomputed .npz files for observation frames."""
        if self.precomputed_dir is None:
            return None

        # Lazy load the precomputed npz for this video
        if video_name not in self._precomputed_cache:
            npz_path = self.precomputed_dir / f"{video_name}.npz"
            if not npz_path.exists():
                self._precomputed_cache[video_name] = None
                return None
            try:
                raw = np.load(npz_path, allow_pickle=True)
                self._precomputed_cache[video_name] = raw["data"].item()
            except Exception:
                self._precomputed_cache[video_name] = None
                return None

        frames_data = self._precomputed_cache.get(video_name)
        if frames_data is None:
            return None

        # Collect per-frame data
        all_bboxes = []
        all_class_names = []
        all_positions = []
        all_velocities = []
        all_class_ids = []

        for fi in obs_frames:
            key = str(int(fi))
            fd = frames_data.get(key)
            if fd is None:
                # Empty frame
                all_bboxes.append(np.zeros((0, 4), dtype=np.float32))
                all_class_names.append([])
                all_positions.append(np.zeros((0, 2), dtype=np.float32))
                all_velocities.append(np.zeros((0, 2), dtype=np.float32))
                all_class_ids.append(np.zeros(0, dtype=np.int32))
            else:
                all_bboxes.append(fd["bboxes"])
                all_class_names.append(list(fd["class_names"]))
                all_positions.append(fd["positions"])
                all_velocities.append(fd["velocities"])
                all_class_ids.append(fd["class_ids"])

        return {
            "bboxes": all_bboxes,           # list of (N_i, 4) arrays
            "class_names": all_class_names, # list of [str, ...]
            "positions": all_positions,     # list of (N_i, 2) arrays
            "velocities": all_velocities,   # list of (N_i, 2) arrays
            "class_ids": all_class_ids,     # list of (N_i,) arrays
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        class_counts = {}
        video_counts = {}
        for s in self.samples:
            c = s["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1
            video_counts[s["video"]] = video_counts.get(s["video"], 0) + 1

        return {
            "total_samples": len(self.samples),
            "num_videos": len(video_counts),
            "class_distribution": class_counts,
        }

    def with_scene_subset(self):
        """Return indices of samples that have scene data."""
        return [i for i, s in enumerate(self.samples) if s.get("has_scene")]


# ======================================================================
# Collate
# ======================================================================

def trajectory_collate_fn(batch: List[dict]) -> dict:
    """Collate function handling optional scene data and domain labels."""
    obs = torch.stack([b["obs_trajectory"] for b in batch], dim=0)
    target = torch.stack([b["target_trajectory"] for b in batch], dim=0)

    result = {
        "obs_trajectory": obs,
        "target_trajectory": target,
        "track_id": [b["track_id"] for b in batch],
        "video": [b["video"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
        "is_violation": torch.tensor([b.get("is_violation", False) for b in batch], dtype=torch.float32),
    }

    # Domain IDs
    domain_ids = [b.get("domain_id", -1) for b in batch]
    if any(d != -1 for d in domain_ids):
        result["domain_id"] = torch.tensor(domain_ids, dtype=torch.long)

    # Collect scene data if present
    # Pass as list of per-sample dicts (not collated) — the perception pipeline
    # processes them individually via disjoint-union batching.
    has_scene = any("scene" in b for b in batch)
    if has_scene:
        result["scene"] = [b.get("scene") for b in batch]

    # GAT condition embeddings
    has_cond = any("cond_embedding" in b for b in batch)
    if has_cond:
        result["cond_embedding"] = torch.stack([b["cond_embedding"] for b in batch], dim=0)

    # Context features (signal + geometry) for conditional FlowChain
    has_signal = any("signal" in b for b in batch)
    if has_signal:
        result["signal"] = torch.stack([b["signal"] for b in batch], dim=0)      # (B, 8, 5)
        result["geom_feat"] = torch.stack([b["geom_feat"] for b in batch], dim=0)  # (B, 8, 6)
        result["goal_label"] = torch.stack([b["goal_label"] for b in batch], dim=0)  # (B, 2)
        result["intent_label"] = torch.stack([b["intent_label"] for b in batch], dim=0)  # (B,)
        result["crossing_label"] = torch.stack([b["crossing_label"] for b in batch], dim=0)  # (B,)

    return result


def _collate_scene_frames(batch: List[dict]) -> dict:
    """Collate per-sample scene lists into batch tensors.

    Each sample's scene dict has:
      bboxes:     list[T] of (N_t, 4) arrays
      positions:  list[T] of (N_t, 2) arrays
      velocities: list[T] of (N_t, 2) arrays
      class_names: list[T] of [str]
      class_ids:  list[T] of (N_t,) arrays

    Output: stacked to (B, T, N_max, D) with a padding mask.
    """
    pass  # simplified: return as list of dicts for per-sample processing
    scene_list = []
    for b in batch:
        if "scene" in b and b["scene"] is not None:
            scene_list.append(b["scene"])
        else:
            scene_list.append(None)
    return scene_list


# ======================================================================
# Crosswalk candidate filtering
# ======================================================================

def _point_in_polygon_px(x: float, y: float, polygon) -> bool:
    """Ray-casting point-in-polygon. polygon: [(x,y), ...] in pixel coords."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside


def _min_dist_to_polygon(x: float, y: float, polygon) -> float:
    """Minimum Euclidean distance from (x,y) to polygon vertices (px)."""
    best = float("inf")
    for px, py in polygon:
        d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        if d < best:
            best = d
    return best


def _point_to_segment_dist(x: float, y: float, seg) -> float:
    """Minimum distance from point (x,y) to segment [x1,y1,x2,y2] (px)."""
    x1, y1, x2, y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return float(np.sqrt((x - x1) ** 2 + (y - y1) ** 2))
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px, py = x1 + t * dx, y1 + t * dy
    return float(np.sqrt((x - px) ** 2 + (y - py) ** 2))


# Traffic-signal vocabulary (order is fixed; index 4 = 'unknown' fallback)
SIGNAL_CLASSES = ["red", "green", "yellow", "off", "unknown"]
_SIGNAL_IDX = {s: i for i, s in enumerate(SIGNAL_CLASSES)}


def signal_states_to_one_hot(states: List[str]) -> torch.Tensor:
    """Map a list of signal-state strings (len T) to (T, 5) one-hot tensor."""
    idx = [_SIGNAL_IDX.get(str(s), 4) for s in states]
    return torch.eye(len(SIGNAL_CLASSES))[idx]


def is_crossing_candidate(
    obs_trajectory: np.ndarray,
    target_trajectory: np.ndarray = None,
    crosswalk_roi=None,
    stop_line=None,
    junction_roi=None,
    angle_min_deg: float = 80.0,
    angle_max_deg: float = 90.0,
) -> bool:
    """
    Determine if a pedestrian is a potential crossing candidate.

    Conditions (any one satisfied → keep):
      A) (Training only) Future GT enters junction ROI.
      B) Motion direction nearly perpendicular to stop line (80°–90°).
         i.e. the pedestrian is clearly walking across, not along the road.

    This tight filter ensures FlowChain only sees crossing trajectories,
    improving its ability to predict junction entry.
    """
    obs = np.asarray(obs_trajectory)

    # A: future GT enters junction (train only)
    if target_trajectory is not None and junction_roi is not None and len(junction_roi) >= 3:
        tgt = np.asarray(target_trajectory)
        for i in range(tgt.shape[0]):
            if _point_in_polygon_px(float(tgt[i, 0]), float(tgt[i, 1]), junction_roi):
                return True

    # B: heading nearly perpendicular to stop line (80°–90°)
    if stop_line is not None and len(stop_line) >= 4 and len(obs) >= 4:
        vel = obs[-1] - obs[-4]
        v_norm = np.sqrt(float(vel[0])**2 + float(vel[1])**2)
        if v_norm < 1e-6:
            return False  # stationary → not crossing
        sl_dx = float(stop_line[2]) - float(stop_line[0])
        sl_dy = float(stop_line[3]) - float(stop_line[1])
        sl_norm = np.sqrt(sl_dx**2 + sl_dy**2)
        if sl_norm < 1e-6:
            return False

        # cos(angle) = |v·d| / (|v|·|d|), angle ∈ [0°, 90°]
        cos_angle = abs(float(vel[0]) * sl_dx + float(vel[1]) * sl_dy) / (v_norm * sl_norm)
        # For angle ∈ [80°, 90°]: cos ∈ [0, cos(80°)] ≈ [0, 0.1736]
        if cos_angle <= np.cos(np.deg2rad(angle_min_deg)):
            return True

    return False


def filter_crosswalk_candidates(
    samples: list,
    crosswalk_roi=None,
    stop_line=None,
    use_future_gt: bool = True,
) -> list:
    """
    Filter a list of dataset samples, keeping only crossing candidates.
    Set use_future_gt=False for test set.
    """
    kept = []
    for s in samples:
        obs = s["obs_trajectory"].numpy() if hasattr(s["obs_trajectory"], "numpy") else np.asarray(s["obs_trajectory"])
        tgt = s.get("target_trajectory")
        if tgt is not None:
            tgt = tgt.numpy() if hasattr(tgt, "numpy") else np.asarray(tgt)
        if not use_future_gt:
            tgt = None
        if is_crossing_candidate(obs, tgt, crosswalk_roi, stop_line):
            kept.append(s)
    return kept
