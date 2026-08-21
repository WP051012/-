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

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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

        # Build samples
        self.samples = self._build_all_samples(min_trajectory_len)
        logger.info(f"TrajectoryDataset: {len(self.samples)} samples (mode={mode})")

        # Scene cache: video_name → {frame_id: [{cls, xc, yc, w, h, tid}, ...]}
        self._scene_cache: Dict[str, Dict[int, list]] = {}
        self._loaded_videos: set = set()

        # If with_scene mode, preload scene data for samples (capped)
        if mode == "with_scene" and self.label_dir and self.samples:
            self._preload_scene_data(max_scene_samples)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _load_violation_map(self) -> dict:
        """Build {(video_name, track_key): is_violation} from all violation_labels.csv."""
        import csv as _csv
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

                    sample = {
                        "video": video_name,
                        "track_id": track_id,
                        "class_name": class_name,
                        "obs_start": int(start),
                        "obs_positions": positions[start:obs_end].astype(np.float32),
                        "target_positions": positions[obs_end:pred_end].astype(np.float32),
                        "is_violation": is_viol,
                    }

                    if is_viol:
                        n_viol_labeled += 1

                    if frames is not None:
                        sample["obs_frames"] = frames[start:obs_end].astype(np.int32)
                    else:
                        sample["obs_frames"] = np.arange(start, obs_end, dtype=np.int32)

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

    def _get_scene_data(self, video_name: str, obs_frames: np.ndarray):
        """Extract per-frame scene data for a sample's observation window."""
        self._ensure_scene_loaded(video_name)
        frames_data = self._scene_cache.get(video_name, {})

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
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

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
            "img_width": self.img_width,
            "img_height": self.img_height,
        }

        # Scene data for perception graph
        if self.mode == "with_scene" and sample.get("has_scene"):
            scene = self._get_scene_data(sample["video"], sample["obs_frames"])
            result["scene"] = scene
            # Find target_idx in the scene
            result["scene"]["target_id"] = sample["track_id"]

        return result

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
    """Collate function handling optional scene data."""
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

    # Collect scene data if present
    has_scene = any("scene" in b for b in batch)
    if has_scene:
        scene_list = []
        for b in batch:
            if "scene" in b:
                scene_list.append(b["scene"])
            else:
                scene_list.append(None)
        result["scene_list"] = scene_list

    return result
