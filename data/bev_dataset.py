"""
BEV dataset — image + pseudo-BEV + pseudo camera mask per frame.

Built on the project's existing detection+tracking output (Ultralytics label
``.txt`` files) — it does NOT re-run detection. Each sample is a single video
frame with

    image         : (3, input_h, input_w) RGB, ImageNet-normalised
    pseudo_bev    : (C, H_bev, W_bev)  homography+detection weak supervision
    camera_mask   : (C, mask_h, mask_w) pseudo camera mask for the cycle loss
    (temporal mode adds the previous frame's image + pseudo_bev)

The homography is the *geometry teacher*; outputs are ``pseudo_*``, never
``gt_*``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.geometry.homography import Homography
from src.geometry.coordinate import BEVGrid
from src.bev.pseudo_bev import generate_pseudo_bev
from src.bev.label_reader import LabelReader
from src.bev.camera_bev_projection import rasterize_camera_mask

logger = logging.getLogger(__name__)

# ImageNet statistics (ResNet pretrained normalisation).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".ts")


class FrameReader:
    """Per-video frame source.

    Decodes each video *sequentially* once and (when ``cache=True``) keeps the
    decoded frames in memory at ``resize`` resolution. Random access — needed by
    shuffled training and temporal sampling — then becomes a cheap array index
    instead of a slow random-seek into compressed H.264.
    """

    def __init__(self, video_dir, cache: bool = True, resize=None):
        self.video_dir = Path(video_dir)
        self.cache = cache
        self.resize = resize  # (w, h); decoded frames are resized to this size
        self._frames: Dict[str, np.ndarray] = {}          # video_name -> (N, H, W, 3) uint8 BGR
        self._caps: Dict[str, cv2.VideoCapture] = {}      # used only when cache=False

    def find_video(self, video_name: str) -> Optional[Path]:
        for ext in VIDEO_EXTENSIONS:
            p = self.video_dir / f"{video_name}{ext}"
            if p.exists():
                return p
        return None

    def _decode_all(self, video_name: str) -> np.ndarray:
        """Sequentially decode the whole video once into a (N, H, W, 3) array."""
        path = self.find_video(video_name)
        if path is None:
            raise FileNotFoundError(f"video not found for '{video_name}' in {self.video_dir}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open video {path}")
        logger.info(f"FrameReader: decoding {path.name} ...")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)
            frames.append(frame)
            if len(frames) % 1000 == 0:
                logger.info(f"FrameReader: {len(frames)} frames decoded from {video_name} ...")
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        arr = np.stack(frames, axis=0)  # (N, H, W, 3) uint8 BGR
        logger.info(f"FrameReader: decoded {len(arr)} frames from {video_name} "
                    f"({arr.shape[2]}x{arr.shape[1]})")
        return arr

    def read(self, video_name: str, frame_id: int) -> np.ndarray:
        if self.cache:
            if video_name not in self._frames:
                self._frames[video_name] = self._decode_all(video_name)
            arr = self._frames[video_name]
            idx = int(frame_id) - 1  # 1-based frame id -> 0-based index
            if idx < 0 or idx >= len(arr):
                raise RuntimeError(
                    f"frame {frame_id} out of range for {video_name} ({len(arr)} frames)")
            return arr[idx]  # BGR (H, W, 3)

        # Non-cache fallback: single VideoCapture with seeking (slow for shuffled access).
        cap = self._caps.get(video_name)
        if cap is None:
            path = self.find_video(video_name)
            if path is None:
                raise FileNotFoundError(f"video not found for '{video_name}' in {self.video_dir}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError(f"cannot open video {path}")
            self._caps[video_name] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) - 1)  # 0-based
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"failed to read frame {frame_id} of {video_name}")
        return frame

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._frames.clear()


class BEVDataset(Dataset):
    """Frame-level BEV dataset over a list of videos.

    Parameters
    ----------
    video_dir, label_dir : str — video files and Ultralytics label ``.txt`` dir.
    video_names : Sequence[str] — video stems in this split.
    homography, grid : Homography, BEVGrid — the geometry teacher + BEV lattice.
    input_h, input_w : network input resolution.
    img_h, img_w : full-resolution frame size the homography is calibrated on.
    mask_h, mask_w : camera-mask (cycle) resolution (defaults to input size).
    sigma : Gaussian sigma for the pseudo-BEV heatmap (grid cells).
    temporal : sample consecutive frames (previous frame for L_temporal).
    normalize : apply ImageNet normalisation.
    """

    def __init__(
        self,
        video_dir,
        label_dir,
        video_names: Sequence[str],
        homography: Homography,
        grid: BEVGrid,
        input_h: int = 480,
        input_w: int = 640,
        img_h: int = 2160,
        img_w: int = 3840,
        mask_h: Optional[int] = None,
        mask_w: Optional[int] = None,
        sigma: float = 1.5,
        temporal: bool = False,
        normalize: bool = True,
    ):
        self.video_dir = Path(video_dir)
        self.label_dir = Path(label_dir)
        self.video_names = list(video_names)
        self.homography = homography
        self.grid = grid
        self.input_h, self.input_w = int(input_h), int(input_w)
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.mask_h = int(mask_h or input_h)
        self.mask_w = int(mask_w or input_w)
        self.sigma = float(sigma)
        self.temporal = temporal
        self.normalize = normalize

        self.reader = FrameReader(video_dir, cache=True, resize=(self.input_w, self.input_h))
        self.label_reader = LabelReader(img_w, img_h)
        self._label_cache: Dict[str, Dict[int, list]] = {}

        self.samples: List[Tuple[str, int]] = self._enumerate_samples()
        logger.info(f"BEVDataset: {len(self.samples)} frames "
                    f"across {len(self.video_names)} videos (temporal={temporal})")

    # -- construction -------------------------------------------------------

    def _enumerate_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for name in self.video_names:
            label_path = self.label_dir / f"{name}.txt"
            if not label_path.exists():
                continue
            frames = self.label_reader.load(label_path)
            for fid in sorted(frames.keys()):
                if self.temporal:
                    ok = fid >= 2 and (fid - 1) in frames  # needs a previous frame
                else:
                    ok = fid >= 1
                if ok:
                    samples.append((name, fid))
        return samples

    def get_detections(self, video_name: str, frame_id: int) -> list:
        if video_name not in self._label_cache:
            label_path = self.label_dir / f"{video_name}.txt"
            self._label_cache[video_name] = self.label_reader.load(label_path)
        return self._label_cache[video_name].get(int(frame_id), [])

    # -- Dataset interface --------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame_rgb(self, video_name: str, frame_id: int) -> torch.Tensor:
        frame = self.reader.read(video_name, frame_id)          # BGR, already at input size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img = (img - mean) / std
        return img

    def _build_targets(self, detections: list):
        pb = generate_pseudo_bev(detections, self.homography, self.grid, self.sigma)
        cam_mask = rasterize_camera_mask(
            detections, self.mask_h, self.mask_w, self.img_h, self.img_w
        )
        return torch.from_numpy(pb.heatmap), torch.from_numpy(cam_mask)

    def __getitem__(self, idx: int) -> dict:
        video_name, frame_id = self.samples[idx]
        detections = self.get_detections(video_name, frame_id)

        image = self._load_frame_rgb(video_name, frame_id)
        pseudo_bev, camera_mask = self._build_targets(detections)

        sample = {
            "image": image,
            "pseudo_bev": pseudo_bev,
            "camera_mask": camera_mask,
            "video": video_name,
            "frame_id": int(frame_id),
        }

        if self.temporal:
            det_prev = self.get_detections(video_name, frame_id - 1)
            image_prev = self._load_frame_rgb(video_name, frame_id - 1)
            pseudo_prev, _ = self._build_targets(det_prev)
            sample["image_prev"] = image_prev
            sample["pseudo_bev_prev"] = pseudo_prev

        return sample


def bev_collate_fn(batch: List[dict]) -> dict:
    """Collate BEV samples (image + pseudo_bev + camera_mask [+ temporal])."""
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "pseudo_bev": torch.stack([b["pseudo_bev"] for b in batch], dim=0),
        "camera_mask": torch.stack([b["camera_mask"] for b in batch], dim=0),
        "video": [b["video"] for b in batch],
        "frame_id": [b["frame_id"] for b in batch],
    }
    if "image_prev" in batch[0]:
        out["image_prev"] = torch.stack([b["image_prev"] for b in batch], dim=0)
        out["pseudo_bev_prev"] = torch.stack([b["pseudo_bev_prev"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Split + factory
# ---------------------------------------------------------------------------

def list_label_videos(label_dir) -> List[str]:
    """Return sorted video stems that have a label ``.txt`` file."""
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return []
    return sorted(p.stem for p in label_dir.glob("*.txt"))


def split_videos(video_names: Sequence[str], split_ratio, seed: int = 42) -> Dict[str, list]:
    """Deterministic train/val/test split of video stems."""
    names = sorted(video_names)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(names))
    names = [names[i] for i in perm]
    r1, r2 = float(split_ratio[0]), float(split_ratio[0] + split_ratio[1])
    n = len(names)
    n_train = int(round(n * r1))
    n_val = int(round(n * r2))
    return {
        "train": names[:n_train],
        "val": names[n_train:n_val],
        "test": names[n_val:],
    }


def build_bev_datasets(config: dict, homography, grid, temporal: bool = False,
                       splits=("train", "val", "test")):
    """Build train/val/test BEVDatasets from a config dict.

    Reads ``config['data']['bev']`` for video/label dirs, split ratio or explicit
    per-split video lists, and input/image/mask sizes.
    """
    bev_cfg = config.get("data", {}).get("bev", {})
    video_dir = bev_cfg.get("video_dir", config.get("data", {}).get("raw_video_dir", "."))
    label_dir = bev_cfg.get("label_dir", str(Path(video_dir) / "labels"))

    input_h = int(bev_cfg.get("input_h", 480))
    input_w = int(bev_cfg.get("input_w", 640))
    img_h = int(bev_cfg.get("img_h", 2160))
    img_w = int(bev_cfg.get("img_w", 3840))
    mask_h = bev_cfg.get("mask_h")
    mask_w = bev_cfg.get("mask_w")
    sigma = float(bev_cfg.get("sigma", 1.5))
    normalize = bool(bev_cfg.get("normalize", True))

    # Resolve the per-split video lists.
    all_names = list_label_videos(label_dir)
    explicit = {s: bev_cfg.get(f"{s}_videos") for s in splits}
    if any(explicit[s] for s in splits):
        split_map = {s: (explicit[s] or []) for s in splits}
    else:
        ratio = bev_cfg.get("split_ratio", [0.7, 0.15, 0.15])
        split_map = split_videos(all_names, ratio, seed=int(bev_cfg.get("split_seed", 42)))

    datasets = {}
    for s in splits:
        datasets[s] = BEVDataset(
            video_dir=video_dir,
            label_dir=label_dir,
            video_names=split_map[s],
            homography=homography,
            grid=grid,
            input_h=input_h, input_w=input_w,
            img_h=img_h, img_w=img_w,
            mask_h=mask_h, mask_w=mask_w,
            sigma=sigma, temporal=temporal, normalize=normalize,
        )
    return datasets
