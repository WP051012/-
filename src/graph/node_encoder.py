"""
Node feature encoders for traffic perception graph.

References:
    STRR: Spatiotemporal Relationship Reasoning (spatial encoding, edge weights)
    Social-STGCNN: social-STGCNN (graph building from trajectories)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Spatial Position Encoding (8-dim, from STRR)
# ======================================================================

class SpatialEncoder(nn.Module):
    """
    8-dimensional spatial encoding from bounding boxes.

    Encodes:
        [x1, y1, x2, y2, w, h, area, aspect_ratio]

    All values are normalised relative to the image / scene dimensions.

    Parameters
    ----------
    img_width, img_height : float
        Reference dimensions for normalisation.
    """

    def __init__(self, img_width: float = 1920.0, img_height: float = 1080.0):
        super().__init__()
        self.img_w = img_width
        self.img_h = img_height

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        bboxes : Tensor (..., 4)
            [x1, y1, x2, y2] in pixel coordinates.

        Returns
        -------
        Tensor (..., 8)
        """
        x1 = bboxes[..., 0] / self.img_w
        y1 = bboxes[..., 1] / self.img_h
        x2 = bboxes[..., 2] / self.img_w
        y2 = bboxes[..., 3] / self.img_h

        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / (h + 1e-6)

        return torch.stack([x1, y1, x2, y2, w, h, area, aspect], dim=-1)

    @staticmethod
    def from_bbox_list(
        bboxes: List[Tuple[float, float, float, float]],
        img_width: float = 1920.0,
        img_height: float = 1080.0,
    ) -> np.ndarray:
        """Numpy version for non-gradient contexts."""
        arr = np.array(bboxes, dtype=np.float32)
        x1 = arr[:, 0] / img_width
        y1 = arr[:, 1] / img_height
        x2 = arr[:, 2] / img_width
        y2 = arr[:, 3] / img_height
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / (h + 1e-6)
        return np.stack([x1, y1, x2, y2, w, h, area, aspect], axis=-1)


# ======================================================================
# Motion Feature Encoder (6-dim)
# ======================================================================

class MotionEncoder(nn.Module):
    """
    6-dimensional motion encoding.

    Encodes:
        [vx, vy, speed, angle, acc_x, acc_y]

    Velocities are computed as finite differences between consecutive
    trajectory positions.
    """

    def __init__(self, fps: float = 30.0):
        super().__init__()
        self.fps = fps

    def forward(
        self,
        positions: torch.Tensor,      # (T, 2)  or (B, T, 2)
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute motion features from position sequence.

        Parameters
        ----------
        positions : Tensor (..., T, 2)
            Centre positions (cx, cy).
        velocities : Tensor (..., T, 2), optional
            Pre-computed velocities. If None, derived from positions.

        Returns
        -------
        Tensor (..., 6)
        """
        if velocities is None:
            velocities = positions[..., 1:, :] - positions[..., :-1, :]
            # Pad first frame with zeros
            pad = torch.zeros_like(velocities[..., :1, :])
            velocities = torch.cat([pad, velocities], dim=-2)

        vx = velocities[..., 0]
        vy = velocities[..., 1]
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        angle = torch.atan2(vy, vx)

        # Acceleration
        acc = velocities[..., 1:, :] - velocities[..., :-1, :]
        pad = torch.zeros_like(acc[..., :1, :])
        acc = torch.cat([pad, acc], dim=-2)
        acc_x = acc[..., 0]
        acc_y = acc[..., 1]

        return torch.stack([vx, vy, speed, angle, acc_x, acc_y], dim=-1)

    @staticmethod
    def from_positions(
        positions: np.ndarray,          # (T, 2)
        fps: float = 30.0,
    ) -> np.ndarray:
        """Numpy version."""
        if len(positions) < 2:
            return np.zeros(6, dtype=np.float32)

        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([np.zeros((1, 2)), velocities])

        vx = velocities[:, 0]
        vy = velocities[:, 1]
        speed = np.sqrt(vx ** 2 + vy ** 2)
        angle = np.arctan2(vy, vx)

        acc = np.diff(velocities, axis=0)
        acc = np.vstack([np.zeros((1, 2)), acc])
        acc_x = acc[:, 0]
        acc_y = acc[:, 1]

        # Return the last frame's features
        return np.array([vx[-1], vy[-1], speed[-1], angle[-1], acc_x[-1], acc_y[-1]],
                        dtype=np.float32)


# ======================================================================
# Node Feature Encoder (aggregates per-node-type encodings)
# ======================================================================

class NodeFeatureEncoder(nn.Module):
    """
    Unified node feature encoder for all traffic participant types.

    Feature dimensions (configurable):
        - pedestrian:   appearance(256) + spatial(8) + motion(6) = 270
        - vehicle:      spatial(8) + motion(6) = 14
        - traffic_light: color(3) + remaining_time(1) = 4
        - traffic_sign:  position(4) + type_embed(4) = 8
        - lane:          distance(1) + direction(2) + type(1) = 4

    All feature vectors are projected to a common dimension `output_dim`.

    Parameters
    ----------
    output_dim : int
        Common output dimension for all node types.
    feat_dims : dict
        Per-type feature dimensions. If None, uses defaults above.
    """

    def __init__(
        self,
        output_dim: int = 128,
        feat_dims: Optional[Dict[str, int]] = None,
        img_width: float = 1920.0,
        img_height: float = 1080.0,
        fps: float = 30.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.feat_dims = feat_dims or {
            "pedestrian":     14,   # spatial(8) + motion(6), appearance(256) removed (needs video frames)
            "bicycle":        14,
            "motorcycle":     14,
            "car":            14,
            "bus":            14,
            "truck":          14,
            "traffic_light":  4,
            "traffic_sign":   8,
            "lane_line":      4,
        }

        # Projectors: per-type → common dim
        self.projectors = nn.ModuleDict()
        for cls_name, dim in self.feat_dims.items():
            self.projectors[cls_name] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim),
            )

        # Sub-encoders
        self.spatial_encoder = SpatialEncoder(img_width, img_height)
        self.motion_encoder = MotionEncoder(fps)

        # Learnable type embedding
        self.type_embedding = nn.Embedding(
            len(self.feat_dims), output_dim,
        )
        self.type_to_idx = {name: i for i, name in enumerate(self.feat_dims.keys())}

        # Traffic light colour embedding
        self.tl_color_embed = nn.Linear(3, 3, bias=False)  # simple linear

    # ------------------------------------------------------------------
    # Public encode method — dispatches by node type
    # ------------------------------------------------------------------

    def forward(
        self,
        bboxes: torch.Tensor,               # (N, 4)  [x1, y1, x2, y2]
        class_names: List[str],              # [N]
        positions: Optional[torch.Tensor] = None,   # (N, 2) centre positions
        velocities: Optional[torch.Tensor] = None,  # (N, 2)
        appearance_features: Optional[torch.Tensor] = None,  # (N, 256)
        traffic_light_states: Optional[torch.Tensor] = None, # (N, 4) [r,g,y,t]
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Encode a batch of heterogeneous nodes into a common feature space.

        Returns
        -------
        Tensor (N, output_dim)
        """
        if bboxes.shape[0] == 0:
            return torch.empty(0, self.output_dim, device=device)

        encoded_list: List[torch.Tensor] = []

        for i, cls_name in enumerate(class_names):
            raw_feat = self._build_raw_features(
                cls_name=cls_name,
                bbox=bboxes[i],
                position=positions[i] if positions is not None else None,
                velocity=velocities[i] if velocities is not None else None,
                appearance=appearance_features[i] if appearance_features is not None else None,
                tl_state=traffic_light_states[i] if traffic_light_states is not None else None,
            )
            # Project to common dim
            proj = self.projectors[cls_name](raw_feat.unsqueeze(0))
            # Add type embedding
            type_idx = self.type_to_idx.get(cls_name, 0)
            type_emb = self.type_embedding(torch.tensor(type_idx, device=device))
            encoded = proj + type_emb
            encoded_list.append(encoded)

        return torch.cat(encoded_list, dim=0)  # (N, output_dim)

    # ------------------------------------------------------------------
    # Raw feature construction per type
    # ------------------------------------------------------------------

    def _build_raw_features(
        self,
        cls_name: str,
        bbox: torch.Tensor,                    # (4,)
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        appearance: Optional[torch.Tensor] = None,
        tl_state: Optional[torch.Tensor] = None,  # traffic light [r,g,y,remaining]
        lane_info: Optional[torch.Tensor] = None,  # lane [distance, dir_x, dir_y, type]
    ) -> torch.Tensor:
        """Build the raw feature vector for a single node based on its type."""

        # --- Spatial encoding (common to almost all types) ---
        spatial = self.spatial_encoder(bbox.unsqueeze(0)).squeeze(0)  # (8,)

        if cls_name == "pedestrian":
            # spatial(8) + motion(6) = 14 (appearance removed — needs video ROI pooling)
            if velocity is not None:
                vx, vy = velocity[0], velocity[1]
                speed = torch.sqrt(vx ** 2 + vy ** 2)
                angle = torch.atan2(vy, vx)
                motion = torch.tensor([vx, vy, speed, angle, 0.0, 0.0], device=bbox.device)
            else:
                motion = torch.zeros(6, device=bbox.device)
            return torch.cat([spatial, motion], dim=0)

        elif cls_name in ("bicycle", "motorcycle", "car", "bus", "truck"):
            # spatial(8) + motion(6) = 14
            if velocity is not None:
                vx, vy = velocity[0], velocity[1]
                speed = torch.sqrt(vx ** 2 + vy ** 2)
                angle = torch.atan2(vy, vx)
                motion = torch.tensor([vx, vy, speed, angle, 0.0, 0.0],
                                      device=bbox.device)
            else:
                motion = torch.zeros(6, device=bbox.device)
            return torch.cat([spatial, motion], dim=0)

        elif cls_name == "traffic_light":
            # color(3) + remaining_time(1) = 4
            if tl_state is not None:
                color = tl_state[:3]     # (r, g, y) one-hot-ish
                remaining = tl_state[3:4]
            else:
                color = torch.zeros(3, device=bbox.device)
                remaining = torch.zeros(1, device=bbox.device)
            return torch.cat([color, remaining], dim=0)

        elif cls_name == "traffic_sign":
            # position(4) + type_embed(4) = 8
            cx = (bbox[0] + bbox[2]) / 2 / self.spatial_encoder.img_w
            cy = (bbox[1] + bbox[3]) / 2 / self.spatial_encoder.img_h
            w = (bbox[2] - bbox[0]) / self.spatial_encoder.img_w
            h = (bbox[3] - bbox[1]) / self.spatial_encoder.img_h
            pos = torch.tensor([cx, cy, w, h], device=bbox.device)
            # Placeholder type embedding (would come from sign classifier)
            type_emb = torch.zeros(4, device=bbox.device)
            return torch.cat([pos, type_emb], dim=0)

        elif cls_name == "lane_line":
            # distance(1) + direction(2) + type(1) = 4
            if lane_info is not None:
                return lane_info
            else:
                return torch.zeros(4, device=bbox.device)

        else:
            # Unknown type — just spatial
            return torch.cat([spatial, torch.zeros(6, device=bbox.device)], dim=0)


# ======================================================================
# Convenience: encode single node from trajectory data
# ======================================================================

def encode_node_from_trajectory(
    track_id: int,
    class_name: str,
    trajectory_data: dict,        # from TrajectoryManager.export_numpy()
    frame_idx: int = -1,           # which frame to encode (-1 = latest)
    appearance_feature: Optional[np.ndarray] = None,
    traffic_light_state: Optional[np.ndarray] = None,
    lane_info: Optional[np.ndarray] = None,
    img_width: float = 1920.0,
    img_height: float = 1080.0,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Encode a single node from raw trajectory data (numpy).

    Returns feature vector as numpy array.
    """
    data = trajectory_data
    fi = frame_idx

    bbox = data["bboxes"][fi]                                                  # (4,)
    spatial = SpatialEncoder.from_bbox_list([bbox], img_width, img_height)[0]  # (8,)

    positions = data["positions"]
    motion = MotionEncoder.from_positions(positions[:fi + 1], fps) if len(positions) > 0 \
        else np.zeros(6, dtype=np.float32)

    if class_name == "pedestrian":
        app = appearance_feature if appearance_feature is not None else np.zeros(256, dtype=np.float32)
        return np.concatenate([app, spatial, motion])

    elif class_name in ("bicycle", "motorcycle", "car", "bus", "truck"):
        return np.concatenate([spatial, motion])

    elif class_name == "traffic_light":
        color = traffic_light_state[:3] if traffic_light_state is not None else np.zeros(3, dtype=np.float32)
        remaining = traffic_light_state[3:4] if traffic_light_state is not None else np.zeros(1, dtype=np.float32)
        return np.concatenate([color, remaining])

    elif class_name == "traffic_sign":
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        pos = np.array([cx, cy, w, h], dtype=np.float32)
        type_emb = np.zeros(4, dtype=np.float32)
        return np.concatenate([pos, type_emb])

    elif class_name == "lane_line":
        return lane_info if lane_info is not None else np.zeros(4, dtype=np.float32)

    else:
        return np.concatenate([spatial, np.zeros(6, dtype=np.float32)])
