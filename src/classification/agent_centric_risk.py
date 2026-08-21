"""
Agent-Centric Red-Light Violation Risk Classifier.

Treats the pedestrian as a decision-making agent, using:
  1. Motion state    — velocity/acceleration/speed-trend from obs trajectory
  2. Environment     — nearby vehicle distances from scene data
  3. Predicted trajectory — stop-line crossing geometry from FlowChain samples

Key insight: deceleration in the observation window = likely stopping,
              constant speed or acceleration = likely violation.
              This is the single most important signal for reducing false negatives.

Design: pure post-hoc classifier — NO dependency on internal model memory vectors,
        works with any trajectory predictor that outputs samples + log_probs.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Geometric helpers (reused from risk_estimator.py)
# ======================================================================

def _signed_dist_to_line(x: float, y: float, sl) -> float:
    """Signed distance from (x,y) to stop_line [x1,y1,x2,y2]. Positive = junction side."""
    A = float(sl[1]) - float(sl[3])
    B = float(sl[2]) - float(sl[0])
    C = float(sl[0]) * float(sl[3]) - float(sl[2]) * float(sl[1])
    return (A * x + B * y + C) / math.sqrt(A**2 + B**2 + 1e-8)


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting point-in-polygon test."""
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


# ======================================================================
# Feature Extractors
# ======================================================================

def extract_motion_features(
    obs_trajectory: Tensor,   # (obs_len, 2)  normalized coords
    norm: Tensor,              # (2,) — [W, H] for un-normalising
) -> np.ndarray:
    """
    Extract 13-dim motion statistics from observed trajectory.

    Returns: np.ndarray of shape (13,)  — all normalised to roughly [0, 1] or [-1, 1]
      [0:7]   speed_seq       — per-frame speed / 200 px/frame
      [7]     speed_trend     — linear slope of last 3 speeds (>0 = accelerating)
      [8]     acc_mean        — mean acceleration magnitude / 200
      [9]     acc_std         — std of acceleration magnitude / 200
      [10]    speed_last      — last-frame speed / 200
      [11]    heading_change  — cosine similarity [-1, 1]
      [12]    is_decelerating — 1 if last 3 frames show clear deceleration
    """
    obs = obs_trajectory.detach().cpu().numpy() * norm.cpu().numpy()  # → pixel coords

    if obs.shape[0] < 3:
        return np.zeros(13, dtype=np.float32)

    # Velocity (7 frames)
    vel = np.diff(obs, axis=0)                           # (obs_len-1, 2)
    speed = np.sqrt((vel ** 2).sum(axis=-1))             # (obs_len-1,)

    # Acceleration (6 frames)
    acc = np.diff(vel, axis=0)                           # (obs_len-2, 2)
    acc_mag = np.sqrt((acc ** 2).sum(axis=-1)) if len(acc) > 0 else np.array([0.0])

    # --- Feature computation ---
    # speed_seq: pad to 7 or truncate; normalise by typical 200 px/frame
    TYPICAL_SPEED = 200.0
    speed_seq = np.zeros(7, dtype=np.float32)
    n_speed = min(len(speed), 7)
    speed_seq[:n_speed] = speed[-n_speed:] / TYPICAL_SPEED if n_speed <= len(speed) else speed / TYPICAL_SPEED

    # speed_trend: linear regression slope of last 3 speeds (normalised)
    if len(speed) >= 3:
        last3 = speed[-3:] / TYPICAL_SPEED
        x = np.arange(3)
        slope = np.polyfit(x, last3, 1)[0]  # change in normalised speed per frame
    elif len(speed) >= 2:
        slope = (speed[-1] - speed[-2]) / TYPICAL_SPEED
    else:
        slope = 0.0

    # acc stats (normalised)
    acc_mean = float(np.mean(acc_mag)) / TYPICAL_SPEED if len(acc_mag) > 0 else 0.0
    acc_std = float(np.std(acc_mag)) / TYPICAL_SPEED if len(acc_mag) > 0 else 0.0

    # speed_last (normalised)
    speed_last = float(speed[-1]) / TYPICAL_SPEED if len(speed) > 0 else 0.0

    # heading_change: cos similarity between first and last velocity
    if len(vel) >= 2:
        v0 = vel[0]
        v1 = vel[-1]
        dot = v0[0] * v1[0] + v0[1] * v1[1]
        norm_prod = math.sqrt((v0**2).sum() * (v1**2).sum()) + 1e-8
        heading_change = float(dot / norm_prod)  # 1 = same direction, -1 = reversed
    else:
        heading_change = 1.0

    # is_decelerating: 1 if speed consistently decreases in last 3 frames
    if len(speed) >= 3:
        last3 = speed[-3:]
        decelerating = 1 if (last3[0] > last3[1] > last3[2]) else 0
    else:
        decelerating = 0

    feats = np.array([
        *speed_seq.tolist(),
        slope,
        acc_mean,
        acc_std,
        speed_last,
        heading_change,
        decelerating,
    ], dtype=np.float32)

    return feats


def extract_environment_features(
    scene_data: dict,
    norm: Tensor,
    target_idx: int = 0,
    crosswalk_roi: Optional[list] = None,
    junction_roi: Optional[list] = None,
) -> np.ndarray:
    """
    Extract 8-dim environment features from scene graph.

    ONLY considers vehicles within the crosswalk/junction zone
    (not all vehicles in the entire frame).

    Returns: np.ndarray of shape (8,)
      [0] d_min_last       — distance from target to nearest IN-ZONE vehicle, last frame
      [1] d_min_mean       — mean of per-frame min in-zone vehicle distances
      [2] d_min_trend      — slope of d_min over 8 frames (>0 = vehicle getting farther)
      [3] n_zone_vehicles  — mean number of vehicles IN the crossing zone per frame
      [4] has_vehicle_in_zone — 1 if any frame has a vehicle in the crossing zone
      [5] ttc_approx       — approximate TTC to nearest in-zone vehicle, capped
      [6] d_min_min        — minimum distance to any in-zone vehicle across all frames
      [7] zone_occupancy   — fraction of frames with at least 1 vehicle in zone
    """
    positions = scene_data.get("positions")
    class_names = scene_data.get("class_names", [])

    if positions is None:
        return np.zeros(8, dtype=np.float32)

    # Handle various dimensionalities
    ndim = positions.dim()
    if ndim == 5:
        # (1, 1, T, N, 2) → (T, N, 2)
        positions = positions.squeeze(0).squeeze(0)
        ndim = 3
    if ndim == 4:
        # (1, T, N, 2) or (B, T, N, 2)
        positions = positions.squeeze(0)  # → (T, N, 2)
        ndim = 3
    if ndim == 3:
        T = positions.shape[0]
        N = positions.shape[1]
        pos_np = positions.detach().cpu().numpy()
    elif ndim == 2:
        # (N, 2) — single frame
        pos_np = positions.detach().cpu().numpy()[np.newaxis, :, :]
        T, N = 1, pos_np.shape[1]
    else:
        return np.zeros(8, dtype=np.float32)

    norm_np = norm.cpu().numpy() if isinstance(norm, Tensor) else np.array([3840., 2160.])

    # Resolve class_names to per-frame list of names
    cn_per_frame = []
    for t in range(T):
        names_t = []
        if isinstance(class_names, list) and len(class_names) > 0:
            if isinstance(class_names[0], list):
                # [T][N] or [B][T][N]
                row = class_names[0] if len(class_names) == 1 else class_names
                if len(row) <= T:
                    # [B=1][T] → row[t]
                    row_t = row[min(t, len(row) - 1)]
                else:
                    row_t = row[min(t, len(row) - 1)]
                if isinstance(row_t, list):
                    names_t = [str(n) for n in row_t]
            elif N <= len(class_names):
                names_t = [str(class_names[n]) for n in range(N)]
        cn_per_frame.append(names_t)

    # Vehicle classes
    vehicle_classes = {"car", "bus", "truck", "bicycle", "motorcycle"}

    # Build junction/crossing zone: which pixels are "near the crossing area"
    # Priority: junction_roi (covers the intersection where vehicles wait),
    #           fall back to crosswalk_roi, then use both if both available.
    # Vehicles don't drive ON the crosswalk — they stop BEFORE it at the stop line,
    # so junction_roi is the right zone for filtering relevant vehicles.
    zone_polygon = None
    ZONE_RADIUS = 300.0  # pixels — max distance from zone boundary to include vehicle
    if junction_roi and len(junction_roi) >= 3:
        if isinstance(junction_roi[0], (list, tuple)):
            zone_polygon = [(float(p[0]), float(p[1])) for p in junction_roi]
        else:
            zone_polygon = [(float(junction_roi[i]), float(junction_roi[i+1]))
                           for i in range(0, len(junction_roi)//2*2, 2)]
    elif crosswalk_roi and len(crosswalk_roi) >= 3:
        zone_polygon = [(float(p[0]), float(p[1])) for p in crosswalk_roi]

    def _dist_to_zone(vx: float, vy: float) -> float:
        """Minimum distance from vehicle center to the crossing zone."""
        if zone_polygon is None:
            return 0.0  # no zone defined → include all vehicles (fallback)
        # Use point-in-polygon check; if inside, distance = 0
        if _point_in_polygon(vx, vy, zone_polygon):
            return 0.0
        # Otherwise, minimum distance to polygon edges
        best = float("inf")
        n_pts = len(zone_polygon)
        for i in range(n_pts):
            px, py = zone_polygon[i]
            d = math.sqrt((vx - px) ** 2 + (vy - py) ** 2)
            if d < best:
                best = d
        return best

    d_min_seq = []
    vehicle_counts = []

    for t in range(T):
        tx = pos_np[t, target_idx, 0] * norm_np[0]
        ty = pos_np[t, target_idx, 1] * norm_np[1]

        d_min_t = float("inf")
        n_veh = 0

        for n in range(N):
            if n == target_idx:
                continue
            # Check if vehicle
            cn = cn_per_frame[t][n] if t < len(cn_per_frame) and n < len(cn_per_frame[t]) else ""
            if cn not in vehicle_classes:
                continue

            vx = pos_np[t, n, 0] * norm_np[0]
            vy = pos_np[t, n, 1] * norm_np[1]

            # --- Key filter: only consider vehicles in/near the crossing zone ---
            d_to_zone = _dist_to_zone(vx, vy)
            if d_to_zone > ZONE_RADIUS:
                continue  # skip vehicles far from the crossing area

            n_veh += 1
            d = math.sqrt((tx - vx) ** 2 + (ty - vy) ** 2)
            if d < d_min_t:
                d_min_t = d

        d_min_seq.append(d_min_t if d_min_t != float("inf") else 9999.0)
        vehicle_counts.append(n_veh)

    d_min_arr = np.array(d_min_seq, dtype=np.float32)
    valid_mask = d_min_arr < 9990.0

    # Compute features
    d_min_last = float(d_min_arr[-1]) if valid_mask[-1] else 9999.0
    d_min_mean = float(d_min_arr[valid_mask].mean()) if valid_mask.any() else 9999.0
    d_min_min = float(d_min_arr[valid_mask].min()) if valid_mask.any() else 9999.0

    # d_min_trend: slope of d_min vs frame index (positive = vehicle getting farther)
    if valid_mask.sum() >= 3:
        x = np.arange(T, dtype=np.float32)[valid_mask]
        y = d_min_arr[valid_mask]
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0

    zone_veh_count = float(np.mean(vehicle_counts)) if vehicle_counts else 0.0
    has_vehicle_in_zone = 1.0 if d_min_min < ZONE_RADIUS else 0.0

    # TTC approximation: d / relative_speed (relative speed estimated from d_min trend)
    if slope < -1.0 and valid_mask[-1]:
        ttc = d_min_last / abs(slope)
        ttc = min(ttc, 999.0)
    else:
        ttc = 999.0

    # Zone occupancy: fraction of frames where at least 1 vehicle is in the zone
    zone_occupancy = float(np.mean([1 if c > 0 else 0 for c in vehicle_counts])) if vehicle_counts else 0.0

    # Normalize distances to [0, ~1] range
    feats = np.array([
        min(d_min_last / 2000.0, 5.0),
        min(d_min_mean / 2000.0, 5.0),
        slope / 200.0,
        zone_veh_count,
        has_vehicle_in_zone,
        ttc / 999.0,
        min(d_min_min / 2000.0, 5.0),
        zone_occupancy,
    ], dtype=np.float32)

    return feats


def extract_trajectory_features(
    samples: Tensor,            # (N, 12, 2) normalised coords
    log_probs: Optional[Tensor],# (N,) log probabilities
    norm: Tensor,               # (2,) [W, H]
    stop_line: Optional[list],  # [x1, y1, x2, y2]
) -> np.ndarray:
    """
    Extract 8-dim trajectory geometry features from FlowChain samples.

    Returns: np.ndarray of shape (8,)
      [0] endpoint_dist    — distance from mean endpoint to stop_line (pixels)
      [1] min_dist         — minimum distance from any point on mean traj to stop_line
      [2] crossing_prob    — fraction of MC samples that cross the stop_line
      [3] crossing_frame   — first frame where mean traj crosses (-1 = never)
      [4] traj_length      — total length of mean predicted trajectory
      [5] traj_direction   — dot product of mean traj direction with crossing direction
      [6] prob_weighted_dist — log_prob-weighted mean endpoint distance
      [7] endpoint_std     — std of endpoint distances across samples
    """
    N = samples.shape[0]

    # Convert to pixel space
    samples_np = samples.detach().cpu().numpy() * norm.cpu().numpy()  # (N, 12, 2)
    mean_traj = samples_np.mean(axis=0)                                # (12, 2)

    # --- Mean trajectory features ---
    endpoint = mean_traj[-1]   # (2,)
    start = mean_traj[0]       # (2,)

    if stop_line is not None:
        # Endpoint distance to stop_line
        d_end = _signed_dist_to_line(endpoint[0], endpoint[1], stop_line)
        endpoint_dist = abs(d_end)

        # Min distance along trajectory
        min_dist = float("inf")
        crossed = False
        first_cross_frame = -1
        for i in range(mean_traj.shape[0]):
            d = _signed_dist_to_line(mean_traj[i, 0], mean_traj[i, 1], stop_line)
            min_dist = min(min_dist, abs(d))
            if d > 0 and not crossed:
                crossed = True
                first_cross_frame = i

        # Crossing probability — fraction of samples that cross
        n_cross = 0
        for n in range(N):
            for i in range(samples_np.shape[1]):
                d = _signed_dist_to_line(samples_np[n, i, 0], samples_np[n, i, 1], stop_line)
                if d > 0:
                    n_cross += 1
                    break
        crossing_prob = n_cross / N
    else:
        endpoint_dist = 0.0
        min_dist = 0.0
        crossing_prob = 0.0
        first_cross_frame = -1

    # Trajectory length
    traj_length = float(np.sqrt(((np.diff(mean_traj, axis=0)) ** 2).sum(axis=-1)).sum())

    # Trajectory direction vs crossing direction
    # Crossing direction is perpendicular to stop_line pointing toward junction
    traj_dir = endpoint - start
    traj_dir_norm = math.sqrt((traj_dir ** 2).sum()) + 1e-8
    traj_dir_unit = traj_dir / traj_dir_norm

    if stop_line is not None:
        # Perpendicular to stop line: (A, B) from Ax+By+C line equation
        A = float(stop_line[1]) - float(stop_line[3])
        B = float(stop_line[2]) - float(stop_line[0])
        cross_dir = np.array([A, B])
        cross_dir_norm = math.sqrt((cross_dir ** 2).sum()) + 1e-8
        cross_dir_unit = cross_dir / cross_dir_norm
        traj_direction = float(np.dot(traj_dir_unit, cross_dir_unit))  # -1 to 1
    else:
        traj_direction = 0.0

    # Prob-weighted endpoint distance
    if log_probs is not None and N > 1:
        lp = log_probs.detach().cpu().numpy()
        lp = lp - lp.max()
        probs = np.exp(lp) / np.exp(lp).sum()
        prob_dist = 0.0
        for n in range(N):
            d = abs(_signed_dist_to_line(samples_np[n, -1, 0], samples_np[n, -1, 1],
                                         stop_line)) if stop_line else 0.0
            prob_dist += probs[n] * d
    else:
        prob_dist = endpoint_dist

    # Endpoint std
    if N > 1:
        endpoints = samples_np[:, -1, :]  # (N, 2)
        endpoint_std = float(np.std(np.sqrt((endpoints ** 2).sum(axis=-1))))
    else:
        endpoint_std = 0.0

    # Normalize
    feats = np.array([
        min(endpoint_dist / 2000.0, 5.0),
        min(min_dist / 2000.0, 5.0),
        crossing_prob,                      # already [0, 1]
        first_cross_frame / 12.0,           # scaled to [-1/12, 1]
        min(traj_length / 2000.0, 5.0),
        traj_direction,                     # already [-1, 1]
        min(prob_dist / 2000.0, 5.0),
        min(endpoint_std / 2000.0, 5.0),
    ], dtype=np.float32)

    return feats


# ======================================================================
# Agent-Centric Risk Classifier
# ======================================================================

class AgentCentricRiskClassifier(nn.Module):
    """
    Classify red-light violation risk using agent-centric physical features.

    Takes observable physical quantities (motion, environment, trajectory geometry)
    and fuses them through a small MLP to produce a violation risk probability.

    This is PURELY post-hoc — it does not depend on any internal model state
    and can be used with any trajectory predictor that outputs samples.

    Parameters
    ----------
    stop_line : [x1, y1, x2, y2] or None — stop-line geometry
    motion_dim : int — dimension of motion features (default 13)
    env_dim : int — dimension of environment features (default 8)
    traj_dim : int — dimension of trajectory features (default 8)
    hidden_dims : list — hidden layer dimensions for fusion MLP
    dropout : float — dropout rate
    """

    def __init__(
        self,
        stop_line: Optional[list] = None,
        crosswalk_roi: Optional[list] = None,
        junction_roi: Optional[list] = None,
        motion_dim: int = 13,
        env_dim: int = 8,
        traj_dim: int = 8,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stop_line = stop_line
        self.crosswalk_roi = crosswalk_roi
        self.junction_roi = junction_roi
        self.motion_dim = motion_dim
        self.env_dim = env_dim
        self.traj_dim = traj_dim

        # Per-feature-group projections (for ablation / interpretability)
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, 16),
            nn.ReLU(inplace=True),
        )
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, 8),
            nn.ReLU(inplace=True),
        )
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, 8),
            nn.ReLU(inplace=True),
        )
        # Rebuild fusion MLP with projected dims
        proj_dim = 16 + 8 + 8  # 32
        self.fusion_proj = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Threshold for binary classification
        self.register_buffer("threshold", torch.tensor(0.5))

    def forward(
        self,
        obs_trajectory: Tensor,          # (obs_len, 2) normalised coords
        scene_data: Optional[dict],       # scene graph data
        samples: Tensor,                  # (N, pred_len, 2) normalised coords
        log_probs: Optional[Tensor],      # (N,) log probabilities
        norm: Tensor,                     # (2,) [W, H] for un-normalising
    ) -> Dict[str, Tensor]:
        """
        Compute violation risk from agent-centric features.

        Returns dict with:
          - violation_risk: scalar tensor in [0, 1]
          - motion_feat: raw motion features (13,)
          - env_feat: raw environment features (8,)
          - traj_feat: raw trajectory features (8,)
          - motion_risk: motion-only risk contribution (interpretability)
        """
        device = obs_trajectory.device if isinstance(obs_trajectory, Tensor) else samples.device

        # 1. Extract features
        motion_feat = extract_motion_features(obs_trajectory, norm)
        env_feat = extract_environment_features(
            scene_data or {}, norm, crosswalk_roi=self.crosswalk_roi, junction_roi=self.junction_roi)
        traj_feat = extract_trajectory_features(samples, log_probs, norm, self.stop_line)

        # Convert to tensors
        m_t = torch.from_numpy(motion_feat).float().to(device)
        e_t = torch.from_numpy(env_feat).float().to(device)
        t_t = torch.from_numpy(traj_feat).float().to(device)

        # 2. Per-group projections
        m_proj = self.motion_proj(m_t)     # (16,)
        e_proj = self.env_proj(e_t)        # (8,)
        t_proj = self.traj_proj(t_t)       # (8,)

        # 3. Fuse
        fused = torch.cat([m_proj, e_proj, t_proj], dim=-1)  # (32,)
        violation_risk = self.fusion_proj(fused).squeeze(-1)   # scalar

        # 4. Interpretability: motion-only risk (for ablation / analysis)
        with torch.no_grad():
            motion_only = torch.cat([
                m_proj,
                torch.zeros_like(e_proj),
                torch.zeros_like(t_proj),
            ], dim=-1)
            motion_risk = self.fusion_proj(motion_only).squeeze(-1)

        return {
            "violation_risk": violation_risk,
            "motion_feat": m_t,
            "env_feat": e_t,
            "traj_feat": t_t,
            "motion_risk": motion_risk,
        }

    def set_threshold(self, th: float):
        """Set classification threshold."""
        self.threshold.fill_(th)

    def classify(self, obs_trajectory, scene_data, samples, log_probs, norm):
        """Binary classification."""
        result = self.forward(obs_trajectory, scene_data, samples, log_probs, norm)
        pred = (result["violation_risk"] >= self.threshold).long()
        return pred, result["violation_risk"], result

    # ------------------------------------------------------------------
    # Training support (for later joint training)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        risk_prob: Tensor,
        labels: Tensor,
        pos_weight: float = 20.0,
    ) -> Tensor:
        """Weighted BCE loss for training the fusion MLP."""
        weight = torch.where(labels > 0.5, pos_weight, 1.0)
        return F.binary_cross_entropy(risk_prob, labels.float(), weight=weight)


# ======================================================================
# Threshold search (using the classifier)
# ======================================================================

def search_threshold_agent_centric(
    classifier: AgentCentricRiskClassifier,
    model,
    val_loader,
    device: str,
    norm: Tensor,
    num_samples: int = 100,
) -> Tuple[float, float]:
    """
    Search best threshold for AgentCentricRiskClassifier on validation set.
    Returns (best_threshold, best_f1).
    """
    classifier.eval()
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs_trajectory"].to(device) / norm.to(device)
            target = batch["target_trajectory"].to(device) / norm.to(device)
            scene_list = batch.get("scene_list", [None] * obs.shape[0])
            labels = batch.get("is_violation",
                               torch.zeros(obs.shape[0], device=device))

            for b in range(obs.shape[0]):
                model.reset_state()
                sc = scene_list[b] if b < len(scene_list) else None
                scene_data = None
                if sc is not None:
                    scene_data = {
                        "bboxes": sc["bboxes"].unsqueeze(0).to(device),
                        "positions": sc["positions"].unsqueeze(0).to(device),
                        "class_names": sc["class_names"],
                        "target_idx": 0,
                    }

                pred = model(
                    obs_trajectory=obs[b:b+1],
                    scene_data=scene_data,
                    num_samples=num_samples,
                )
                samples = pred.get("samples")
                if samples is None:
                    continue

                log_probs = pred.get("log_probs")
                result = classifier(
                    obs_trajectory=obs[b],
                    scene_data=scene_data,
                    samples=samples[:, 0] if samples.dim() == 4 else samples,
                    log_probs=log_probs[:, 0] if log_probs is not None and log_probs.dim() >= 2 else log_probs,
                    norm=norm,
                )
                all_probs.append(float(result["violation_risk"].item()))
                lbl = float(labels[b].item()) if labels.numel() > b else float(labels.item())
                all_labels.append(lbl)

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    pos_mask = labels == 1

    if pos_mask.sum() == 0:
        logger.warning("No positive samples in validation set")
        return 0.5, 0.0

    best_thresh, best_f1 = 0.5, 0.0
    for th in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= th).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((labels == 1) & (preds == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    return best_thresh, best_f1
