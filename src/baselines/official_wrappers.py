"""
Adapters wrapping official baseline implementations into our experiment interface.

Each wrapper:
    Imports the official model class from the cloned repo.
    Converts our data format to theirs.
    Extracts the mean trajectory from their output (5-dim Gaussian → 2-dim mean).

Our interface:
    Input:  obs_trajectory (B, obs_len, 2)  [normalized coords]
    Output: dict with "mean" (B, pred_len, 2)
"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_from_file(module_name: str, file_path: Path):
    """Import a module from a specific file path (avoids sys.path conflicts)."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Paths to official repos (cloned on cloud)
# ======================================================================

_HERE = Path(__file__).resolve().parent
_SOCIAL_LSTM_PATH = _HERE / "social_lstm_official"
_SOCIAL_STGCNN_PATH = _HERE / "social_stgcnn_official"


# ======================================================================
# 1. Social-LSTM Wrapper — uses official SocialModel
# ======================================================================

class SocialLSTMOfficial(nn.Module):
    """
    Wraps the official SocialModel from social_lstm_official/model.py.

    The official model uses grid-based social pooling and outputs
    bivariate Gaussian parameters (5 dims). We extract the mean (μx, μy).

    Single-pedestrian mode: V=1, identity grid, no social pooling.
    """

    def __init__(self, obs_len=8, pred_len=12, rnn_size=128, grid_size=4,
                 embedding_size=64, dropout=0.1, use_cuda=True):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.rnn_size = rnn_size
        self.grid_size = grid_size
        self.use_cuda = use_cuda
        self.seq_length = obs_len  # official model uses seq_length internally

        # Build args namespace (official model expects argparse-style)
        args = Namespace(
            rnn_size=rnn_size,
            grid_size=grid_size,
            embedding_size=embedding_size,
            input_size=2,           # (x, y)
            output_size=5,          # bivariate Gaussian: μx, μy, σx, σy, ρ
            maxNumPeds=1,           # single pedestrian
            seq_length=obs_len + pred_len,
            gru=False,
            use_cuda=use_cuda,
            dropout=dropout,
        )

        social_lstm_mod = _import_from_file(
            "social_lstm_model",
            _SOCIAL_LSTM_PATH / "model.py",
        )
        self.model = social_lstm_mod.SocialModel(args, infer=False)
        self.infer_model = social_lstm_mod.SocialModel(args, infer=True)

        # Zero-init output layers: model starts from "standing still" prior.
        # Without this, random init produces delta ~500px/step, and the
        # autoregressive MSE loss struggles to converge in 10 epochs.
        nn.init.zeros_(self.model.output_layer.weight)
        nn.init.zeros_(self.model.output_layer.bias)
        nn.init.zeros_(self.infer_model.output_layer.weight)
        nn.init.zeros_(self.infer_model.output_layer.bias)

        # After training, switch to inference mode
        self._trained = False

    def forward(self, obs_trajectory, **kwargs):
        """
        obs_trajectory: (B, obs_len, 2)  normalized

        Note: The official SocialModel requires multi-pedestrian scenes for
        grid-based social pooling. For single-pedestrian (V=1), we use a
        simplified prediction path that bypasses the social tensor.
        """
        B, T, _ = obs_trajectory.shape
        device = obs_trajectory.device

        # For single-pedestrian: simple LSTM encode → decode (no social pooling)
        all_preds = []
        for b in range(B):
            traj = obs_trajectory[b]  # (obs_len, 2)
            trajectory = self._predict_simple(traj, device)  # (pred_len, 2) absolute coords
            all_preds.append(trajectory)

        return {"mean": torch.stack(all_preds, dim=0)}

    def _predict_simple(self, traj, device):
        """Simplified single-pedestrian prediction bypassing social pooling."""
        T = self.obs_len
        emb = self.model.input_embedding_layer(traj)           # (T, emb_dim)
        emb = self.model.dropout(self.model.relu(emb))

        # Encode: run LSTM cell over observation
        h = torch.zeros(1, self.rnn_size, device=device)
        c = torch.zeros(1, self.rnn_size, device=device)
        for t in range(T):
            inp = torch.cat([emb[t:t+1], torch.zeros(1, emb.shape[1], device=device)], dim=-1)
            h, c = self.model.cell(inp, (h, c))

        # Decode autoregressively — accumulate positions to match teacher-forcing
        current_pos = traj[-1:].clone()  # (1, 2) last observed position
        outputs = []
        for _ in range(self.pred_len):
            out = self.model.output_layer(h)  # (1, 5) Gaussian params
            delta = out[:, :2]                # (1, 2) displacement delta
            current_pos = current_pos + delta  # accumulate absolute position
            outputs.append(current_pos)
            # Feed accumulated absolute position (not fixed last_pos)
            next_emb = self.model.input_embedding_layer(current_pos)
            next_emb = self.model.dropout(self.model.relu(next_emb))
            inp = torch.cat([next_emb, torch.zeros(1, emb.shape[1], device=device)], dim=-1)
            h, c = self.model.cell(inp, (h, c))

        pred = torch.cat(outputs, dim=0)  # (pred_len, 2) — already absolute positions
        return pred


# ======================================================================
# 2. Social-STGCNN Wrapper — uses official social_stgcnn
# ======================================================================

class SocialSTGCNNOfficial(nn.Module):
    """
    Wraps the official social_stgcnn from social_stgcnn_official/model.py.

    The official model uses ST-GCNN layers (spatial-temporal graph convolutions)
    and outputs bivariate Gaussian parameters.

    Input:  (N, C, T, V)  — batch, channels, time, vertices
    Output: (N, C, T_pred, V)  — 5-channel Gaussian params

    Single-pedestrian: V=1, identity adjacency A.
    """

    def __init__(self, obs_len=8, pred_len=12, n_stgcnn=1, n_txpcnn=1,
                 kernel_size=3):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        stgcnn_mod = _import_from_file(
            "social_stgcnn_model",
            _SOCIAL_STGCNN_PATH / "model.py",
        )
        self.model = stgcnn_mod.social_stgcnn(
            n_stgcnn=n_stgcnn,
            n_txpcnn=n_txpcnn,
            input_feat=2,          # (x, y) positions
            output_feat=5,         # bivariate Gaussian
            seq_len=obs_len,
            pred_seq_len=pred_len,
            kernel_size=kernel_size,
        )

    def forward(self, obs_trajectory, **kwargs):
        """Standard trajectory prediction interface."""
        result = self.forward_gaussian(obs_trajectory, **kwargs)
        return {"mean": result["mean"]}

    def forward_gaussian(self, obs_trajectory, **kwargs):
        """
        Full Gaussian parameter output.

        Returns
        -------
        dict with:
            "mean":  (B, pred_len, 2)   — μx, μy  (absolute coords)
            "mu":    (B, pred_len, 2)   — μx, μy  (displacement deltas)
            "sigma": (B, pred_len, 2)   — σx, σy
            "rho":   (B, pred_len)      — correlation ρ
        """
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        # Reshape: (B, obs_len, 2) → (B, 2, obs_len, 1)
        v = obs_trajectory.permute(0, 2, 1).unsqueeze(-1)  # (B, 2, T, 1)

        # Identity adjacency for single pedestrian
        K = self.model.st_gcns[0].gcn.kernel_size
        A = torch.eye(1, device=device).unsqueeze(0).repeat(K, 1, 1)  # (K, 1, 1)

        # Run official model
        v_out, _ = self.model(v, A)
        # v_out: (B, 5, pred_len, 1)

        # Unpack Gaussian: channel 0=μx, 1=μy, 2=σx, 3=σy, 4=ρ
        mu_x  = v_out[:, 0, :, 0]   # (B, pred_len)
        mu_y  = v_out[:, 1, :, 0]   # (B, pred_len)
        sig_x = torch.abs(v_out[:, 2, :, 0]) + 1e-4   # ensure positive
        sig_y = torch.abs(v_out[:, 3, :, 0]) + 1e-4
        rho   = torch.tanh(v_out[:, 4, :, 0])           # bound to [-1, 1]

        mu = torch.stack([mu_x, mu_y], dim=-1)           # (B, pred_len, 2)
        sigma = torch.stack([sig_x, sig_y], dim=-1)       # (B, pred_len, 2)

        last_obs = obs_trajectory[:, -1:]                 # (B, 1, 2)
        mean_abs = last_obs + mu                          # absolute coords

        return {"mean": mean_abs, "mu": mu, "sigma": sigma, "rho": rho}

    def sample(self, obs_trajectory, num_samples=100, **kwargs):
        """
        Monte Carlo sampling from predicted bivariate Gaussian distribution.

        Returns
        -------
        samples : (num_samples, B, pred_len, 2)   MC trajectory samples
        """
        g = self.forward_gaussian(obs_trajectory, **kwargs)
        mu, sigma, rho = g["mu"], g["sigma"], g["rho"]
        B, T, _ = mu.shape
        device = mu.device

        samples_all = []
        for _ in range(num_samples):
            eps = torch.randn(B, T, 2, device=device)
            # Reparameterise:  s_x = μ_x + σ_x * ε_x
            #                  s_y = μ_y + σ_y * (ρ*ε_x + √(1-ρ²)*ε_y)
            sx = mu[..., 0] + sigma[..., 0] * eps[..., 0]
            sy = mu[..., 1] + sigma[..., 1] * (rho * eps[..., 0] +
                                               torch.sqrt(1 - rho**2 + 1e-8) * eps[..., 1])
            sample_xy = torch.stack([sx, sy], dim=-1)  # (B, T, 2)

            last_obs = obs_trajectory[:, -1:]
            sample_abs = last_obs + sample_xy           # absolute coords
            samples_all.append(sample_abs)

        return torch.stack(samples_all, dim=0)  # (N, B, T, 2)

    def log_prob(self, obs_trajectory, target, **kwargs):
        """
        Negative log-likelihood of target under predicted bivariate Gaussian.

        Parameters
        ----------
        obs_trajectory : (B, obs_len, 2)
        target : (B, pred_len, 2) — absolute coords or displacement deltas

        Returns
        -------
        log_prob : (B,) — log P(target | μ, Σ) per batch element
        """
        g = self.forward_gaussian(obs_trajectory, **kwargs)
        mu, sigma, rho = g["mu"], g["sigma"], g["rho"]  # all in delta space
        # mu: (B, T, 2), sigma: (B, T, 2), rho: (B, T)

        # Convert target to delta space
        last_obs = obs_trajectory[:, -1:]  # (B, 1, 2)
        target_delta = target - last_obs   # (B, T, 2)

        sx = sigma[..., 0] + 1e-6  # (B, T)
        sy = sigma[..., 1] + 1e-6
        r = rho

        dx = target_delta[..., 0] - mu[..., 0]  # (B, T)
        dy = target_delta[..., 1] - mu[..., 1]

        # Bivariate Gaussian NLL per timestep:
        # NLL = log(2π) + log(σx·σy) + 0.5·log(1-ρ²)
        #     + 1/(2(1-ρ²)) · [dx²/σx² - 2ρ·dx·dy/(σx·σy) + dy²/σy²]
        r2 = 1.0 - r ** 2 + 1e-8
        mahal = (dx ** 2) / (sx ** 2) - 2 * r * dx * dy / (sx * sy) + (dy ** 2) / (sy ** 2)
        nll_per_step = (math.log(2 * math.pi) + torch.log(sx) + torch.log(sy)
                        + 0.5 * torch.log(r2) + 0.5 * mahal / r2)

        return -nll_per_step.sum(dim=1)  # (B,)  sum over timesteps


# ======================================================================
# 3. STRR — Spatiotemporal Relationship Reasoning (STR-PIP)
# ======================================================================

class STRROfficial(nn.Module):
    """
    Faithful reimplementation of STRR (STR-PIP, Liu et al., IEEE RA-L 2020).

    Core architecture:
        1. Spatial encoder (replaces ResNet backbone)
        2. Per-frame graph: target + surrounding agents + infrastructure
        3. Inner-product edge weights: A_ij = σ(φ(h_i)^T · θ(h_j))
        4. 2-layer GCN message passing with learned adjacency
        5. Pedestrian GRU: temporal smoothing of target node
        6. Temporal GRU: sequence-level encoding
        7. Binary classifier: violation / no-violation

    Input:  obs_trajectory (B, obs_len, 2) + scene_data (optional)
    Output: violation logits (B,)
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        conv_dim: int = 512,
        hidden_dim: int = 256,
        node_feat_dim: int = 128,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.conv_dim = conv_dim
        self.node_feat_dim = node_feat_dim

        # ---- 1. Spatial encoder (8-dim → node_feat_dim) ----
        self.pos_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, node_feat_dim),
        )

        # ---- 2. Inner-product edge weight projections (STRR eq. 3-4) ----
        # A_ij = σ( φ(h_i)^T · θ(h_j) / √d )
        self.phi = nn.Linear(node_feat_dim, node_feat_dim, bias=False)
        self.theta = nn.Linear(node_feat_dim, node_feat_dim, bias=False)

        # ---- 3. GCN layers (2-layer with residual, operates on node_feat_dim) ----
        self.gcn_w1 = nn.Linear(node_feat_dim, node_feat_dim)
        self.gcn_w2 = nn.Linear(node_feat_dim, node_feat_dim)
        self.gcn_norm1 = nn.LayerNorm(node_feat_dim)
        self.gcn_norm2 = nn.LayerNorm(node_feat_dim)

        # ---- 4. Target projector (node_feat_dim → conv_dim for GRU input) ----
        self.target_proj = nn.Linear(node_feat_dim, conv_dim)

        # ---- 5. Pedestrian GRU (temporal smoothing, STRR eq. 5) ----
        self.ped_gru = nn.GRU(conv_dim, conv_dim, num_layers=2, batch_first=True)

        # ---- 6. Temporal GRU (sequence encoding) ----
        self.gru = nn.GRU(conv_dim, hidden_dim, num_layers=2, batch_first=True)

        # ---- 7. Classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        # Graph builder (shared with perception_model)
        from src.graph import PerceptionGraphBuilder
        self.graph_builder = PerceptionGraphBuilder(max_distance=30.0)

    # ------------------------------------------------------------------
    # Spatial encoding (STRR helper_get_pos_vec equivalent)
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_encode(traj: torch.Tensor) -> torch.Tensor:
        """
        8-dim spatial features: [x, y, vx, vy, |v|, θ, ax, ay]

        Parameters
        ----------
        traj : (B, T, 2) or (N, 2) — trajectory positions

        Returns
        -------
        (B, T, 8) or (N, 8)
        """
        if traj.dim() == 2:
            traj = traj.unsqueeze(0)  # (1, N, 2)
            squeeze_out = True
        else:
            squeeze_out = False

        B, T_coord, _ = traj.shape
        device = traj.device

        v = traj[:, 1:] - traj[:, :-1]
        v = torch.cat([torch.zeros(B, 1, 2, device=device), v], dim=1)
        speed = torch.norm(v, dim=-1, keepdim=True)
        angle = torch.atan2(v[..., 1:2], v[..., 0:1])

        a = v[:, 1:] - v[:, :-1]
        a = torch.cat([torch.zeros(B, 1, 2, device=device), a], dim=1)

        feat = torch.cat([traj, v, speed, angle], dim=-1)  # (B, T, 6)
        # Add acceleration (2 dims) to reach exactly 8 spatial features
        feat = torch.cat([feat, a[..., :2]], dim=-1)[:, :, :8]  # (B, T, 8)

        return feat.squeeze(0) if squeeze_out else feat

    # ------------------------------------------------------------------
    # Build graph adjacency matrix (inner-product, STRR eq. 3-4)
    # ------------------------------------------------------------------

    def _build_adj(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute inner-product adjacency matrix (STRR eq. 3-4).

        A_ij = σ( φ(h_i)^T · θ(h_j) / √d )

        NOTE: This is DIRECTIONAL (asymmetric). Since φ and θ are different
        projections, A_ij ≠ A_ji in general — matching STRR's unidirectional
        relationship modeling. The matrix form φ·θᵀ gives a full N×N directed
        adjacency, not a symmetric similarity matrix.

        Parameters
        ----------
        node_feats : (N, D) — all node features in this frame

        Returns
        -------
        A : (N, N) — row-normalized directional adjacency
        """
        N = node_feats.shape[0]
        device = node_feats.device

        phi_out = self.phi(node_feats)    # (N, D)
        theta_out = self.theta(node_feats) # (N, D)

        # Inner product: (N, D) @ (D, N) → (N, N)
        logits = (phi_out @ theta_out.T) / (self.node_feat_dim ** 0.5)
        A = torch.sigmoid(logits)

        # Row-normalize (like GCN)
        A = A + torch.eye(N, device=device)  # self-loops
        deg = A.sum(dim=-1, keepdim=True) + 1e-6
        A = A / deg

        return A

    # ------------------------------------------------------------------
    # GCN message passing (2 layers)
    # ------------------------------------------------------------------

    def _gcn_forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Two-layer GCN with residual connections.

        Parameters
        ----------
        x : (N, D) — node features
        A : (N, N) — normalized adjacency

        Returns
        -------
        (N, D) — updated node features
        """
        # Layer 1
        h1 = self.gcn_w1(A @ x)
        h1 = self.gcn_norm1(h1)
        h1 = F.relu(h1)

        # Layer 2 + residual (same dim, simple skip connection)
        h2 = self.gcn_w2(A @ h1)
        h2 = self.gcn_norm2(h2)
        h2 = F.relu(h2 + x)

        return h2

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_trajectory: torch.Tensor,         # (B, obs_len, 2)
        scene_data: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B,) — violation logits (before sigmoid)
        """
        B, T, _ = obs_trajectory.shape
        device = obs_trajectory.device
        D = self.node_feat_dim

        # ---- Per-frame GCN → target features ----
        target_feats = []

        for t in range(T):
            # Encode target pedestrian spatial features
            target_spatial = self._spatial_encode(
                obs_trajectory[:, t:t+1, :]
            )  # (B, 1, 8)
            target_enc = self.pos_encoder(target_spatial).squeeze(1)  # (B, D)

            # Try to get scene data for multi-node graph
            frame_data = None
            if scene_data is not None:
                frame_data = self._get_frame_data(scene_data, t, device)

            if frame_data is not None and frame_data["bboxes"].shape[0] > 1:
                # ---- Multi-node graph (target + scene entities) ----
                pos_np = frame_data["positions"].detach().cpu().numpy()
                edge_index, node_types, _ = self.graph_builder.build(
                    positions=pos_np,
                    class_names=frame_data["class_names"],
                    target_idx=frame_data.get("target_idx", 0),
                )

                # Encode ALL nodes (including target) with same encoder
                all_positions = frame_data["positions"]
                all_spatial = self._spatial_encode(all_positions)  # (N, 8)
                all_feats = self.pos_encoder(all_spatial)          # (N, D)

                target_idx = frame_data.get("target_idx", 0)
                target_idx_safe = min(target_idx, all_feats.shape[0] - 1)

                # Inner-product adjacency + GCN
                A = self._build_adj(all_feats)                     # (N, N)
                gcn_out = self._gcn_forward(all_feats, A)          # (N, D)

                # Extract target node → project to conv_dim
                t_feat = gcn_out[target_idx_safe]                   # (D,)
                t_feat = self.target_proj(t_feat)                   # (conv_dim,)
                if t_feat.dim() == 1:
                    t_feat = t_feat.unsqueeze(0)                    # (1, conv_dim)
            else:
                # ---- Single-node (no scene data or only target) ----
                # Process each sample independently with self-loop (1×1 identity).
                # Using a (B,B) adjacency here would mix unrelated pedestrians
                # from different scenes — semantically wrong.
                gcn_out_list = []
                for b in range(B):
                    x = target_enc[b:b+1]                      # (1, D)
                    A_self = torch.eye(1, device=device)       # (1, 1) self-loop
                    gcn_b = self._gcn_forward(x, A_self)       # (1, D)
                    gcn_out_list.append(gcn_b)
                gcn_out = torch.cat(gcn_out_list, dim=0)       # (B, D)
                t_feat = self.target_proj(gcn_out)             # (B, conv_dim)

            # Normalize to (B, conv_dim)
            if t_feat.shape[0] != B:
                t_feat = t_feat.expand(B, -1)
            target_feats.append(t_feat)

        # Stack: (B, T, conv_dim)
        target_seq = torch.stack(target_feats, dim=1)

        # ---- Pedestrian GRU (temporal smoothing of target node) ----
        target_seq, _ = self.ped_gru(target_seq)   # (B, T, conv_dim)

        # ---- Temporal GRU ----
        gru_out, _ = self.gru(target_seq)           # (B, T, hidden_dim)
        h_last = gru_out[:, -1, :]                   # (B, hidden_dim)

        # ---- Classify ----
        logits = self.classifier(h_last).squeeze(-1)  # (B,)
        return logits

    # ------------------------------------------------------------------
    # Frame data extraction (mirrors perception_model._get_frame_data)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame_data(
        scene_data: dict, t: int, device
    ) -> Optional[dict]:
        """
        Extract single-frame scene data for graph building.

        Handles both batched (B, T, N, F) and unbatched (T, N, F) scene tensors.
        With scene data, caller should use B=1 (one scene = one graph);
        batched multi-graph is not supported here — if B>1, only the first
        sample's scene data is used.
        """
        try:
            bboxes = scene_data.get("bboxes")
            if bboxes is None:
                return None

            ndim = bboxes.dim()
            if ndim == 4:
                # (B, T, N, 4) → take first batch element: (N, 4)
                b_t = bboxes[0, t]
            elif ndim == 3:
                # (T, N, 4) → (N, 4)
                b_t = bboxes[t]
            else:
                return None

            if b_t.numel() == 0 or b_t.shape[-2] == 0:
                return None

            positions = scene_data.get("positions")
            if positions is not None:
                if positions.dim() == 4:
                    p_t = positions[0, t]       # (N, 2)
                elif positions.dim() >= 3:
                    p_t = positions[t]           # (N, 2)
                else:
                    p_t = positions
            else:
                p_t = b_t[:, :2]

            class_names_all = scene_data.get("class_names", [])
            if isinstance(class_names_all, list) and len(class_names_all) > 0 \
                    and isinstance(class_names_all[0], list):
                # nested list: first level = batch, second = frames
                cn_t = class_names_all[0][t] if len(class_names_all) <= 1 else class_names_all[0][t]
            elif isinstance(class_names_all, list):
                cn_t = class_names_all[t] if t < len(class_names_all) else class_names_all
            else:
                cn_t = class_names_all

            return {
                "bboxes": b_t.to(device),
                "class_names": cn_t,
                "positions": p_t.to(device),
                "target_idx": scene_data.get("target_idx", 0),
            }
        except (KeyError, IndexError, AttributeError):
            return None


# ======================================================================
# Gaussian MC Violation Classifier
# ======================================================================

class GaussianMCViolationClassifier(nn.Module):
    """
    Wraps a Gaussian-output trajectory model (STGCNN, SocialLSTM) with
    Monte Carlo violation checking for binary classification.

    Pipeline:
        1. Model predicts bivariate Gaussian params (μ, σ, ρ) per timestep
        2. Sample N=100 trajectories from the Gaussian
        3. Each sample → RedLightViolationChecker.geometric_check()
        4. P(violation) = count(violated) / N
        5. Threshold → binary prediction

    Parameters
    ----------
    trajectory_model : nn.Module
        Must support .sample(obs, num_samples=100) → (N, B, T, 2).
    violation_checker : RedLightViolationChecker, optional
    threshold : float
    """

    def __init__(
        self,
        trajectory_model: nn.Module,
        violation_checker=None,
        threshold: float = 0.5,
    ):
        super().__init__()
        from src.classification.red_light_classifier import (
            RedLightViolationChecker, RedLightProbabilityEstimator,
        )
        self.trajectory_model = trajectory_model
        self.violation_checker = violation_checker or RedLightViolationChecker()
        self.estimator = RedLightProbabilityEstimator(
            violation_checker=self.violation_checker,
            threshold=threshold,
        )

    def forward(self, obs_trajectory, num_samples=100, **kwargs):
        """
        Returns
        -------
        dict with:
            "violation_probability": (B,)   P(violation)
            "is_violation":         (B,)   0/1 binary prediction
            "samples":              (N, B, T, 2)  MC trajectory samples
        """
        # 1. Sample trajectories from Gaussian model
        samples = self.trajectory_model.sample(
            obs_trajectory, num_samples=num_samples, **kwargs
        )  # (N, B, T, 2)

        # 2. MC probability estimation
        prob, stats = self.estimator.estimate_probability(samples)

        # 3. Threshold
        pred = (prob >= self.estimator.current_threshold).long()

        return {
            "violation_probability": prob,
            "is_violation": pred,
            "samples": samples,
            "mc_stats": stats,
        }
