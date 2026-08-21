"""
Baseline model implementations for comparison experiments.

Each baseline has consistent I/O:
    Input:  obs_trajectory (B, obs_len, 2)
    Output: dict with "mean" (B, pred_len, 2) and optionally "log_probs", "samples"

Reference:
    Experiment Plan Section 5, 8
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Social-LSTM
# ======================================================================

class SocialLSTM(nn.Module):
    """
    Social-LSTM: LSTM with social pooling for pedestrian trajectory prediction.

    Simplified version — pools hidden states of nearby pedestrians via
    a max-pooling layer, then LSTM decoder predicts future trajectory.

    Reference:
        Alahi et al., CVPR 2016
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pool_dim: int = 32,
        neighborhood_size: int = 32,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        # Position embedding
        self.embedding = nn.Linear(2, embedding_dim)

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Social pooling (simplified: pool over learnable grid)
        self.pool_fc = nn.Sequential(
            nn.Linear(hidden_dim, pool_dim),
            nn.ReLU(),
            nn.Linear(pool_dim, pool_dim),
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=embedding_dim + pool_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Output projection: hidden -> (dx, dy)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        neighbor_trajectories: Optional[torch.Tensor] = None,  # (B, N, obs_len, 2) optional
        **kwargs,
    ) -> dict:
        B, T, _ = obs_trajectory.shape

        # Embed positions
        emb = self.embedding(obs_trajectory)        # (B, T, emb_dim)
        _, (h_n, _) = self.encoder(emb)
        h_last = h_n[-1]                             # (B, hidden_dim)

        # Social pooling (if neighbors available)
        if neighbor_trajectories is not None and neighbor_trajectories.shape[1] > 0:
            # Pool neighbor features — simplified max-pool
            B, N, Tn, _ = neighbor_trajectories.shape
            neighbor_flat = neighbor_trajectories.view(B * N, Tn, 2)
            n_emb = self.embedding(neighbor_flat)
            _, (n_h, _) = self.encoder(n_emb)
            n_hidden = n_h[-1].view(B, N, -1)         # (B, N, hidden_dim)
            pooled = n_hidden.max(dim=1)[0]            # (B, hidden_dim)
            pool_feat = self.pool_fc(pooled)            # (B, pool_dim)
        else:
            pool_feat = torch.zeros(B, self.pool_fc[-1].out_features,
                                   device=obs_trajectory.device)

        # Decode future trajectory autoregressively
        decoder_input = emb[:, -1:]                     # (B, 1, emb_dim)
        h_dec = h_last.unsqueeze(0).repeat(1, 1, 1).contiguous()  # (1, B, hidden_dim)
        c_dec = torch.zeros_like(h_dec)
        hidden = (h_dec, c_dec)

        outputs = []
        for _ in range(self.pred_len):
            dec_in = torch.cat([decoder_input[:, -1:],
                                pool_feat.unsqueeze(1).expand(-1, 1, -1)], dim=-1)
            out, hidden = self.decoder(dec_in, hidden)
            delta = self.output_fc(out)                  # (B, 1, 2)
            outputs.append(delta)

            # Next input: embed the predicted position
            if obs_trajectory.shape[1] > 0:
                last_pos = obs_trajectory[:, -1:]        # (B, 1, 2)
                next_pos = last_pos + delta
                decoder_input = self.embedding(next_pos)

        pred = torch.cat(outputs, dim=1)                 # (B, pred_len, 2) — deltas

        # Cumulative sum to get absolute positions
        last_obs = obs_trajectory[:, -1:]                # (B, 1, 2)
        trajectory = last_obs + pred.cumsum(dim=1)

        return {"mean": trajectory, "pred_deltas": pred}


# ======================================================================
# 2. Social-STGCNN
# ======================================================================

class SocialSTGCNN(nn.Module):
    """
    Social-STGCNN: Spatiotemporal graph CNN for trajectory prediction.

    Uses spatial graph convolutions (GCN) over pedestrian positions
    followed by temporal convolutions (TCN) and a temporal-preserving
    decoder (analogous to TXP-CNN in the original paper).

    Reference:
        Mohamed et al., CVPR 2020
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        # Spatial graph convolution layers (simplified GCN)
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 1),
            nn.ReLU(),
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
        )

        # Temporal convolution (TCN over time axis)
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Temporal-preserving decoder (replaces mean-pool + flat Linear)
        # Analogous to TXP-CNN: keeps temporal structure, upsamples T→pred_len
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.temporal_upsample = nn.AdaptiveAvgPool1d(pred_len)  # T=8 → pred_len=12
        self.output_proj = nn.Linear(hidden_dim, 2)               # per-frame (dx, dy)

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        neighbor_positions: Optional[torch.Tensor] = None,  # (B, N, 2) current positions
        **kwargs,
    ) -> dict:
        B, T, _ = obs_trajectory.shape

        # Treat single pedestrian as a 1-node graph
        # Reshape: (B, 2, 1, T) for spatial conv
        x = obs_trajectory.permute(0, 2, 1).unsqueeze(2)  # (B, 2, 1, T)

        # Spatial convs
        x = self.spatial_conv1(x)   # (B, H, 1, T)
        x = self.spatial_conv2(x)   # (B, H, 1, T)
        x = x.squeeze(2)            # (B, H, T)

        # Temporal conv
        x = self.tcn(x)             # (B, H, T)

        # Temporal-preserving decoder: keep time structure, upsample to pred_len
        x = self.temporal_proj(x)        # (B, H, T)
        x = self.temporal_upsample(x)    # (B, H, pred_len)
        x = x.transpose(1, 2)            # (B, pred_len, H)
        pred = self.output_proj(x)       # (B, pred_len, 2) — per-frame delta

        # Absolute positions (relative to last observed position)
        last_obs = obs_trajectory[:, -1:]  # (B, 1, 2)
        trajectory = last_obs + pred

        return {"mean": trajectory}


# ======================================================================
# 3. FlowChain Baseline (without traffic perception)
# ======================================================================

class FlowChainBase(nn.Module):
    """
    Vanilla FlowChain without traffic perception conditioning.

    Wraps FlowChainPredictor (→ TransformerFlowChain) with zero condition.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        d_model: int = 64,
        nvp_hidden_dim: int = 128,
        nvp_num_blocks: int = 3,
        condition_dim: int = 256,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        from src.prediction import FlowChainPredictor
        self.predictor = FlowChainPredictor(
            obs_len=obs_len,
            pred_len=pred_len,
            trajectory_dim=2,
            hidden_dim=d_model,
            condition_dim=condition_dim,
            num_flows=nvp_num_blocks,
        )
        self._cond_dim = condition_dim

    def forward(
        self,
        obs_trajectory: torch.Tensor,
        num_samples: int = 20,
        **kwargs,
    ) -> dict:
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device
        c = torch.zeros(B, self._cond_dim, device=device)

        return self.predictor(
            obs_trajectory=obs_trajectory,
            perception_c=c,
            num_samples=num_samples,
        )


# ======================================================================
# 4. LSTM Classifier (trajectory → violation)
# ======================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM-based red-light violation classifier.

    Input: full trajectory (obs + ground-truth future)
    Output: violation probability
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        trajectory : Tensor (B, T, 2)

        Returns
        -------
        Tensor (B,) — violation logits
        """
        _, (h_n, _) = self.lstm(trajectory)
        h_last = h_n[-1]                    # (B, hidden_dim)
        logits = self.classifier(h_last).squeeze(-1)  # (B,)
        return logits


# ======================================================================
# 5. GRU Classifier (trajectory → violation)
# ======================================================================

class GRUClassifier(nn.Module):
    """
    GRU-based red-light violation classifier.

    Input: full trajectory (obs + ground-truth future)
    Output: violation probability
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(trajectory)
        h_last = h_n[-1]
        logits = self.classifier(h_last).squeeze(-1)
        return logits


# ======================================================================
# 6. STRR Classifier (graph-based violation prediction)
# ======================================================================

class STRRClassifier(nn.Module):
    """
    STRR-style spatiotemporal relationship reasoning for violation prediction.

    Simplified STRR: builds graph from target + context, applies GCN reasoning,
    then classifies violation probability.

    Reference:
        STRR: Spatiotemporal Relationship Reasoning for Pedestrian Intent Prediction
    """

    def __init__(
        self,
        node_feat_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim

        # Position encoder
        self.pos_encoder = nn.Linear(8, node_feat_dim)  # 8-dim spatial

        # Pedestrian / context embeddings
        self.ped_embed = nn.Linear(node_feat_dim, node_feat_dim)
        self.ctxt_embed = nn.Linear(node_feat_dim, node_feat_dim)

        # Graph convolution weight
        self.W_graph = nn.Linear(node_feat_dim, node_feat_dim)

        # GRU for temporal encoding
        self.gru = nn.GRU(node_feat_dim, hidden_dim, 2, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        context_bboxes: Optional[torch.Tensor] = None,  # (B, obs_len, N, 4)
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns
        -------
        Tensor (B,) — violation logits
        """
        B, T, _ = obs_trajectory.shape
        device = obs_trajectory.device

        # Encode pedestrian positions (simplified: use trajectory as features)
        ped_feats = self.pos_encoder(
            self._spatial_encode(obs_trajectory)
        )  # (B, T, feat_dim)

        # Temporal GRU
        ped_feats, _ = self.gru(ped_feats)       # (B, T, hidden_dim)
        h_last = ped_feats[:, -1, :]             # (B, hidden_dim)

        logits = self.classifier(h_last).squeeze(-1)
        return logits

    @staticmethod
    def _spatial_encode(traj: torch.Tensor) -> torch.Tensor:
        """Simple 8-dim spatial encoding from trajectory."""
        B, T, _ = traj.shape
        v = traj[:, 1:] - traj[:, :-1]
        v = torch.cat([torch.zeros(B, 1, 2, device=traj.device), v], dim=1)
        speed = torch.norm(v, dim=-1, keepdim=True)
        angle = torch.atan2(v[..., 1:2], v[..., 0:1])
        return torch.cat([traj, v, speed, angle], dim=-1)[:, :, :8]  # (B, T, 8)


# ======================================================================
# 7. Our Method Wrapper
# ======================================================================

# ======================================================================
# 8. Transformer Baseline (Trajectory-Transformer, official architecture)
# ======================================================================

class TransformerBaseline(nn.Module):
    """
    Trajectory-Transformer baseline for trajectory prediction.

    Wraps the official IndividualTF model from FGiuliari/Trajectory-Transformer,
    adapting it to the unified interface: (B, obs_len, 2) → {"mean": (B, pred_len, 2)}.

    Architecture (from "Attention is All You Need"):
        - Linear embedding of (x,y) deltas → d_model
        - Positional encoding (sinusoidal)
        - N-layer Transformer encoder (self-attention + FFN)
        - N-layer Transformer decoder (self-attn + cross-attn + FFN)
        - Linear projection to output deltas

    Reference:
        Giuliari et al., "Transformer Networks for Trajectory Forecasting", ICPR 2020
        https://github.com/FGiuliari/Trajectory-Transformer
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        d_model: int = 512,
        d_ff: int = 2048,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Import the official IndividualTF model
        import sys
        from pathlib import Path
        _tf_dir = Path(__file__).resolve().parent / "trajectory_transformer"
        if str(_tf_dir) not in sys.path:
            sys.path.insert(0, str(_tf_dir.parent))

        from src.baselines.trajectory_transformer import subsequent_mask

        # Build the official model components manually for clarity
        import copy
        from trajectory_transformer.multihead_attention import MultiHeadAttention
        from trajectory_transformer.pointerwise_feedforward import PointerwiseFeedforward
        from trajectory_transformer.positional_encoding import PositionalEncoding
        from trajectory_transformer.encoder_decoder import EncoderDecoder
        from trajectory_transformer.encoder import Encoder
        from trajectory_transformer.encoder_layer import EncoderLayer
        from trajectory_transformer.decoder import Decoder
        from trajectory_transformer.decoder_layer import DecoderLayer

        c = copy.deepcopy
        attn = MultiHeadAttention(heads, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layers),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), layers),
            nn.Sequential(self._LinearEmbedding(2, d_model), c(position)),
            nn.Sequential(self._LinearEmbedding(3, d_model), c(position)),
            self._Generator(d_model, 3),  # outputs (dx, dy, confidence)
        )

        # Weight initialization (from official code)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Track normalization stats (set during training)
        self.register_buffer("delta_mean", torch.zeros(2))
        self.register_buffer("delta_std", torch.ones(2))

        # Save subsequent_mask reference for inference
        self._subsequent_mask = subsequent_mask

    class _LinearEmbedding(nn.Module):
        """Linear projection + sqrt(d_model) scaling (from official code)."""
        def __init__(self, inp_size, d_model):
            super().__init__()
            self.lut = nn.Linear(inp_size, d_model)
            self.d_model = d_model

        def forward(self, x):
            return self.lut(x) * math.sqrt(self.d_model)

    class _Generator(nn.Module):
        """Final linear projection (from official code)."""
        def __init__(self, d_model, out_size):
            super().__init__()
            self.proj = nn.Linear(d_model, out_size)

        def forward(self, x):
            return self.proj(x)

    def _to_deltas(self, traj: torch.Tensor) -> torch.Tensor:
        """Convert absolute positions to frame-to-frame deltas."""
        # traj: (B, T, 2)
        B, T, _ = traj.shape
        deltas = traj[:, 1:] - traj[:, :-1]  # (B, T-1, 2)
        # Prepend origin (0,0) for the first frame
        origin = torch.zeros(B, 1, 2, device=traj.device, dtype=traj.dtype)
        return torch.cat([origin, deltas], dim=1)  # (B, T, 2)

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        num_samples: int = 1,                # ignored; deterministic
        **kwargs,
    ) -> dict:
        """
        Forward pass with autoregressive decoding.

        Returns dict with "mean": (B, pred_len, 2) in absolute coordinates.
        """
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        # Convert to deltas and normalize
        obs_deltas = self._to_deltas(obs_trajectory)  # (B, obs_len, 2)
        std = self.delta_std.clamp(min=1e-6)
        obs_deltas_norm = (obs_deltas - self.delta_mean.to(device)) / std.to(device)

        # Encoder
        src_att = torch.ones((B, 1, obs_deltas_norm.shape[1]), device=device)

        # Decoder: start token (from official code: [0, 0, 1])
        start_token = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3).repeat(B, 1, 1)
        dec_inp = start_token  # (B, 1, 3)

        # Autoregressive decoding
        for _ in range(self.pred_len):
            trg_att = self._subsequent_mask(dec_inp.shape[1]).repeat(B, 1, 1).to(device)
            raw = self.model(obs_deltas_norm, dec_inp, src_att, trg_att)  # (B, t+1, d_model)
            out = self.model.generator(raw)                                # (B, t+1, 3)  ← official: project to output dim
            dec_inp = torch.cat([dec_inp, out[:, -1:, :]], dim=1)

        # Extract predicted deltas (skip start token, take first 2 dims)
        pred_deltas_norm = dec_inp[:, 1:, :2]  # (B, pred_len, 2)

        # Denormalize
        pred_deltas = pred_deltas_norm * std.to(device) + self.delta_mean.to(device)

        # Cumulative sum → absolute positions relative to last observed position
        last_obs = obs_trajectory[:, -1:]  # (B, 1, 2)
        trajectory = last_obs + pred_deltas.cumsum(dim=1)

        return {"mean": trajectory}

    def fit_normalization(self, train_loader, device, norm=None):
        """
        Compute delta mean/std from training data (in normalized space).

        If norm is provided (e.g. [3840, 2160]), statistics are computed in
        normalized [0,1] coordinates to match the training loop's /NORM step.

        Call once before training (mirrors official code).
        """
        all_deltas = []
        norm_t = torch.tensor(norm, device=device) if norm is not None else None
        for batch in train_loader:
            obs = batch["obs_trajectory"].to(device)
            if norm_t is not None:
                obs = obs / norm_t
            deltas = self._to_deltas(obs)
            all_deltas.append(deltas.view(-1, 2))
        all_deltas = torch.cat(all_deltas, dim=0)
        self.delta_mean = all_deltas.mean(0)
        self.delta_std = all_deltas.std(0)


# ======================================================================
# 9. Vanilla RNN Seq2Seq Baseline (from uestc-db/TrajectoryPrediction RNN/)
# ======================================================================

class RNNBaseline(nn.Module):
    """
    Vanilla RNN Seq2Seq encoder-decoder for trajectory prediction.

    PyTorch port of the RNN model from uestc-db/TrajectoryPrediction.
    Architecture: RNN encoder over observed trajectory → final hidden state
    initializes RNN decoder → autoregressive delta prediction.

    Uses plain tanh-RNN (no gates), consistent with the reference implementation.

    Reference:
        https://github.com/uestc-db/TrajectoryPrediction (RNN/)
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Encoder: vanilla RNN (tanh activation, no gates)
        self.encoder = nn.RNN(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.encoder_dropout = nn.Dropout(dropout)

        # Decoder: vanilla RNN
        self.decoder = nn.RNN(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Output projection: hidden → (dx, dy)
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        **kwargs,
    ) -> dict:
        """
        Forward pass with autoregressive decoding.

        Feeds back the previous predicted absolute position (not delta)
        to keep decoder input scale consistent across all time steps.

        Returns dict with "mean": (B, pred_len, 2) in absolute coordinates.
        """
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        # Encoder: RNN over obs trajectory
        _, h_n = self.encoder(obs_trajectory)  # h_n: (1, B, hidden_dim)
        h_n = self.encoder_dropout(h_n)

        # Decoder: autoregressively predict future positions
        # Feed back absolute position at each step to keep input scale ~0.5
        dec_input = obs_trajectory[:, -1:]       # (B, 1, 2) last absolute position
        state = h_n                                # (1, B, hidden_dim)
        pred_pos = dec_input.clone()
        outputs = []

        for _ in range(self.pred_len):
            dec_out, state = self.decoder(dec_input, state)
            delta = self.output_proj(dec_out)      # (B, 1, 2) position delta
            pred_pos = pred_pos + delta             # accumulate to absolute position
            outputs.append(pred_pos)
            dec_input = pred_pos                    # feed back absolute position

        trajectory = torch.cat(outputs, dim=1)     # (B, pred_len, 2)

        return {"mean": trajectory}


# ======================================================================
# 10. Our Method Wrapper
# ======================================================================

class OurMethodWrapper(nn.Module):
    """
    Full pipeline wrapper: perception graph + GAT + CM-GRU + FlowChain.

    This loads the trained RedLightPredictionModel (Stage 3) and exposes
    the same forward() interface as baselines for evaluation.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 256,
        condition_dim: int = 256,
        num_flows: int = 3,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self._cond_dim = condition_dim

        from src.prediction import FlowChainPredictor
        self.predictor = FlowChainPredictor(
            obs_len=obs_len,
            pred_len=pred_len,
            trajectory_dim=2,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_flows=num_flows,
        )

    def forward(
        self,
        obs_trajectory: torch.Tensor,
        perception_c: Optional[torch.Tensor] = None,
        num_samples: int = 20,
        **kwargs,
    ) -> dict:
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        if perception_c is None:
            perception_c = torch.zeros(
                B, self._cond_dim, device=device,
            )

        return self.predictor(
            obs_trajectory=obs_trajectory,
            perception_c=perception_c,
            num_samples=num_samples,
        )
