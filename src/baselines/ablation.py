"""
Ablation study model variants.

Each variant removes or replaces one component of the full pipeline
to measure its contribution.

Variants:
    AblationNoGraph:       Remove perception graph (no traffic cognition)
    AblationNoGRU:         GRU without cognitive context (trajectory + GAT only)
    AblationNoFlowChain:   Replace FlowChain with deterministic MLP prediction
"""

import torch
import torch.nn as nn


class AblationNoGraph(nn.Module):
    """
    Ablation: Remove perception graph.

    Uses only trajectory coordinates without any traffic cognition.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        d_model: int = 64,
        condition_dim: int = 256,
        num_flows: int = 3,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self._cond_dim = condition_dim

        from src.prediction import FlowChainPredictor
        self.predictor = FlowChainPredictor(
            obs_len=obs_len, pred_len=pred_len, trajectory_dim=2,
            hidden_dim=d_model, condition_dim=condition_dim,
            num_flows=num_flows,
        )

    def forward(self, obs_trajectory, num_samples=20, **kwargs):
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device
        c = torch.zeros(B, self._cond_dim, device=device)
        return self.predictor(
            obs_trajectory=obs_trajectory,
            perception_c=c, num_samples=num_samples,
        )


class AblationNoGRU(nn.Module):
    """
    Ablation: Replace CM-GRU with standard GRU for temporal encoding.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        input_dim: int = 2,
        hidden_dim: int = 256,
        condition_dim: int = 256,
        num_flows: int = 3,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        # Standard GRU (no perception injection)
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True)
        self.hidden_to_cond = nn.Linear(hidden_dim, condition_dim)

        from src.prediction import FlowChainPredictor
        self.predictor = FlowChainPredictor(
            obs_len=obs_len, pred_len=pred_len, trajectory_dim=2,
            hidden_dim=hidden_dim, condition_dim=condition_dim,
            num_flows=num_flows,
        )

    def forward(self, obs_trajectory, num_samples=20, **kwargs):
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        _, h_n = self.gru(obs_trajectory)
        h = h_n[-1]
        c = self.hidden_to_cond(h)

        return self.predictor(
            obs_trajectory=obs_trajectory,
            perception_c=c, num_samples=num_samples,
        )


class AblationNoFlowChain(nn.Module):
    """
    Ablation: Replace FlowChain with deterministic MLP prediction.

    Uses an MLP to directly predict future trajectory mean + variance
    instead of probabilistic FlowChain.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Temporal encoder
        self.encoder = nn.GRU(2, hidden_dim, 2, batch_first=True)

        # Deterministic decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_len * 2),
        )

        self.pred_len = pred_len

    def forward(self, obs_trajectory, num_samples=20, **kwargs):
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        _, h_n = self.encoder(obs_trajectory)
        h = h_n[-1]

        # Predict DELTAS from last observed position (like STGCNN)
        deltas = self.decoder(h).view(B, self.pred_len, 2)
        last_obs = obs_trajectory[:, -1:]  # (B, 1, 2)
        pred = last_obs + deltas            # delta → absolute

        # Deterministic → no log_probs
        return {
            "mean": pred,
            "samples": pred.unsqueeze(0).repeat(num_samples, 1, 1, 1),
            "log_probs": torch.zeros(num_samples, B),
            "std": torch.ones_like(pred),
        }
