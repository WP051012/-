"""
Traffic perception memory module.

Simulates the formation of three memory types in human traffic perception:
    1. Behavioral Memory    — what the target person intends to do
    2. Environmental Memory  — what traffic rules / infrastructure constrain
    3. Interactive Memory   — how surrounding agents influence / interact

These are fused into a unified traffic perception vector **c**, which
conditions the subsequent trajectory prediction (FlowChain).

References:
    Paper Section 3: 交通感知记忆模块
    STRR:             spatiotemporal relationship reasoning (GRU structure)
    MAGELLAN:         cognitive map (memory confidence / decay concept)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# 1. Behavioral Memory
# ======================================================================

class BehavioralMemory(nn.Module):
    """
    Encodes the target pedestrian's own behavioral intent.

    Source: target pedestrian node features from the perception graph.
    Represents: the pedestrian's movement pattern, speed, heading, etc.

    Parameters
    ----------
    input_dim : int
        Dimension of target node features from perception graph.
    memory_dim : int
        Output behavioural memory dimension.
    """

    def __init__(self, input_dim: int = 128, memory_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(inplace=True),
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )

    def forward(self, target_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        target_feat : Tensor (..., input_dim)
            Target pedestrian features from perception graph.

        Returns
        -------
        Tensor (..., memory_dim)
        """
        return self.encoder(target_feat)


# ======================================================================
# 2. Environmental Memory
# ======================================================================

class EnvironmentalMemory(nn.Module):
    """
    Encodes environmental constraints from infrastructure elements.

    Source: traffic lights, traffic signs, lane lines → attention-weighted
    fusion. Additionally calibrated against target pedestrian state.

    Per paper Section 3.2:
        - Infrastructure node features are attended over (key, value)
        - Target pedestrian features serve as query
        - Output represents current traffic rule constraints

    Parameters
    ----------
    infra_feat_dim : int
        Dimension of infrastructure node features.
    target_feat_dim : int
        Dimension of target pedestrian features.
    memory_dim : int
        Output environmental memory dimension.
    attention_dim : int
        Hidden dimension for attention computation.
    """

    def __init__(
        self,
        infra_feat_dim: int = 128,
        target_feat_dim: int = 128,
        memory_dim: int = 128,
        attention_dim: int = 64,
    ):
        super().__init__()

        # Attention: target (query) → infrastructure (key/value)
        self.query_proj = nn.Linear(target_feat_dim, attention_dim)
        self.key_proj = nn.Linear(infra_feat_dim, attention_dim)
        self.value_proj = nn.Linear(infra_feat_dim, memory_dim)

        self.attention_dim = attention_dim

        # Calibration gate: how much does the environment constrain the target?
        self.calibrate = nn.Sequential(
            nn.Linear(target_feat_dim + memory_dim, memory_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )

    def forward(
        self,
        target_feat: Tensor,               # (..., target_feat_dim)
        infra_feats: Tensor,               # (..., num_infra, infra_feat_dim)
        infra_mask: Optional[Tensor] = None,  # (..., num_infra)  bool mask
    ) -> Tensor:
        """
        Parameters
        ----------
        target_feat : Tensor
            Target pedestrian features.
        infra_feats : Tensor
            Stacked infrastructure features.
        infra_mask : Tensor, optional
            Boolean mask for valid infrastructure nodes (True = valid).

        Returns
        -------
        Tensor (..., memory_dim)
        """
        if infra_feats.size(-2) == 0:
            # No infrastructure nodes → empty environment memory
            return torch.zeros(*target_feat.shape[:-1], self.output[-1].normalized_shape[0],
                               device=target_feat.device)

        # --- Attention ---
        Q = self.query_proj(target_feat).unsqueeze(-2)   # (..., 1, attn_dim)
        K = self.key_proj(infra_feats)                     # (..., N, attn_dim)
        V = self.value_proj(infra_feats)                   # (..., N, memory_dim)

        attn_scores = (Q * K).sum(dim=-1) / (self.attention_dim ** 0.5)  # (..., N)

        if infra_mask is not None:
            attn_scores = attn_scores.masked_fill(~infra_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)     # (..., N)

        env_context = (attn_weights.unsqueeze(-1) * V).sum(dim=-2)  # (..., memory_dim)

        # --- Calibration gate ---
        calib_input = torch.cat([target_feat, env_context], dim=-1)
        gate = self.calibrate(calib_input)

        env_memory = gate * env_context

        return self.output(env_memory)


# ======================================================================
# 3. Interactive Memory
# ======================================================================

class InteractiveMemory(nn.Module):
    """
    Encodes interaction relationships between traffic participants.

    Source: GAT-processed interaction graph (from PerceptionGraph).
    Represents: how surrounding agents influence / interact with the target.

    Per paper Section 3.3:
        - GAT replaces STRR's GCN for interaction modelling
        - Simultaneously considers node features, edge weights, edge features

    Parameters
    ----------
    node_feat_dim : int
        Dimension of node features after GAT.
    memory_dim : int
        Output interactive memory dimension.
    """

    def __init__(
        self,
        node_feat_dim: int = 128,
        memory_dim: int = 128,
    ):
        super().__init__()

        # Aggregate surrounding agent features via attention
        self.attn_query = nn.Linear(node_feat_dim, memory_dim)
        self.attn_key = nn.Linear(node_feat_dim, memory_dim)
        self.attn_value = nn.Linear(node_feat_dim, memory_dim)

        self.memory_dim = memory_dim

        # Output
        self.output = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(memory_dim),
        )

    def forward(
        self,
        target_feat: Tensor,              # (..., node_feat_dim)
        all_node_feats: Tensor,           # (..., N, node_feat_dim)
        target_idx: int = 0,
        agent_mask: Optional[Tensor] = None,  # (..., N)  True = is agent (not infra)
    ) -> Tensor:
        """
        Parameters
        ----------
        target_feat : Tensor
        all_node_feats : Tensor
        target_idx : int
        agent_mask : Tensor, optional
            Boolean mask selecting only agent nodes (vehicles + pedestrians).

        Returns
        -------
        Tensor (..., memory_dim)
        """
        N = all_node_feats.size(-2)
        if N <= 1:
            return torch.zeros(*target_feat.shape[:-1], self.memory_dim,
                               device=target_feat.device)

        # --- Aggregate other agents via attention ---
        Q = self.attn_query(target_feat).unsqueeze(-2)   # (..., 1, memory_dim)
        K = self.attn_key(all_node_feats)                  # (..., N, memory_dim)
        V = self.attn_value(all_node_feats)                # (..., N, memory_dim)

        attn_scores = (Q * K).sum(dim=-1) / (self.memory_dim ** 0.5)

        # Mask out self
        self_mask = torch.ones_like(attn_scores, dtype=torch.bool)
        self_mask[..., target_idx] = False
        attn_scores = attn_scores.masked_fill(~self_mask, -1e9)

        # Mask out non-agent nodes (infrastructure)
        if agent_mask is not None:
            attn_scores = attn_scores.masked_fill(~agent_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)      # (..., N)

        interact_context = (attn_weights.unsqueeze(-1) * V).sum(dim=-2)

        return self.output(interact_context)


# ======================================================================
# 4. Memory Fusion → Traffic Perception Vector c
# ======================================================================

class MemoryFusion(nn.Module):
    """
    Fuses behavioral, environmental, and interactive memories into
    a unified traffic perception vector **c**.

    Fusion strategy:
        c = LayerNorm( W_fuse [ m_behavioral || m_env || m_interactive ] )

    Additionally provides a confidence-weighted variant where each
    memory branch has a learned confidence score.

    Parameters
    ----------
    behavioral_dim, environmental_dim, interactive_dim : int
    fusion_dim : int
        Output dimension of traffic perception vector c.
    """

    def __init__(
        self,
        behavioral_dim: int = 128,
        environmental_dim: int = 128,
        interactive_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        total_dim = behavioral_dim + environmental_dim + interactive_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Learned memory confidence weights
        self.confidence = nn.Parameter(torch.ones(3) / 3)  # uniform prior

    def forward(
        self,
        behavioral: Tensor,     # (..., D_behavioral)
        environmental: Tensor,  # (..., D_env)
        interactive: Tensor,    # (..., D_interactive)
        use_confidence: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns
        -------
        c : Tensor (..., fusion_dim)
            Traffic perception vector.
        memory_info : dict
            Individual (possibly confidence-weighted) memory tensors
            for interpretability / logging.
        """
        if use_confidence:
            # Normalise confidence to sum to 1
            conf = F.softmax(self.confidence, dim=0)
            b = behavioral * conf[0]
            e = environmental * conf[1]
            i = interactive * conf[2]
        else:
            conf = torch.ones(3, device=behavioral.device) / 3
            b, e, i = behavioral, environmental, interactive

        fused = torch.cat([b, e, i], dim=-1)
        c = self.fusion(fused)

        info = {
            "behavioral": b,
            "environmental": e,
            "interactive": i,
            "confidence_weights": conf.detach(),
        }

        return c, info


# ======================================================================
# 5. Full Perception Memory Module
# ======================================================================

class TrafficPerceptionMemory(nn.Module):
    """
    Complete traffic perception memory module.

    Pipeline:
        1. BehavioralMemory   ← target pedestrian features
        2. EnvironmentalMemory ← infrastructure features (attended by target)
        3. InteractiveMemory   ← surrounding agent features (attended by target)
        4. MemoryFusion        → traffic perception vector **c**

    Parameters
    ----------
    node_feat_dim : int
        Input dimension of graph node features.
    behavioral_dim, environmental_dim, interactive_dim, fusion_dim : int
    """

    def __init__(
        self,
        node_feat_dim: int = 128,
        behavioral_dim: int = 128,
        environmental_dim: int = 128,
        interactive_dim: int = 128,
        fusion_dim: int = 256,
        attention_dim: int = 64,
    ):
        super().__init__()

        self.behavioral = BehavioralMemory(
            input_dim=node_feat_dim,
            memory_dim=behavioral_dim,
        )
        self.environmental = EnvironmentalMemory(
            infra_feat_dim=node_feat_dim,
            target_feat_dim=node_feat_dim,
            memory_dim=environmental_dim,
            attention_dim=attention_dim,
        )
        self.interactive = InteractiveMemory(
            node_feat_dim=node_feat_dim,
            memory_dim=interactive_dim,
        )
        self.fusion = MemoryFusion(
            behavioral_dim=behavioral_dim,
            environmental_dim=environmental_dim,
            interactive_dim=interactive_dim,
            fusion_dim=fusion_dim,
        )

    def forward(
        self,
        node_embeddings: Tensor,          # (N, node_feat_dim)
        target_idx: int = 0,
        infra_indices: Optional[List[int]] = None,
        agent_indices: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        node_embeddings : Tensor (N, D)
            All node embeddings from perception graph.
        target_idx : int
            Index of target pedestrian.
        infra_indices : list of int, optional
            Indices of infrastructure nodes.
        agent_indices : list of int, optional
            Indices of agent nodes (vehicles + other pedestrians).

        Returns
        -------
        c : Tensor (fusion_dim,)
            Traffic perception vector.
        memory_info : dict
        """
        N = node_embeddings.size(0)
        device = node_embeddings.device

        target_feat = node_embeddings[target_idx]  # (D,)

        # --- 1. Behavioral ---
        b = self.behavioral(target_feat)

        # --- 2. Environmental ---
        if infra_indices is not None and len(infra_indices) > 0:
            infra_feats = node_embeddings[infra_indices]  # (N_infra, D)
            infra_mask = torch.ones(len(infra_indices), dtype=torch.bool,
                                    device=device)
        else:
            infra_feats = node_embeddings[[]]   # empty
            infra_mask = None

        e = self.environmental(target_feat, infra_feats, infra_mask)

        # --- 3. Interactive ---
        if agent_indices is not None and len(agent_indices) > 0:
            agent_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for idx in agent_indices:
                if idx < N:
                    agent_mask[idx] = True
        else:
            agent_mask = None

        i = self.interactive(target_feat, node_embeddings, target_idx, agent_mask)

        # --- 4. Fusion ---
        c, memory_info = self.fusion(b, e, i)

        return c, memory_info

    # ------------------------------------------------------------------
    # Convenience: run from raw graph output
    # ------------------------------------------------------------------

    def from_graph_output(
        self,
        node_embeddings: Tensor,          # (N, D)
        node_types: List[int],            # [N]  type codes (0=person, 1=vehicle, 2=infra)
        target_idx: int = 0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Wrapper that auto-infers infra/agent indices from node types."""
        infra_indices = [i for i, t in enumerate(node_types) if t == 2]
        agent_indices = [i for i, t in enumerate(node_types)
                         if t in (0, 1) and i != target_idx]
        return self.forward(
            node_embeddings,
            target_idx=target_idx,
            infra_indices=infra_indices,
            agent_indices=agent_indices,
        )
