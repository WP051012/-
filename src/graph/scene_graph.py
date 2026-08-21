"""
Scene graph for traffic context encoding.

A simplified graph containing only pedestrians and vehicles (no
infrastructure), focusing on inter-agent spatial and motion
relationships without traffic-rules context.

Used for:
    - Providing traffic participant context to the interaction memory
    - Scene-level understanding (who is where, moving how)

References:
    Paper Section 2(4): 场景图 — 以行人和车辆为节点,刻画交通参与者信息
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .gat_conv import EdgeWeightedGATLayer


class SceneGraph(nn.Module):
    """
    Scene graph with pedestrian / vehicle nodes and spatial edges.

    Edge features encode pairwise:
        [relative_distance, relative_direction_x, relative_direction_y, speed_diff]

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    hidden_dim : int
        Hidden / output dimension.
    heads : int
        GAT attention heads.
    max_distance : float
        Maximum distance for creating edges.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        heads: int = 4,
        max_distance: float = 30.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_distance = max_distance

        self.gat = EdgeWeightedGATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            use_edge_features=True,
            edge_feat_dim=4,
        )

        self.output_proj = nn.Linear(hidden_dim * heads, hidden_dim)

    def forward(
        self,
        node_feats: Tensor,          # (N, in_dim)  PFN
        positions: Tensor,           # (N, 2)
        velocities: Optional[Tensor] = None,  # (N, 2)
    ) -> Tensor:
        """
        Returns
        -------
        Tensor (N, hidden_dim)
            Scene-aware node embeddings.
        """
        N = node_feats.size(0)
        device = node_feats.device

        if N <= 1:
            return self.output_proj(node_feats) if N == 1 else node_feats

        # --- Build fully-connected spatial graph within distance threshold ---
        pos_np = positions.detach().cpu().numpy()
        edges_src, edges_dst = [], []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dist = np.linalg.norm(pos_np[i] - pos_np[j])
                if dist < self.max_distance:
                    edges_src.append(i)
                    edges_dst.append(j)

        if not edges_src:
            return self.output_proj(node_feats)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long,
                                  device=device)
        src, dst = edges_src, edges_dst

        # --- Edge features: [rel_dist_norm, rel_dir_x, rel_dir_y, speed_diff] ---
        delta = positions[dst] - positions[src]              # (E, 2)
        rel_dist = torch.norm(delta, dim=-1, keepdim=True)   # (E, 1)
        rel_dist_norm = rel_dist / self.max_distance

        rel_dir = delta / (rel_dist + 1e-6)                  # (E, 2)

        if velocities is not None:
            speed_src = torch.norm(velocities[src], dim=-1, keepdim=True)
            speed_dst = torch.norm(velocities[dst], dim=-1, keepdim=True)
            speed_diff = (speed_dst - speed_src) / (speed_src + 1e-6)
        else:
            speed_diff = torch.zeros(len(edges_src), 1, device=device)

        edge_attr = torch.cat([rel_dist_norm, rel_dir, speed_diff], dim=-1)

        # --- GAT ---
        out = self.gat(node_feats, edge_index, edge_weight=None, edge_attr=edge_attr)
        return self.output_proj(out)  # (N, hidden_dim)
