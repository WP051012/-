"""
Traffic perception graph construction and GAT-based reasoning.

Builds a heterogeneous spatiotemporal graph of traffic participants
and infrastructure, then applies edge-weighted GAT for relational
reasoning.

References:
    STRR:          spatiotemporal relationship reasoning (graph structure + edge weights)
    Social-STGCNN: social spatiotemporal graph CNN (graph building from positions)
    Paper Section 2: 交通感知图构建 (traffic perception graph construction)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .node_encoder import NodeFeatureEncoder, SpatialEncoder
from .gat_conv import EdgeWeightedGAT, STREdgeWeight, EdgeFeatureEncoder, RelativeSpatialEncoder

logger = logging.getLogger(__name__)


# ======================================================================
# Graph Builder — constructs adjacency from node positions & types
# ======================================================================

class PerceptionGraphBuilder:
    """
    Builds the heterogeneous traffic perception graph.

    Node types (as defined in paper Section 2.1):
        - core:      target pedestrian (1 node)
        - vehicle:   cars, buses, trucks, bicycles, motorcycles
        - person:    other pedestrians
        - infra:     traffic lights, traffic signs, lane lines

    Edge types:
        - core ↔ all_others       (full connectivity from target)
        - infra ↔ infra           (infrastructure-to-infrastructure)
        - vehicle ↔ vehicle       (if within distance threshold)

    No edges between non-target pedestrians/vehicles and infrastructure
    (consistent with STRR's design).

    Parameters
    ----------
    max_distance : float
        Maximum spatial distance for creating edges (in pixel or metres).
    max_neighbors : int
        Maximum number of neighbours per node.
    """

    def __init__(
        self,
        max_distance: float = 30.0,
        max_neighbors: int = 16,
    ):
        self.max_distance = max_distance
        self.max_neighbors = max_neighbors

        # Node type constants
        self.CORE = 0
        self.VEHICLE = 1
        self.PERSON = 2
        self.INFRA = 3

    def build(
        self,
        positions: np.ndarray,        # (N, 2)  centre positions
        class_names: List[str],       # [N]      traffic class names
        target_idx: int = 0,          # index of target pedestrian node
    ) -> Tuple[Tensor, List[int], Tensor]:
        """
        Build edge_index, node type labels, and edge type labels.

        Edge types:
            0 = Core ↔ Vehicle
            1 = Core ↔ Person (other pedestrian)
            2 = Core ↔ Infra (traffic light)
            3 = Vehicle ↔ Vehicle
            4 = Infra ↔ Infra
            5 = Person ↔ Person

        Returns
        -------
        edge_index : Tensor (2, E)
        node_types : list of int per node
        edge_types : Tensor (E,) long — per-edge type code
        """
        N = len(class_names)
        node_types = [self._get_node_type(cn) for cn in class_names]

        edges_src, edges_dst, edge_type_list = [], [], []

        # --- 1. Core ↔ Vehicle (type 0) ---
        for i in range(N):
            if i == target_idx:
                continue
            if node_types[i] == self.VEHICLE:
                edges_src.append(target_idx); edges_dst.append(i)
                edge_type_list.append(0)
                edges_src.append(i); edges_dst.append(target_idx)
                edge_type_list.append(0)

        # --- 2. Core ↔ Person (type 1) ---
        for i in range(N):
            if i == target_idx:
                continue
            if node_types[i] == self.PERSON:
                edges_src.append(target_idx); edges_dst.append(i)
                edge_type_list.append(1)
                edges_src.append(i); edges_dst.append(target_idx)
                edge_type_list.append(1)

        # --- 3. Core ↔ Infra (type 2) ---
        for i in range(N):
            if i == target_idx:
                continue
            if node_types[i] == self.INFRA:
                edges_src.append(target_idx); edges_dst.append(i)
                edge_type_list.append(2)
                edges_src.append(i); edges_dst.append(target_idx)
                edge_type_list.append(2)

        # --- 4. Vehicle ↔ Vehicle (type 3) ---
        veh_indices = [i for i, t in enumerate(node_types) if t == self.VEHICLE]
        for i in range(len(veh_indices)):
            for j in range(i + 1, len(veh_indices)):
                u, v = veh_indices[i], veh_indices[j]
                dist = np.linalg.norm(positions[u] - positions[v])
                if dist < self.max_distance:
                    edges_src.append(u); edges_dst.append(v)
                    edge_type_list.append(3)
                    edges_src.append(v); edges_dst.append(u)
                    edge_type_list.append(3)

        # --- 5. Infra ↔ Infra (type 4) ---
        infra_indices = [i for i, t in enumerate(node_types) if t == self.INFRA]
        for i in range(len(infra_indices)):
            for j in range(i + 1, len(infra_indices)):
                u, v = infra_indices[i], infra_indices[j]
                edges_src.append(u); edges_dst.append(v)
                edge_type_list.append(4)
                edges_src.append(v); edges_dst.append(u)
                edge_type_list.append(4)

        # --- 6. Person ↔ Person (type 5) ---
        person_indices = [i for i, t in enumerate(node_types)
                          if t == self.PERSON and i != target_idx]
        for i in range(len(person_indices)):
            for j in range(i + 1, len(person_indices)):
                u, v = person_indices[i], person_indices[j]
                dist = np.linalg.norm(positions[u] - positions[v])
                if dist < self.max_distance:
                    edges_src.append(u); edges_dst.append(v)
                    edge_type_list.append(5)
                    edges_src.append(v); edges_dst.append(u)
                    edge_type_list.append(5)

        if not edges_src:
            return (
                torch.empty(2, 0, dtype=torch.long),
                node_types,
                torch.empty(0, dtype=torch.long),
            )

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_types = torch.tensor(edge_type_list, dtype=torch.long)
        return edge_index, node_types, edge_types

    def _get_node_type(self, class_name: str) -> int:
        if class_name == "pedestrian":
            return self.PERSON   # (will be overridden for target)
        elif class_name in ("bicycle", "motorcycle", "car", "bus", "truck"):
            return self.VEHICLE
        elif class_name in ("traffic_light", "traffic_sign", "lane_line"):
            return self.INFRA
        else:
            return self.INFRA  # fallback


# ======================================================================
# Main Perception Graph Network
# ======================================================================

class TrafficPerceptionGraph(nn.Module):
    """
    Traffic perception graph with node encoding + edge-weighted GAT.

    Pipeline:
        1. Encode heterogeneous node features  → common dim
        2. Compute STRR-style edge weights
        3. Compute edge features (relative position, direction, speed)
        4. 2-layer edge-weighted GAT → node embeddings

    Parameters
    ----------
    node_feat_dim : int
        Common node feature dimension after projection.
    gat_hidden_dim : int
        GAT hidden dimension.
    gat_heads : int
        GAT attention heads.
    spatial_dim : int, default 8
        STRR spatial encoding dimension.
    """

    def __init__(
        self,
        node_feat_dim: int = 128,
        gat_hidden_dim: int = 64,
        gat_out_dim: int = 128,
        gat_heads: int = 4,
        spatial_dim: int = 8,
        dropout: float = 0.1,
        max_distance: float = 30.0,
        use_ped_gru: bool = True,
    ):
        super().__init__()

        # Node encoder
        self.node_encoder = NodeFeatureEncoder(output_dim=node_feat_dim)

        # Graph builder (not a nn.Module — no params)
        self.graph_builder = PerceptionGraphBuilder(
            max_distance=max_distance,
        )

        # STRR edge weight
        self.edge_weight_calc = STREdgeWeight(
            feat_dim=node_feat_dim,
            spatial_dim=spatial_dim,
            hidden_dim=64,
        )

        # Edge feature encoder
        self.edge_feat_encoder = EdgeFeatureEncoder(max_distance=max_distance)

        # Relative spatial encoder for edge weight inputs (STRR style)
        self.rel_spatial_encoder = RelativeSpatialEncoder()

        # 2-layer GAT
        self.gat = EdgeWeightedGAT(
            in_dim=node_feat_dim,
            hidden_dim=gat_hidden_dim,
            out_dim=gat_out_dim,
            heads=gat_heads,
            dropout=dropout,
            use_edge_features=True,
        )

        # Spatial encoder (standalone, for node features)
        self.spatial_encoder = SpatialEncoder()

        # STR-style pedestrian GRU after each GAT layer
        self.use_ped_gru = use_ped_gru
        if use_ped_gru:
            self.ped_gru = nn.GRU(
                node_feat_dim, node_feat_dim,
                num_layers=2, batch_first=True,
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        bboxes: Tensor,                      # (N, 4)
        class_names: List[str],              # [N]
        positions: Tensor,                   # (N, 2)  centre coords
        velocities: Optional[Tensor] = None, # (N, 2)
        appearance_features: Optional[Tensor] = None,  # (N, 256) 行人外观
        traffic_light_states: Optional[Tensor] = None, # (N_tl, 4)
        target_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        node_embeddings : Tensor (N, gat_out_dim)
            Updated node embeddings after GAT reasoning.
        target_embedding : Tensor (gat_out_dim,)
            Embedding of the target pedestrian node.
        """
        N = bboxes.size(0)
        device = bboxes.device

        if N == 0:
            empty = torch.empty(0, self.gat.conv2.out_dim, device=device)
            return empty, torch.zeros(self.gat.conv2.out_dim, device=device)

        # --- 1. Encode nodes ---
        node_feats = self.node_encoder(
            bboxes=bboxes,
            class_names=class_names,
            positions=positions,
            velocities=velocities,
            appearance_features=appearance_features,
            traffic_light_states=traffic_light_states,
            device=str(device),
        )  # (N, node_feat_dim)

        # --- 2. Build graph ---
        pos_np = positions.detach().cpu().numpy()
        edge_index, node_types, edge_types = self.graph_builder.build(
            positions=pos_np,
            class_names=class_names,
            target_idx=target_idx,
        )
        edge_index = edge_index.to(device)
        edge_types = edge_types.to(device)

        if edge_index.numel() == 0:
            # No edges — return raw encoded features
            target_emb = node_feats[target_idx]
            return node_feats, target_emb

        # --- 3. Compute edge weights (STRR-style with relative spatial) ---
        src, dst = edge_index[0], edge_index[1]

        rel_spatial = self.rel_spatial_encoder(
            bbox_src=bboxes[src],
            bbox_dst=bboxes[dst],
        )  # (E, 8)

        edge_weight = self.edge_weight_calc(
            h_src=node_feats[src],
            h_dst=node_feats[dst],
            spatial_src=rel_spatial,
            spatial_dst=rel_spatial,
        )  # (E,)

        # --- 4. Compute per-edge-type features ---
        # Prepare traffic light states per edge
        tl_edge_states = None
        if traffic_light_states is not None and traffic_light_states.shape[0] > 0:
            # Build mapping: node_index → position in traffic_light_states
            tl_node_to_idx = {}
            tl_pos = 0
            for n, cn in enumerate(class_names):
                if cn == "traffic_light" and tl_pos < traffic_light_states.shape[0]:
                    tl_node_to_idx[n] = tl_pos
                    tl_pos += 1

            if tl_node_to_idx:
                tl_edge_states = torch.zeros(len(src), 4, device=device)
                for e in range(len(src)):
                    src_node, dst_node = int(src[e]), int(dst[e])
                    for n in (src_node, dst_node):
                        if n in tl_node_to_idx:
                            tl_edge_states[e] = traffic_light_states[tl_node_to_idx[n]]
                            break

        # Prepare vehicle speeds for Core↔Vehicle edges (type 0)
        veh_speeds_edge = None
        if velocities is not None:
            veh_speeds_edge = torch.norm(velocities[dst], dim=-1)  # (E,)

        edge_attr = self.edge_feat_encoder(
            pos_src=positions[src],
            pos_dst=positions[dst],
            edge_types=edge_types,
            vel_src=velocities[src] if velocities is not None else None,
            vel_dst=velocities[dst] if velocities is not None else None,
            tl_states=tl_edge_states,
            vehicle_speeds=veh_speeds_edge,
        )  # (E, output_dim)

        # --- 5. GAT message passing ---
        node_embeddings = self.gat(
            x=node_feats,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
        )  # (N, gat_out_dim)

        # --- 6. Pedestrian GRU (STRR style): update target node temporally ---
        safe_idx = min(target_idx, node_embeddings.size(0) - 1) if node_embeddings.size(0) > 0 else 0
        if self.use_ped_gru and target_idx < node_embeddings.size(0):
            target_emb = node_embeddings[safe_idx:safe_idx + 1].unsqueeze(0)  # (1, 1, D)
            target_emb, _ = self.ped_gru(target_emb)
            target_emb = target_emb.squeeze(0).squeeze(0)  # (D,)
            node_embeddings = node_embeddings.clone()
            node_embeddings[safe_idx] = target_emb
        else:
            target_emb = node_embeddings[safe_idx]

        return node_embeddings, target_emb


# ======================================================================
# Temporal Graph — stacks perception graphs over T frames
# ======================================================================

class TemporalPerceptionGraph(nn.Module):
    """
    Processes a temporal window of perception graphs.

    For each frame t in the observation window:
        1. Build perception graph
        2. Apply GAT → per-frame embeddings
        3. Stack temporally

    This provides the input for the GRU-based temporal encoder.

    Parameters
    ----------
    obs_len : int
        Number of observation frames.
    **graph_kwargs
        Arguments for TrafficPerceptionGraph.
    """

    def __init__(
        self,
        obs_len: int = 8,
        **graph_kwargs,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.perception_graph = TrafficPerceptionGraph(**graph_kwargs)

    def forward(
        self,
        bboxes_seq: Tensor,                   # (T, N, 4)
        class_names_seq: List[List[str]],     # [T][N]
        positions_seq: Tensor,                # (T, N, 2)
        velocities_seq: Optional[Tensor] = None,  # (T, N, 2)
        target_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        temporal_embeddings : Tensor (T, N, gat_out_dim)
            Per-frame node embeddings.
        target_embeddings : Tensor (T, gat_out_dim)
            Per-frame target node embeddings.
        """
        T = bboxes_seq.size(0)
        device = bboxes_seq.device

        temporal_nodes = []
        temporal_target = []

        for t in range(min(T, self.obs_len)):
            vel_t = velocities_seq[t] if velocities_seq is not None else None
            node_emb, target_emb = self.perception_graph(
                bboxes=bboxes_seq[t],
                class_names=class_names_seq[t],
                positions=positions_seq[t],
                velocities=vel_t,
                target_idx=target_idx,
            )
            temporal_nodes.append(node_emb)
            temporal_target.append(target_emb)

        # Pad if fewer frames than obs_len
        while len(temporal_nodes) < self.obs_len:
            temporal_nodes.append(torch.zeros_like(temporal_nodes[0]))
            temporal_target.append(torch.zeros_like(temporal_target[0]))

        return (
            torch.stack(temporal_nodes, dim=0),    # (T, N, D)
            torch.stack(temporal_target, dim=0),   # (T, D)
        )
