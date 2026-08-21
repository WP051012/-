"""
Edge-aware GAT convolution for traffic perception graph.

Extends standard GAT to incorporate:
    1. Learned edge weights (STRR-style inner-product weights)
    2. Edge features in attention computation (edge-aware attention)
    3. Edge features in message passing (edge-conditioned messages)

GAT with edge features (this version):
    α_ij = softmax_j(LeakyReLU(a_src·W·h_i + a_dst·W·h_j + a_edge·W_edge·e_ij))
    msg_ij = α_ij · (W·h_j + W_e·e_ij)

Backward compatible: if edge_attr is None, falls back to standard GAT.

References:
    STRR: Spatiotemporal Relationship Reasoning (edge weights via inner product)
    GAT: Graph Attention Networks (Velickovic et al., ICLR 2018)
    EGAT: Edge-featured Graph Attention Networks (Chen et al., 2021)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Edge-Weighted GAT Layer
# ======================================================================

class EdgeWeightedGATLayer(nn.Module):
    """
    Edge-aware GAT layer with edge features in attention AND message.

    Attention (edge-aware):
        α_ij = softmax_j( LeakyReLU( a_src·W·h_i + a_dst·W·h_j + a_edge·(W_e·e_ij) ) )
        Then scaled by optional STRR edge weight w_ij.

    Message (edge-conditioned):
        msg_ij = α_ij · ( W·h_j + W_msg·e_ij )

    Backward compatible: if edge_attr=None, falls back to standard GAT.

    Parameters
    ----------
    in_dim : int
    out_dim : int
    heads : int
        Number of attention heads.
    dropout : float
    alpha : float
        LeakyReLU negative slope.
    concat : bool
        If True, concatenate heads. If False, average them.
    use_edge_features : bool
        If True, edge features participate in attention AND message.
    edge_feat_dim : int
        Dimension of edge features (only used if use_edge_features=True).
    edge_in_message : bool
        If True, edge features are also added to source node messages.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
        use_edge_features: bool = False,
        edge_feat_dim: int = 4,
        edge_in_message: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.use_edge_features = use_edge_features
        self.edge_in_message = edge_in_message and use_edge_features

        # Linear transform for node features
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))

        # Edge feature in attention: project edge → (heads * out_dim), attend with att_edge
        if use_edge_features:
            self.edge_att_proj = nn.Linear(edge_feat_dim, heads * out_dim, bias=False)
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_dim))

        # Edge feature in message: W_e · e_ij added to source node message
        if self.edge_in_message:
            self.W_e = nn.Linear(edge_feat_dim, heads * out_dim, bias=False)

        # Edge weight projection (integrates STRR-computed weights)
        self.edge_weight_scale = nn.Parameter(torch.ones(1))

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.att_src, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.att_dst, gain=math.sqrt(2))
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_att_proj.weight)
            nn.init.xavier_uniform_(self.att_edge, gain=math.sqrt(2))
        if self.edge_in_message:
            nn.init.xavier_uniform_(self.W_e.weight, gain=math.sqrt(2))

    def forward(
        self,
        x: Tensor,                        # (N, in_dim)  node features
        edge_index: Tensor,               # (2, E)  (src, dst) indices
        edge_weight: Optional[Tensor] = None,  # (E,)  STRR-style edge weights
        edge_attr: Optional[Tensor] = None,    # (E, edge_feat_dim)  edge features
    ) -> Tensor:
        """
        Edge-aware GAT forward pass.

        - Edge features in attention: a_edge · (W_edge · e_ij) added to attention scores
        - Edge features in message: W_e · e_ij added to source node message
        - Backward compatible: edge_attr=None → standard GAT

        Returns
        -------
        Tensor (N, heads * out_dim) if concat=True else (N, out_dim)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # --- Linear projection ---
        Wh = self.W(x).view(N, self.heads, self.out_dim)  # (N, H, D')

        Wh_src = Wh[src]  # (E, H, D')
        Wh_dst = Wh[dst]  # (E, H, D')

        # --- Attention scores ---
        # Decompose: a^T [W h_src || W h_dst] = att_src·Wh_src + att_dst·Wh_dst
        e_src = (Wh_src * self.att_src).sum(dim=-1)  # (E, H)
        e_dst = (Wh_dst * self.att_dst).sum(dim=-1)  # (E, H)
        e = e_src + e_dst                               # (E, H)

        # --- Edge features in attention ---
        if self.use_edge_features and edge_attr is not None:
            # Project edge features to (heads * out_dim), then attend
            edge_emb = self.edge_att_proj(edge_attr).view(-1, self.heads, self.out_dim)  # (E, H, D')
            e_edge = (edge_emb * self.att_edge).sum(dim=-1)  # (E, H)
            e = e + e_edge

        e = F.leaky_relu(e, self.alpha)

        # --- Incorporate STRR-style edge weights ---
        if edge_weight is not None:
            w = edge_weight.unsqueeze(-1) * self.edge_weight_scale  # (E, 1)
            e = e * w

        # --- Softmax per destination node ---
        alpha = self._scatter_softmax(e, dst, N)  # (E, H)

        # --- Dropout on attention ---
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # --- Message: Wh_src + edge_in_message ---
        msg_src = Wh_src  # (E, H, D')
        if self.edge_in_message and edge_attr is not None:
            edge_msg = self.W_e(edge_attr).view(-1, self.heads, self.out_dim)  # (E, H, D')
            msg_src = msg_src + edge_msg

        msg = msg_src * alpha.unsqueeze(-1)  # (E, H, D')

        # --- Aggregate per destination ---
        out = torch.zeros(N, self.heads, self.out_dim,
                          device=x.device, dtype=x.dtype)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim)
        out = out.scatter_add(0, dst_expanded, msg)

        if self.concat:
            out = out.view(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        return out + self.bias

    @staticmethod
    def _scatter_softmax(
        scores: Tensor,     # (E, H)
        index: Tensor,      # (E,)
        num_nodes: int,
    ) -> Tensor:
        """Softmax over scores grouped by destination node index."""
        # Subtract max per group for numerical stability
        max_scores = torch.zeros(num_nodes, scores.size(1),
                                 device=scores.device, dtype=scores.dtype)
        max_scores = max_scores.scatter_reduce(
            0, index.unsqueeze(-1).expand_as(scores), scores,
            reduce="amax", include_self=False,
        )
        scores = scores - max_scores[index]

        exp_scores = torch.exp(scores)

        sum_exp = torch.zeros(num_nodes, scores.size(1),
                              device=scores.device, dtype=scores.dtype)
        sum_exp = sum_exp.scatter_add(
            0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores,
        )
        return exp_scores / (sum_exp[index] + 1e-6)


# ======================================================================
# Edge Feature Encoder
# ======================================================================

class EdgeFeatureEncoder(nn.Module):
    """
    Encode per-edge-type relationship features between traffic participants.

    Edge types and their features:
        type 0 — Core ↔ Vehicle:
            [rel_dist, rel_dir_x, rel_dir_y, speed_diff_ratio, vehicle_speed_norm]
        type 1 — Core ↔ Person (other pedestrian):
            [rel_dist, rel_dir_x, rel_dir_y, speed_diff_ratio]
        type 2 — Core ↔ Infra (traffic light):
            [rel_dist, rel_dir_x, rel_dir_y, tl_is_red, tl_is_green, tl_is_yellow]
        type 3 — Vehicle ↔ Vehicle:
            [rel_dist, rel_dir_x, rel_dir_y, speed_diff_ratio]
        type 4 — Infra ↔ Infra:
            [rel_dist, rel_dir_x, rel_dir_y]
        type 5 — Person ↔ Person:
            [rel_dist, rel_dir_x, rel_dir_y, speed_diff_ratio]

    All edge types are projected to a common `output_dim` (default 4 for GAT).

    References:
        Paper Section 2(4): 边特征 — 按节点类型使用不同特征组合
    """

    # Max raw feature dim per edge type
    RAW_DIMS = {0: 5, 1: 4, 2: 6, 3: 4, 4: 3, 5: 4}

    def __init__(self, max_distance: float = 30.0, output_dim: int = 4,
                 max_speed: float = 100.0):
        super().__init__()
        self.max_distance = max_distance
        self.max_speed = max_speed
        self.output_dim = output_dim

        # Per-edge-type projectors: raw_dim → output_dim
        self.projectors = nn.ModuleDict({
            str(t): nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(inplace=True),
            )
            for t, dim in self.RAW_DIMS.items()
        })

    def forward(
        self,
        pos_src: Tensor,       # (E, 2)
        pos_dst: Tensor,       # (E, 2)
        edge_types: Tensor,    # (E,) long — per-edge type code
        vel_src: Optional[Tensor] = None,              # (E, 2)
        vel_dst: Optional[Tensor] = None,              # (E, 2)
        tl_states: Optional[Tensor] = None,            # (E, 4) [r,g,y,remaining]
        vehicle_speeds: Optional[Tensor] = None,       # (E,)  dst node speed
    ) -> Tensor:
        """
        Compute per-edge-type features and project to common dimension.

        Returns
        -------
        Tensor (E, output_dim)
        """
        E = pos_src.shape[0]
        device = pos_src.device

        # --- Common spatial features for ALL edges ---
        delta = pos_dst - pos_src                              # (E, 2)
        rel_distance = torch.norm(delta, dim=-1, keepdim=True)  # (E, 1)
        rel_dist_norm = rel_distance / self.max_distance        # (E, 1)
        rel_dir = delta / (rel_distance + 1e-6)                # (E, 2)

        # --- Build raw features per edge type ---
        raw_feats = torch.zeros(E, self.output_dim, device=device)

        for etype in self.RAW_DIMS:
            mask = (edge_types == etype)  # (E,)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            feats = []

            # All types: spatial base
            feats.append(rel_dist_norm[idx])   # (n, 1)
            feats.append(rel_dir[idx])          # (n, 2)

            if etype in (0, 1, 3, 5):
                # Types with speed diff
                if vel_src is not None and vel_dst is not None:
                    s_src = torch.norm(vel_src[idx], dim=-1, keepdim=True)
                    s_dst = torch.norm(vel_dst[idx], dim=-1, keepdim=True)
                    sd = (s_dst - s_src) / (s_src + 1e-6)
                else:
                    sd = torch.zeros(len(idx), 1, device=device)
                feats.append(sd)                # (n, 1)

            if etype == 0:
                # Core ↔ Vehicle: add vehicle speed info
                if vehicle_speeds is not None:
                    vs = vehicle_speeds[idx].unsqueeze(-1) / self.max_speed
                else:
                    vs = torch.zeros(len(idx), 1, device=device)
                feats.append(vs)                # (n, 1)

            if etype == 2:
                # Core ↔ Infra (traffic light): add TL state
                if tl_states is not None:
                    tl = tl_states[idx]         # (n, 4)  [r, g, y, remaining]
                else:
                    tl = torch.zeros(len(idx), 3, device=device)
                feats.append(tl[:, :3])         # (n, 3)  r, g, y channels

            # Concatenate and project
            raw = torch.cat(feats, dim=-1)              # (n, raw_dim)
            proj = self.projectors[str(etype)](raw)     # (n, output_dim)
            # Under AMP autocast proj may be FP16 while raw_feats is FP32
            if raw_feats.dtype != proj.dtype:
                raw_feats = raw_feats.to(proj.dtype)
            raw_feats[idx] = proj

        return raw_feats


# ======================================================================
# STRR-style Edge Weight Calculator
# ======================================================================

class STREdgeWeight(nn.Module):
    """
    STRR-style edge weight computation.

    For each pair (i, j):
        embed_i = ReLU( W_src [h_i || spatial_i] )
        embed_j = ReLU( W_dst [h_j || spatial_j] )
        w_ij   = σ( embed_i · embed_j / sqrt(d) )

    where σ = sigmoid and d = hidden_dim.

    This implements the cross-inner-product edge weight from STRR:
    separately embed source and destination nodes, then compute
    sigmoid of their inner-product similarity.

    Reference:
        STRR model_graph.py getA():  sigmoid(embed_ped · embed_objs)
        Paper Section 2(5): 边权重 — 拼接后内积 + sigmoid

    Parameters
    ----------
    feat_dim : int
        Node feature dimension.
    spatial_dim : int
        Spatial encoding dimension (default 8).
    hidden_dim : int
        Hidden dimension for the embedding.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        spatial_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        src_input_dim = feat_dim + spatial_dim
        dst_input_dim = feat_dim + spatial_dim

        # Separate embeddings for source and destination (STRR style)
        self.src_embed = nn.Sequential(
            nn.Linear(src_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dst_embed = nn.Sequential(
            nn.Linear(dst_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        h_src: Tensor,          # (E, feat_dim)
        h_dst: Tensor,          # (E, feat_dim)
        spatial_src: Tensor,    # (E, spatial_dim)
        spatial_dst: Tensor,    # (E, spatial_dim)
    ) -> Tensor:
        """
        Compute STRR-style edge weights via cross-inner-product.

        Returns
        -------
        Tensor (E,)  — edge weights in (0, 1).
        """
        # Embed source and destination separately
        embed_src = self.src_embed(
            torch.cat([h_src, spatial_src], dim=-1)
        )  # (E, hidden_dim)
        embed_dst = self.dst_embed(
            torch.cat([h_dst, spatial_dst], dim=-1)
        )  # (E, hidden_dim)

        # Cross-inner-product similarity (STRR style)
        # w_ij = σ(embed_i · embed_j / sqrt(d))
        similarity = (embed_src * embed_dst).sum(dim=-1)  # (E,)

        # Sigmoid to (0, 1)
        weight = torch.sigmoid(similarity / math.sqrt(self.hidden_dim))
        return weight


# ======================================================================
# Relative Spatial Encoder (STRR-compatible 8-dim)
# ======================================================================

class RelativeSpatialEncoder(nn.Module):
    """
    STRR-style RELATIVE spatial encoding between two bounding boxes.

    Encodes pairwise spatial relationship as 8 dimensions:
        [dxmin, dymin, dxmax, dymax, dxc, dyc, w_obj, h_obj]

    where:
        dxmin = |x1_ped - x1_obj|  (top-left x difference)
        dymin = |y1_ped - y1_obj|  (top-left y difference)
        dxmax = |x2_ped - x2_obj|  (bottom-right x difference)
        dymax = |y2_ped - y2_obj|  (bottom-right y difference)
        dxc   = |cx_ped - cx_obj|  (center x difference)
        dyc   = |cy_ped - cy_obj|  (center y difference)
        w_obj = obj width
        h_obj = obj height

    All values normalised by image dimensions.

    Reference:
        STRR model_graph.py helper_get_pos_vec()
    """

    def __init__(self, img_width: float = 3840.0, img_height: float = 2160.0):
        super().__init__()
        self.img_w = img_width
        self.img_h = img_height

    def forward(
        self,
        bbox_src: Tensor,   # (..., 4)  [x1, y1, x2, y2]  pedestrian/target
        bbox_dst: Tensor,   # (..., 4)  [x1, y1, x2, y2]  context object
    ) -> Tensor:
        """
        Compute relative spatial encoding.

        Returns
        -------
        Tensor (..., 8)
        """
        # Normalise
        src = bbox_src / torch.tensor(
            [self.img_w, self.img_h, self.img_w, self.img_h],
            device=bbox_src.device, dtype=bbox_src.dtype,
        )
        dst = bbox_dst / torch.tensor(
            [self.img_w, self.img_h, self.img_w, self.img_h],
            device=bbox_dst.device, dtype=bbox_dst.dtype,
        )

        # Top-left differences
        dxmin = (src[..., 0] - dst[..., 0]).abs()
        dymin = (src[..., 1] - dst[..., 1]).abs()

        # Bottom-right differences
        dxmax = (src[..., 2] - dst[..., 2]).abs()
        dymax = (src[..., 3] - dst[..., 3]).abs()

        # Center differences
        cx_src = (src[..., 0] + src[..., 2]) / 2
        cy_src = (src[..., 1] + src[..., 3]) / 2
        cx_dst = (dst[..., 0] + dst[..., 2]) / 2
        cy_dst = (dst[..., 1] + dst[..., 3]) / 2
        dxc = (cx_src - cx_dst).abs()
        dyc = (cy_src - cy_dst).abs()

        # Object size
        w_obj = dst[..., 2] - dst[..., 0]
        h_obj = dst[..., 3] - dst[..., 1]

        return torch.stack(
            [dxmin, dymin, dxmax, dymax, dxc, dyc, w_obj, h_obj], dim=-1,
        )


# ======================================================================
# Two-layer Edge-Weighted GAT (as used in the paper)
# ======================================================================

class EdgeWeightedGAT(nn.Module):
    """
    Two-layer edge-aware GAT as described in the paper.

    Edge features participate in both attention computation and message
    passing (edge-conditioned messages).

    Backward compatible: edge_attr=None → standard GAT without edge features.

    Parameters
    ----------
    in_dim : int
    hidden_dim : int
    out_dim : int
    heads : int
    dropout : float
    use_edge_features : bool
    edge_feat_dim : int
    edge_in_message : bool
        Whether edge features are added to messages (W_e · e_ij).
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_feat_dim: int = 4,
        edge_in_message: bool = True,
    ):
        super().__init__()

        self.conv1 = EdgeWeightedGATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim,
            edge_in_message=edge_in_message,
        )

        self.conv2 = EdgeWeightedGATLayer(
            in_dim=hidden_dim * heads,
            out_dim=out_dim,
            heads=1,                     # single head for final output
            dropout=dropout,
            concat=False,                # average (only 1 head anyway)
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim,
            edge_in_message=edge_in_message,
        )

        # Residual projection (if in_dim != out_dim)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, in_dim)
        edge_index : Tensor (2, E)
        edge_weight : Tensor (E,) — STRR-style weights
        edge_attr : Tensor (E, 4) — edge features

        Returns
        -------
        Tensor (N, out_dim)
        """
        # --- Layer 1 ---
        h1 = self.conv1(x, edge_index, edge_weight, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # --- Layer 2 ---
        h2 = self.conv2(h1, edge_index, edge_weight, edge_attr)
        h2 = self.norm2(h2)

        # --- Residual ---
        residual = self.residual(x)
        h2 = h2 + residual
        h2 = F.relu(h2)

        return h2


# ======================================================================
# Enhanced Traffic Edge Encoder
# ======================================================================

class TrafficEdgeEncoder(nn.Module):
    """
    Enhanced edge feature encoder with rich per-edge-type semantic features.

    Edge types and their features:
        type 0 — Pedestrian ↔ Vehicle:
            [rel_dist, rel_dir(2), speed_diff, rel_speed, ttc, veh_speed]
            → 7 raw features
        type 1 — Pedestrian ↔ Person (other pedestrian):
            [rel_dist, rel_dir(2), speed_diff]
            → 4 raw features
        type 2 — Pedestrian ↔ Infra (traffic light):
            [rel_dist, rel_dir(2), tl_color(3), tl_remaining]
            → 7 raw features
        type 3 — Vehicle ↔ Vehicle:
            [rel_dist, rel_dir(2), speed_diff, ttc]
            → 5 raw features
        type 4 — Infra ↔ Infra:
            [rel_dist, rel_dir(2)]
            → 3 raw features
        type 5 — Person ↔ Person:
            [rel_dist, rel_dir(2), speed_diff]
            → 4 raw features

    Additional edge types for new agent pairs:
        type 6 — Vehicle ↔ Infra (traffic light / crosswalk):
            [rel_dist, rel_dir(2), veh_speed]
            → 4 raw features
        type 7 — Pedestrian ↔ Crosswalk:
            [rel_dist, rel_dir(2)]
            → 3 raw features

    All edge types are projected to a common ``output_dim`` via per-type MLPs.

    References:
        Paper Section 2 (revised): 边特征 — 交通语义边特征编码
    """

    # Max raw feature dim per edge type
    RAW_DIMS = {0: 7, 1: 4, 2: 7, 3: 5, 4: 3, 5: 4, 6: 4, 7: 3}

    def __init__(
        self,
        max_distance: float = 30.0,
        output_dim: int = 8,
        max_speed: float = 100.0,
        ttc_max: float = 10.0,
    ):
        super().__init__()
        self.max_distance = max_distance
        self.max_speed = max_speed
        self.ttc_max = ttc_max
        self.output_dim = output_dim

        # Per-edge-type projectors: raw_dim → output_dim
        self.projectors = nn.ModuleDict({
            str(t): nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim),
            )
            for t, dim in self.RAW_DIMS.items()
        })

    def forward(
        self,
        pos_src: Tensor,             # (E, 2)
        pos_dst: Tensor,             # (E, 2)
        edge_types: Tensor,          # (E,) long — per-edge type code
        vel_src: Optional[Tensor] = None,        # (E, 2)
        vel_dst: Optional[Tensor] = None,        # (E, 2)
        tl_states: Optional[Tensor] = None,      # (E, 4) [r,g,y,remaining]
        vehicle_speeds: Optional[Tensor] = None, # (E,)  dst node speed norm
    ) -> Tensor:
        """
        Compute per-edge-type traffic-semantic features and project.

        Returns
        -------
        Tensor (E, output_dim)
        """
        E = pos_src.shape[0]
        device = pos_src.device

        # --- Common spatial features for ALL edges ---
        delta = pos_dst - pos_src                              # (E, 2)
        rel_distance = torch.norm(delta, dim=-1, keepdim=True) # (E, 1)
        rel_dist_norm = rel_distance / self.max_distance       # (E, 1)
        rel_dir = delta / (rel_distance + 1e-6)                # (E, 2)

        # --- Speed-related features ---
        s_src = torch.norm(vel_src, dim=-1, keepdim=True) if vel_src is not None else None  # (E, 1)
        s_dst = torch.norm(vel_dst, dim=-1, keepdim=True) if vel_dst is not None else None  # (E, 1)

        # --- Build raw features per edge type ---
        raw_feats = torch.zeros(E, self.output_dim, device=device)

        for etype in self.RAW_DIMS:
            mask = (edge_types == etype)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            feats = []

            # All types: spatial base (3 features)
            feats.append(rel_dist_norm[idx])   # (n, 1)
            feats.append(rel_dir[idx])          # (n, 2)

            if etype in (0, 1, 3, 5):
                # --- Speed difference ---
                if s_src is not None and s_dst is not None:
                    sd = (s_dst[idx] - s_src[idx]) / (s_src[idx] + 1e-6)
                else:
                    sd = torch.zeros(len(idx), 1, device=device)
                feats.append(sd)  # (n, 1)

            if etype == 0:
                # Pedestrian ↔ Vehicle: additional speed + TTC
                if s_dst is not None:
                    vs = s_dst[idx] / self.max_speed  # (n, 1)
                else:
                    vs = torch.zeros(len(idx), 1, device=device)
                feats.append(vs)  # (n, 1) rel_speed

                # TTC (Time-To-Collision): distance / relative speed along line-of-sight
                if s_src is not None and s_dst is not None:
                    # Relative velocity projected onto line-of-sight direction
                    delta_idx = delta[idx]                       # (n, 2)
                    rel_vel = vel_dst[idx] - vel_src[idx] if vel_dst is not None and vel_src is not None \
                        else torch.zeros(len(idx), 2, device=device)
                    los_speed = (rel_vel * delta_idx / (rel_distance[idx] + 1e-6)).sum(dim=-1, keepdim=True)
                    # TTC = distance / closing_speed, clamped
                    ttc = rel_distance[idx] / (los_speed.abs() + 1e-6)
                    ttc = torch.clamp(ttc / self.ttc_max, 0.0, 1.0)
                else:
                    ttc = torch.zeros(len(idx), 1, device=device)
                feats.append(ttc)  # (n, 1)

                # Vehicle speed norm
                if s_dst is not None:
                    vs_norm = s_dst[idx] / self.max_speed
                else:
                    vs_norm = torch.zeros(len(idx), 1, device=device)
                feats.append(vs_norm)  # (n, 1)

            if etype == 2:
                # Pedestrian ↔ Traffic Light: color + remaining time
                if tl_states is not None:
                    tl = tl_states[idx]  # (n, 4) [r, g, y, remaining]
                    feats.append(tl[:, :3])      # (n, 3) color
                    feats.append(tl[:, 3:4])     # (n, 1) remaining time
                else:
                    feats.append(torch.zeros(len(idx), 3, device=device))
                    feats.append(torch.zeros(len(idx), 1, device=device))

            if etype == 3:
                # Vehicle ↔ Vehicle: TTC
                if s_src is not None and s_dst is not None:
                    delta_idx = delta[idx]
                    rel_vel = vel_dst[idx] - vel_src[idx] if vel_dst is not None and vel_src is not None \
                        else torch.zeros(len(idx), 2, device=device)
                    los_speed = (rel_vel * delta_idx / (rel_distance[idx] + 1e-6)).sum(dim=-1, keepdim=True)
                    ttc = rel_distance[idx] / (los_speed.abs() + 1e-6)
                    ttc = torch.clamp(ttc / self.ttc_max, 0.0, 1.0)
                else:
                    ttc = torch.zeros(len(idx), 1, device=device)
                feats.append(ttc)  # (n, 1)

            if etype == 6:
                # Vehicle ↔ Infra: vehicle speed
                if s_dst is not None:
                    vs = s_dst[idx] / self.max_speed
                else:
                    vs = torch.zeros(len(idx), 1, device=device)
                feats.append(vs)  # (n, 1)

            # Concatenate and project
            raw = torch.cat(feats, dim=-1)
            proj = self.projectors[str(etype)](raw)
            raw_feats[idx, :proj.shape[-1]] = proj

        return raw_feats[:, :self.output_dim]
