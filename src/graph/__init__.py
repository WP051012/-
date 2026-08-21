"""
Traffic Perception Graph Module
--------------------------------
Heterogeneous spatiotemporal graph for traffic scene understanding.

Classes:
    SpatialEncoder          — 8-dim spatial position encoding
    MotionEncoder           — 6-dim motion feature encoding
    NodeFeatureEncoder      — multi-type node feature projector
    STREdgeWeight           — STRR-style edge weight via inner product
    EdgeFeatureEncoder      — pairwise relationship edge features
    EdgeWeightedGATLayer    — single GAT layer with edge weights
    EdgeWeightedGAT         — 2-layer edge-weighted GAT
    PerceptionGraphBuilder  — edge construction by node type
    TrafficPerceptionGraph  — full perception graph network
    TemporalPerceptionGraph — temporal stack of perception graphs
    SceneGraph              — simplified pedestrian/vehicle scene graph
"""

from .node_encoder import (
    SpatialEncoder,
    MotionEncoder,
    NodeFeatureEncoder,
    encode_node_from_trajectory,
)
from .gat_conv import (
    EdgeWeightedGATLayer,
    EdgeWeightedGAT,
    STREdgeWeight,
    EdgeFeatureEncoder,
    TrafficEdgeEncoder,
    RelativeSpatialEncoder,
)
from .perception_graph import (
    PerceptionGraphBuilder,
    TrafficPerceptionGraph,
    TemporalPerceptionGraph,
)
from .scene_graph import SceneGraph
