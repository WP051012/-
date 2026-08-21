"""
Trajectory Prediction Module
-----------------------------
Cognitive-state enhanced trajectory prediction.

Classes:
    CognitiveEnhancedGRU        — NEW: standard GRU with enriched input
    CognitiveEnhancedGRUNoContext — NEW: GRU without cognitive context
    PerceptionGRUCell           — [DEPRECATED] GRU cell with gate injection
    PerceptionGRU               — [DEPRECATED] perception-infused GRU
    PerceptionContextEncoder     — temporal perception context encoder
    StructuralChangeDetector     — rule-based structural change detection
    CognitiveStateChangeDetector — NEW: c-state change detection
    CognitiveConflictDetector   — NEW: memory conflict detection
    ConceptDriftDetector        — data-driven concept drift detection
    PerceptionChangeDetector    — unified change detector (REVISED)
    TransformerFlowChain        — official FlowChain (Transformer + RealNVP)
    RealNVP                     — MADE-style conditional normalizing flow
    FlowChainPredictor          — wrapper with standard (B,T,2) interface
    MeanScaler                  — adaptive per-trajectory scaler
    joint_nll_mse_loss          — joint NLL + MSE training loss
    flow_chain_nll_loss         — joint loss alias (backward compat)
"""

from .perception_gru import (
    CognitiveEnhancedGRU,
    CognitiveEnhancedGRUNoContext,
    PerceptionGRUCell,         # deprecated — kept for checkpoint compat
    PerceptionGRU,             # deprecated — kept for checkpoint compat
    PerceptionContextEncoder,
)
from .change_detector import (
    StructuralChangeDetector,
    CognitiveStateChangeDetector,
    CognitiveConflictDetector,
    ConceptDriftDetector,
    PerceptionChangeDetector,
    ChangeType,
    ChangeEvent,
)
from .flow_chain import (
    FiLMAffineCoupling,         # deprecated — kept for checkpoint compat
    FiLMConditionProjector,     # deprecated
    ConditionalRealNVP,         # deprecated
    FlowChainPredictor,
    flow_chain_nll_loss,
)
from .flow_chain_official import (
    TransformerFlowChain,
    RealNVP,
    Flow,
    MeanScaler,
    LinearMaskedCoupling,
    BatchNorm,
    FlowSequential,
    joint_nll_mse_loss,
    transformer_flow_nll_loss,
)
