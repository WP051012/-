"""
Baseline models for comparison experiments.

Trajectory prediction baselines:
    SocialLSTM     — LSTM with social pooling
    SocialSTGCNN   — Spatiotemporal graph CNN
    FlowChainBase  — Vanilla FlowChain without traffic perception
    OurMethod      — Full pipeline (perception graph + GAT + CM-GRU + FlowChain)

Classification baselines:
    LSTMClassifier — LSTM over trajectory → violation probability
    GRUClassifier  — GRU over trajectory → violation probability
    STRRClassifier — STRR-style graph reasoning → violation probability
"""
from .baseline_models import (
    SocialLSTM,
    SocialSTGCNN,
    FlowChainBase,
    LSTMClassifier,
    GRUClassifier,
    STRRClassifier,
    OurMethodWrapper,
)
from .ablation import (
    AblationNoGraph,
    AblationNoGRU,
    AblationNoFlowChain,
)
