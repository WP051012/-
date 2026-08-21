"""
Red-Light Running Classification Module
----------------------------------------
Converts trajectory distribution → violation probability → binary decision.

Classes:
    StopLine                     — geometric stop line
    JunctionRegion               — junction polygon
    RedLightViolationChecker     — per-trajectory violation check
    RedLightProbabilityEstimator — Monte Carlo probability estimator
    CounterfactualSimulator      — "what-if" simulation
"""

from .red_light_classifier import (
    StopLine,
    JunctionRegion,
    RedLightViolationChecker,
    RedLightProbabilityEstimator,
    CounterfactualSimulator,
)

from .crossing_probability import (
    CrossingProbabilityEstimator,
    compute_signal_factor,
    point_in_polygon,
)

from .risk_regression_head import (
    RiskRegressionHead,
    prepare_stage2_features,
    search_threshold_regression,
)
