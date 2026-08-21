"""
Loss presets for the 3 modes and ablations A0–A5.

The full research loss is

    L = λ_pseudo·L_pseudo + λ_cvp·L_cvp_cycle + λ_cycle·L_cycle
        + λ_corr·L_corr + λ_temporal·L_temporal

The three *modes* and the ablations differ only in which λ terms are switched
on — the network graph is identical, nothing is deleted:

    geometry : no learned BEV network (homography projection only; eval only)
    yang     : CVP/CVT supervised by pseudo-BEV alone (λ_pseudo > 0, rest 0)
    proposed : all five terms active

Ablations (A0 = full proposed):
    a0 : full proposed
    a1 : w/o L_cycle        (no camera⇄BEV cycle)
    a2 : w/o L_cvp_cycle    (no CVP feature cycle)
    a3 : w/o L_corr         (no object-position correspondence)
    a4 : w/o L_temporal     (no temporal velocity consistency)
    a5 : w/o L_pseudo       (cycle + corr + temporal only — no direct pseudo)
"""

from __future__ import annotations

# All five terms active (proposed / A0).
PROPOSED_LOSS = {
    "pseudo_weight": 1.0,
    "pseudo_mode": "focal",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "dice_weight": 1.0,
    "cvp_cycle_weight": 1.0,
    "cycle_weight": 1.0,
    "cycle_dice_weight": 1.0,
    "corr_weight": 1.0,
    "temporal_weight": 1.0,
}

# Each ablation disables exactly one term (weight → 0).
ABLATION_DISABLED = {
    "a0": [],
    "a1": ["cycle"],
    "a2": ["cvp_cycle"],
    "a3": ["corr"],
    "a4": ["temporal"],
    "a5": ["pseudo"],
}

VALID_MODES = ("geometry", "yang", "proposed")
VALID_ABLATIONS = tuple(ABLATION_DISABLED.keys())


def resolve_loss_cfg(config: dict) -> dict:
    """Return the effective ``loss:`` block for a config.

    Reads ``mode`` and ``ablation`` from the config and applies the matching
    preset on top of any explicit ``loss:`` weights. A config value of ``None``
    for a weight falls back to the preset default.
    """
    mode = config.get("mode", "proposed")
    ablation = config.get("ablation", None) or "a0"
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode '{mode}' (expected one of {VALID_MODES})")
    if ablation not in VALID_ABLATIONS:
        raise ValueError(f"unknown ablation '{ablation}' (expected one of {VALID_ABLATIONS})")

    explicit = dict(config.get("loss", {}) or {})
    # Start from the proposed defaults, then overlay any explicit non-None values.
    base = PROPOSED_LOSS.copy()
    base.update({k: v for k, v in explicit.items() if v is not None})

    if mode == "yang":
        base["cvp_cycle_weight"] = 0.0
        base["cycle_weight"] = 0.0
        base["corr_weight"] = 0.0
        base["temporal_weight"] = 0.0

    for term in ABLATION_DISABLED.get(ablation, []):
        base[f"{term}_weight"] = 0.0

    return base


def mode_uses_network(mode: str) -> bool:
    """Whether a mode trains a learned BEV network (geometry does not)."""
    return mode != "geometry"
