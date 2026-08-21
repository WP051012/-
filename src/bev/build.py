"""
Config-driven construction helpers shared by the train/eval/inference scripts.

    load_config   : read a YAML config
    build_geometry : homography + BEV grid from a config
    build_model   : MonocularBEV (None for mode=geometry)
    build_loaders : DataLoaders over the BEV dataset
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.homography import homography_from_config
from src.geometry.coordinate import BEVGrid
from src.bev.monocular_bev import build_monocular_bev
from src.bev.losses import mode_uses_network
from data.bev_dataset import build_bev_datasets, bev_collate_fn

from torch.utils.data import DataLoader


_ENV_PAT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_string(s: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in a string from os.environ."""

    def _repl(m):
        name, default = m.group(1), m.group(2)
        val = os.environ.get(name)
        if val:
            return val
        return default if default is not None else ""

    return _ENV_PAT.sub(_repl, s)


def _expand_env(value):
    """Recursively expand ``${VAR}`` patterns in a parsed YAML object."""
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _expand_env(config)


def build_geometry(config):
    """Return (homography, BEVGrid) from a config dict."""
    homography = homography_from_config(config.get("homography", {}) or {})
    grid = BEVGrid.from_config(config["bev"])
    return homography, grid


def build_model(config, homography, grid):
    """Build a MonocularBEV, or None for mode=geometry (no learned network)."""
    if not mode_uses_network(config.get("mode", "proposed")):
        return None
    bev_cfg = config.get("data", {}).get("bev", {})
    return build_monocular_bev(
        config, homography, grid,
        mask_h=bev_cfg.get("mask_h"),
        mask_w=bev_cfg.get("mask_w"),
        img_h=int(bev_cfg.get("img_h", 2160)),
        img_w=int(bev_cfg.get("img_w", 3840)),
    )


def build_loaders(config, homography, grid, splits=("train", "val"), temporal=None):
    """Return {split: DataLoader}. temporal defaults from config (train only)."""
    bev_cfg = config.get("data", {}).get("bev", {})
    if temporal is None:
        temporal = bool(bev_cfg.get("temporal", False))
    # L_temporal is a train-only loss: val/test never consume image_prev /
    # pseudo_bev_prev, so only the "train" split should build previous-frame
    # targets. Pass a per-split spec instead of a single flag to avoid the
    # val loader doing 2x wasted target work.
    temporal_spec = {s: (bool(temporal) and s == "train") for s in splits}
    datasets = build_bev_datasets(config, homography, grid, temporal=temporal_spec,
                                  splits=splits)
    tr_cfg = config.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 0))
    loaders = {}
    for s in splits:
        if s not in datasets:
            continue
        loaders[s] = DataLoader(
            datasets[s],
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=num_workers,
            collate_fn=bev_collate_fn,
        )
    return loaders
