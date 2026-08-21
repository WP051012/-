#!/usr/bin/env python3
"""
训练单目相机 → BEV 模型
======================
Geometry-guided weakly-supervised monocular BEV（Yang CVP/CVT 骨干 + 循环/时序一致性）。

用法:
    py scripts/train_bev.py --config configs/bev_proposed.yaml
    py scripts/train_bev.py --config configs/bev_yang.yaml
    py scripts/train_bev.py --config configs/bev_proposed.yaml --ablation a1

mode=geometry 无学习网络，请用 evaluate_bev.py。
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bev.build import load_config, build_geometry, build_model, build_loaders
from src.bev.trainer import BEVTrainer, set_seed
from src.bev.losses import mode_uses_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="训练单目相机→BEV模型")
    parser.add_argument("--config", default="configs/bev_proposed.yaml")
    parser.add_argument("--ablation", default=None,
                        help="覆盖 config 的 ablation 字段（a0..a5）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None, help="从检查点恢复")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.ablation is not None:
        config["ablation"] = args.ablation
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    mode = config.get("mode", "proposed")
    if not mode_uses_network(mode):
        raise SystemExit("mode='geometry' 无学习网络；请用 scripts/evaluate_bev.py")

    tr_cfg = config["training"]
    set_seed(int(tr_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, mode={mode}, ablation={config.get('ablation', 'a0')}")

    homography, grid = build_geometry(config)
    model = build_model(config, homography, grid).to(device)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    temporal = bool(config["data"]["bev"].get("temporal", False))
    loaders = build_loaders(config, homography, grid, splits=("train", "val"),
                            temporal=temporal)
    logger.info(f"train batches={len(loaders['train'])}, val batches={len(loaders['val'])}")

    trainer = BEVTrainer(model, config, device)
    if args.resume:
        trainer.load(args.resume)

    trainer.fit(
        loaders["train"], loaders["val"],
        epochs=int(tr_cfg.get("epochs", 50)),
        resolution=float(config["bev"]["resolution"]),
    )
    trainer.save(tr_cfg.get("checkpoint_dir", "checkpoints/bev/"), tag="last")


if __name__ == "__main__":
    main()
