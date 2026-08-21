#!/usr/bin/env python3
"""
评估单目相机 → BEV 模型
=======================
在测试集上比较模型预测与 pseudo_bev（弱监督参考，非 GT）。

    py scripts/evaluate_bev.py --config configs/bev_proposed.yaml --checkpoint checkpoints/bev/best.pt
    py scripts/evaluate_bev.py --config configs/bev_geometry.yaml    # 几何基线（恒等参考）

mode=geometry 时预测即 pseudo_bev（单应投影本身），作为弱监督参考上界。
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bev.build import load_config, build_geometry, build_model, build_loaders
from src.bev.metrics import compute_bev_metrics, activation_stats
from src.bev.losses import mode_uses_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(config, model, loader, device, resolution, mode):
    agg = {}
    act_sum = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = 0
    for batch in loader:
        pseudo = batch["pseudo_bev"].to(device)
        if mode == "geometry":
            pred = pseudo                                   # identity reference
        else:
            with torch.no_grad():
                pred = model(batch["image"].to(device))["pred_bev"]

        m = compute_bev_metrics(pred, pseudo, resolution)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + (v if v == v else 0.0)
        stats = activation_stats(pred)
        for k, v in stats.items():
            act_sum[k] += v
        n += 1

    metrics = {k: v / max(1, n) for k, v in agg.items()}
    act = {k: v / max(1, n) for k, v in act_sum.items()}
    return metrics, act


def main():
    parser = argparse.ArgumentParser(description="评估单目相机→BEV模型")
    parser.add_argument("--config", default="configs/bev_proposed.yaml")
    parser.add_argument("--checkpoint", default=None, help="模型检查点（learned 模式必填）")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = config.get("mode", "proposed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    homography, grid = build_geometry(config)
    model = build_model(config, homography, grid)

    if mode_uses_network(mode):
        if args.checkpoint is None:
            raise SystemExit("learned 模式需 --checkpoint 指定检查点")
        model = model.to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])
        model.eval()

    loaders = build_loaders(config, homography, grid, splits=(args.split,),
                            temporal=False)
    loader = loaders[args.split]
    logger.info(f"evaluating on '{args.split}' ({len(loader)} batches), mode={mode}")

    metrics, act = evaluate(config, model, loader, device,
                            float(config["bev"]["resolution"]), mode)

    print("\n================ BEV 评估结果（vs pseudo_bev，非 GT）================")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.4f}")
    print("  激活统计 (pred_bev): "
          f"mean={act['mean']:.4f} std={act['std']:.4f} "
          f"min={act['min']:.4f} max={act['max']:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
