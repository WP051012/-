#!/usr/bin/env python3
"""
推理单目相机 → BEV 模型，逐帧保存 BEV 热力图
============================================
    py scripts/inference_bev.py --config configs/bev_proposed.yaml \
        --checkpoint checkpoints/bev/best.pt --output output/bev_pred/ --split test

每帧保存 `{output}/{video}/frame_{id}.npz`，含 keys:
    pred_bev   : (C, H_bev, W_bev) 预测热力图 [0,1]
    pseudo_bev : (C, H_bev, W_bev) 伪标签（弱监督参考）
可用 tools/visualize_bev.py 渲染。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bev.build import load_config, build_geometry, build_model, build_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="单目相机→BEV推理")
    parser.add_argument("--config", default="configs/bev_proposed.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="output/bev_pred/")
    parser.add_argument("--max-frames", type=int, default=0, help="0=全部")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    homography, grid = build_geometry(config)
    model = build_model(config, homography, grid).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])
    model.eval()

    loaders = build_loaders(config, homography, grid, splits=(args.split,), temporal=False)
    loader = loaders[args.split]

    out_root = Path(args.output)
    n_saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred = model(images)["pred_bev"].cpu().numpy()       # (B,C,H,W)
            pseudo = batch["pseudo_bev"].numpy()
            for b in range(pred.shape[0]):
                video = batch["video"][b]
                fid = batch["frame_id"][b]
                d = out_root / video
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    d / f"frame_{fid}.npz",
                    pred_bev=pred[b].astype(np.float32),
                    pseudo_bev=pseudo[b].astype(np.float32),
                )
                n_saved += 1
                if args.max_frames and n_saved >= args.max_frames:
                    break
            if args.max_frames and n_saved >= args.max_frames:
                break

    logger.info(f"Saved {n_saved} frames to {out_root}")


if __name__ == "__main__":
    main()
