#!/usr/bin/env python3
"""
可视化 BEV 热力图
=================
渲染预测/伪 BEV 热力图（对象中心高斯）。输入为 inference_bev.py 生成的 npz
（含 pred_bev 与 pseudo_bev），或任意 (C,H,W) 热力图 npz。

    py tools/visualize_bev.py --npz output/bev_pred/{video}/frame_12.npz --out frame_12.png
    py tools/visualize_bev.py --dir output/bev_pred/{video}/ --out vis/   # 批量
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bev.pseudo_bev import BEV_CLASSES


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_heatmap(heatmap, classes, out_path, title=""):
    """Render a (C, H, W) heatmap stack to a PNG file."""
    plt = _plt()
    C = heatmap.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(4.5 * C, 4.5), squeeze=False)
    for c in range(C):
        ax = axes[0, c]
        im = ax.imshow(heatmap[c], cmap="jet", vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(classes[c] if c < len(classes) else f"ch{c}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="可视化 BEV 热力图")
    parser.add_argument("--npz", default=None, help="单个 npz 文件")
    parser.add_argument("--dir", default=None, help="批量：目录下所有 frame_*.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.dir:
        out_dir = Path(args.out)
        files = sorted(Path(args.dir).glob("frame_*.npz"))
        for f in files:
            data = np.load(f)
            pred = data.get("pred_bev", data.get("heatmap"))
            if pred is None:
                continue
            out = out_dir / f"{f.stem}.png"
            render_heatmap(pred, BEV_CLASSES, out, title=f.stem)
            print(f"  {out}")
        print(f"渲染 {len(files)} 帧 → {out_dir}")
        return

    if args.npz:
        data = np.load(args.npz)
        pred = data.get("pred_bev", data.get("heatmap"))
        if pred is None:
            raise SystemExit("npz 缺少 pred_bev/heatmap")
        render_heatmap(pred, BEV_CLASSES, args.out, title=Path(args.npz).stem)
        print(f"渲染完成 → {args.out}")
        return

    raise SystemExit("请指定 --npz 或 --dir")


if __name__ == "__main__":
    main()
