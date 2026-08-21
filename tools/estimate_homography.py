#!/usr/bin/env python3
"""
估计单应矩阵（图像像素 → 地面米）
================================
单目固定相机的标定工具。三种用法:

1) 从点对应计算（推荐）:
      py tools/estimate_homography.py --points points.json --out homography.json --method ransac
   points.json 格式:
      {"correspondences": [{"pixel": [u, v], "ground": [X, Y]}, ...]}   # 至少 4 对

2) 交互式点击像素点（生成模板，地面坐标手工填）:
      py tools/estimate_homography.py --click frame.jpg --out points.json

3) 占位 planar_scale（无真实标定时的近似）:
      py tools/estimate_homography.py --planar-scale 47.5 --origin 1920,2160 --out homography.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.homography import compute_homography, Homography


def compute_from_points(points, method="ransac"):
    """points: list of {'pixel':[u,v], 'ground':[X,Y]} → Homography."""
    if len(points) < 4:
        raise ValueError("至少需要 4 对点对应来估计单应矩阵")
    src = np.array([p["pixel"] for p in points], dtype=np.float64)
    dst = np.array([p["ground"] for p in points], dtype=np.float64)
    H = compute_homography(src, dst, method)
    homo = Homography(H)
    ok = homo.validate_homography()
    err = homo.round_trip_error(src)
    return homo, ok, err


def interactive_click(image_path, out_json):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    pts = []
    canvas = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([float(x), float(y)])
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("click (ESC 结束)", canvas)

    cv2.namedWindow("click (ESC 结束)")
    cv2.setMouseCallback("click (ESC 结束)", on_mouse)
    while True:
        cv2.imshow("click (ESC 结束)", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    points = [{"pixel": p, "ground": [0.0, 0.0]} for p in pts]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"correspondences": points}, f, indent=2)
    print(f"已记录 {len(pts)} 个像素点 → {out_json}（请手工填写每个点的地面坐标 ground=[X,Y] 米）")


def main():
    parser = argparse.ArgumentParser(description="估计单应矩阵")
    parser.add_argument("--points", default=None, help="点对应 JSON")
    parser.add_argument("--method", default="ransac", choices=["dlt", "ransac"])
    parser.add_argument("--click", default=None, help="交互式点击的图像路径")
    parser.add_argument("--planar-scale", type=float, default=None, help="像素/米")
    parser.add_argument("--origin", default="1920,2160", help="原点像素 u,v（planar-scale）")
    parser.add_argument("--out", required=True, help="输出 homography .json/.npy/.txt")
    args = parser.parse_args()

    if args.click:
        interactive_click(args.click, args.out)
        return

    if args.planar_scale is not None:
        ou, ov = (float(x) for x in args.origin.split(","))
        ppm = args.planar_scale
        homo = Homography.from_planar_scale(meters_per_pixel=1.0 / ppm,
                                           origin_u=ou, origin_v=ov)
        print(f"planar_scale 单应矩阵（近似，{ppm} px/m, 原点=({ou},{ov})）")
    else:
        with open(args.points, "r") as f:
            data = json.load(f)
        points = data["correspondences"]
        homo, ok, err = compute_from_points(points, args.method)
        print(f"由 {len(points)} 对点计算（method={args.method}）")
        print(f"  校验: {'通过' if ok else '失败'}")
        print(f"  往返重投影误差: {err:.6f} px")

    homo.save(args.out)
    print(f"单应矩阵已保存 → {args.out}")


if __name__ == "__main__":
    main()
