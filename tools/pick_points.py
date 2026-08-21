#!/usr/bin/env python3
"""
交互式标定点拾取（Windows 本地用，需要 GUI）
==============================================
打开一帧，鼠标点击记录地面标定点，输出 points.json 模板。

用法:
    py tools/pick_points.py calib_frame.png points.json

点击后终端会打印每个点的全分辨率像素坐标；按 ESC 结束。
生成的 points.json 里 ground=[0,0] 是占位，需要你手工填成真实地面坐标（米）。

显示时等比缩放到宽 1280，但记录的坐标会换算回原图全分辨率，不影响精度。
"""

import sys
import json
from pathlib import Path

import cv2

DISP_W = 1280  # 显示宽度（像素），只影响显示，不影响记录的坐标


def main():
    if len(sys.argv) < 3:
        print("用法: py tools/pick_points.py <帧图片> <输出points.json>")
        sys.exit(1)
    img_path = sys.argv[1]
    out_path = sys.argv[2]

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image {img_path}")
    H, W = img.shape[:2]
    scale = DISP_W / W
    disp_h = int(H * scale)
    disp = cv2.resize(img, (DISP_W, disp_h))

    pts = []
    canvas = disp.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = x / scale, y / scale
            pts.append([round(fx, 1), round(fy, 1)])
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(len(pts)), (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"  点{len(pts)}: pixel=[{fx:.0f}, {fy:.0f}]")

    win = "click (按 ESC 结束)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("点击地面点（按 ESC 结束）。全分辨率像素坐标如下：")
    while True:
        cv2.imshow(win, canvas)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    correspondences = [{"pixel": p, "ground": [0.0, 0.0]} for p in pts]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"correspondences": correspondences}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n已记录 {len(pts)} 个点 → {out_path}")
    print("下一步：把每个点的 ground=[X,Y] 米手工填进 JSON，然后：")
    print("  py tools/estimate_homography.py --points points.json "
          "--out data/calibration/homography.json --method ransac")


if __name__ == "__main__":
    main()
