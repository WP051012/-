#!/usr/bin/env python3
"""
交互式路口标注工具
===================
在视频帧上标注停止线和路口区域，用于闯红灯判定。

操作说明:
    1. 运行脚本，会显示视频第一帧
    2. 用鼠标点击两个点定义停止线 (绿色线段)
    3. 用鼠标拖动矩形定义路口区域 (蓝色矩形)
    4. 按 's' 保存当前标注到配置文件
    5. 按 'r' 重置当前标注
    6. 按 'n' 跳到下一帧验证
    7. 按 'q' 退出

用法:
    python scripts/annotate_intersection.py --video path/to/video.mp4
    python scripts/annotate_intersection.py --video path/to/video.mp4 --config configs/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# Global state
# ======================================================================

class AnnotationState:
    def __init__(self, display_scale: float = 0.5):
        self.stop_line_pts: list = []       # [(x1,y1), (x2,y2)] in ORIGINAL coords
        self.junction_pts: list = []        # [(x1,y1), (x2,y2)] in ORIGINAL coords
        self.traffic_light_rois: list = []  # [{x1,y1,x2,y2,direction}] in ORIGINAL coords
        self.current_action = "stop_line"
        self.drawing = False
        self.temp_pt = None
        self.frame = None                   # original full-res frame
        self.display_frame = None           # scaled-down display frame
        self.frame_id: int = 0
        self.video_path: str = ""
        self.window_name = "Intersection Annotation Tool"
        self.message = "Click 2 points for STOP LINE (green). Press 'j' for junction mode."
        self.scale = display_scale          # display_scale factor
        self.orig_w: int = 0
        self.orig_h: int = 0

    def to_original(self, x: int, y: int) -> tuple:
        """Convert display coordinates to original image coordinates."""
        return (int(x / self.scale), int(y / self.scale))

    def to_display(self, x: int, y: int) -> tuple:
        """Convert original coordinates to display coordinates."""
        return (int(x * self.scale), int(y * self.scale))


state = AnnotationState(display_scale=0.4)  # scale 4K to fit screen


# ======================================================================
# Mouse callback
# ======================================================================

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events. x,y are DISPLAY coordinates — convert to original."""
    ox, oy = state.to_original(x, y)

    if state.current_action == "stop_line":
        if event == cv2.EVENT_LBUTTONDOWN and len(state.stop_line_pts) < 2:
            state.stop_line_pts.append((ox, oy))
            if len(state.stop_line_pts) == 2:
                p1, p2 = state.stop_line_pts
                state.message = (
                    f"Stop line: ({p1[0]},{p1[1]}) -> ({p2[0]},{p2[1]})"
                    " | Press 'j' for junction, 's' to save, 'r' to reset"
                )

    elif state.current_action == "junction":
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.temp_pt = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            if state.temp_pt:
                x1, y1 = state.temp_pt
                state.junction_pts = [(x1, y1), (ox, oy)]
                state.message = (
                    f"Junction: ({x1},{y1}) -> ({ox},{oy})"
                    " | Press 's' to save, 'r' to reset"
                )
                state.temp_pt = None

    elif state.current_action == "traffic_light":
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.temp_pt = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            if state.temp_pt:
                x1, y1 = state.temp_pt
                state.traffic_light_rois.append({
                    "x1": min(x1, ox), "y1": min(y1, oy),
                    "x2": max(x1, ox), "y2": max(y1, oy),
                    "direction": "vertical",
                })
                state.message = (
                    f"Traffic light ROI #{len(state.traffic_light_rois)} added"
                    " | Press 's' to save, 'r' to reset, 't' to add another TL"
                )
                state.temp_pt = None


# ======================================================================
# Drawing
# ======================================================================

def draw_annotations():
    """Render annotations on the display frame."""
    if state.display_frame is None:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    vis = state.display_frame.copy()
    h, w = vis.shape[:2]

    # Instructions panel
    panel_h = 100
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    cv2.putText(panel, state.message, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(panel, "[s]save [r]reset [j]junction [t]traffic_light [l]stop_line [n]next_frame [q]quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"Frame: {state.frame_id} | Action: {state.current_action} | Scale: {state.scale:.0%}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Draw stop line (green) - convert to display coords
    if len(state.stop_line_pts) == 1:
        dp = state.to_display(*state.stop_line_pts[0])
        cv2.circle(vis, dp, 8, (0, 255, 0), -1)
        cv2.putText(vis, "Click second point", dp,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    elif len(state.stop_line_pts) == 2:
        dp1 = state.to_display(*state.stop_line_pts[0])
        dp2 = state.to_display(*state.stop_line_pts[1])
        cv2.line(vis, dp1, dp2, (0, 255, 0), 3)
        cv2.circle(vis, dp1, 6, (0, 255, 0), -1)
        cv2.circle(vis, dp2, 6, (0, 255, 0), -1)
        mid = ((dp1[0] + dp2[0]) // 2, (dp1[1] + dp2[1]) // 2)
        cv2.putText(vis, "STOP LINE", (mid[0] - 40, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw junction ROI (blue)
    if len(state.junction_pts) == 2:
        dp1 = state.to_display(*state.junction_pts[0])
        dp2 = state.to_display(*state.junction_pts[1])
        cv2.rectangle(vis, dp1, dp2, (255, 0, 0), 2)
        cv2.putText(vis, "JUNCTION", (dp1[0], dp1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw traffic light ROIs (yellow)
    for i, roi in enumerate(state.traffic_light_rois):
        dp1 = state.to_display(roi["x1"], roi["y1"])
        dp2 = state.to_display(roi["x2"], roi["y2"])
        cv2.rectangle(vis, dp1, dp2, (0, 255, 255), 2)
        cv2.putText(vis, f"TL{i+1}", (dp1[0], dp1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    combined = np.vstack([vis, panel])
    return combined


# ======================================================================
# Main
# ======================================================================

def run_annotation_tool(video_path: str, config_path: Optional[str] = None):
    """Run the interactive annotation tool."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"\nAnnotation Instructions:")
    print(f"  1. Stop Line: Click TWO points to define the stop line")
    print(f"  2. Junction:   Press 'j', then drag a rectangle for the junction area")
    print(f"  3. Traffic Lights: Press 't', then drag rectangles for each traffic light")
    print(f"  [s] Save  [r] Reset  [n] Next frame  [q] Quit\n")

    state.video_path = video_path
    state.frame_id = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        cap.release()
        sys.exit(1)

    state.frame = frame
    # Create scaled display frame
    disp_w = int(width * state.scale)
    disp_h = int(height * state.scale)
    state.display_frame = cv2.resize(frame, (disp_w, disp_h))
    state.orig_w = width
    state.orig_h = height

    cv2.namedWindow(state.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(state.window_name, disp_w, disp_h + 100)
    cv2.setMouseCallback(state.window_name, mouse_callback)

    while True:
        display = draw_annotations()
        cv2.imshow(state.window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            _save_annotations(config_path)
            state.message = "✓ Saved! | Press 'q' to quit or continue annotating"

        elif key == ord('r'):
            state.stop_line_pts = []
            state.junction_pts = []
            state.traffic_light_rois = []
            state.message = "Reset. Click 2 points for STOP LINE."

        elif key == ord('j'):
            state.current_action = "junction"
            state.message = "Junction mode: DRAG rectangle for junction area"

        elif key == ord('t'):
            state.current_action = "traffic_light"
            state.message = "Traffic light mode: DRAG rectangle around a traffic light"

        elif key == ord('l'):
            state.current_action = "stop_line"
            state.message = "Stop line mode: Click 2 points"

        elif key == ord('n'):
            state.frame_id += 25
            if state.frame_id >= total_frames:
                state.frame_id = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.frame_id)
            ret, frame = cap.read()
            if ret:
                state.frame = frame
                state.display_frame = cv2.resize(frame, (int(width * state.scale), int(height * state.scale)))
                state.message = f"Frame {state.frame_id} | Verify. Press 's' to save."

    cap.release()
    cv2.destroyAllWindows()


def _save_annotations(config_path: Optional[str] = None):
    """Save annotations to a JSON file and optionally update config."""
    video_stem = Path(state.video_path).stem

    annotation = {
        "video": str(state.video_path),
        "frame_annotated": state.frame_id,
        "stop_line": {
            "x1": state.stop_line_pts[0][0] if len(state.stop_line_pts) >= 2 else None,
            "y1": state.stop_line_pts[0][1] if len(state.stop_line_pts) >= 2 else None,
            "x2": state.stop_line_pts[1][0] if len(state.stop_line_pts) >= 2 else None,
            "y2": state.stop_line_pts[1][1] if len(state.stop_line_pts) >= 2 else None,
        },
        "junction_roi": {
            "x1": state.junction_pts[0][0] if len(state.junction_pts) >= 2 else None,
            "y1": state.junction_pts[0][1] if len(state.junction_pts) >= 2 else None,
            "x2": state.junction_pts[1][0] if len(state.junction_pts) >= 2 else None,
            "y2": state.junction_pts[1][1] if len(state.junction_pts) >= 2 else None,
        },
        "traffic_light_rois": state.traffic_light_rois,
    }

    # Save annotation file
    out_path = Path("data/annotations") / f"{video_stem}_annotation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"\n[OK] Annotation saved to: {out_path}")
    print(f"  Stop line: {annotation['stop_line']}")
    print(f"  Junction:  {annotation['junction_roi']}")
    print(f"  Traffic light ROIs: {len(annotation['traffic_light_rois'])}")

    # Print config snippet for copy-paste
    sl = annotation["stop_line"]
    jr = annotation["junction_roi"]
    if sl["x1"] is not None:
        print(f"\n--- Config snippet ---")
        print(f"  stop_line: [{sl['x1']}, {sl['y1']}, {sl['x2']}, {sl['y2']}]")
        print(f"  junction_roi: [{jr['x1']}, {jr['y1']}, {jr['x2']}, {jr['y2']}]")
        if state.traffic_light_rois:
            print(f"  traffic_light_rois:")
            for roi in state.traffic_light_rois:
                print(f"    - {{x1: {roi['x1']}, y1: {roi['y1']}, x2: {roi['x2']}, y2: {roi['y2']}}}")

    # Update config if specified
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Determine which intersection this belongs to
            if "timing_" in video_stem:
                section = "intersection_A"
            else:
                section = "intersection_B"

            if section in config:
                config[section]["stop_line"] = [
                    sl["x1"], sl["y1"], sl["x2"], sl["y2"],
                ]
                config[section]["junction_roi"] = [
                    jr["x1"], jr["y1"], jr["x2"], jr["y2"],
                ]
                config[section]["traffic_light_rois"] = state.traffic_light_rois

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            print(f"✓ Config updated: {config_path}#{section}")
        except Exception as e:
            print(f"Warning: Could not update config: {e}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="交互式路口标注工具 — 标记停止线和路口区域"
    )
    parser.add_argument("--video", required=True, help="视频路径 (用于标注)")
    parser.add_argument("--config", default=None, help="自动更新配置文件")
    parser.add_argument("--frame", type=int, default=0, help="起始帧 (默认第0帧)")

    args = parser.parse_args()

    if args.frame > 0:
        state.frame_id = args.frame

    run_annotation_tool(args.video, args.config)


if __name__ == "__main__":
    main()
