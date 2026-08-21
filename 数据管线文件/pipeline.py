#!/usr/bin/env python3
"""
闯红灯预测 Pipeline 总入口
=============================
完整的端到端流程编排。

工作流:
    Step 1: 标注路口 — 交互式标注停止线和路口区域
        python pipeline.py annotate --video <样例视频>

    Step 2: 预处理 — 从953个视频提取轨迹+交通灯状态
        python pipeline.py preprocess --all
        python pipeline.py preprocess --dry-run          # 仅统计

    Step 3: 训练 — 分阶段训练模型
        python pipeline.py train --stage 1               # FlowChain预训练
        python pipeline.py train --stage 2 --resume ...  # 加入感知图
        python pipeline.py train --stage 3 --resume ...  # 加入分类器

    Step 4: 推理 — 对新视频做闯红灯预测
        python pipeline.py inference --video <video.mp4> --visualize

快速开始:
    # 1. 先标注两个路口
    python pipeline.py annotate --video "D:/Red-Light视频数据/2026_01_21/ch01_timing_20260121080748-20260121081042.mp4"
    python pipeline.py annotate --video "D:/Red-Light视频数据/2026_01_22/ch01_00000000003000000_20260122080500_20260122080754_224561.mp4"

    # 2. 预处理全部数据 (耗时较长，建议先跑一天测试)
    python pipeline.py preprocess --start-date 2026_01_21 --end-date 2026_01_21

    # 3. 训练
    python pipeline.py train --stage 1

配置文件: configs/default.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def cmd_annotate(args):
    """Run the intersection annotation tool."""
    script = PROJECT_ROOT / "scripts" / "annotate_intersection.py"
    cmd = [sys.executable, str(script), "--video", args.video]
    if args.config:
        cmd += ["--config", args.config]
    subprocess.run(cmd)


def cmd_preprocess(args):
    """Run offline preprocessing."""
    script = PROJECT_ROOT / "scripts" / "preprocess.py"
    cmd = [sys.executable, str(script), "--config",
           args.config or str(PROJECT_ROOT / "configs" / "default.yaml")]

    if args.all:
        pass  # process all videos
    elif args.video:
        cmd += ["--video", args.video]
    else:
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        if args.end_date:
            cmd += ["--end-date", args.end_date]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_frames > 1:
        cmd += ["--skip-frames", str(args.skip_frames)]
    if args.mode:
        cmd += ["--mode", args.mode]

    subprocess.run(cmd)


def cmd_train(args):
    """Run model training."""
    script = PROJECT_ROOT / "scripts" / "train.py"
    data_dir = args.data or str(PROJECT_ROOT / "data" / "processed" / "trajectories")

    cmd = [
        sys.executable, str(script),
        "--config", args.config or str(PROJECT_ROOT / "configs" / "default.yaml"),
        "--data", data_dir,
        "--stage", str(args.stage),
    ]

    if args.resume:
        cmd += ["--resume", args.resume]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.lr:
        cmd += ["--lr", str(args.lr)]
    if args.device:
        cmd += ["--device", args.device]

    subprocess.run(cmd)


def cmd_inference(args):
    """Run inference on a video."""
    script = PROJECT_ROOT / "scripts" / "inference.py"
    cmd = [
        sys.executable, str(script),
        "--video", args.video,
        "--config", args.config or str(PROJECT_ROOT / "configs" / "default.yaml"),
    ]

    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.stop_line:
        cmd += ["--stop-line", args.stop_line]
    if args.junction_roi:
        cmd += ["--junction-roi", args.junction_roi]
    if args.output:
        cmd += ["--output", args.output]
    if args.visualize:
        cmd.append("--visualize")
    if args.device:
        cmd += ["--device", args.device]

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="闯红灯预测 Pipeline — 端到端流程编排",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # --- annotate ---
    p_ann = sub.add_parser("annotate", help="交互式路口标注")
    p_ann.add_argument("--video", required=True, help="标注用视频路径")
    p_ann.add_argument("--config", default=None, help="配置文件路径")

    # --- preprocess ---
    p_pre = sub.add_parser("preprocess", help="离线预处理 (检测+追踪+轨迹+交通灯)")
    p_pre.add_argument("--config", default=None, help="配置文件路径")
    p_pre.add_argument("--all", action="store_true", help="处理所有视频")
    p_pre.add_argument("--video", default=None, help="处理单个视频")
    p_pre.add_argument("--start-date", default=None, help="起始日期")
    p_pre.add_argument("--end-date", default=None, help="结束日期")
    p_pre.add_argument("--dry-run", action="store_true", help="仅统计")
    p_pre.add_argument("--skip-frames", type=int, default=1, help="跳帧")
    p_pre.add_argument("--mode", default="full",
                       choices=["full", "track_and_trafficlight", "trajectories_only"])

    # --- train ---
    p_tr = sub.add_parser("train", help="训练模型")
    p_tr.add_argument("--config", default=None)
    p_tr.add_argument("--data", default=None, help="预处理数据目录")
    p_tr.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p_tr.add_argument("--resume", default=None)
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--batch-size", type=int, default=None)
    p_tr.add_argument("--lr", type=float, default=None)
    p_tr.add_argument("--device", default="cuda")

    # --- inference ---
    p_inf = sub.add_parser("inference", help="推理")
    p_inf.add_argument("--video", required=True)
    p_inf.add_argument("--config", default=None)
    p_inf.add_argument("--checkpoint", default=None)
    p_inf.add_argument("--stop-line", default=None)
    p_inf.add_argument("--junction-roi", default=None)
    p_inf.add_argument("--output", default=None)
    p_inf.add_argument("--visualize", action="store_true")
    p_inf.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "inference":
        cmd_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
