#!/usr/bin/env bash
# ============================================================
# BEV 数据路径环境变量（HPC 上使用）
#
# config 里 data.bev.video_dir / label_dir 写成 ${BEV_VIDEO_DIR}、
# ${BEV_LABEL_DIR}（带 D:/ 默认值）。本地 Windows 不 source 本文件即可；
# 在 HPC 上先 source 本文件，把路径指向真实数据目录：
#
#     source scripts/hpc_env.sh
#
# 用 $HOME 自动定位家目录，无需手写绝对路径。
# ============================================================
export BEV_VIDEO_DIR="$HOME/red-light-prediction/data_bev/videos/"
export BEV_LABEL_DIR="$HOME/red-light-prediction/data_bev/labels/"
