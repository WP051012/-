#!/usr/bin/env bash
# ============================================================
# HPC 后台训练（Web 终端 + tmux，浏览器关了也不会断）
#   bash scripts/hpc_run.sh [config路径] [GPU编号] [conda环境名]
#
# 示例:
#   bash scripts/hpc_run.sh configs/bev_proposed.yaml 0 bev
# ============================================================
set -euo pipefail

CONFIG="${1:-configs/bev_proposed.yaml}"
GPU="${2:-0}"
ENV_NAME="${3:-bev}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU}"

# 注入数据路径环境变量（config 里 ${BEV_VIDEO_DIR}/${BEV_LABEL_DIR} 引用）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hpc_env.sh"

SESSION="bev_train"

# 避免重名会话
tmux kill-session -t "${SESSION}" 2>/dev/null || true

# 后台启动训练，日志同时打到终端和 train_bev.log
tmux new-session -d -s "${SESSION}" \
  "python scripts/train_bev.py --config ${CONFIG} 2>&1 | tee train_bev.log"

echo "✅ 已在 tmux 会话 '${SESSION}' 启动训练，日志 → train_bev.log"
echo ""
echo "  查看实时日志:  tail -f train_bev.log"
echo "  进入会话:      tmux attach -t ${SESSION}     (退出按 Ctrl-b 再按 d)"
echo "  看显存:        nvidia-smi"
echo "  结束训练:      tmux kill-session -t ${SESSION}"
echo ""
echo "⚠️  数据路径由 scripts/hpc_env.sh 注入；首次运行前确认该文件里的"
echo "    BEV_VIDEO_DIR / BEV_LABEL_DIR 指向正确目录（默认 $HOME/red-light-prediction/data_bev/）。"
