#!/usr/bin/env bash
# ============================================================
# HPC 环境一键搭建（在 Web 终端里执行）
#   bash scripts/hpc_setup.sh [环境名] [python版本] [cuda版本]
#
# 示例:
#   bash scripts/hpc_setup.sh bev 3.10 cu121
#   bash scripts/hpc_setup.sh bev 3.10 cu124
#
# cuda 版本务必按 `nvidia-smi` 右上角显示的 CUDA 版本填：
#   12.1 → cu121    12.4 → cu124    11.8 → cu118
# ============================================================
set -euo pipefail

ENV_NAME="${1:-bev}"
PY_VER="${2:-3.10}"
CUDA="${3:-cu121}"

# 1) 先看 GPU 与 CUDA 驱动版本
echo "===== nvidia-smi ====="
nvidia-smi

# 2) 建 conda 环境（装在 ~ 家目录，容器重启不丢）
echo "===== 创建 conda 环境: ${ENV_NAME} (python ${PY_VER}) ====="
conda create -n "${ENV_NAME}" python="${PY_VER}" -y

# 让 conda activate 在脚本里可用
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 3) 升级 pip 并装 CUDA 版 PyTorch（单独装，指定官方 CUDA 源）
echo "===== 安装 PyTorch (${CUDA}) ====="
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"

# 4) 安装其余依赖
echo "===== 安装其余依赖 ====="
pip install -r requirements_bev.txt

# 5) 验证
echo "===== 验证 ====="
python -c "import torch, torchvision, cv2, numpy, yaml, matplotlib; \
print('torch', torch.__version__); \
print('cuda available:', torch.cuda.is_available()); \
print('device count :', torch.cuda.device_count())"

echo ""
echo "✅ 环境 ${ENV_NAME} 搭建完成。"
echo "   后续每次进入新终端先执行:  conda activate ${ENV_NAME}"
