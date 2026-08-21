#!/bin/bash
# ================================================================
# 闯红灯预测 — AutoDL 云端运行脚本 (2026-07-31 消融实验版)
# ================================================================
#
# 用法说明：
#   消融实验 6 个变体需要逐个手动跑（每个 --segment 2 安全释放内存）。
#   每个变体跑完后，下一个变体直接从 checkpoint 自动 resume。
#   所有结果导出到 ablation_study_results.csv。
#
#   总体耗时: 6 变体 × 10 epoch × 18 min/epoch ≈ 18 小时
#   (加上 segment 重启开销 ≈ 20 小时)
#
#   快速测试: 加 --epochs 2 --segment 2 --quick (2 min/epoch)
# ================================================================

CONFIG="configs/default.yaml"
PROCESSED_DIR="data/processed/trajectories"

echo "============================================"
echo " 闯红灯预测 — 消融实验"
echo " 开始时间: $(date)"
echo " 日志文件: train.log"
echo "============================================"

# ---- Step 0: 确认依赖 ----
echo ""
echo "[Step 0] Checking dependencies..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import yaml, tqdm; print('OK')"

# ================================================================
# 消融实验 (逐个变体手动跑)
# ================================================================
#
# 运行方式: 复制下面其中一行到终端执行
#   每个变体内部自动 seg2→resume→seg2→...→完→eval
#   一个变体跑完再复制下一个

echo ""
echo "==== Ablation: 逐个变体运行指令 ===="
echo ""
echo "# 变体 1/6: FullModel (完整 OurMethod, baseline)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant FullModel --epochs 10 --segment 2 --resume checkpoints/ablation_fullmodel.pt"
echo ""
echo "# 变体 2/6: NoGraph (去感知图 GAT → 简单编码)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant NoGraph --epochs 10 --segment 2 --resume checkpoints/ablation_nograph.pt"
echo ""
echo "# 变体 3/6: NoMemory (去三支记忆 → 直接投影)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant NoMemory --epochs 10 --segment 2 --resume checkpoints/ablation_nomemory.pt"
echo ""
echo "# 变体 4/6: NoCogContext (GRU不含认知状态c)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant NoCogContext --epochs 10 --segment 2"
echo ""
echo "# 变体 5/6: NoFlowChain (FlowChain → MLP)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant NoFlowChain --epochs 10 --segment 2 --resume checkpoints/ablation_noflowchain.pt"
echo ""
echo "# 变体 6/6: NoChange (去变化检测+衰减)"
echo "python scripts/run_experiments.py --config $CONFIG --processed-dir $PROCESSED_DIR --exp ablation --variant NoChange --epochs 10 --segment 2 --resume checkpoints/ablation_nochange.pt"
echo ""

# ---- 完成后 ----
echo "============================================"
echo " 结果文件:"
echo "   ablation_study_results.csv  — 各变体 ADE/FDE/NLL"
echo "   experiment_results.json     — 完整 JSON"
echo ""
echo " 查看结果:"
echo "   cat ablation_study_results.csv"
echo "============================================"
