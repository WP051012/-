"""串联运行：OurMethod 轨迹训练(Stage2) → 闯红灯分类评估(Stage3)"""
import os, sys, subprocess

BASE = "/root/miniconda3/bin/python"
SCRIPT = "scripts/run_experiments.py"
COMMON = "--config configs/default.yaml --processed-dir data/processed/trajectories --label-dir labels/"

# ═══════════════════════════════════════════════════════
# Phase 1: 轨迹预测训练 (Stage 2)
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("Phase 1: Training OurMethod Trajectory (Stage 2)")
print("=" * 60)

cmd1 = (f"{BASE} {SCRIPT} {COMMON} "
        f"--exp trajectory "
        f"--skip social-lstm,stgcnn,transformer,rnn,flowchain "
        f"--epochs 10 --segment 0")
print(f"Running: {cmd1}")
rc1 = os.system(cmd1)
print(f"Phase 1 exit: {rc1}")

# Check checkpoint exists
if not os.path.exists("checkpoints/ablation_fullmodel.pt"):
    print("ERROR: Phase 1 did not produce checkpoint!")
    sys.exit(1)

# ═══════════════════════════════════════════════════════
# Phase 2: 闯红灯预测分类 (Stage 3)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Red-Light Violation Classification (Stage 3)")
print("=" * 60)

# Run only classification (STRR + OurMethod)
cmd2 = (f"{BASE} {SCRIPT} {COMMON} "
        f"--exp classification "
        f"--epochs 10 --segment 0")
print(f"Running: {cmd2}")
rc2 = os.system(cmd2)
print(f"Phase 2 exit: {rc2}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
