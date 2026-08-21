#!/usr/bin/env bash
# FOMAML v4 (ADE-only objective) train + eval, chained so you can leave it running.
# Run:  bash run_v4_train_eval.sh
set -euo pipefail
cd /root/red-light-prediction
mkdir -p logs

echo "=== [1/2] Training FOMAML v4 (ADE-only objective) ==="
python train_fomaml.py \
  --save-dir checkpoints/fomaml_v4 \
  --filter-crossing \
  --epochs 30 \
  --batch-size 64 \
  --inner-lr 0.01 \
  --inner-steps 5 \
  --ada-alpha 0.3 \
  --seed 42 \
  2>&1 | tee logs/fomaml_v4_train.log

echo "=== [2/2] Evaluating FOMAML v4 (best-of-100) ==="
# NOTE: --flowchain-ckpt must match train_fomaml.py's default backbone
# (flowchain_best_finetuned.pt). eval_fomaml.py's default is
# flowchain_domain_filtered.pt, which mismatches the frozen training backbone.
python scripts/eval_fomaml.py \
  --checkpoint checkpoints/fomaml_v4/best_fomaml.pt \
  --flowchain-ckpt checkpoints/flowchain_best_finetuned.pt \
  --num-mc 100 \
  2>&1 | tee logs/fomaml_v4_eval.log

echo "=== Done. Logs: logs/fomaml_v4_train.log, logs/fomaml_v4_eval.log ==="
