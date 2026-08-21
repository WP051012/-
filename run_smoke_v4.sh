#!/usr/bin/env bash
# FOMAML v4 smoke test — ADE-only objective, modulation ON, filter-crossing ON.
# Verifies the full training pipeline on a tiny subset before the ~1h full run:
#   (1) filter_crossing fix   (dataset.samples[idx] instead of dataset[idx])
#   (2) ADE-only loss         (no NLL / log_prob)
#   (3) modulation enabled    (no --no-modulation)
# Uses a separate save-dir so it never clobbers fomaml_v3 / fomaml_v4.
set -euo pipefail
cd /root/red-light-prediction

# Purge stale bytecode so the updated sources are guaranteed to be used.
find /root/red-light-prediction -name '*.pyc' -delete 2>/dev/null || true
find /root/red-light-prediction -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

python -B train_fomaml.py \
  --epochs 2 \
  --max-samples 500 \
  --val-interval 1 \
  --batch-size 32 \
  --flowchain-ckpt checkpoints/flowchain_best_finetuned.pt \
  --save-dir checkpoints/smoke_v4 \
  --filter-crossing \
  --inner-lr 0.01 \
  --inner-steps 5 \
  --ada-alpha 0.3 \
  --seed 42 \
  --anomaly \
  2>&1 | tee logs/fomaml_v4_smoke.log

echo "=== smoke done. see logs/fomaml_v4_smoke.log ==="
