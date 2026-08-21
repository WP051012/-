#!/usr/bin/env bash
# Quick runtime check for the FIXED per-domain FOMAML eval.
# Verifies: adaptation is fast (~seconds) + sampling runs at ~baseline it/s,
# so the full D5 eval is ~80 min, not the old ~16 h per-sample loop.
set -e
cd /root/red-light-prediction
mkdir -p logs

python scripts/test_eval_runtime.py \
  --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
  --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
  --num-mc 100 \
  --n-timing 600 \
  --full-n 128412 \
  2>&1 | tee logs/fomaml_runtime_test.log
