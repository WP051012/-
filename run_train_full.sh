#!/usr/bin/env bash
# FOMAML v2 full run (few-shot meta-learning scale).
# --max-samples 5000  →  meta-train ~2342 / meta-val 59 / meta-test ~1049.
# This is the intended FOMAML scale. Do NOT drop --max-samples (default is
# 500000 → ~1h/epoch, not the few-shot regime).
set -e
cd /root/red-light-prediction
mkdir -p logs

python train_fomaml.py \
  --epochs 50 \
  --max-samples 5000 \
  --val-interval 5 \
  --batch-size 32 \
  --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
  --save-dir checkpoints/fomaml_v2 \
  --no-modulation \
  --ada-alpha 0.7 \
  --inner-lr 0.001 \
  --inner-steps 3 \
  --lambda-feat 0.01 \
  --max-delta-norm 0.1 \
  2>&1 | tee logs/fomaml_train_full.log
