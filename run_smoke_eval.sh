#!/usr/bin/env bash
# FOMAML eval smoke test — verify model load + inner-loop backward + sampling
# on a tiny subset BEFORE the ~1h full run. Exits non-zero on FAIL.
set -e
cd /root/red-light-prediction
mkdir -p logs

python scripts/smoke_fomaml.py \
  --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
  --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
  --num-samples 8 \
  --num-mc 20 \
  --max-samples 128 \
  --max-scene 128 \
  2>&1 | tee logs/fomaml_smoke_eval.log
