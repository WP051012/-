#!/bin/bash
# FOMAML v2 smoke test — adapter + BN only (no modulation)
set -e
cd /root/red-light-prediction
python train_fomaml.py \
    --epochs 3 \
    --max-samples 200 \
    --val-interval 1 \
    --batch-size 32 \
    --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
    --save-dir checkpoints/fomaml_v2 \
    --no-modulation \
    --ada-alpha 0.7 \
    --inner-lr 0.001 \
    --inner-steps 3 \
    --lambda-feat 0.01 \
    --lambda-dist 0.01 \
    --max-delta-norm 0.1 \
    --anomaly
