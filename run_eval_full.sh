#!/usr/bin/env bash
# FOMAML v2 best-of-100 evaluation — compare against the FULL FlowChain baseline.
#
# Run AFTER the 50-epoch training (checkpoints/fomaml_v2/best_fomaml.pt) finishes.
#
# --num-mc 100 → matches eval_flowchain_domain.py's NUM_MC=100 protocol. The
#                authoritative FlowChain baseline (full 128K test set, same ckpt)
#                is: ADE mean=35.20px / FDE mean=47.99px / AUC=0.9931 (best-of-100).
#                FOMAML must beat those numbers to claim a win.
set -e
cd /root/red-light-prediction
mkdir -p logs

python scripts/eval_fomaml.py \
  --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
  --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
  --num-mc 100 \
  2>&1 | tee logs/fomaml_eval_best100.log
