#!/usr/bin/env bash
# Conditional FlowChain (signal+geometry+scene+goal) train + eval, chained.
# Run:  bash run_conditional.sh
set -euo pipefail
cd /root/red-light-prediction
mkdir -p logs

# Preflight: backbone checkpoint + scene embeddings must exist on A1.
if [ ! -f checkpoints/flowchain_best_finetuned.pt ]; then
  echo "ERROR: checkpoints/flowchain_best_finetuned.pt missing" >&2
  exit 1
fi
if [ ! -f data/gat_conditions.pt ]; then
  echo "WARN: data/gat_conditions.pt missing (scene encoder falls back to zeros)" >&2
fi

# Conditioning strategy (frozen backbone): inject the 256-dim context through a
# Linear(256→16) into the flow's NATIVE conditioning (dist_args), keeping the
# trajectory encoder clean. Load the 28px baseline flowchain_best_finetuned.pt,
# FREEZE the flow backbone, and train only flow_cond_proj (zero-initialized, so
# the model starts exactly at the 28px baseline) + context encoders + aux heads.
# The ONLY difference vs pure FlowChain is the injected condition.
# Fast run: 10 epochs, half the filtered samples (25k).
COND_FLOW_ARGS="--flow-lr 1e-4 --freeze-flow"

echo "=== [1/2] Training conditional FlowChain (frozen backbone) ==="
python scripts/train_conditional.py \
  --config configs/default.yaml \
  --gat-conditions data/gat_conditions.pt \
  --save-dir checkpoints/conditional \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-samples 20 \
  --max-filtered 25000 \
  $COND_FLOW_ARGS \
  2>&1 | tee logs/conditional_train.log

echo "=== [2/2] Evaluating conditional FlowChain (single-sample + best-of-100 + NLL) ==="
python scripts/eval_conditional.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/conditional/best_conditional.pt \
  --gat-conditions data/gat_conditions.pt \
  --num-samples 100 \
  --max-filtered 25000 \
  2>&1 | tee logs/conditional_eval.log

echo "=== Done. Logs: logs/conditional_train.log, logs/conditional_eval.log ==="
