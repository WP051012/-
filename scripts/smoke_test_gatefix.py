"""Smoke test for the signal-gate fix (window-level label + force_scene).

Verifies, on a SMALL subset (--max-samples), that the two recent fixes work
before spending 25-45 min on the full eval:

  1. Dataset builds with force_scene=True + crossing_region (no crash)
  2. is_violation_window is computed (window-level label, not per-track)
  3. pred_traffic_light_states is populated with real light states (not "unknown")
  4. Consistency: every window-violation sample has red in its pred window
  5. The P_cross x is_red classifier runs end-to-end (mini AUC/F1)

Run (from /root/red-light-prediction):
    python scripts/smoke_test_gatefix.py \
        --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt \
        --max-samples 500 --num-mc 50
"""

import sys, os, json, argparse, logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts dir

from data.dataset import TrajectoryDataset
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import CrossingProbabilityEstimator
from src.evaluation import compute_classification_metrics

# Reuse the exact same geometry/split/filter/gate logic as the real eval.
from eval_flowchain_domain import (
    parse_geometry, split_by_domain, filter_candidates,
    search_threshold, evaluate_split,
    META_TRAIN, META_VAL, META_TEST, NORM, DEVICE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke")


def report_label_stats(dataset, indices, name):
    """Compare per-track vs window-level violation counts, and check the
    red-in-pred-window consistency for window-violation samples."""
    n = len(indices)
    n_track = sum(1 for i in indices if dataset.samples[i].get("is_violation", False))
    n_window = sum(1 for i in indices if dataset.samples[i].get("is_violation_window", False))

    # Consistency: every window-violation sample MUST have red in its pred window
    # (both the label and the gate read the same target_frames + traffic_lights).
    bad = 0
    for i in indices:
        s = dataset.samples[i]
        if s.get("is_violation_window", False):
            tgt = s.get("target_positions")
            tf = s.get("target_frames")
            if tgt is None or tf is None:
                bad += 1
                continue
            # replicate the gate's red check directly on the sample
            scene = dataset._get_scene_data(s["video"], s["obs_frames"], tf)
            states = scene.get("pred_traffic_light_states", [])
            if not any(st == "red" for st in states):
                bad += 1

    print(f"  [{name}] n={n}")
    print(f"    per-track viol : {n_track} ({100*n_track/max(1,n):.1f}%)")
    print(f"    window   viol : {n_window} ({100*n_window/max(1,n):.1f}%)")
    print(f"    window-viol but NO red in pred window (should be 0): {bad}")


def report_light_state_coverage(dataset, indices, name):
    """Distribution of pred_traffic_light_states over samples in this split."""
    state_counter = Counter()
    n_scene = 0
    n_any_red = 0
    n_all_unknown = 0
    for i in indices:
        s = dataset.samples[i]
        tf = s.get("target_frames")
        if tf is None:
            continue
        scene = dataset._get_scene_data(s["video"], s["obs_frames"], tf)
        states = scene.get("pred_traffic_light_states", [])
        if not states:
            continue
        n_scene += 1
        state_counter.update(states)
        if any(st == "red" for st in states):
            n_any_red += 1
        if all(st == "unknown" for st in states):
            n_all_unknown += 1

    print(f"  [{name}] pred-window light states over {n_scene} samples:")
    for k, v in state_counter.most_common():
        print(f"    {k:10s} {v:6d}")
    print(f"    samples with >=1 red frame : {n_any_red} ({100*n_any_red/max(1,n_scene):.1f}%)")
    print(f"    samples all-unknown (no csv): {n_all_unknown} ({100*n_all_unknown/max(1,n_scene):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed/trajectories")
    ap.add_argument("--label-dir", default="labels/")
    ap.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_domain_filtered.pt")
    ap.add_argument("--num-mc", type=int, default=50)
    ap.add_argument("--skip-classify", action="store_true",
                    help="Only run the data/label diagnostics, skip the classifier")
    args = ap.parse_args()

    junction_roi, crosswalk_roi, stop_line = parse_geometry()
    logger.info(f"Geometry: junction={junction_roi is not None}, stop_line={stop_line is not None}")

    with open(args.domain_map) as f:
        domain_label_map = json.load(f)

    logger.info("Building eval dataset (with_scene + force_scene, small subset)...")
    ds = TrajectoryDataset(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="with_scene",
        max_scene_samples=0,          # force_scene loads lazily; skip preload
        max_samples=args.max_samples,
        domain_label_map=domain_label_map,
        force_scene=True,
        crossing_region=junction_roi,
    )
    logger.info(f"  {len(ds)} total samples (capped)")

    # Domain split + crossing-candidate filter (same as real eval)
    test_all = split_by_domain(ds, META_TEST, "test_all(D5)")
    val_all = split_by_domain(ds, META_VAL, "val_all(D3)")
    train_all = split_by_domain(ds, META_TRAIN, "train_all")

    print("\n" + "=" * 60)
    print("LABEL STATISTICS (per-track vs window-level)")
    print("=" * 60)
    for idxs, nm in [(train_all, "train"), (val_all, "val"), (test_all, "test")]:
        report_label_stats(ds, idxs, nm)

    print("\n" + "=" * 60)
    print("LIGHT-STATE COVERAGE (pred window)")
    print("=" * 60)
    for idxs, nm in [(val_all, "val(D3)"), (test_all, "test(D5)")]:
        report_light_state_coverage(ds, idxs, nm)

    if args.skip_classify:
        print("\nSkipped classifier (--skip-classify).")
        return

    test_idx = filter_candidates(ds, test_all, junction_roi, stop_line,
                                 crosswalk_roi, use_future_gt=True, name="test")
    val_idx = filter_candidates(ds, val_all, junction_roi, stop_line,
                                crosswalk_roi, use_future_gt=True, name="val")

    if len(test_idx) == 0:
        logger.error("Test set empty after filtering — bump --max-samples.")
        sys.exit(1)

    # Load FlowChain
    logger.info(f"Loading FlowChain: {args.flowchain_ckpt}")
    flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    ckpt = torch.load(args.flowchain_ckpt, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state") or ckpt.get("model") or ckpt
    if any(k.startswith("flow_chain.") for k in sd.keys()):
        sd = {k.replace("flow_chain.", "predictor."): v for k, v in sd.items()}
    flowchain.load_state_dict(sd, strict=False)
    flowchain.eval()

    crossing_region = junction_roi if junction_roi else crosswalk_roi
    p_cross_est = CrossingProbabilityEstimator(crossing_region=crossing_region)
    norm_tensor = NORM.to(DEVICE)

    logger.info(f"Evaluating val (D3): {len(val_idx)} samples...")
    _, val_risks, val_labels, _, val_sig = evaluate_split(
        ds, val_idx, flowchain, p_cross_est, norm_tensor,
        num_mc=args.num_mc, split_name="Val(D3)")
    best_th, best_f1 = search_threshold(val_risks, val_labels)
    logger.info(f"  Best threshold: {best_th:.3f} (F1={best_f1:.4f})")

    logger.info(f"Evaluating test (D5): {len(test_idx)} samples...")
    test_traj, test_risks, test_labels, test_pc, test_sig = evaluate_split(
        ds, test_idx, flowchain, p_cross_est, norm_tensor,
        num_mc=args.num_mc, split_name="Test(D5)")
    test_preds = (test_risks >= best_th).astype(int)
    cls = compute_classification_metrics(test_labels, test_preds, test_risks)
    cls_05 = compute_classification_metrics(test_labels, (test_risks >= 0.5).astype(int), test_risks)
    _tb_th, f1_test_best = search_threshold(test_risks, test_labels)

    print("\n" + "=" * 60)
    print("SMOKE CLASSIFIER RESULT (small subset — not final numbers)")
    print("=" * 60)
    print(f"  Test:      {len(test_idx)} samples, {int(test_labels.sum())} window-viol "
          f"({100*test_labels.sum()/max(1,len(test_labels)):.1f}%)")
    print(f"  AUC:         {cls.get('AUC', 0):.4f} (threshold-free)")
    print(f"  F1@0.5:      {cls_05.get('F1', 0):.4f}")
    print(f"  F1@val_th:   {best_f1:.4f} (th={best_th:.3f})")
    print(f"  F1@test-best:{f1_test_best:.4f} (th={_tb_th:.3f})")
    print(f"  Accuracy:    {cls.get('Accuracy', 0):.4f}")
    red_mask = test_sig >= 0.99
    print(f"  Red(pred win): {red_mask.sum()} samples, viol={((test_labels == 1) & red_mask).sum()}")
    print(f"  Pos risk:  {test_risks[test_labels == 1].mean():.4f} (n={int((test_labels == 1).sum())})")
    print(f"  Neg risk:  {test_risks[test_labels == 0].mean():.4f} (n={int((test_labels == 0).sum())})")
    print("\nDone!")


if __name__ == "__main__":
    main()
