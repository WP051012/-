"""
Diagnose Stage 1 Feature Distributions — Why did two-stage training fail?

Checks P_cross, signal_factor, env_feat distributions on val/test data
to identify which feature(s) provide zero discriminative signal.

Usage:
    python scripts/diagnose_features.py
"""

import sys, os, yaml, argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import (
    CrossingProbabilityEstimator,
    compute_signal_factor,
    point_in_polygon,
)
from src.classification.agent_centric_risk import extract_environment_features
from scripts.run_experiments import load_split_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
NUM_MC = 100  # Use 100 samples for reliable P_cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default="data/processed/trajectories")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--checkpoint", default="checkpoints/flowchain_best.pt")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples to diagnose (0=all)")
    args = parser.parse_args()

    # ── Config & Geometry ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    junction_roi, crosswalk_roi = None, None
    for key in ("intersection_A", "intersection_B"):
        c = config.get(key, {})
        cw = c.get("crosswalk_roi")
        jr = c.get("junction_roi")
        if cw and len(cw) >= 3:
            if isinstance(cw[0], (list, tuple)):
                crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
            else:
                crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
        if jr and len(jr) >= 3:
            junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
        if junction_roi:
            break

    crossing_region = junction_roi if junction_roi else crosswalk_roi
    print(f"Crossing region (junction_roi): {crossing_region}")
    print(f"Crosswalk ROI: {crosswalk_roi}")
    print(f"Device: {DEVICE}")

    # ── Load Data ──
    print("\nLoading datasets...")
    _, _, _, train_scene, val_scene, test_scene = load_split_datasets(
        args.processed_dir, label_dir=args.label_dir, quick=False,
    )

    # Use val set (2630 samples, 63 violations)
    ds = val_scene
    max_n = min(len(ds), args.max_samples) if args.max_samples > 0 else len(ds)
    print(f"Diagnosing {max_n} samples from val set (total: {len(ds)})")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=trajectory_collate_fn)

    # ── Load FlowChain ──
    print(f"\nLoading FlowChain from {args.checkpoint}...")
    flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        flowchain.load_state_dict(ckpt)
        print("  Loaded checkpoint")
    else:
        print(f"  WARNING: Checkpoint not found: {args.checkpoint}")
    flowchain.eval()
    for p in flowchain.parameters():
        p.requires_grad = False

    # ── Stage 1 Estimator ──
    p_cross_estimator = CrossingProbabilityEstimator(crossing_region=crossing_region)
    norm_tensor = NORM.to(DEVICE)

    # ── Collect Features ──
    all_p_cross = []
    all_signals = []
    all_env_feat = []  # list of (8,) arrays
    all_labels = []
    all_tl_states = []  # raw traffic light states for debugging
    all_n_entered = []  # how many MC samples entered the polygon

    print("\nRunning Stage 1 on samples...")
    for i, batch in enumerate(tqdm(loader, total=max_n)):
        if i >= max_n:
            break

        obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
        scene_list = batch.get("scene_list", [None])
        labels = batch.get("is_violation")

        # FlowChain forward (single sample, batched dim)
        with torch.no_grad():
            pred = flowchain(obs_trajectory=obs, num_samples=NUM_MC)
        samples = pred.get("samples")  # (N, 1, 12, 2)

        if samples is None:
            continue

        s_b = samples[:, 0]  # (N, 12, 2)

        # P_cross + detail
        p_cross = p_cross_estimator.compute_p_cross(s_b)
        detail = p_cross_estimator.compute_details(s_b)

        # Signal factor
        sc = scene_list[0] if scene_list else None
        tl_states = sc.get("traffic_light_states", []) if sc is not None else []
        signal_factor = compute_signal_factor(tl_states)

        # Vehicle features
        sc_dict = {}
        if sc is not None:
            sc_dict = {
                "bboxes": sc["bboxes"].unsqueeze(0).to(DEVICE),
                "positions": sc["positions"].unsqueeze(0).to(DEVICE),
                "class_names": sc["class_names"],
                "target_idx": 0,
            }
        env_np = extract_environment_features(
            scene_data=sc_dict, norm=norm_tensor, target_idx=0,
            junction_roi=junction_roi, crosswalk_roi=crosswalk_roi,
        )

        lbl = float(labels.item()) if hasattr(labels, 'item') else float(labels[0]) if isinstance(labels, (list, tuple)) else 0.0

        all_p_cross.append(float(p_cross.item()))
        all_signals.append(float(signal_factor))
        all_env_feat.append(env_np)
        all_labels.append(lbl)
        all_tl_states.append(tl_states)
        all_n_entered.append(detail["n_entered"])

    # ── Analysis ──
    all_p_cross = np.array(all_p_cross)
    all_signals = np.array(all_signals)
    all_env_feat = np.stack(all_env_feat, axis=0)  # (N, 8)
    all_labels = np.array(all_labels)
    all_n_entered = np.array(all_n_entered)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(all_labels)} samples, {n_pos} violations, {n_neg} non-violations")
    print(f"{'='*70}")

    # ── P_cross Distribution ──
    print(f"\n--- P_cross Distribution ---")
    print(f"Overall:   mean={all_p_cross.mean():.4f}, std={all_p_cross.std():.4f}, "
          f"min={all_p_cross.min():.4f}, max={all_p_cross.max():.4f}")
    print(f"  zeros:  {(all_p_cross == 0).sum()}/{len(all_p_cross)} ({(all_p_cross == 0).mean()*100:.1f}%)")
    print(f"  >0:     {(all_p_cross > 0).sum()}/{len(all_p_cross)} ({(all_p_cross > 0).mean()*100:.1f}%)")
    print(f"  >0.1:   {(all_p_cross > 0.1).sum()}/{len(all_p_cross)} ({(all_p_cross > 0.1).mean()*100:.1f}%)")
    print(f"  Pos:     mean={all_p_cross[pos_mask].mean():.4f} std={all_p_cross[pos_mask].std():.4f}" if n_pos > 0 else "  Pos: N/A")
    print(f"  Neg:     mean={all_p_cross[neg_mask].mean():.4f} std={all_p_cross[neg_mask].std():.4f}" if n_neg > 0 else "  Neg: N/A")

    # Histogram buckets
    buckets = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print(f"  P_cross histogram (Pos/Neg):")
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        mask = (all_p_cross >= lo) & (all_p_cross < hi)
        pos_in = mask & pos_mask
        neg_in = mask & neg_mask
        print(f"    [{lo:.2f}, {hi:.2f}): {mask.sum():5d} total, {pos_in.sum():3d} pos, {neg_in.sum():4d} neg")

    # ── N_entered distribution ──
    print(f"\n--- N_entered (MC samples crossing the polygon) ---")
    print(f"Overall:   mean={all_n_entered.mean():.2f}/{NUM_MC}, max={all_n_entered.max()}")
    print(f"  Pos:     mean={all_n_entered[pos_mask].mean():.2f}" if n_pos > 0 else "  Pos: N/A")
    print(f"  Neg:     mean={all_n_entered[neg_mask].mean():.2f}" if n_neg > 0 else "  Neg: N/A")

    # ── Signal Factor Distribution ──
    print(f"\n--- Signal Factor Distribution ---")
    unique_signals = Counter(all_signals)
    for val in sorted(unique_signals.keys()):
        pos_cnt = ((all_signals == val) & pos_mask).sum()
        neg_cnt = ((all_signals == val) & neg_mask).sum()
        label_map = {0.0: "Green", 0.5: "Unknown", 0.7: "Yellow", 1.0: "Red"}
        lbl_name = label_map.get(val, f"{val}")
        print(f"  {lbl_name} ({val}): {unique_signals[val]:5d} total, {pos_cnt:3d} pos, {neg_cnt:4d} neg")

    # Raw TL states
    all_tl_raw = [s[-1] if s else "empty" for s in all_tl_states]  # last frame
    tl_counter = Counter(all_tl_raw)
    print(f"\n  Raw last-frame TL states:")
    for st, cnt in tl_counter.most_common():
        pos_cnt = sum(1 for i, s in enumerate(all_tl_raw) if s == st and all_labels[i] == 1)
        print(f"    '{st}': {cnt} total, {pos_cnt} pos")

    # ── Env Feature Distribution ──
    print(f"\n--- Vehicle Environment Features (8 dimensions) ---")
    feat_names = [
        "d_min_last", "d_min_mean", "d_min_trend",
        "vehicle_count", "has_close_vehicle",
        "ttc_approx", "rel_vel", "density"
    ]
    for j in range(min(all_env_feat.shape[1], 8)):
        col = all_env_feat[:, j]
        print(f"\n  [{j}] {feat_names[j]}:")
        print(f"    Overall: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"min={col.min():.4f}, max={col.max():.4f}, zeros={(col==0).sum()}/{len(col)}")
        if n_pos > 0:
            print(f"    Pos:     mean={col[pos_mask].mean():.4f}, std={col[pos_mask].std():.4f}")
        if n_neg > 0:
            print(f"    Neg:     mean={col[neg_mask].mean():.4f}, std={col[neg_mask].std():.4f}")

    # ── Quick AUC check per feature ──
    print(f"\n--- Per-Feature AUC (univariate) ---")
    def quick_auc(feature, labels):
        """Compute AUC by sorting."""
        order = np.argsort(feature)
        sorted_labels = labels[order]
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        # Count pos-neg pairs where pos > neg
        pos_ranks = np.where(sorted_labels == 1)[0]
        # Mann-Whitney U → AUC
        u = pos_ranks.sum() - n_pos * (n_pos - 1) / 2
        auc = u / (n_pos * n_neg)
        return max(auc, 1 - auc)  # ensure > 0.5

    print(f"  P_cross:       AUC={quick_auc(all_p_cross, all_labels):.4f}")
    print(f"  Signal_factor: AUC={quick_auc(all_signals, all_labels):.4f}")
    for j in range(min(all_env_feat.shape[1], 8)):
        print(f"  env[{j}] {feat_names[j]:18s}: AUC={quick_auc(all_env_feat[:, j], all_labels):.4f}")

    # ── Correlation matrix ──
    print(f"\n--- Feature-Label Correlations ---")
    print(f"  P_cross vs label:       {np.corrcoef(all_p_cross, all_labels)[0,1]:.4f}")
    print(f"  Signal vs label:        {np.corrcoef(all_signals, all_labels)[0,1]:.4f}")
    for j in range(min(all_env_feat.shape[1], 8)):
        col = all_env_feat[:, j]
        if col.std() > 0:
            print(f"  env[{j}] vs label:        {np.corrcoef(col, all_labels)[0,1]:.4f}")

    print(f"\n{'='*70}")
    print("Diagnosis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
