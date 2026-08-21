"""
FOMAML v2 eval smoke test — verify the full eval code path on a tiny subset
before committing to the ~1h full run.

Exercises the exact chain that previously crashed:
    1. load_fomaml_model   (adapter + flow BN unfreeze, ada_alpha, checkpoint load)
    2. dataset build + domain split + candidate filter
    3. inner-loop SGD      (loss.backward() — the RuntimeError source)
    4. best-of-N sampling  (forward-only, raw pixels)

Prints PASS/FAIL plus key diagnostics (Trainable count, AdaBN alpha, NaN check).

Usage:
    python scripts/smoke_fomaml.py \
        --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
        --num-samples 8 --num-mc 20
"""

import sys, os, argparse, logging
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/ dir

from data.dataset import TrajectoryDataset
from eval_fomaml import (
    load_fomaml_model,
    split_by_domain,
    filter_candidates,
    evaluate_split,
    parse_geometry,
    META_TEST,
    META_VAL,
    NORM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_fomaml")

FAIL = False


def check(name, cond, detail=""):
    global FAIL
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL = True
    logger.info(f"  [{status}] {name} {detail}")
    return cond


def main():
    global FAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--perception-ckpt", default="checkpoints/stage1_best.pt")
    ap.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_domain_filtered.pt")
    ap.add_argument("--data-dir", default="data/processed/trajectories")
    ap.add_argument("--label-dir", default="labels/")
    ap.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    ap.add_argument("--annotations-dir", default="data/annotations")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--num-mc", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--max-scene", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- 1. Model load (adapter + BN + ada_alpha) ----------
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model, trainable_params, mod_net, domain_conditions, inner_cfg = \
        load_fomaml_model(config, args.perception_ckpt, args.flowchain_ckpt,
                          args.checkpoint, device)

    n_train = sum(p.numel() for p in trainable_params.values())
    # Adapter (~2K) + flow BN (~16). If unfreeze is wrong it balloons to ~50K.
    check("Trainable count sane (<10000)", n_train < 10000, f"(n={n_train})")
    logger.info(f"  trainable params = {n_train:,}  "
                f"inner_steps={inner_cfg['inner_steps']} inner_lr={inner_cfg['inner_lr']}")

    # ---------- 2. Geometry + dataset + split + filter ----------
    junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
    check("geometry loaded", junction_roi is not None)

    import json
    with open(args.domain_map) as f:
        domain_label_map = json.load(f)

    ds = TrajectoryDataset(
        data_dir=args.data_dir, label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="with_scene",
        max_scene_samples=args.max_scene, max_samples=args.max_samples,
        domain_label_map=domain_label_map, force_scene=True,
        crossing_region=junction_roi,
    )
    logger.info(f"  dataset: {len(ds)} samples")

    test_all = split_by_domain(ds, META_TEST, "test_all(D5)")
    test_idx = filter_candidates(ds, test_all, junction_roi, stop_line,
                                 crosswalk_roi, use_future_gt=True, name="test")
    check("test candidates non-empty", len(test_idx) > 0, f"(n={len(test_idx)})")

    # ---------- 3. Inner-loop + sampling on a tiny subset ----------
    subset = test_idx[:args.num_samples]
    from src.classification.crossing_probability import CrossingProbabilityEstimator
    crossing_region = junction_roi if junction_roi else crosswalk_roi
    p_cross_est = CrossingProbabilityEstimator(crossing_region=crossing_region)
    norm_tensor = NORM.to(device)

    logger.info(f"Adapting {len(subset)} samples (inner_steps={inner_cfg['inner_steps']}, "
                f"num_mc={args.num_mc})...")
    traj, risks, labels, pc, sig = evaluate_split(
        ds, subset, model, trainable_params, mod_net, domain_conditions,
        inner_cfg, p_cross_est, norm_tensor,
        num_mc=args.num_mc, split_name="smoke",
    )

    # ---------- 4. NaN / finite checks ----------
    check("ADE finite", np.all(np.isfinite(traj["ade_mean"])),
          f"(mean={traj['ade_mean']:.2f}px median={traj['ade_median']:.2f}px)")
    check("FDE finite", np.all(np.isfinite(traj["fde_mean"])),
          f"(mean={traj['fde_mean']:.2f}px median={traj['fde_median']:.2f}px)")
    check("risks finite", np.all(np.isfinite(risks)), f"(n={len(risks)})")
    check("labels present", len(labels) == len(subset), f"(n_viol={int(labels.sum())})")

    # Param sanity after adaptation + restore
    bad = 0
    for name, p in trainable_params.items():
        if not torch.isfinite(p.data).all():
            bad += 1
    check("params finite after adapt/restore", bad == 0, f"(bad={bad})")

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULT: {'FAIL' if FAIL else 'PASS'}  "
                f"(ADE mean={traj['ade_mean']:.2f}px, n_viol={int(labels.sum())}/{len(subset)})")
    logger.info("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
