"""
Simplified risk evaluation: P_cross + signal_factor → optimal threshold → AUC/F1.

Concept:
    1. Non-red light → risk = 0 (pass, not a violation)
    2. Red light → risk = P_cross (fine-tuned FlowChain junction-crossing probability)
    3. Find optimal P_cross threshold on val set, evaluate on test set.

No training required. Only loads scene subset (~10K samples), no full 2.47M dataset.

Usage:
    python scripts/eval_simple_risk.py
    python scripts/eval_simple_risk.py --quick
"""

import sys, os, yaml, logging, argparse, json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import (
    CrossingProbabilityEstimator,
    compute_signal_factor,
)
from src.evaluation import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
NUM_MC = 100  # Monte Carlo samples for P_cross

TRAIN_DATES = {"2026_01_15", "2026_01_21", "2026_01_22", "2026_01_23"}
VAL_DATES = {"2026_01_26"}
TEST_DATES = {"2026_01_27"}


def parse_roi(config):
    """Extract junction_roi, crosswalk_roi, stop_line from config."""
    crosswalk_roi = None; junction_roi = None; stop_line = None
    for key in ("intersection_A", "intersection_B"):
        c = config.get(key, {})
        cw = c.get("crosswalk_roi"); jr = c.get("junction_roi"); sl = c.get("stop_line")
        if cw and len(cw) >= 3:
            if isinstance(cw[0], (list, tuple)):
                crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
            else:
                crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
        if jr and len(jr) >= 3:
            junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
            if len(junction_roi) == 2:
                x1,y1=junction_roi[0]; x2,y2=junction_roi[1]
                junction_roi = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        if sl and len(sl) >= 4: stop_line = [float(x) for x in sl]
        if crosswalk_roi and junction_roi: break
    return junction_roi, crosswalk_roi, stop_line


def load_scene_split(config, args):
    """
    Load scene-enabled dataset and split by date.
    Returns (train_samples, val_samples, test_samples) as lists of dataset __getitem__ results.
    """
    ds = TrajectoryDataset(
        data_dir=args.processed_dir,
        label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="with_scene",
        max_scene_samples=3000 if args.quick else 10000,
    )
    scene_indices = ds.with_scene_subset()
    logger.info(f"Scene dataset: {len(ds)} total, {len(scene_indices)} with scene data")

    def _by_date(dates, indices):
        result = []
        for i in indices:
            video = ds.samples[i].get("video", "")
            if any(d.replace("_", "") in video for d in dates):
                result.append(i)
        return result

    train_idx = _by_date(TRAIN_DATES, scene_indices)
    val_idx = _by_date(VAL_DATES, scene_indices)
    test_idx = _by_date(TEST_DATES, scene_indices)

    logger.info(f"Split: T={len(train_idx)} V={len(val_idx)} Te={len(test_idx)}")

    # Load actual samples (with scene data) into memory, then free the big dataset
    def _load_samples(indices):
        samples = []
        for i in tqdm(indices, desc="Loading samples", leave=False):
            samples.append(ds[i])
        return samples

    train = _load_samples(train_idx)
    val = _load_samples(val_idx)
    test = _load_samples(test_idx)

    # Free the full dataset — samples are now independent dicts
    del ds
    import gc; gc.collect()

    return train, val, test


@torch.no_grad()
def compute_features(samples, model, p_cross_est, norm_tensor):
    """
    For each sample, compute P_cross and signal_factor.
    Returns two lists: p_cross_values, signal_values, labels.
    """
    p_cross_list = []
    signal_list = []
    labels = []

    for s in tqdm(samples, desc="Computing features", leave=False):
        obs = s["obs_trajectory"].to(DEVICE).unsqueeze(0) / norm_tensor.to(DEVICE)
        is_viol = s.get("is_violation", False)
        labels.append(1.0 if is_viol else 0.0)

        # Signal factor
        scene = s.get("scene", {})
        tl_states = scene.get("traffic_light_states", [])
        signal_list.append(compute_signal_factor(tl_states))

        # P_cross
        pred = model(obs_trajectory=obs, num_samples=NUM_MC)
        samples_t = pred.get("samples", pred.get("best_sample"))
        if samples_t is None:
            p_cross_list.append(0.0)
            continue
        if samples_t.dim() == 4:
            samples_t = samples_t[:, 0, :, :]  # (N, T, 2)
        # Denormalize
        samples_px = samples_t * norm_tensor.to(samples_t.device)
        p_cross_list.append(float(p_cross_est.compute_p_cross(samples_px)))

    return np.array(p_cross_list), np.array(signal_list), np.array(labels)


def compute_risk(p_cross, signal_factor, red_only=True):
    """
    Compute risk score.

    Args:
        p_cross: array of crossing probabilities
        signal_factor: array of signal factors (1.0=red, 0.7=yellow, 0.5=unknown, 0.0=green)
        red_only: if True, non-red signals → risk = 0 (hard gate)
    """
    if red_only:
        # Hard gate: only red light matters
        is_red = (signal_factor >= 0.99)  # red=1.0
        return p_cross * is_red.astype(np.float32)
    else:
        return p_cross * signal_factor


def search_threshold(risks, labels):
    """Find optimal threshold maximizing F1."""
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.01, 1.0, 0.01):
        preds = (risks >= th).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((labels == 1) & (preds == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def evaluate_at_threshold(risks, labels, threshold):
    """Compute classification metrics at given threshold."""
    preds = (risks >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc = (tp + tn) / len(labels)

    # AUC via sklearn if available
    auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, risks)
    except ImportError:
        pass

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser(description="Simplified P_cross + signal risk evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default="data/processed/trajectories")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--checkpoint", default="checkpoints/flowchain_best_finetuned.pt")
    parser.add_argument("--num-mc", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="Fast test with fewer samples")
    parser.add_argument("--red-only", action="store_true", default=True,
                       help="Only red light counts as violation risk")
    args = parser.parse_args()

    global NUM_MC
    NUM_MC = args.num_mc

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    junction_roi, crosswalk_roi, stop_line = parse_roi(config)
    logger.info(f"Junction ROI: {junction_roi}")

    # Load FlowChain
    logger.info(f"Loading FlowChain from {args.checkpoint}")
    model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    p_cross_est = CrossingProbabilityEstimator(crossing_region=junction_roi)

    # Load data
    logger.info("Loading scene data...")
    train, val, test = load_scene_split(config, args)

    # Count violations per split
    for name, split in [("train", train), ("val", val), ("test", test)]:
        nv = sum(1 for s in split if s.get("is_violation", False))
        logger.info(f"  {name}: {len(split)} samples, {nv} violations")

    # Compute features
    logger.info("Computing features on train set...")
    pc_tr, sig_tr, lbl_tr = compute_features(train, model, p_cross_est, NORM)
    logger.info("Computing features on val set...")
    pc_val, sig_val, lbl_val = compute_features(val, model, p_cross_est, NORM)
    logger.info("Computing features on test set...")
    pc_te, sig_te, lbl_te = compute_features(test, model, p_cross_est, NORM)

    # Compute risk
    risk_tr = compute_risk(pc_tr, sig_tr, red_only=args.red_only)
    risk_val = compute_risk(pc_val, sig_val, red_only=args.red_only)
    risk_te = compute_risk(pc_te, sig_te, red_only=args.red_only)

    # ── Threshold search on val ──
    # Only consider red-light samples (others are auto-passed)
    red_val = sig_val >= 0.99
    n_red_val = red_val.sum()
    n_red_viol_val = ((sig_val >= 0.99) & (lbl_val == 1)).sum()
    logger.info(f"Val red-light samples: {n_red_val}, violations: {n_red_viol_val}")

    best_th, best_f1 = search_threshold(risk_val, lbl_val)
    logger.info(f"Best threshold (val): {best_th:.3f}, F1={best_f1:.4f}")

    # ── Evaluate on test ──
    test_metrics = evaluate_at_threshold(risk_te, lbl_te, best_th)

    # Also show signal breakdown
    red_test = sig_te >= 0.99
    yellow_test = (sig_te >= 0.69) & (sig_te <= 0.71)
    green_test = sig_te <= 0.01

    print("\n" + "=" * 60)
    print("SIMPLIFIED RISK EVALUATION (P_cross + signal gate)")
    print("=" * 60)
    print(f"\nModel: {args.checkpoint}")
    print(f"Gate strategy: {'Red-only (hard gate)' if args.red_only else 'Soft gate (× signal_factor)'}")
    print(f"Threshold (from val): {best_th:.3f}")
    print(f"\nTest set: {test_metrics['n_samples']} samples, "
          f"{test_metrics['n_positive']} violations "
          f"({100*test_metrics['n_positive']/max(1,test_metrics['n_samples']):.1f}%)")
    print(f"\nSignal breakdown (test):")
    print(f"  Red:    {red_test.sum()} samples, {((sig_te >= 0.99) & (lbl_te == 1)).sum()} violations")
    print(f"  Yellow: {yellow_test.sum()} samples, {((sig_te >= 0.69) & (sig_te <= 0.71) & (lbl_te == 1)).sum()} violations")
    print(f"  Green:  {green_test.sum()} samples, {((sig_te <= 0.01) & (lbl_te == 1)).sum()} violations")
    print(f"\nP_cross stats (red-light samples only):")
    pc_red = pc_te[red_test]
    lbl_red = lbl_te[red_test]
    print(f"  Violations:   mean P_cross={pc_red[lbl_red == 1].mean():.4f}, "
          f"median={np.median(pc_red[lbl_red == 1]):.4f}")
    print(f"  Non-viol:     mean P_cross={pc_red[lbl_red == 0].mean():.4f}, "
          f"median={np.median(pc_red[lbl_red == 0]):.4f}")
    print(f"\nResults:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
          f"FN={test_metrics['fn']}, TN={test_metrics['tn']}")

    # ── Also try soft-gate for comparison ──
    if args.red_only:
        risk_te_soft = compute_risk(pc_te, sig_te, red_only=False)
        soft_th, soft_f1 = search_threshold(risk_val, lbl_val)  # re-search on val with soft risk
        risk_val_soft = compute_risk(pc_val, sig_val, red_only=False)
        soft_th, soft_f1 = search_threshold(risk_val_soft, lbl_val)
        soft_metrics = evaluate_at_threshold(risk_te_soft, lbl_te, soft_th)
        print(f"\nComparison — Soft gate (× signal_factor):")
        print(f"  Threshold: {soft_th:.3f}")
        print(f"  F1:        {soft_metrics['f1']:.4f}")
        print(f"  AUC:       {soft_metrics['auc']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
