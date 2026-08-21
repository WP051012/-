"""
Two-Stage Risk Regression Training Pipeline (FlowChain version).

Stage 1 (no training, frozen FlowChain):
    FlowChain trajectory samples → P_cross (polygon check) + Signal_factor + Vehicle features

Stage 2 (trainable RiskRegressionHead):
    Risk = MLP(P_cross, Vehicle_features) × Signal_factor
    Trained with SmoothL1(Risk, binary_label)

Key properties:
    - FlowChain stays FROZEN → trajectory quality preserved (ADE=20px, best predictor)
    - FlowChain is unconditional (zero condition) → no perception model needed
    - SmoothL1 gives continuous gradients to ALL samples (vs sparse BCE)
    - Signal_factor gates: green light → Risk → 0 (legal crossing ≠ violation)

Usage:
    python scripts/train_risk_regression.py [--epochs 50] [--lr 1e-3] [--quick]
"""

import sys, os, yaml, logging, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import (
    CrossingProbabilityEstimator,
    compute_signal_factor,
)
from src.classification.risk_regression_head import (
    RiskRegressionHead,
    prepare_stage2_features,
    search_threshold_regression,
)
from src.classification.agent_centric_risk import extract_environment_features
from src.evaluation import compute_classification_metrics
from scripts.run_experiments import load_split_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])
NUM_MC_TRAIN = 50   # Monte Carlo samples during training (faster)
NUM_MC_EVAL = 100   # Monte Carlo samples during evaluation (more accurate)


# ══════════════════════════════════════════════════════════════════════
# Stage 1: Compute features from frozen FlowChain
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_stage1_features_batch(
    flowchain: FlowChainBase,
    obs_batch: torch.Tensor,           # (B, obs_len, 2) normalized
    scene_list: list,                   # list of scene_data dicts (or None) per sample
    norm_tensor: torch.Tensor,
    p_cross_estimator: CrossingProbabilityEstimator,
    num_samples: int,
    junction_roi: list = None,
    crosswalk_roi: list = None,
) -> dict:
    """
    Stage 1: batched FlowChain forward → per-sample P_cross + signal + vehicle features.

    FlowChain forward is batched (B samples at once, much faster than B individual calls).
    Returns dict with keys: p_cross (B,), signal_factor (B,), env_feat (B, 8), samples, log_probs
    """
    B = obs_batch.shape[0]

    # Batched FlowChain forward (unconditional)
    pred = flowchain(
        obs_trajectory=obs_batch,
        num_samples=num_samples,
    )

    samples = pred.get("samples")       # (N, B, pred_len, 2)
    log_probs = pred.get("log_probs")   # (N, B)

    if samples is None:
        return {
            "p_cross": torch.zeros(B, device=DEVICE),
            "signal_factor": torch.full((B,), 0.5, device=DEVICE),
            "env_feat": torch.zeros(B, 8, device=DEVICE),
            "samples": None,
            "log_probs": None,
        }

    # Per-sample features (CPU, relatively fast)
    p_cross_list, signal_list, env_list = [], [], []
    for b in range(B):
        s_b = samples[:, b]  # (N, pred_len, 2)
        p_cross_list.append(p_cross_estimator.compute_p_cross(s_b))

        sc = scene_list[b] if b < len(scene_list) else None
        sc_dict = {}
        tl_states = []
        if sc is not None:
            sc_dict = {
                "bboxes": sc["bboxes"].unsqueeze(0).to(DEVICE),
                "positions": sc["positions"].unsqueeze(0).to(DEVICE),
                "class_names": sc["class_names"],
                "target_idx": 0,
                "traffic_light_states": sc.get("traffic_light_states", []),
            }
            tl_states = sc.get("traffic_light_states", [])

        signal_list.append(compute_signal_factor(tl_states))
        env_np = extract_environment_features(
            scene_data=sc_dict, norm=norm_tensor, target_idx=0,
            junction_roi=junction_roi, crosswalk_roi=crosswalk_roi,
        )
        env_list.append(env_np)

    return {
        "p_cross": torch.tensor(p_cross_list, device=DEVICE, dtype=torch.float32),
        "signal_factor": torch.tensor(signal_list, device=DEVICE, dtype=torch.float32),
        "env_feat": torch.from_numpy(np.stack(env_list, axis=0)).float().to(DEVICE),
        "samples": samples,
        "log_probs": log_probs,
    }


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════

def train_epoch(
    flowchain: FlowChainBase,
    risk_head: RiskRegressionHead,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    p_cross_estimator: CrossingProbabilityEstimator,
    norm_tensor: torch.Tensor,
    junction_roi: list = None,
    crosswalk_roi: list = None,
) -> float:
    """Train RiskRegressionHead for one epoch. FlowChain stays frozen, batched forward."""
    flowchain.eval()
    risk_head.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        B = batch["obs_trajectory"].shape[0]
        obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor.to(DEVICE)
        scene_list = batch.get("scene_list", [None] * B)
        labels = batch.get("is_violation")
        if isinstance(labels, torch.Tensor):
            lbl_t = labels.float().to(DEVICE)
        else:
            lbl_t = torch.zeros(B, device=DEVICE)

        # Stage 1: batched FlowChain forward + per-sample features (no grad)
        feat = compute_stage1_features_batch(
            flowchain, obs, scene_list, norm_tensor,
            p_cross_estimator, NUM_MC_TRAIN, junction_roi, crosswalk_roi,
        )

        # Stage 2: forward + loss
        optimizer.zero_grad()

        risk_out = risk_head.forward(
            p_cross=feat["p_cross"],
            env_feat=feat["env_feat"],
            signal_factor=feat["signal_factor"],
        )
        risk = risk_out["risk"]  # (B,)

        loss = risk_head.compute_loss(risk, lbl_t)

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(risk_head.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    flowchain: FlowChainBase,
    risk_head: RiskRegressionHead,
    test_loader: DataLoader,
    p_cross_estimator: CrossingProbabilityEstimator,
    norm_tensor: torch.Tensor,
    junction_roi: list = None,
    crosswalk_roi: list = None,
) -> dict:
    """Evaluate on test set. Batched forward. Returns metrics dict."""
    flowchain.eval()
    risk_head.eval()

    all_ade, all_fde = [], []
    all_risks, all_labels = [], []
    all_p_cross, all_signals = [], []

    for batch in tqdm(test_loader, desc="Eval", leave=False):
        B = batch["obs_trajectory"].shape[0]
        obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor.to(DEVICE)
        target = batch["target_trajectory"].to(DEVICE) / norm_tensor.to(DEVICE)
        scene_list = batch.get("scene_list", [None] * B)
        labels = batch.get("is_violation")

        # Stage 1: batched
        feat = compute_stage1_features_batch(
            flowchain, obs, scene_list, norm_tensor,
            p_cross_estimator, NUM_MC_EVAL, junction_roi, crosswalk_roi,
        )

        # Stage 2: batched
        risk_out = risk_head.forward(
            p_cross=feat["p_cross"],
            env_feat=feat["env_feat"],
            signal_factor=feat["signal_factor"],
        )
        risks = risk_out["risk"]  # (B,)

        for b in range(B):
            all_risks.append(float(risks[b].item()))
            lbl = float(labels[b].item()) if isinstance(labels, torch.Tensor) and labels.numel() > b else 0.0
            all_labels.append(lbl)
            all_p_cross.append(float(feat["p_cross"][b].item()))
            all_signals.append(float(feat["signal_factor"][b].item()))

        # Trajectory metrics (Best-of-N, pixel space)
        if feat["samples"] is not None:
            samples = feat["samples"]        # (N, B, pred_len, 2)
            for b in range(B):
                s = samples[:, b]            # (N, pred_len, 2)
                samples_px = s * norm_tensor.to(DEVICE)
                target_px = (target[b] * norm_tensor.to(DEVICE)).unsqueeze(0)
                diff = samples_px - target_px
                l2 = torch.sqrt((diff ** 2).sum(dim=-1))
                ade_per_sample = l2.mean(dim=-1)
                best_idx = ade_per_sample.argmin()
                all_ade.append(float(ade_per_sample[best_idx]))
                all_fde.append(float(l2[best_idx, -1]))

    risks = np.array(all_risks)
    labels = np.array(all_labels)
    p_cross_arr = np.array(all_p_cross)
    signals_arr = np.array(all_signals)

    # Threshold search
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
            best_f1, best_th = f1, th

    # Classification metrics at best threshold
    best_preds = (risks >= best_th).astype(int)
    metrics = compute_classification_metrics(labels, best_preds, risks)

    # Distribution stats
    pos_mask = labels == 1
    neg_mask = labels == 0

    return {
        "ade": float(np.median(all_ade)) if all_ade else 0.0,
        "fde": float(np.median(all_fde)) if all_fde else 0.0,
        "ade_mean": float(np.mean(all_ade)) if all_ade else 0.0,
        "fde_mean": float(np.mean(all_fde)) if all_fde else 0.0,
        "auc": metrics.get("AUC", 0.0),
        "f1": best_f1,
        "best_threshold": best_th,
        "accuracy": metrics.get("Accuracy", 0.0),
        "n_samples": len(labels),
        "n_positive": int(pos_mask.sum()),
        "pos_mean_risk": float(risks[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0,
        "neg_mean_risk": float(risks[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0,
        "pos_mean_p_cross": float(p_cross_arr[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0,
        "neg_mean_p_cross": float(p_cross_arr[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0,
        "pos_mean_signal": float(signals_arr[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0,
        "neg_mean_signal": float(signals_arr[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Risk Regression Training (FlowChain)")
    parser.add_argument("--quick", action="store_true", help="Use reduced dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default="data/processed/trajectories")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--checkpoint", default="checkpoints/flowchain_best.pt",
                        help="Path to pre-trained FlowChain checkpoint")
    parser.add_argument("--save-path", default="checkpoints/risk_regression_head_flowchain.pt")
    args = parser.parse_args()

    # ── Config & Geometry ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    stop_line, crosswalk_roi, junction_roi = None, None, None
    for key in ("intersection_A", "intersection_B"):
        c = config.get(key, {})
        sl = c.get("stop_line")
        cw = c.get("crosswalk_roi")
        jr = c.get("junction_roi")
        if sl and len(sl) >= 4:
            stop_line = [float(x) for x in sl]
        if cw and len(cw) >= 3:
            if isinstance(cw[0], (list, tuple)):
                crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
            else:
                crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
        if jr and len(jr) >= 3:
            junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
        if stop_line or crosswalk_roi:
            break

    # Use junction_roi as crossing region for P_cross (covers the road area)
    crossing_region = junction_roi if junction_roi else crosswalk_roi
    logger.info(f"Crossing region: {crossing_region is not None}, "
                f"Junction: {junction_roi is not None}, "
                f"Stop line: {stop_line is not None}")

    # ── Load Data ──
    logger.info("Loading datasets...")
    _, _, _, train_scene, val_scene, test_scene = load_split_datasets(
        args.processed_dir, label_dir=args.label_dir, quick=args.quick,
    )

    def count_violations(ds):
        n = 0
        for i in range(len(ds)):
            s = ds.dataset[ds.indices[i]] if hasattr(ds, 'indices') else ds[i]
            if s.get("is_violation", False):
                n += 1
        return n

    for name, ds in [("train", train_scene), ("val", val_scene), ("test", test_scene)]:
        nv = count_violations(ds)
        logger.info(f"  {name}: {len(ds)} samples, {nv} violations ({100*nv/max(1,len(ds)):.1f}%)")

    # ── Filter to crossing candidates ──
    #   GT enters junction  OR  heading 80-90° to stop line
    from data.dataset import is_crossing_candidate as _icc

    def _filter_candidates(ds, use_future_gt):
        base = ds.dataset if hasattr(ds, 'dataset') else ds
        indices = list(ds.indices) if hasattr(ds, 'indices') else list(range(len(ds)))
        kept = []
        for idx in indices:
            s = base[idx]
            obs = s["obs_trajectory"].numpy() if hasattr(s["obs_trajectory"], "numpy") else s["obs_trajectory"]
            tgt = s.get("target_trajectory")
            if tgt is not None:
                tgt = tgt.numpy() if hasattr(tgt, "numpy") else tgt
            if not use_future_gt:
                tgt = None
            if _icc(obs, tgt, crosswalk_roi, stop_line, junction_roi):
                kept.append(idx)
        from torch.utils.data import Subset
        return Subset(base, kept)

    train_scene = _filter_candidates(train_scene, use_future_gt=True)
    val_scene = _filter_candidates(val_scene, use_future_gt=True)
    test_scene = _filter_candidates(test_scene, use_future_gt=True)

    for name, ds in [("train", train_scene), ("val", val_scene), ("test", test_scene)]:
        nv_val = sum(1 for i in range(len(ds))
                     if (ds.dataset[ds.indices[i]] if hasattr(ds, 'indices') else ds[i]).get("is_violation", False))
        logger.info(f"  {name} (filtered): {len(ds)} samples, {nv_val} violations "
                    f"({100*nv_val/max(1,len(ds)):.1f}%)")

    train_loader = DataLoader(train_scene, batch_size=16, shuffle=True,
                              collate_fn=trajectory_collate_fn)
    test_loader = DataLoader(test_scene, batch_size=1, shuffle=False,
                             collate_fn=trajectory_collate_fn)
    val_loader = DataLoader(val_scene, batch_size=1, shuffle=False,
                            collate_fn=trajectory_collate_fn)

    # ── Load FlowChain (frozen, unconditional) ──
    # Prefer fine-tuned checkpoint, fallback to original
    ft_checkpoint = args.checkpoint.replace(".pt", "_finetuned.pt")
    ckpt_path = ft_checkpoint if os.path.exists(ft_checkpoint) else args.checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_path = args.checkpoint  # let it fail below with warning
    logger.info(f"Loading FlowChain from {ckpt_path}...")
    flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        flowchain.load_state_dict(ckpt)
        logger.info(f"  Loaded checkpoint")
    else:
        logger.warning(f"  Checkpoint not found: {args.checkpoint} — using random init!")

    flowchain.eval()
    for p in flowchain.parameters():
        p.requires_grad = False  # FREEZE

    n_params = sum(p.numel() for p in flowchain.parameters())
    logger.info(f"  FlowChain params: {n_params:,} (FROZEN, unconditional)")

    # ── Init Stage 1 & Stage 2 ──
    p_cross_estimator = CrossingProbabilityEstimator(crossing_region=crossing_region)
    risk_head = RiskRegressionHead(env_dim=8, hidden_dim=32, dropout=0.1).to(DEVICE)

    n_risk_params = sum(p.numel() for p in risk_head.parameters())
    logger.info(f"  RiskRegressionHead params: {n_risk_params:,} (TRAINABLE)")

    norm_tensor = NORM.to(DEVICE)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(risk_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Pre-evaluation (before training) ──
    logger.info("Pre-training evaluation...")
    pre_metrics = evaluate(
        flowchain, risk_head, test_loader, p_cross_estimator,
        norm_tensor, junction_roi, crosswalk_roi,
    )
    logger.info(f"  Pre-train: AUC={pre_metrics['auc']:.4f}, "
                f"F1={pre_metrics['f1']:.4f}, "
                f"PosRisk={pre_metrics['pos_mean_risk']:.4f}, "
                f"NegRisk={pre_metrics['neg_mean_risk']:.4f}, "
                f"ADE={pre_metrics['ade']:.2f}px, FDE={pre_metrics['fde']:.2f}px")

    # ── Train ──
    logger.info(f"Training RiskRegressionHead for {args.epochs} epochs...")
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            flowchain, risk_head, train_loader, optimizer,
            p_cross_estimator, norm_tensor, junction_roi, crosswalk_roi,
        )
        scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_metrics = evaluate(
                flowchain, risk_head, val_loader, p_cross_estimator,
                norm_tensor, junction_roi, crosswalk_roi,
            )
            logger.info(f"  E{epoch+1:3d}: loss={train_loss:.4f}, "
                        f"val AUC={val_metrics['auc']:.4f}, "
                        f"val F1={val_metrics['f1']:.4f}")

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                torch.save({
                    "model_state": risk_head.state_dict(),
                    "epoch": epoch + 1,
                    "val_f1": best_val_f1,
                    "val_auc": val_metrics['auc'],
                }, args.save_path)
        else:
            logger.info(f"  E{epoch+1:3d}: loss={train_loss:.4f}")

    # ── Final Evaluation ──
    logger.info("Final evaluation...")

    # Load best checkpoint
    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=DEVICE)
        risk_head.load_state_dict(ckpt["model_state"])
        logger.info(f"  Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}, val F1={ckpt.get('val_f1', 0):.4f})")

    test_metrics = evaluate(
        flowchain, risk_head, test_loader, p_cross_estimator,
        norm_tensor, junction_roi, crosswalk_roi,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("TWO-STAGE RISK REGRESSION (FLOWCHAIN) — RESULTS")
    print("=" * 60)

    print(f"\nDataset: {test_metrics['n_samples']} samples, "
          f"{test_metrics['n_positive']} violations "
          f"({100*test_metrics['n_positive']/max(1,test_metrics['n_samples']):.1f}%)")
    print(f"\nTrajectory (FlowChain, Best-of-100, pixel):")
    print(f"  ADE: median={test_metrics['ade']:.2f}px, mean={test_metrics['ade_mean']:.2f}px")
    print(f"  FDE: median={test_metrics['fde']:.2f}px, mean={test_metrics['fde_mean']:.2f}px")
    print(f"\nRisk Regression (Stage 2):")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"  F1:       {test_metrics['f1']:.4f}  (threshold={test_metrics['best_threshold']:.2f})")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nRisk Distribution:")
    print(f"  Positive: mean={test_metrics['pos_mean_risk']:.4f}")
    print(f"  Negative: mean={test_metrics['neg_mean_risk']:.4f}")
    print(f"\nP_cross Distribution (Stage 1):")
    print(f"  Positive: mean={test_metrics['pos_mean_p_cross']:.4f}")
    print(f"  Negative: mean={test_metrics['neg_mean_p_cross']:.4f}")
    print(f"\nSignal Distribution:")
    print(f"  Positive: mean={test_metrics['pos_mean_signal']:.4f}")
    print(f"  Negative: mean={test_metrics['neg_mean_signal']:.4f}")

    # ── Save ──
    with open("risk_regression_results_flowchain.csv", "w") as f:
        f.write("Method,ADEmedian,ADEmean,FDEmedian,FDEmean,AUC,F1,BestTh,Accuracy,PosRisk,NegRisk,PosPcross,NegPcross\n")
        f.write(f"FlowChain-TwoStage,{test_metrics['ade']:.2f},{test_metrics['ade_mean']:.2f},")
        f.write(f"{test_metrics['fde']:.2f},{test_metrics['fde_mean']:.2f},")
        f.write(f"{test_metrics['auc']:.4f},{test_metrics['f1']:.4f},{test_metrics['best_threshold']:.2f},")
        f.write(f"{test_metrics['accuracy']:.4f},{test_metrics['pos_mean_risk']:.4f},{test_metrics['neg_mean_risk']:.4f},")
        f.write(f"{test_metrics['pos_mean_p_cross']:.4f},{test_metrics['neg_mean_p_cross']:.4f}\n")
    logger.info("Saved to risk_regression_results_flowchain.csv")

    return test_metrics


if __name__ == "__main__":
    main()
