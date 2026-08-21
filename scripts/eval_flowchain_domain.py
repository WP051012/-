"""
Pure FlowChain 训练 + 线3 分类器评估（域划分，匹配 FOMAML 条件）
===============================================================
Phase 1 — 在穿越候选上重新训练 FlowChain:
  - 域划分: train [0,1,2,4,6], val [3], test [5]
  - 穿越候选过滤: is_crossing_candidate
  - 损失: NLL + λ·ADE
  - 保存: checkpoints/flowchain_domain_filtered.pt

Phase 2 — 评估:
  - 轨迹指标: Best-of-100 ADE/FDE (pixel)
  - 分类: P_cross + signal_factor gate → 阈值搜索 → AUC/F1

用法:
    # 完整流程: 训练 + 评估
    python scripts/eval_flowchain_domain.py --train --epochs 50

    # 仅评估 (用已有 checkpoint)
    python scripts/eval_flowchain_domain.py --flowchain-ckpt checkpoints/flowchain_domain_filtered.pt
"""

import sys, os, yaml, logging, argparse, json, time, gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import (
    CrossingProbabilityEstimator,
)
from src.evaluation import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])

# FOMAML 域划分
META_TRAIN = [0, 1, 2, 4, 6]
META_VAL = [3]
META_TEST = [5]


# ======================================================================
# Config helpers
# ======================================================================

def parse_geometry(annotations_dir="data/annotations"):
    """从 annotation JSON 加载 geometry.

    JSON 格式: {"stop_line": {x1,y1,x2,y2}, "junction_roi": {x1,y1,x2,y2}(矩形)}
    junction_roi 矩形 → 4顶点多边形. 无 crosswalk_roi → 用 junction_roi 替代.
    """
    annot_dir = Path(annotations_dir)
    if not annot_dir.exists():
        logger.warning(f"Annotations dir not found: {annotations_dir}")
        return None, None, None

    geo_a = None   # intersection_A (timing)
    geo_b = None   # intersection_B (numbered)

    for af in sorted(annot_dir.glob("*.json")):
        data = json.loads(af.read_text())
        video = data.get("video", af.stem)
        sl = data.get("stop_line", {})
        jr = data.get("junction_roi", {})

        sl_list = None
        jr_poly = None
        if sl and all(k in sl for k in ("x1","y1","x2","y2")):
            sl_list = [float(sl["x1"]), float(sl["y1"]), float(sl["x2"]), float(sl["y2"])]
        if jr and all(k in jr for k in ("x1","y1","x2","y2")):
            x1,y1 = float(jr["x1"]), float(jr["y1"])
            x2,y2 = float(jr["x2"]), float(jr["y2"])
            jr_poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]

        geo = {"stop_line": sl_list, "junction_roi": jr_poly}
        if "timing" in video:
            geo_a = geo
        else:
            geo_b = geo

    # 取第一个完整的 geometry
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"] and geo["stop_line"]:
            logger.info(f"  Using geometry: stop_line={geo['stop_line']}, "
                        f"junction_roi={len(geo['junction_roi'])}pts")
            # crosswalk_roi 不存在, 用 junction_roi 替代
            return geo["junction_roi"], geo["junction_roi"], geo["stop_line"]

    # 降级
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"]:
            return geo["junction_roi"], geo["junction_roi"], geo.get("stop_line")
    return None, None, None


def split_by_domain(dataset, domains, name=""):
    indices = []
    for i, s in enumerate(dataset.samples):
        if s.get("domain_id", -1) in domains:
            indices.append(i)
    n_viol = sum(1 for i in indices if dataset.samples[i].get("is_violation", False))
    logger.info(f"  {name}: {len(indices)} samples, {n_viol} violations "
                f"({100*n_viol/max(1,len(indices)):.1f}%)")
    return indices


def filter_candidates(dataset, indices, junction_roi, stop_line, crosswalk_roi,
                       use_future_gt=True, name=""):
    kept = []
    for idx in indices:
        s = dataset.samples[idx]
        obs = s["obs_positions"]
        tgt = s.get("target_positions") if use_future_gt else None
        if is_crossing_candidate(obs, tgt, crosswalk_roi, stop_line, junction_roi):
            kept.append(idx)
    n_viol = sum(1 for i in kept if dataset.samples[i].get("is_violation", False))
    logger.info(f"  {name} (filtered): {len(kept)}/{len(indices)} samples, "
                f"{n_viol} violations ({100*n_viol/max(1,len(kept)):.1f}%)")
    return kept


def search_threshold(risks, labels):
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
    return best_th, best_f1


def confusion_metrics(y_true, y_pred):
    """Confusion matrix + 检出率(Recall)/漏检率(FNR)/错检率(FPR) at a threshold.

    漏检率 FNR = FN/(TP+FN) = 1 - Recall      (真实违规中被漏掉的比例)
    错检率 FPR = FP/(FP+TN) = 1 - Specificity (真实非违规中被误报的比例)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn + 1e-8)
    fnr = fn / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr = fp / (tn + fp + 1e-8)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Recall=recall, FNR=fnr, Specificity=specificity, FPR=fpr)


# ======================================================================
# Phase 1: FlowChain Training
# ======================================================================

def train_flowchain(
    flowchain: FlowChainBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    ade_weight: float,
    save_path: str,
    resume_ckpt: dict = None,
):
    """Train FlowChain with NLL + λ·ADE loss. Save best checkpoint by val loss.

    Args:
        resume_ckpt: if given, resume from this checkpoint dict with keys
            model_state, epoch, optimizer_state, scheduler_state, best_val_loss
    """
    optimizer = torch.optim.AdamW(flowchain.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    norm_tensor = NORM.to(DEVICE)
    cond_dim = flowchain._cond_dim
    best_val_loss = float("inf")
    best_state = None
    start_epoch = 0

    # Resume if checkpoint has training state
    if resume_ckpt is not None:
        start_epoch = resume_ckpt.get("epoch", 0)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from epoch {start_epoch}/{epochs} "
                    f"(best_val_loss={best_val_loss:.4f})")
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            logger.info("  Optimizer & scheduler state restored")
        except (KeyError, ValueError) as e:
            logger.warning(f"  Could not restore optimizer/scheduler: {e}")
        # Restore best model state
        if "model_state" in resume_ckpt and start_epoch > 0:
            best_state = {k: v.clone() for k, v in resume_ckpt["model_state"].items()}

    def _compute_nll(flowchain, obs, target):
        """Teacher-forced NLL on FlowChain (unconditional)."""
        B = obs.shape[0]
        zero_cond = torch.zeros(B, cond_dim, device=obs.device)
        log_prob = flowchain.predictor.log_prob(
            obs_trajectory=obs, target=target, perception_c=zero_cond)
        return -log_prob.mean()

    for epoch in range(start_epoch, epochs):
        # ── Train ──
        flowchain.train()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"FC-E{epoch}", leave=False):
            obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
            target = batch["target_trajectory"].to(DEVICE) / norm_tensor

            optimizer.zero_grad()

            nll = _compute_nll(flowchain, obs, target)
            pred = flowchain(obs_trajectory=obs, num_samples=1)
            ade = torch.sqrt(((pred["mean"] - target) ** 2).sum(dim=-1) + 1e-8).mean()

            loss = nll + ade_weight * ade

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flowchain.parameters(), 10.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # ── Validate ──
        flowchain.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
                target = batch["target_trajectory"].to(DEVICE) / norm_tensor

                nll = _compute_nll(flowchain, obs, target)
                pred = flowchain(obs_trajectory=obs, num_samples=1)
                ade = torch.sqrt(((pred["mean"] - target) ** 2).sum(dim=-1) + 1e-8).mean()
                val_loss += (nll + ade_weight * ade).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        logger.info(f"  E{epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in flowchain.state_dict().items()}
            logger.info(f"    *best*")

        # Periodic save every 5 epochs (safety net)
        if (epoch + 1) % 5 == 0:
            ckpt_dir = os.path.dirname(save_path) or "."
            os.makedirs(ckpt_dir, exist_ok=True)
            save_dict = {
                "model_state": {k: v.cpu().clone() for k, v in flowchain.state_dict().items()},
                "epoch": epoch + 1,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(save_dict, save_path)
            logger.info(f"  [checkpoint saved at epoch {epoch + 1}]")

    # Load best state
    flowchain.load_state_dict(best_state)
    flowchain.eval()

    # Final save
    ckpt_dir = os.path.dirname(save_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    save_dict = {
        "model_state": best_state,
        "epoch": epochs,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(save_dict, save_path)
    logger.info(f"Saved best FlowChain to {save_path} (val_loss={best_val_loss:.4f})")

    return best_val_loss


# ======================================================================
# Phase 2: Evaluation
# ======================================================================

@torch.no_grad()
def evaluate_split(dataset, indices, flowchain, p_cross_est, norm_tensor,
                    num_mc, split_name=""):
    all_ade, all_fde = [], []
    all_risks, all_labels = [], []
    all_p_cross, all_signals = [], []

    for idx in tqdm(indices, desc=split_name, leave=False):
        sample = dataset[idx]

        obs = sample["obs_trajectory"].to(DEVICE).unsqueeze(0) / norm_tensor.to(DEVICE)
        target_px = sample["target_trajectory"].to(DEVICE).unsqueeze(0)
        is_viol = float(sample.get("is_violation_window", sample.get("is_violation", False)))

        # FlowChain 采样
        pred = flowchain(obs_trajectory=obs, num_samples=num_mc)
        samples = pred.get("samples")

        if samples is None:
            all_risks.append(0.0); all_labels.append(is_viol)
            all_p_cross.append(0.0); all_signals.append(0.5)
            continue

        if samples.dim() == 4:
            s = samples[:, 0]
        else:
            s = samples

        # Trajectory (Best-of-N)
        samples_px = s * norm_tensor.to(DEVICE)
        diff = samples_px - target_px
        l2 = torch.sqrt((diff ** 2).sum(dim=-1))
        ade_per = l2.mean(dim=-1)
        best_idx = ade_per.argmin()
        all_ade.append(float(ade_per[best_idx]))
        all_fde.append(float(l2[best_idx, -1]))

        # P_cross
        p_cross = float(p_cross_est.compute_p_cross(samples_px))
        all_p_cross.append(p_cross)

        # Signal gate: red during the prediction window (where crossing occurs).
        # Aligned with the label definition (violation = crossing while red),
        # instead of the old last-obs-frame lookup which missed lights turning
        # red between observation end and the crossing.
        scene = sample.get("scene", {})
        pred_tl_states = scene.get("pred_traffic_light_states", [])
        is_red = 1.0 if any(s == "red" for s in pred_tl_states) else 0.0
        all_signals.append(is_red)

        # Risk = P_cross × is_red
        all_risks.append(p_cross * is_red)
        all_labels.append(is_viol)

    ade_arr = np.array(all_ade) if all_ade else np.array([float('nan')])
    fde_arr = np.array(all_fde) if all_fde else np.array([float('nan')])

    return {
        "ade_median": float(np.median(ade_arr)),
        "ade_mean": float(np.mean(ade_arr)),
        "fde_median": float(np.median(fde_arr)),
        "fde_mean": float(np.mean(fde_arr)),
    }, np.array(all_risks), np.array(all_labels), np.array(all_p_cross), np.array(all_signals)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FlowChain Train + Eval (domain split)")
    # ── Data ──
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    parser.add_argument("--max-samples", type=int, default=0, help="数据集最大样本数 (0=全部)")
    parser.add_argument("--max-scene", type=int, default=20000, help="Scene data 最大加载数")

    # ── Training ──
    parser.add_argument("--train", action="store_true", help="启用 Phase 1: FlowChain 训练")
    parser.add_argument("--resume", action="store_true", help="从 --flowchain-ckpt 断点续训")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ade-weight", type=float, default=1.0)

    # ── Checkpoint ──
    parser.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_best_finetuned.pt",
                        help="预训练 FlowChain (--train 时作为初始化, eval 时直接加载)")
    parser.add_argument("--save-ckpt", default="checkpoints/flowchain_domain_filtered.pt",
                        help="训练后保存路径")

    # ── Evaluation ──
    parser.add_argument("--num-mc", type=int, default=100, help="Monte Carlo 采样数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (采样可复现)")
    parser.add_argument("--skip-eval", action="store_true", help="仅训练, 跳过评估")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Geometry (从 annotation JSON 加载) ──
    junction_roi, crosswalk_roi, stop_line = parse_geometry()
    logger.info(f"Geometry: junction={junction_roi is not None}, "
                f"crosswalk={crosswalk_roi is not None}, stop_line={stop_line is not None}")

    # ── Domain map ──
    with open(args.domain_map) as f:
        domain_label_map = json.load(f)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: 在 trajectory_only 上训练 FlowChain
    # ═══════════════════════════════════════════════════════════
    if args.train:
        logger.info("=" * 55)
        logger.info("PHASE 1: Train FlowChain (filtered, domain split)")
        logger.info("=" * 55)

        logger.info("Loading trajectory dataset...")
        ds_train = TrajectoryDataset(
            data_dir=args.data_dir,
            obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
            target_classes=["pedestrian"],
            mode="trajectory_only",
            max_samples=args.max_samples,
            domain_label_map=domain_label_map,
        )
        logger.info(f"  {len(ds_train)} total samples")

        # 域划分 + 过滤
        train_all = split_by_domain(ds_train, META_TRAIN, "train_all")
        val_all = split_by_domain(ds_train, META_VAL, "val_all")
        test_all_for_ref = split_by_domain(ds_train, META_TEST, "test_all(ref)")

        train_idx = filter_candidates(ds_train, train_all, junction_roi, stop_line,
                                       crosswalk_roi, use_future_gt=True, name="train")
        val_idx = filter_candidates(ds_train, val_all, junction_roi, stop_line,
                                     crosswalk_roi, use_future_gt=True, name="val")

        if len(train_idx) == 0:
            logger.error("Training set empty after filtering!")
            sys.exit(1)

        train_loader = DataLoader(
            Subset(ds_train, train_idx), batch_size=args.batch_size,
            shuffle=True, collate_fn=trajectory_collate_fn, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(ds_train, val_idx), batch_size=args.batch_size,
            shuffle=False, collate_fn=trajectory_collate_fn, num_workers=2, pin_memory=True,
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Init FlowChain (从预训练 checkpoint 或随机)
        flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
        resume_ckpt = None
        if os.path.exists(args.flowchain_ckpt):
            logger.info(f"Loading checkpoint: {args.flowchain_ckpt}")
            ckpt = torch.load(args.flowchain_ckpt, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("model_state") or ckpt.get("model") or ckpt
            # FlowChainBase 的 key 是 predictor.model.*
            # flowchain_best_finetuned.pt 保存的就是 FlowChainBase.state_dict()
            # 如果 key 是 flow_chain.model.* → strip flow_chain. → predictor.model.*
            if any(k.startswith("flow_chain.") for k in sd.keys()):
                sd = {k.replace("flow_chain.", "predictor."): v for k, v in sd.items()}
                logger.info("  Remapped flow_chain. → predictor.")
            missing, unexpected = flowchain.load_state_dict(sd, strict=False)
            if missing:
                logger.info(f"  Missing keys: {len(missing)} (first: {missing[:3]})")
            # Check if resume
            if args.resume:
                resume_ckpt = ckpt
                trained_epochs = resume_ckpt.get("epoch", 0)
                if trained_epochs >= args.epochs:
                    logger.info(f"  Already trained {trained_epochs}/{args.epochs} epochs. Skipping training.")
                    args.train = False  # skip to eval
                else:
                    logger.info(f"  Resume mode: will train epochs {trained_epochs+1}..{args.epochs}")
        elif args.resume:
            logger.warning("--resume set but no checkpoint found, training from scratch")

        n_params = sum(p.numel() for p in flowchain.parameters())
        logger.info(f"FlowChain params: {n_params:,}")
        logger.info(f"Training: epochs={args.epochs}, lr={args.lr}, "
                    f"batch={args.batch_size}, ade_weight={args.ade_weight}")

        train_flowchain(
            flowchain, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, ade_weight=args.ade_weight,
            save_path=args.save_ckpt, resume_ckpt=resume_ckpt,
        )

        logger.info("Phase 1 complete.\n")

        # Free trajectory dataset memory
        del ds_train
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 在 with_scene 上评估分类
    # ═══════════════════════════════════════════════════════════
    if args.skip_eval:
        logger.info("Skipping evaluation (--skip-eval).")
        return

    logger.info("=" * 55)
    logger.info("PHASE 2: Evaluate (domain split, with_scene)")
    logger.info("=" * 55)

    # 加载 FlowChain (训练后的 或 指定的)
    ckpt_path = args.save_ckpt if (args.train and os.path.exists(args.save_ckpt)) else args.flowchain_ckpt
    logger.info(f"Loading FlowChain: {ckpt_path}")
    flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state") or ckpt.get("model") or ckpt
    if any(k.startswith("flow_chain.") for k in sd.keys()):
        sd = {k.replace("flow_chain.", "predictor."): v for k, v in sd.items()}
    flowchain.load_state_dict(sd, strict=False)
    flowchain.eval()
    logger.info(f"  Params: {sum(p.numel() for p in flowchain.parameters()):,}")

    # 加载 scene 数据集 (eval 需要 traffic_light_states)
    logger.info("Loading eval dataset (with_scene mode)...")
    ds_eval = TrajectoryDataset(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="with_scene",
        max_scene_samples=args.max_scene,
        max_samples=args.max_samples,
        domain_label_map=domain_label_map,
        force_scene=True,
        crossing_region=junction_roi,
    )
    logger.info(f"  {len(ds_eval)} total samples")

    # 域划分 + 过滤
    test_all = split_by_domain(ds_eval, META_TEST, "test_all(D5)")
    val_all = split_by_domain(ds_eval, META_VAL, "val_all(D3)")

    test_idx = filter_candidates(ds_eval, test_all, junction_roi, stop_line,
                                  crosswalk_roi, use_future_gt=True, name="test")
    val_idx = filter_candidates(ds_eval, val_all, junction_roi, stop_line,
                                 crosswalk_roi, use_future_gt=True, name="val")

    if len(test_idx) == 0:
        logger.error("Test set empty after filtering! Check domain/geometry config.")
        sys.exit(1)

    # P_cross estimator
    crossing_region = junction_roi if junction_roi else crosswalk_roi
    p_cross_est = CrossingProbabilityEstimator(crossing_region=crossing_region)
    norm_tensor = NORM.to(DEVICE)

    # ── Val: 搜阈值 ──
    logger.info(f"\nEvaluating val (D3): {len(val_idx)} samples...")
    val_traj, val_risks, val_labels, val_pc, val_sig = evaluate_split(
        ds_eval, val_idx, flowchain, p_cross_est, norm_tensor,
        num_mc=args.num_mc, split_name="Val(D3)",
    )
    best_th, best_f1 = search_threshold(val_risks, val_labels)
    logger.info(f"  Best threshold: {best_th:.3f} (F1={best_f1:.4f})")

    # ── Test: 最终评估 ──
    logger.info(f"\nEvaluating test (D5): {len(test_idx)} samples...")
    test_traj, test_risks, test_labels, test_pc, test_sig = evaluate_split(
        ds_eval, test_idx, flowchain, p_cross_est, norm_tensor,
        num_mc=args.num_mc, split_name="Test(D5)",
    )
    test_preds = (test_risks >= best_th).astype(int)
    cls_metrics = compute_classification_metrics(test_labels, test_preds, test_risks)
    cm = confusion_metrics(test_labels, test_preds)                                   # @ val threshold
    cm_05 = confusion_metrics(test_labels, (test_risks >= 0.5).astype(int))           # @ fixed 0.5

    # Threshold-robust F1 reporting (val D3 is a different domain and may be
    # small, so we do not hang the headline F1 on the val-searched threshold):
    #   F1@0.5        : fixed operational threshold (reproducible)
    #   F1@val_th     : threshold searched on val (subject to domain shift)
    #   F1@test-best  : best achievable F1 on test (upper-bound reference)
    cls_05 = compute_classification_metrics(test_labels, (test_risks >= 0.5).astype(int), test_risks)
    f1_at_05 = cls_05.get("F1", 0.0)
    _test_best_th, f1_test_best = search_threshold(test_risks, test_labels)

    # ── Print Results ──
    print("\n" + "=" * 65)
    print("PURE FLOWCHAIN + 线3 分类器 (域划分, 穿越候选过滤)")
    print("=" * 65)
    print(f"\nFlowChain: {ckpt_path} ({sum(p.numel() for p in flowchain.parameters()):,} params)")
    print(f"Training:  {'filtered ' + str(len(test_idx)) if args.train else 'pretrained'}")
    print(f"Split:     train=D{set(META_TRAIN)}, val=D{set(META_VAL)}, test=D{set(META_TEST)}")
    print(f"Test set:  {len(test_idx)} samples, {int(test_labels.sum())} violations "
          f"({100*test_labels.sum()/max(1,len(test_labels)):.1f}%)")
    print(f"Threshold: {best_th:.3f} (from val D3, F1={best_f1:.4f})")
    print(f"NUM_MC:    {args.num_mc}")

    print(f"\n--- Trajectory (Best-of-{args.num_mc}, pixel) ---")
    print(f"  ADE: median={test_traj['ade_median']:.2f}px  mean={test_traj['ade_mean']:.2f}px")
    print(f"  FDE: median={test_traj['fde_median']:.2f}px  mean={test_traj['fde_mean']:.2f}px")

    print(f"\n--- Classification (线3: P_cross × signal gate) ---")
    print(f"  AUC:         {cls_metrics.get('AUC', 0):.4f}  (threshold-free, primary)")
    print(f"  F1@0.5:      {f1_at_05:.4f}  (fixed threshold)")
    print(f"  F1@val_th:   {best_f1:.4f}  (threshold={best_th:.3f}, from val D3)")
    print(f"  F1@test-best:{f1_test_best:.4f}  (threshold={_test_best_th:.3f}, upper bound)")
    print(f"  Accuracy:    {cls_metrics.get('Accuracy', 0):.4f}  (at val threshold)")

    print(f"\n--- 检出 / 漏检 / 错检 (test D5) ---")
    print(f"  @val_th({best_th:.3f}): Recall(检出)={cm['Recall']:.4f}  "
          f"FNR(漏检)={cm['FNR']:.4f}  FPR(错检)={cm['FPR']:.4f}")
    print(f"  @0.5      : Recall(检出)={cm_05['Recall']:.4f}  "
          f"FNR(漏检)={cm_05['FNR']:.4f}  FPR(错检)={cm_05['FPR']:.4f}")
    print(f"  混淆矩阵 @val_th: TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")

    # Per-signal breakdown (signal = red during prediction window)
    red_mask = test_sig >= 0.99
    print(f"\n--- P_cross Distribution (test D5) ---")
    print(f"  Red(pred win): {red_mask.sum()} samples, viol={((test_labels == 1) & red_mask).sum()}")
    if red_mask.sum() > 0:
        pc_red = test_pc[red_mask]; lbl_red = test_labels[red_mask]
        print(f"    Pos P_cross={pc_red[lbl_red == 1].mean():.4f}  Neg P_cross={pc_red[lbl_red == 0].mean():.4f}")

    non_red_mask = test_sig <= 0.01
    print(f"  Non-red(pred win): {non_red_mask.sum()} samples, viol={((test_labels == 1) & non_red_mask).sum()}")

    pos_risk = test_risks[test_labels == 1]
    neg_risk = test_risks[test_labels == 0]
    print(f"\n--- Risk Distribution ---")
    print(f"  Pos mean: {pos_risk.mean():.4f} (n={len(pos_risk)})")
    print(f"  Neg mean: {neg_risk.mean():.4f} (n={len(neg_risk)})")

    # Save CSV
    csv_path = "flowchain_domain_results.csv"
    with open(csv_path, "w") as f:
        f.write("Method,Split,ADEmedian,ADEmean,FDEmedian,FDEmean,AUC,F1_05,F1_val,F1_testbest,BestTh,Accuracy,"
                "NSamples,NPos,PosRisk,NegRisk,PosPcross,NegPcross,Checkpoint,NUM_MC\n")
        f.write(f"FlowChain,Domain-D5,"
                f"{test_traj['ade_median']:.2f},{test_traj['ade_mean']:.2f},"
                f"{test_traj['fde_median']:.2f},{test_traj['fde_mean']:.2f},"
                f"{cls_metrics.get('AUC',0):.4f},{f1_at_05:.4f},{best_f1:.4f},{f1_test_best:.4f},{best_th:.3f},"
                f"{cls_metrics.get('Accuracy',0):.4f},"
                f"{len(test_labels)},{int(test_labels.sum())},"
                f"{pos_risk.mean():.4f},{neg_risk.mean():.4f},"
                f"{test_pc[test_labels==1].mean():.4f},{test_pc[test_labels==0].mean():.4f},"
                f"{ckpt_path},{args.num_mc}\n")
    logger.info(f"Saved to {csv_path}")

    # Save per-sample risks + labels so any future metric (FNR/FPR/ROC/etc.)
    # can be computed post-hoc without re-running the full eval.
    np.savez("flowchain_test_preds.npz", risks=test_risks, labels=test_labels)
    logger.info("Saved per-sample predictions to flowchain_test_preds.npz")

    print("\nDone!")


if __name__ == "__main__":
    main()
