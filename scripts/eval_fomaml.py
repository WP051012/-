"""
FOMAML v2 Evaluation — domain-adaptive trajectory prediction
=============================================================
Matches eval_flowchain_domain.py conditions:
  - Same geometry, crossing-candidate filter
  - Same domain split (val D3, test D5)
  - Same metrics: Best-of-N ADE/FDE + P_cross classifier

Loads a FOMAML checkpoint, applies domain-conditioned modulation,
adapts with inner-loop SGD, then evaluates.

Usage:
    python scripts/eval_fomaml.py \
        --checkpoint checkpoints/fomaml_v2/best_fomaml.pt \
        --num-mc 100
"""

import sys, os, json, argparse, logging, time
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.perception_model import TrafficPerceptionModel
from src.modulation_net import ModulationNet
from src.classification.crossing_probability import CrossingProbabilityEstimator
from src.evaluation import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])

META_TRAIN = [0, 1, 2, 4, 6]
META_VAL = [3]
META_TEST = [5]


# ======================================================================
# Geometry (same as eval_flowchain_domain.py)
# ======================================================================

def parse_geometry(annotations_dir="data/annotations"):
    annot_dir = Path(annotations_dir)
    if not annot_dir.exists():
        return None, None, None
    geo_a, geo_b = None, None
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
            x1,y1,x2,y2 = float(jr["x1"]),float(jr["y1"]),float(jr["x2"]),float(jr["y2"])
            jr_poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
        geo = {"stop_line": sl_list, "junction_roi": jr_poly}
        if "timing" in video:
            geo_a = geo
        else:
            geo_b = geo
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"] and geo["stop_line"]:
            return geo["junction_roi"], geo["junction_roi"], geo["stop_line"]
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"]:
            return geo["junction_roi"], geo["junction_roi"], geo.get("stop_line")
    return None, None, None


# ======================================================================
# Trainable param enumeration (must match train_fomaml.py exactly)
# ======================================================================

def _enum_trainable_params(model: TrafficPerceptionModel):
    """Generator yielding (name, param) for all trainable FlowChain params
    in a fixed, deterministic order. MUST match train_fomaml.py exactly.

    Architecture: encoder (frozen) → encoder_adapter (trainable) → decoder → flow.
    Only adapter + flow BN are trained by FOMAML.
    """
    fc = model.flow_chain.model

    # Encoder adapter (trainable, ~2K params)
    if fc.use_adapter:
        for pname, p in fc.encoder_adapter.named_parameters():
            if p.requires_grad:
                yield f"enc_adapter.{pname}", p

    # Flow BatchNorm (trainable, ~16 params)
    for name, p in fc.flow.named_parameters():
        if ('log_gamma' in name or 'beta' in name) and p.requires_grad:
            yield f"flow.{name}", p


# ======================================================================
# Filter helpers
# ======================================================================

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
    logger.info(f"  {name}: {len(kept)}/{len(indices)} samples, "
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
    """Confusion matrix + 检出率(Recall)/漏检率(FNR)/错检率(FPR) at a threshold."""
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
# Model loading
# ======================================================================

def load_fomaml_model(config, perception_ckpt, flowchain_ckpt, fomaml_ckpt, device):
    """Build model, load all checkpoints, return (model, trainable_params,
    mod_net, domain_conditions, inner_cfg)."""
    # Build model (same as train_fomaml)
    model = TrafficPerceptionModel(config, stage=2).to(device).eval()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Load perception checkpoint
    logger.info(f"Loading perception: {perception_ckpt}")
    p_ckpt = torch.load(perception_ckpt, map_location=device, weights_only=False)
    p_sd = p_ckpt.get("model_state") or p_ckpt.get("model") or p_ckpt
    p_loaded = 0
    for k, v in p_sd.items():
        if k.startswith("flow_chain."):
            continue
        try:
            target = model
            for part in k.split(".")[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    break
            else:
                if hasattr(target, k.split(".")[-1]):
                    param = getattr(target, k.split(".")[-1])
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(v)
                        p_loaded += 1
        except Exception:
            pass
    logger.info(f"  {p_loaded} perception params")

    # Load FlowChain checkpoint
    logger.info(f"Loading FlowChain: {flowchain_ckpt}")
    f_ckpt = torch.load(flowchain_ckpt, map_location=device, weights_only=False)
    f_sd = f_ckpt.get("model") or f_ckpt.get("model_state") or f_ckpt
    if any(k.startswith("flow_chain.") for k in f_sd.keys()):
        f_sd = {k.replace("flow_chain.", ""): v for k, v in f_sd.items()}
    if any(k.startswith("predictor.") for k in f_sd.keys()):
        f_sd = {k.replace("predictor.", ""): v for k, v in f_sd.items()}
    fc_missing, _ = model.flow_chain.load_state_dict(f_sd, strict=False)
    logger.info(f"  FlowChain missing: {len(fc_missing)}")

    # Unfreeze trainable groups (same as build_model in train_fomaml.py):
    # encoder_adapter (trainable) + flow BatchNorm (log_gamma/beta).
    fc_model = model.flow_chain.model
    if fc_model.use_adapter:
        for p in fc_model.encoder_adapter.parameters():
            p.requires_grad_(True)
    for name, p in fc_model.flow.named_parameters():
        if 'log_gamma' in name or 'beta' in name:
            p.requires_grad_(True)

    trainable_params = OrderedDict(_enum_trainable_params(model))
    n_trainable = sum(p.numel() for p in trainable_params.values())
    logger.info(f"  Trainable: {n_trainable:,} params")

    # Load FOMAML checkpoint
    logger.info(f"Loading FOMAML: {fomaml_ckpt}")
    fml_ckpt = torch.load(fomaml_ckpt, map_location=device, weights_only=False)
    for name, p in trainable_params.items():
        if name in fml_ckpt.get("trainable_params", {}):
            p.data.copy_(fml_ckpt["trainable_params"][name])

    cfg = fml_ckpt.get("config", {})
    inner_cfg = {
        "inner_lr": cfg.get("inner_lr", 0.01),
        "inner_steps": cfg.get("inner_steps", 5),
        "ade_weight": cfg.get("ade_weight", 1.0),
        "batch_size": cfg.get("batch_size", 64),
        "lambda_feat": cfg.get("lambda_feat", 0.01),
        "max_delta_norm": cfg.get("max_delta_norm", 0.1),
    }

    # AdaBN: set alpha < 1 so flow BN blends batch + running stats during the
    # inner-loop adaptation. Pure running stats (alpha=1.0) after the adapter
    # shifts the encoder distribution → NaN. Matches train_fomaml.py.
    ada_alpha = fml_ckpt.get("config", {}).get("ada_alpha", 0.7)
    for m in model.flow_chain.model.flow.net:
        if hasattr(m, 'ada_alpha'):
            m.ada_alpha.fill_(ada_alpha)
    logger.info(f"  AdaBN alpha={ada_alpha}")

    # Modulation net
    mod_net = None
    domain_conditions = {}
    if "modulation_net" in fml_ckpt:
        param_shapes = [(name, p.shape) for name, p in trainable_params.items()]
        mod_net = ModulationNet(cond_dim=64, param_shapes=param_shapes).to(device)
        mod_net.load_state_dict(fml_ckpt["modulation_net"])
        mod_net.eval()
        logger.info(f"  ModulationNet loaded: {sum(p.numel() for p in mod_net.parameters()):,} params")
    if "domain_conditions" in fml_ckpt:
        domain_conditions = {int(k): v.to(device) for k, v in fml_ckpt["domain_conditions"].items()}

    return model, trainable_params, mod_net, domain_conditions, inner_cfg


# ======================================================================
# Evaluation
# ======================================================================

def _compute_loss(model, obs, target, ade_weight, lambda_feat):
    """ade_weight*ADE + lambda_feat*||adapter_residual||² — mirrors
    train_fomaml.compute_loss (ADE-only: NLL dropped). Returns a NaN tensor
    (requires_grad) on error so the caller skips the step."""
    B = obs.shape[0]
    device = obs.device
    # Normalize to [0,1] to match the pretrained FlowChain encoder (trained on
    # normalized coords; baseline eval feeds obs/3840, obs/2160).
    norm = torch.tensor([3840.0, 2160.0], device=device)
    obs_n = obs / norm
    target_n = target / norm
    zero_cond = torch.zeros(B, model.condition_dim, device=device)
    nan = torch.tensor(float('nan'), device=device, requires_grad=True)

    fc_model = model.flow_chain.model

    try:
        pred = model.flow_chain(obs_trajectory=obs_n, perception_c=zero_cond, num_samples=1)
        mean_pred = pred["mean"]
    except (ValueError, RuntimeError):
        return nan
    if not torch.isfinite(mean_pred).all():
        return nan
    diff_px = (mean_pred - target_n) * norm
    ade = torch.sqrt((diff_px ** 2).sum(dim=-1) + 1e-8).mean()

    # Read AFTER forward (see train_fomaml.compute_loss) so feat_shift refers to
    # THIS forward's residual, not the previous one's freed graph.
    feat_shift = torch.tensor(0.0, device=device)
    if fc_model.use_adapter:
        feat_shift = fc_model.encoder_adapter.get_feature_shift()

    return ade_weight * ade + lambda_feat * feat_shift


def evaluate_split(
    dataset, indices, model, trainable_params, mod_net, domain_conditions,
    inner_cfg, p_cross_est, norm_tensor, num_mc, split_name="",
):
    """Per-DOMAIN adaptation (matches train_fomaml.inner_loop) + full forward eval.

    For each domain in `indices`: save meta-state → apply modulation → adapt the
    trainable params ONCE on that domain's 70% support set (inner_steps batches,
    with grad clip + trust-region clamp) → evaluate ALL domain samples by
    best-of-N forward sampling → restore meta-state.

    (The previous per-sample inner loop was ~12× slower and did not match the
    training protocol, which adapts per-domain via support_loaders.)
    """
    inner_lr = inner_cfg["inner_lr"]
    inner_steps = inner_cfg["inner_steps"]
    ade_weight = inner_cfg["ade_weight"]
    lambda_feat = inner_cfg["lambda_feat"]
    max_delta_norm = inner_cfg["max_delta_norm"]
    batch_size = inner_cfg["batch_size"]
    support_ratio = 0.7

    all_ade, all_fde = [], []
    all_risks, all_labels = [], []
    all_p_cross, all_signals = [], []

    # Group indices by domain (val=D3, test=D5 are single-domain; kept general)
    by_domain = {}
    for idx in indices:
        did = dataset.samples[idx].get("domain_id", -1)
        by_domain.setdefault(did, []).append(idx)

    for did, did_indices in by_domain.items():
        # ── Save meta-state ──
        meta_state = {name: p.data.clone() for name, p in trainable_params.items()}

        # ── Apply modulation once (no-op when mod_net is None) ──
        cond = domain_conditions.get(did)
        if mod_net is not None and cond is not None:
            delta_flat = mod_net(cond.unsqueeze(0))
            mod_net.apply_delta(delta_flat, trainable_params, sign=+1)

        # ── Support set: 70% of domain samples, shuffled with seed 42 (train build_domain_split) ──
        rng = np.random.RandomState(42)
        idx_arr = np.array(did_indices)
        rng.shuffle(idx_arr)
        n_support = max(1, int(len(idx_arr) * support_ratio))
        support_idx = idx_arr[:n_support].tolist()

        # ── Inner-loop adaptation on support set (eval mode, blended BN via ada_alpha) ──
        model.eval()
        support_loader = DataLoader(
            Subset(dataset, support_idx),
            batch_size=min(batch_size, len(support_idx)),
            shuffle=True, collate_fn=trajectory_collate_fn,
            num_workers=0,
        )
        p_init = {name: p.data.clone() for name, p in trainable_params.items()}
        n_steps = 0
        for k, batch in enumerate(support_loader):
            if k >= inner_steps:
                break
            obs = batch["obs_trajectory"].to(DEVICE)
            target = batch["target_trajectory"].to(DEVICE)
            loss = _compute_loss(model, obs, target, ade_weight, lambda_feat)
            if torch.isfinite(loss):
                loss.backward()
                for name, p in trainable_params.items():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                        p.data -= inner_lr * p.grad
                        p.grad = None
                        delta = p.data - p_init[name]
                        delta_norm = delta.norm()
                        if delta_norm > max_delta_norm:
                            delta = delta * (max_delta_norm / delta_norm)
                            p.data = p_init[name] + delta
                n_steps += 1
        logger.info(f"  {split_name} domain D{did}: adapted {n_steps} support steps "
                    f"on {len(support_idx)} support samples")

        # ── Evaluate all samples in this domain (forward-only best-of-N) ──
        model.eval()
        zero_cond = torch.zeros(1, model.condition_dim, device=DEVICE)
        for idx in tqdm(did_indices, desc=split_name, leave=False):
            sample = dataset[idx]
            is_viol = float(sample.get("is_violation_window", sample.get("is_violation", False)))

            obs = sample["obs_trajectory"].to(DEVICE).unsqueeze(0) / norm_tensor.to(DEVICE)
            target = sample["target_trajectory"].to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                pred = model.flow_chain(obs_trajectory=obs, perception_c=zero_cond, num_samples=num_mc)
                samples = pred.get("samples")

            if samples is None:
                all_risks.append(0.0); all_labels.append(is_viol)
                all_p_cross.append(0.0); all_signals.append(0.5)
                continue

            if samples.dim() == 4:
                s = samples[:, 0]
            else:
                s = samples

            samples_px = s * norm_tensor.to(DEVICE)   # denormalize to pixels
            target_px = target      # (1, pred_len, 2) raw pixels

            diff = samples_px - target_px
            l2 = torch.sqrt((diff ** 2).sum(dim=-1))
            ade_per = l2.mean(dim=-1)
            best_idx = ade_per.argmin()
            all_ade.append(float(ade_per[best_idx]))
            all_fde.append(float(l2[best_idx, -1]))

            # P_cross
            p_cross = float(p_cross_est.compute_p_cross(samples_px))
            all_p_cross.append(p_cross)

            # Signal gate: red during the prediction window (aligned with the
            # window-level label = crossing while red). This is the corrected gate
            # that fixes the old last-obs-frame lookup (AUC=0.50 root cause).
            scene = sample.get("scene", {})
            pred_tl_states = scene.get("pred_traffic_light_states", [])
            is_red = 1.0 if any(s == "red" for s in pred_tl_states) else 0.0
            all_signals.append(is_red)

            # Risk = P_cross × is_red
            all_risks.append(p_cross * is_red)
            all_labels.append(is_viol)

        # ── Restore meta-state ──
        for name, p in trainable_params.items():
            p.data.copy_(meta_state[name])

    ade_arr = np.array(all_ade) if all_ade else np.array([float('nan')])
    fde_arr = np.array(all_fde) if all_fde else np.array([float('nan')])

    return {
        "ade_median": float(np.median(ade_arr)), "ade_mean": float(np.mean(ade_arr)),
        "fde_median": float(np.median(fde_arr)), "fde_mean": float(np.mean(fde_arr)),
    }, np.array(all_risks), np.array(all_labels), np.array(all_p_cross), np.array(all_signals)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FOMAML v2 Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="FOMAML checkpoint path")
    parser.add_argument("--perception-ckpt", default="checkpoints/stage1_best.pt")
    parser.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_domain_filtered.pt")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--label-dir", default="labels/")
    parser.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-scene", type=int, default=20000)
    parser.add_argument("--num-mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (采样可复现)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  seed={args.seed}")

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Geometry
    junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
    logger.info(f"Geometry: junction={junction_roi is not None}, stop_line={stop_line is not None}")

    # Domain map
    with open(args.domain_map) as f:
        domain_label_map = json.load(f)

    # Load model (no modulation net needed for pure eval)
    model, trainable_params, mod_net, domain_conditions, inner_cfg = \
        load_fomaml_model(config, args.perception_ckpt, args.flowchain_ckpt,
                          args.checkpoint, device)

    # Build eval dataset (with_scene mode for traffic lights)
    logger.info("Loading eval dataset (with_scene mode)...")
    ds_eval = TrajectoryDataset(
        data_dir=args.data_dir, label_dir=args.label_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="with_scene",
        max_scene_samples=args.max_scene, max_samples=args.max_samples,
        domain_label_map=domain_label_map,
        force_scene=True,
        crossing_region=junction_roi,
    )
    logger.info(f"  {len(ds_eval)} total samples")

    # Domain split + filter
    test_all = split_by_domain(ds_eval, META_TEST, "test_all(D5)")
    val_all = split_by_domain(ds_eval, META_VAL, "val_all(D3)")

    test_idx = filter_candidates(ds_eval, test_all, junction_roi, stop_line,
                                  crosswalk_roi, use_future_gt=True, name="test")
    val_idx = filter_candidates(ds_eval, val_all, junction_roi, stop_line,
                                 crosswalk_roi, use_future_gt=True, name="val")

    if len(test_idx) == 0:
        logger.error("Test set empty!")
        sys.exit(1)

    # P_cross estimator
    crossing_region = junction_roi if junction_roi else crosswalk_roi
    p_cross_est = CrossingProbabilityEstimator(crossing_region=crossing_region)
    norm_tensor = NORM.to(device)

    # Val: search threshold
    logger.info(f"\nEvaluating val (D3) with FOMAML adaptation: {len(val_idx)} samples...")
    val_traj, val_risks, val_labels, val_pc, val_sig = evaluate_split(
        ds_eval, val_idx, model, trainable_params, mod_net, domain_conditions,
        inner_cfg, p_cross_est, norm_tensor, num_mc=args.num_mc, split_name="Val(D3)",
    )
    best_th, best_f1 = search_threshold(val_risks, val_labels)
    logger.info(f"  Best threshold: {best_th:.3f} (F1={best_f1:.4f})")

    # Test: final evaluation
    logger.info(f"\nEvaluating test (D5) with FOMAML adaptation: {len(test_idx)} samples...")
    test_traj, test_risks, test_labels, test_pc, test_sig = evaluate_split(
        ds_eval, test_idx, model, trainable_params, mod_net, domain_conditions,
        inner_cfg, p_cross_est, norm_tensor, num_mc=args.num_mc, split_name="Test(D5)",
    )
    test_preds = (test_risks >= best_th).astype(int)
    cls_metrics = compute_classification_metrics(test_labels, test_preds, test_risks)
    cm = confusion_metrics(test_labels, test_preds)                                   # @ val threshold
    cm_05 = confusion_metrics(test_labels, (test_risks >= 0.5).astype(int))           # @ fixed 0.5
    cls_05 = compute_classification_metrics(test_labels, (test_risks >= 0.5).astype(int), test_risks)
    f1_at_05 = cls_05.get("F1", 0.0)
    _test_best_th, f1_test_best = search_threshold(test_risks, test_labels)

    # Print results
    print("\n" + "=" * 65)
    print("FOMAML v2 + GAT Condition (域划分, 穿越候选过滤)")
    print("=" * 65)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Modulation: {'yes' if mod_net is not None else 'no'}")
    print(f"Split:      train=D{META_TRAIN}, val=D{META_VAL}, test=D{META_TEST}")
    print(f"Inner loop: {inner_cfg['inner_steps']} steps, lr={inner_cfg['inner_lr']} "
          f"(per-domain, support_ratio=0.7)")
    print(f"Test set:   {len(test_idx)} samples, {int(test_labels.sum())} violations "
          f"({100*test_labels.sum()/max(1,len(test_labels)):.1f}%)")
    print(f"Threshold:  {best_th:.3f} (from val D3, F1={best_f1:.4f})")
    print(f"NUM_MC:     {args.num_mc}")

    print(f"\n--- Trajectory (Best-of-{args.num_mc}, pixel) ---")
    print(f"  ADE: median={test_traj['ade_median']:.2f}px  mean={test_traj['ade_mean']:.2f}px")
    print(f"  FDE: median={test_traj['fde_median']:.2f}px  mean={test_traj['fde_mean']:.2f}px")

    print(f"\n--- Classification (P_cross × signal gate) ---")
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

    red_mask = test_sig >= 0.99
    non_red_mask = test_sig <= 0.01
    print(f"\n--- P_cross Distribution (test D5) ---")
    print(f"  Red(pred win):    {red_mask.sum()} samples, viol={((test_labels == 1) & red_mask).sum()}")
    if red_mask.sum() > 0:
        pc_red = test_pc[red_mask]; lbl_red = test_labels[red_mask]
        print(f"    Pos P_cross={pc_red[lbl_red == 1].mean():.4f}  Neg P_cross={pc_red[lbl_red == 0].mean():.4f}")
    print(f"  Non-red(pred win):{non_red_mask.sum()} samples, viol={((test_labels == 1) & non_red_mask).sum()}")

    pos_risk = test_risks[test_labels == 1]
    neg_risk = test_risks[test_labels == 0]
    print(f"\n--- Risk Distribution ---")
    print(f"  Pos mean: {pos_risk.mean():.4f} (n={len(pos_risk)})")
    print(f"  Neg mean: {neg_risk.mean():.4f} (n={len(neg_risk)})")

    # Save CSV (columns aligned with eval_flowchain_domain.py)
    csv_path = "fomaml_domain_results.csv"
    with open(csv_path, "w") as f:
        f.write("Method,Split,ADEmedian,ADEmean,FDEmedian,FDEmean,AUC,F1_05,F1_val,F1_testbest,BestTh,Accuracy,"
                "NSamples,NPos,PosRisk,NegRisk,PosPcross,NegPcross,Checkpoint,NUM_MC,InnerSteps,InnerLR,Modulation\n")
        f.write(f"FOMAML,Domain-D5,"
                f"{test_traj['ade_median']:.2f},{test_traj['ade_mean']:.2f},"
                f"{test_traj['fde_median']:.2f},{test_traj['fde_mean']:.2f},"
                f"{cls_metrics.get('AUC',0):.4f},{f1_at_05:.4f},{best_f1:.4f},{f1_test_best:.4f},{best_th:.3f},"
                f"{cls_metrics.get('Accuracy',0):.4f},{len(test_labels)},{int(test_labels.sum())},"
                f"{pos_risk.mean():.4f},{neg_risk.mean():.4f},"
                f"{test_pc[test_labels==1].mean():.4f},{test_pc[test_labels==0].mean():.4f},"
                f"{args.checkpoint},{args.num_mc},{inner_cfg['inner_steps']},{inner_cfg['inner_lr']},"
                f"{'yes' if mod_net is not None else 'no'}\n")
    logger.info(f"Saved to {csv_path}")

    # Save per-sample preds for post-hoc metrics
    np.savez("fomaml_test_preds.npz", risks=test_risks, labels=test_labels)
    logger.info("Saved per-sample predictions to fomaml_test_preds.npz")

    print("\nDone!")


if __name__ == "__main__":
    main()
