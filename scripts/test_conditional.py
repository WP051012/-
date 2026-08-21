"""
Smoke test for the conditional FlowChain (from-scratch path), focused on two
questions before any full training run:

  Q1. Is the condition REAL?  (signal / geom / scene are non-zero, and the
      assembled 256-dim context is non-zero)
  Q2. Does the condition actually ENTER the flow?  (feeding context changes the
      teacher-forced log_prob vs a zero-condition twin; and the from-scratch
      forward + backward runs clean with finite loss + non-zero grads)

This mirrors train_conditional.py's data pipeline (parse_roi -> parse_geometry
fallback, return_context=True, /[3840,2160] normalization) but only touches a
handful of samples so it finishes in minutes.

Run:
    python scripts/test_conditional.py \
        --config configs/default.yaml \
        --gat-conditions data/gat_conditions.pt \
        --n-samples 16 --batch-size 8 --num-samples 4
"""
import argparse, logging, sys, os, random, json
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.conditional_flowchain import ConditionalFlowChain, conditional_flow_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0], device=DEVICE)


def parse_roi(config):
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
                x1, y1 = junction_roi[0]; x2, y2 = junction_roi[1]
                junction_roi = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if sl and len(sl) >= 4:
            stop_line = [float(x) for x in sl]
        if crosswalk_roi and junction_roi:
            break
    return junction_roi, crosswalk_roi, stop_line


def parse_geometry(annotations_dir="data/annotations"):
    annot_dir = Path(annotations_dir)
    if not annot_dir.exists():
        return None, None, None
    geo_a = geo_b = None
    for af in sorted(annot_dir.glob("*.json")):
        try:
            data = json.loads(af.read_text())
        except Exception:
            continue
        video = data.get("video", af.stem)
        sl = data.get("stop_line", {})
        jr = data.get("junction_roi", {})
        sl_list = None; jr_poly = None
        if sl and all(k in sl for k in ("x1", "y1", "x2", "y2")):
            sl_list = [float(sl["x1"]), float(sl["y1"]), float(sl["x2"]), float(sl["y2"])]
        if jr and all(k in jr for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = float(jr["x1"]), float(jr["y1"]), float(jr["x2"]), float(jr["y2"])
            jr_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        geo = {"stop_line": sl_list, "junction_roi": jr_poly}
        if "timing" in video:
            geo_a = geo
        else:
            geo_b = geo
    for geo in (geo_a, geo_b):
        if geo and geo["junction_roi"] and geo["stop_line"]:
            return geo["junction_roi"], geo["junction_roi"], geo["stop_line"]
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default="data/processed/trajectories")
    ap.add_argument("--annotations-dir", default="data/annotations")
    ap.add_argument("--gat-conditions", default="data/gat_conditions.pt")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    junction_roi, crosswalk_roi, stop_line = parse_roi(config)
    if junction_roi is None or stop_line is None:
        logger.info("config lacks intersection_A/B; falling back to data/annotations/*.json")
        junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
    if junction_roi is None or stop_line is None:
        logger.error("No geometry found — aborting.")
        sys.exit(1)
    logger.info(f"junction_roi={junction_roi}")
    logger.info(f"stop_line={stop_line}")

    condition_map = None
    if os.path.exists(args.gat_conditions):
        condition_map = torch.load(args.gat_conditions, map_location="cpu", weights_only=False)
        logger.info(f"GAT conditions loaded: {len(condition_map)} videos")
    else:
        logger.warning("gat_conditions.pt NOT found -> scene will be zero")

    logger.info("Building dataset (return_context=True)...")
    dataset = TrajectoryDataset(
        data_dir=args.data_dir, label_dir="labels",
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="trajectory_only",
        junction_roi=junction_roi, crosswalk_roi=crosswalk_roi, stop_line=stop_line,
        condition_map=condition_map, return_context=True,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    idx = list(range(min(args.n_samples, len(dataset))))
    subset = Subset(dataset, idx)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=trajectory_collate_fn)

    # ============================================================
    # Q1: is the condition data REAL (non-zero)?
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Q1. Condition data is non-zero?")
    logger.info("=" * 60)
    sig_sum = geom_sum = scene_sum = 0.0
    n_scene = 0
    for s in [dataset[i] for i in idx]:
        sig_sum += float(s["signal"].abs().sum())
        geom_sum += float(s["geom_feat"].abs().sum())
        sc = s.get("cond_embedding")
        if sc is not None:
            scene_sum += float(sc.abs().sum())
            n_scene += 1
    n = len(idx)
    logger.info(f"signal   (B,8,5 one-hot): abs-sum/样本 = {sig_sum/n:.2f}  (每样本8个1，期望=8.0)")
    logger.info(f"geom_feat(B,8,6):         abs-sum/样本 = {geom_sum/n:.3f}  (>0 表示几何特征非零)")
    if n_scene:
        logger.info(f"scene    (B,64 GAT):       abs-sum/样本 = {scene_sum/n_scene:.3f}  (>0 表示场景嵌入非零)")
    else:
        logger.warning("scene: 所有样本 cond_embedding 均为 None —— scene 条件缺失！")

    # ============================================================
    # Q2a: does context actually ENTER the flow (change log_prob)?
    # Q2b: from-scratch forward + backward runs clean?
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Q2. Condition enters the flow + from-scratch runs clean?")
    logger.info("=" * 60)

    model = ConditionalFlowChain(
        obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3, condition_dim=256,
        condition_flow=True, condition_norm=False, condition_gate=False,
    ).to(DEVICE)
    model_zero = ConditionalFlowChain(
        obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3, condition_dim=256,
        condition_flow=False, condition_norm=False, condition_gate=False,
    ).to(DEVICE)
    model_zero.load_state_dict(model.state_dict())
    logger.info("Built two twin models (random init, NO checkpoint): "
                "condition_flow=True vs condition_flow=False (zero-condition)")

    for batch in loader:
        obs = batch["obs_trajectory"].to(DEVICE) / NORM
        target = batch["target_trajectory"].to(DEVICE) / NORM
        signal = batch["signal"].to(DEVICE)
        geom = batch["geom_feat"].to(DEVICE)
        scene = batch.get("cond_embedding", None)
        if scene is not None:
            scene = scene.to(DEVICE)

        # --- Q2a: context enters flow (deterministic log_prob comparison) ---
        model.eval(); model_zero.eval()
        with torch.no_grad():
            built = model.build_context(signal, geom, scene)
            ctx = built["context"]
            logger.info(f"assembled context: shape={tuple(ctx.shape)}, "
                        f"abs_mean={float(ctx.abs().mean()):.4f}, "
                        f"std={float(ctx.std()):.4f}")
            lp_cond = model.log_prob(obs, target, signal=signal, geom=geom, scene=scene)
            lp_zero = model_zero.log_prob(obs, target, signal=signal, geom=geom, scene=scene)
        diff = float((lp_cond - lp_zero).abs().max())
        logger.info(f"|Δlog_prob| (cond vs zero-cond) = {diff:.6f}")
        if diff > 1e-5:
            logger.info("  ✓ condition REALLY enters the flow (non-zero context changes the encoder output)")
        else:
            logger.error("  ✗ condition has NO effect on the flow — context is not reaching the encoder!")

        # --- Q2b: from-scratch forward + backward ---
        labels = {
            "goal_label": batch["goal_label"].to(DEVICE),
            "intent_label": batch["intent_label"].to(DEVICE),
            "crossing_label": batch["crossing_label"].to(DEVICE),
        }
        model.train()
        pred, aux = model(obs, signal=signal, geom=geom, scene=scene,
                          num_samples=args.num_samples)
        loss, comps = conditional_flow_loss(pred, target, aux, labels)
        logger.info(f"loss components: " + " ".join(f"{k}={float(v):.3f}" for k, v in comps.items()))
        model.zero_grad()
        loss.backward()
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None and name.split('.')[0] in ("signal_encoder", "geometry_encoder",
                                                            "scene_encoder", "predictor", "goal_embed"):
                grads[name] = float(p.grad.norm())
        finite = bool(torch.isfinite(loss))
        logger.info(f"loss={float(loss):.4f}  finite={finite}")
        if not finite:
            logger.error("  ✗ loss is non-finite (NaN/Inf) — from-scratch backward is broken")
        # print a few representative grad norms to show gradients flow everywhere
        for name in ("predictor.model.encoder_input.weight",
                     "signal_encoder.proj.weight",
                     "geometry_encoder.proj.weight",
                     "goal_embed.proj.weight"):
            if name in grads:
                logger.info(f"  grad||{name}|| = {grads[name]:.6f}")
        logger.info("  ✓ from-scratch forward + backward completed without crash")
        break  # one batch is enough

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: check the ✓/✗ markers above. "
                "If all ✓, the condition is real AND enters the flow AND from-scratch trains clean.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
