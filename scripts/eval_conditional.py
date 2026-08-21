"""
Evaluate the conditional FlowChain.

Reports (matching the FlowChain-baseline reporting conventions):
    - single-sample ADE/FDE (mean prediction, num_samples=1)
    - best-of-N ADE/FDE  (N samples, pick lowest-ADE sample)
    - NLL               (teacher-forced log-prob of the target)
    - auxiliary heads    (intent accuracy, crossing-time accuracy, goal MAE)

Data processing mirrors train_conditional.py exactly (same seed, filter, cap,
split) so the held-out split is identical to the one training never saw.

Usage:
    python scripts/eval_conditional.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/conditional/best_conditional.pt \
        --gat-conditions data/gat_conditions.pt \
        --num-samples 100
"""
import argparse, logging, sys, os, random, json, warnings
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.conditional_flowchain import ConditionalFlowChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])

# The single-sample path calls FlowChain with num_samples=1, which makes
# `preds.std(dim=1)` in flow_chain_official.forward warn about ddof<=0 (std over
# 1 sample = NaN). We never use that std field, so silence the harmless flood.
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")


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
    """Read stop_line + junction_roi from annotation JSONs (no config needed)."""
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
        sl_list = None
        jr_poly = None
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


def filter_indices(dataset, junction_roi, crosswalk_roi, stop_line):
    kept = []
    for i, s in enumerate(tqdm(dataset.samples, desc="Filtering")):
        if is_crossing_candidate(
            s["obs_positions"], s.get("target_positions"),
            crosswalk_roi, stop_line, junction_roi,
        ):
            kept.append(i)
    logger.info(f"Filter: {len(kept)}/{len(dataset.samples)} kept")
    return kept


def main():
    parser = argparse.ArgumentParser(description="Evaluate conditional FlowChain")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--gat-conditions", default="data/gat_conditions.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-filtered", type=int, default=50000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=100, help="best-of-N")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="eval batch size (batch-izes the per-sample loop for "
                             "GPU parallelism; small footprint so 128-256 is safe)")
    parser.add_argument("--no-signal", action="store_true")
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--no-goal", action="store_true")
    parser.add_argument("--no-intent", action="store_true")
    parser.add_argument("--no-crossing", action="store_true")
    parser.add_argument("--no-condition-flow", action="store_true",
                        help="flow was trained with zero condition (context -> aux heads only)")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    junction_roi, crosswalk_roi, stop_line = parse_roi(config)
    if junction_roi is None or stop_line is None:
        logger.info("config lacks intersection_A/B; falling back to data/annotations/*.json")
        junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
    if junction_roi is None or stop_line is None:
        logger.error("No geometry (junction_roi/stop_line) found — filter will keep 0 samples. "
                     "Check --config and --annotations-dir.")
        sys.exit(1)

    condition_map = None
    if not args.no_scene and os.path.exists(args.gat_conditions):
        condition_map = torch.load(args.gat_conditions, map_location="cpu", weights_only=False)

    logger.info("Building dataset (return_context=True)...")
    dataset = TrajectoryDataset(
        data_dir=args.data_dir, label_dir="labels",
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="trajectory_only",
        junction_roi=junction_roi, crosswalk_roi=crosswalk_roi, stop_line=stop_line,
        condition_map=condition_map, return_context=True,
    )

    indices = filter_indices(dataset, junction_roi, crosswalk_roi, stop_line)
    if len(indices) > args.max_filtered:
        random.seed(args.seed); random.shuffle(indices); indices = indices[:args.max_filtered]
    random.seed(args.seed); random.shuffle(indices)
    n_val = int(len(indices) * args.val_frac)
    eval_indices = indices[:n_val]  # same held-out split as train_conditional
    logger.info(f"Eval set: {len(eval_indices)} samples")

    model = ConditionalFlowChain(
        obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3, condition_dim=256,
        use_signal=not args.no_signal,
        use_geom=not args.no_geom,
        use_scene=not args.no_scene,
        use_goal=not args.no_goal,
        use_intent=not args.no_intent,
        use_crossing=not args.no_crossing,
        condition_flow=not args.no_condition_flow,
    ).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logger.info(f"Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    norm = NORM.to(DEVICE)

    # Batch-ize the eval loop: a DataLoader (with collate_fn) replaces the
    # per-sample `dataset[idx]` loop, so each forward processes `batch_size`
    # samples at once. The FlowChain backbone is tiny (d_model=64), so the
    # bottleneck was kernel-launch overhead, not GPU math — batching removes it.
    eval_ds = Subset(dataset, eval_indices)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=trajectory_collate_fn, num_workers=2,
                             pin_memory=True)
    logger.info(f"Eval loader: {len(eval_indices)} samples @ batch {args.batch_size}")

    single_ade, single_fde = [], []
    best_ade, best_fde = [], []
    nlls = []
    intent_correct, crossing_correct = 0, 0
    goal_mae = []
    n_eval = 0

    for batch in tqdm(eval_loader, desc="Eval"):
        obs = batch["obs_trajectory"].to(DEVICE) / norm      # (B, obs_len, 2)
        target = batch["target_trajectory"].to(DEVICE)        # (B, pred_len, 2)
        signal = batch["signal"].to(DEVICE)
        geom = batch["geom_feat"].to(DEVICE)
        scene = batch.get("cond_embedding", None)
        if scene is not None:
            scene = scene.to(DEVICE)
        labels = {
            "goal_label": batch["goal_label"].to(DEVICE),
            "intent_label": batch["intent_label"].to(DEVICE),
            "crossing_label": batch["crossing_label"].to(DEVICE),
        }
        B = obs.shape[0]

        with torch.no_grad():
            # single-sample (mean)
            pred1, aux1 = model(obs, signal=signal, geom=geom, scene=scene, num_samples=1)
            mean_pred = pred1["mean"] * norm  # (B, pred, 2) px
            l2 = torch.sqrt(((mean_pred - target) ** 2).sum(dim=-1))  # (B, pred)
            single_ade.extend(l2.mean(dim=-1).cpu().tolist())
            single_fde.extend(l2[:, -1].cpu().tolist())

            # best-of-N (wrapper guarantees samples is (N, B, pred, 2))
            predN, _ = model(obs, signal=signal, geom=geom, scene=scene, num_samples=args.num_samples)
            s_traj = predN["samples"] * norm  # (N, B, pred, 2) px
            diffN = s_traj - target.unsqueeze(0)  # (N, B, pred, 2)
            l2N = torch.sqrt((diffN ** 2).sum(dim=-1))  # (N, B, pred)
            ade_per = l2N.mean(dim=-1)  # (N, B)
            best_i = ade_per.argmin(dim=0)  # (B,)
            best_ade.extend(ade_per.gather(0, best_i.unsqueeze(0)).squeeze(0).cpu().tolist())
            best_fde.extend(l2N[:, :, -1].gather(0, best_i.unsqueeze(0)).squeeze(0).cpu().tolist())

            # NLL (teacher-forced)
            lp = model.log_prob(obs, target / norm, signal=signal, geom=geom, scene=scene)
            nlls.extend((-lp).cpu().tolist())

            # aux metrics
            if aux1 is not None:
                if "intent" in aux1:
                    pred_intent = aux1["intent"].argmax(dim=1)  # (B,)
                    intent_correct += int((pred_intent == labels["intent_label"].long()).sum())
                if "crossing" in aux1:
                    pred_cross = aux1["crossing"].argmax(dim=1) + 1  # (B,)
                    crossing_correct += int((pred_cross == labels["crossing_label"].long()).sum())
                if "goal" in aux1:
                    goal_pred_px = aux1["goal"] * norm      # (B, 2) px
                    goal_gt_px = labels["goal_label"] * norm  # (B, 2) px
                    goal_mae.extend((goal_pred_px - goal_gt_px).abs().mean(dim=-1).cpu().tolist())

        n_eval += B

    single_ade = np.array(single_ade); single_fde = np.array(single_fde)
    best_ade = np.array(best_ade); best_fde = np.array(best_fde)
    nlls = np.array(nlls)
    goal_mae = np.array(goal_mae)

    print("\n" + "=" * 60)
    print("Conditional FlowChain Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval set:   {n_eval} samples")
    print(f"\n--- Single-sample (mean) ADE/FDE (px) ---")
    print(f"  ADE: mean={single_ade.mean():.2f}  median={np.median(single_ade):.2f}")
    print(f"  FDE: mean={single_fde.mean():.2f}  median={np.median(single_fde):.2f}")
    print(f"\n--- Best-of-{args.num_samples} ADE/FDE (px) ---")
    print(f"  ADE: mean={best_ade.mean():.2f}  median={np.median(best_ade):.2f}")
    print(f"  FDE: mean={best_fde.mean():.2f}  median={np.median(best_fde):.2f}")
    print(f"\n--- NLL (teacher-forced) ---")
    print(f"  NLL: mean={nlls.mean():.4f}")
    print(f"\n--- Auxiliary heads ---")
    print(f"  Intent accuracy:   {intent_correct}/{n_eval} = {intent_correct/max(n_eval,1):.4f}")
    print(f"  Crossing accuracy: {crossing_correct}/{n_eval} = {crossing_correct/max(n_eval,1):.4f}")
    print(f"  Goal MAE:          {goal_mae.mean():.2f} px" if len(goal_mae) else "  Goal MAE: n/a")
    print("\nDone!")


if __name__ == "__main__":
    main()
