"""
Train the conditional FlowChain (signal + geometry + scene + goal context).

Same data processing + FlowChain backbone as `finetune_flowchain.py`:
    - trajectory-only dataset, crossing-candidate filter, coords /[3840,2160]
    - FlowChain predictor init from the fine-tuned baseline checkpoint
      (`flowchain_best_finetuned.pt`), loaded with strict=False so the new
      context encoders / aux heads start from scratch.

The only additions are the 256-dim context (fed as perception_c) and three
auxiliary losses (goal / intent / crossing-time).

Usage:
    python scripts/train_conditional.py --config configs/default.yaml \
        --checkpoint checkpoints/flowchain_best_finetuned.pt \
        --save-dir checkpoints/conditional \
        --gat-conditions data/gat_conditions.pt \
        --epochs 20 --lr 1e-4 --batch-size 64
"""
import argparse, logging, sys, os, random, json
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.conditional_flowchain import ConditionalFlowChain, conditional_flow_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])


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
                x1, y1 = junction_roi[0]; x2, y2 = junction_roi[1]
                junction_roi = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if sl and len(sl) >= 4:
            stop_line = [float(x) for x in sl]
        if crosswalk_roi and junction_roi:
            break
    return junction_roi, crosswalk_roi, stop_line


def parse_geometry(annotations_dir="data/annotations"):
    """Read stop_line + junction_roi from annotation JSONs (no config needed).

    Mirrors scripts/eval_fomaml.py's parse_geometry. The annotation JSONs only
    carry stop_line + junction_roi (no separate crosswalk), so crosswalk_roi is
    set equal to junction_roi here. Values are identical to the config's
    intersection_A/B (the config was derived from these JSONs).
    """
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
    """Return crossing-candidate indices WITHOUT going through __getitem__
    (avoids loading traffic-light / geometry context during filtering)."""
    kept = []
    for i, s in enumerate(tqdm(dataset.samples, desc="Filtering")):
        if is_crossing_candidate(
            s["obs_positions"], s.get("target_positions"),
            crosswalk_roi, stop_line, junction_roi,
        ):
            kept.append(i)
    logger.info(f"Filter: {len(kept)}/{len(dataset.samples)} kept "
                f"({100*len(kept)/len(dataset.samples):.1f}%)")
    return kept


def main():
    parser = argparse.ArgumentParser(description="Train conditional FlowChain")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--gat-conditions", default="data/gat_conditions.pt")
    parser.add_argument("--checkpoint", default="checkpoints/flowchain_best_finetuned.pt")
    parser.add_argument("--save-dir", default="checkpoints/conditional")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate for context encoders + aux heads")
    parser.add_argument("--flow-lr", type=float, default=1e-5,
                        help="learning rate for the FlowChain backbone (slow, "
                             "so conditioning ramps in without corrupting it)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-filtered", type=int, default=50000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Ablation toggles
    parser.add_argument("--no-signal", action="store_true")
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--no-goal", action="store_true")
    parser.add_argument("--no-intent", action="store_true")
    parser.add_argument("--no-crossing", action="store_true")
    # Conditioning strategy
    parser.add_argument("--no-condition-flow", action="store_true",
                        help="do NOT feed context to the flow (context -> aux heads only)")
    parser.add_argument("--freeze-flow", action="store_true",
                        help="freeze the FlowChain backbone (keeps baseline trajectory)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="do NOT load any backbone checkpoint; train the whole "
                             "model (flow + context encoders) from random init")
    # Loss weights
    parser.add_argument("--w-goal", type=float, default=1.0)
    parser.add_argument("--w-intent", type=float, default=1.0)
    parser.add_argument("--w-crossing", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    logger.info(f"Junction ROI: {junction_roi}")
    logger.info(f"Stop line: {stop_line}")

    # ---- GAT scene embeddings ----
    condition_map = None
    if not args.no_scene and os.path.exists(args.gat_conditions):
        logger.info(f"Loading GAT conditions: {args.gat_conditions}")
        condition_map = torch.load(args.gat_conditions, map_location="cpu", weights_only=False)
        logger.info(f"  {len(condition_map)} videos")

    # ---- Dataset (context-enabled) ----
    logger.info("Building dataset (return_context=True)...")
    dataset = TrajectoryDataset(
        data_dir=args.data_dir, label_dir="labels",
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="trajectory_only",
        junction_roi=junction_roi, crosswalk_roi=crosswalk_roi, stop_line=stop_line,
        condition_map=condition_map, return_context=True,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    # ---- Filter + cap + split ----
    indices = filter_indices(dataset, junction_roi, crosswalk_roi, stop_line)
    if len(indices) > args.max_filtered:
        random.seed(args.seed)
        random.shuffle(indices)
        indices = indices[:args.max_filtered]
        logger.info(f"Capped to {args.max_filtered} samples")

    random.seed(args.seed)
    random.shuffle(indices)
    n_val = int(len(indices) * args.val_frac)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    logger.info(f"Split: {len(train_indices)} train / {len(val_indices)} val")

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=trajectory_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=trajectory_collate_fn, num_workers=2, pin_memory=True)

    # ---- Model ----
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
    if args.from_scratch:
        logger.info("Training from scratch: no backbone checkpoint, whole model random-init")
    else:
        logger.info(f"Loading FlowChain backbone from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        n_loaded = sum(1 for k in ckpt if k not in missing)
        logger.info(f"  Loaded {n_loaded}/{len(ckpt)} backbone params; "
                    f"{len(missing)} new modules randomly initialized")

    if not args.from_scratch:
        # flow_cond_proj is a NEW module (absent from the backbone checkpoint).
        # Its random init would inject noise into the frozen flow at step 0 and
        # corrupt the 28px baseline. Zero it so the model starts exactly at the
        # zero-condition baseline (dist_args += 0) and the condition ramps in.
        with torch.no_grad():
            model.predictor.model.flow_cond_proj.weight.zero_()
        logger.info("  Zero-initialized flow_cond_proj (start from zero-condition baseline)")

    if args.freeze_flow:
        frozen = 0
        for name, p in model.predictor.named_parameters():
            if "flow_cond_proj" in name:
                continue  # condition projection stays trainable
            p.requires_grad = False
            frozen += p.numel()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  FlowChain backbone frozen ({frozen:,} params, excl. flow_cond_proj); "
                    f"{n_trainable:,} trainable params remain (context + aux + flow_cond_proj)")

    norm_tensor = NORM.to(DEVICE)
    # Grouped LR: the flow backbone learns slowly so the conditioned encoder
    # columns adapt without corrupting the zero-condition baseline; the context
    # encoders + aux heads learn at full rate. The conditioning gate/LayerNorm
    # (cond_scale / cond_ln) ride with the flow — they shape the *trajectory*
    # condition, so they must ramp in at the same slow rate as the flow.
    def _is_flow_path(name: str) -> bool:
        # flow_cond_proj is the *condition* path, not the frozen flow backbone:
        # it trains at the context LR alongside the context encoders.
        return (name.startswith("predictor.") and "flow_cond_proj" not in name) \
            or name.startswith("cond_ln.") or name == "cond_scale"

    flow_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and _is_flow_path(n)]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not _is_flow_path(n)]
    param_groups = []
    if flow_params:
        param_groups.append({"params": flow_params, "lr": args.flow_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    logger.info(f"Optimizer: flow+gate lr={args.flow_lr} "
                f"({sum(p.numel() for p in flow_params)} params), "
                f"context/aux lr={args.lr} "
                f"({sum(p.numel() for p in other_params)} params)")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0; n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
            target = batch["target_trajectory"].to(DEVICE) / norm_tensor
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

            optimizer.zero_grad()
            pred, aux = model(obs, signal=signal, geom=geom, scene=scene,
                              num_samples=args.num_samples)
            loss, comps = conditional_flow_loss(
                pred, target, aux, labels,
                w_goal=args.w_goal, w_intent=args.w_intent, w_crossing=args.w_crossing,
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_total = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
                target = batch["target_trajectory"].to(DEVICE) / norm_tensor
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
                pred, aux = model(obs, signal=signal, geom=geom, scene=scene,
                                  num_samples=args.num_samples)
                vloss, _ = conditional_flow_loss(
                    pred, target, aux, labels,
                    w_goal=args.w_goal, w_intent=args.w_intent, w_crossing=args.w_crossing,
                )
                if torch.isfinite(vloss):
                    val_total += vloss.item(); val_n += 1
        val_loss = val_total / max(val_n, 1)
        logger.info(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_conditional.pt"))
            logger.info(f"  Saved best checkpoint (val_loss={best_val:.4f})")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_conditional.pt"))
    logger.info(f"Done. Best val_loss={best_val:.4f}. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
