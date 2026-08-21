"""
Fine-tune FlowChain on filtered crossing-candidate data.

Key change from Stage 1 training:
    1. Filter the full ~2.4M trajectory dataset to only keep pedestrians
       whose GT enters junction OR heading is 80-90° to stop line.
    2. Fine-tune FlowChain on this filtered subset, so FlowChain learns
       to predict trajectories that enter junctions.
    3. Use the fine-tuned FlowChain in the two-stage risk framework.

Usage:
    python scripts/finetune_flowchain.py --config configs/default.yaml \
        --checkpoint checkpoints/flowchain_best.pt \
        --save-path checkpoints/flowchain_finetuned.pt \
        --epochs 5 --lr 1e-4 --batch-size 64
"""
import argparse, logging, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.baselines.baseline_models import FlowChainBase
from src.prediction.flow_chain import flow_chain_nll_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def filter_dataset(dataset, junction_roi, crosswalk_roi, stop_line, desc="Filtering"):
    """Filter dataset to only crossing candidates (GT junction OR heading 80-90)."""
    kept = []
    for i in tqdm(range(len(dataset)), desc=desc):
        s = dataset[i]
        obs = s["obs_trajectory"].numpy()
        tgt = s.get("target_trajectory")
        tgt_np = tgt.numpy() if tgt is not None else None

        if is_crossing_candidate(obs, tgt_np, crosswalk_roi, stop_line, junction_roi):
            kept.append(i)

    logger.info(f"{desc}: {len(kept)}/{len(dataset)} kept ({100*len(kept)/len(dataset):.1f}%)")
    return Subset(dataset, kept)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FlowChain on filtered data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--checkpoint", default="checkpoints/flowchain_best.pt")
    parser.add_argument("--save-path", default="checkpoints/flowchain_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-filtered", type=int, default=50000,
                       help="Cap filtered samples (RAM limit)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    junction_roi, crosswalk_roi, stop_line = parse_roi(config)
    logger.info(f"Junction ROI: {junction_roi}")
    logger.info(f"Stop line: {stop_line}")

    # ---- Load full trajectory dataset ----
    logger.info("Loading full trajectory dataset...")
    full_ds = TrajectoryDataset(args.data_dir, mode="trajectory_only")
    logger.info(f"Full dataset: {len(full_ds)} samples")

    # ---- Filter ----
    logger.info("Filtering to junction-crossing candidates...")
    filtered_ds = filter_dataset(
        full_ds, junction_roi, crosswalk_roi, stop_line,
        desc="Filtering full dataset")

    if len(filtered_ds) > args.max_filtered:
        import random; random.seed(42)
        indices = random.sample(list(filtered_ds.indices), args.max_filtered)
        filtered_ds = Subset(full_ds, indices)
        logger.info(f"Capped to {args.max_filtered} samples")

    n_viol = sum(1 for i in filtered_ds.indices if full_ds[i].get("is_violation", False))
    logger.info(f"Filtered dataset: {len(filtered_ds)} samples, {n_viol} violations")

    loader = DataLoader(filtered_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=trajectory_collate_fn, num_workers=2, pin_memory=True)

    # ---- Load FlowChain ----
    logger.info(f"Loading FlowChain from {args.checkpoint}")
    model = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    logger.info(f"  Loaded {sum(p.numel() for p in model.parameters()):,} params")

    norm_tensor = torch.tensor([3840.0, 2160.0], device=DEVICE)

    # ---- Fine-tune ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    logger.info(f"Fine-tuning for {args.epochs} epochs (lr={args.lr})...")

    for epoch in range(args.epochs):
        total_loss = 0.0; n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
            target = batch["target_trajectory"].to(DEVICE) / norm_tensor

            optimizer.zero_grad()
            pred = model(obs_trajectory=obs, num_samples=args.num_samples)
            loss = flow_chain_nll_loss(pred, target, mse_weight=1.0)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    logger.info(f"Saved fine-tuned checkpoint to {args.save_path}")

    # ---- Quick eval: ADE/FDE on filtered data ----
    model.eval()
    all_ade, all_fde = [], []
    eval_loader = DataLoader(
        Subset(filtered_ds, list(range(min(500, len(filtered_ds))))),
        batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval ADE/FDE"):
            obs = batch["obs_trajectory"].to(DEVICE) / norm_tensor
            target = batch["target_trajectory"].to(DEVICE) / norm_tensor
            pred = model(obs_trajectory=obs, num_samples=args.num_samples)
            best_sample = pred.get("best_sample", pred.get("samples"))

            if best_sample is not None and best_sample.shape[0] > 0:
                best = best_sample[:, :, 0]  # (1, pred_len, 2)
                err = best - target
                all_ade.append(float(err.norm(dim=-1).mean()))
                all_fde.append(float(err[:, -1].norm(dim=-1).mean()))

    if all_ade:
        logger.info(f"ADE: median={np.median(all_ade)*3840:.1f}px, mean={np.mean(all_ade)*3840:.1f}px")
        logger.info(f"FDE: median={np.median(all_fde)*3840:.1f}px, mean={np.mean(all_fde)*3840:.1f}px")

    logger.info("Done!")


if __name__ == "__main__":
    main()
