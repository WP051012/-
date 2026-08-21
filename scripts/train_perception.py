"""
训练感知流水线 (GAT → Memory → GRU → FlowChain)

使用预计算场景数据，端到端训练 perception pipeline，产生有意义的 perception_c。

架构:
  Precomputed scene frames → GAT图推理 → MemoryAttentionFusion → GRU
  → perception_c (256维) → FlowChain → 轨迹预测

参数:
  --stage 2:  训练 GAT + Memory + GRU (FlowChain 冻结, load from Stage 1)
  --stage 3:  端到端联合微调 (全部可训练)
  --ablation:  消融实验变体
"""

import os, sys, argparse, logging, time, json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.perception_model import TrafficPerceptionModel

logger = logging.getLogger(__name__)


# ======================================================================
# Training
# ======================================================================

def train_epoch(
    model: TrafficPerceptionModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0; n_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        obs = batch["obs_trajectory"].to(device)       # (B, obs_len, 2)
        target = batch["target_trajectory"].to(device) # (B, pred_len, 2)
        domain_ids = batch.get("domain_id")
        if domain_ids is not None:
            domain_ids = domain_ids.to(device)

        optimizer.zero_grad()

        # AMP: perception pipeline → condition, then teacher-forced log_prob
        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Step 1: perception pipeline → flow_condition (fast, no FlowChain)
                flow_condition, prompts = model.compute_condition(
                    obs_trajectory=obs,
                    scene_list=batch.get("scene"),
                    domain_ids=domain_ids,
                )
                # Step 2: teacher-forced NLL (MUCH faster than autoregressive sampling)
                log_prob = model.flow_chain.log_prob(
                    obs_trajectory=obs,
                    target=target,
                    perception_c=flow_condition,
                    prompts=prompts,
                )
                nll = -log_prob.mean()

            if torch.isfinite(nll) and nll.grad_fn is not None:
                scaler.scale(nll).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += nll.item()
                n_batches += 1
            else:
                logger.warning(f"Batch skipped: finite={torch.isfinite(nll).item()}, "
                               f"has_grad={nll.grad_fn is not None}")
        except Exception as e:
            logger.warning(f"Batch failed, skipping: {e}")
            optimizer.zero_grad()

        pbar.set_postfix({"nll": f"{nll.item():.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: TrafficPerceptionModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    use_amp: bool = True,
) -> float:
    model.eval()
    total_loss = 0.0; n_batches = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        obs = batch["obs_trajectory"].to(device)
        target = batch["target_trajectory"].to(device)
        domain_ids = batch.get("domain_id")
        if domain_ids is not None:
            domain_ids = domain_ids.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            flow_condition, prompts = model.compute_condition(
                obs_trajectory=obs,
                scene_list=batch.get("scene"),
                domain_ids=domain_ids,
            )
            log_prob = model.flow_chain.log_prob(
                obs_trajectory=obs,
                target=target,
                perception_c=flow_condition,
                prompts=prompts,
            )
            nll = -log_prob.mean()

        total_loss += nll.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None,
                        help="Stage 1 checkpoint to load FlowChain weights")
    parser.add_argument("--save-dir", type=str, default="checkpoints/perception")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=[None, "no_graph", "no_memory", "no_cogcontext",
                                 "no_flowchain", "no_change"])
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--precomputed-dir", type=str,
                        default="data/precomputed")
    parser.add_argument("--domain-labels", type=str,
                        default="data/domains/domain_labels_int.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=500000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (FP16)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.save_dir, "train.log")),
        ],
    )

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load domain labels
    domain_label_map = {}
    if os.path.exists(args.domain_labels):
        with open(args.domain_labels) as f:
            domain_label_map = json.load(f)
        logger.info(f"Domain labels: {len(domain_label_map)} videos, "
                     f"{len(set(domain_label_map.values()))} domains")

    # --- Load data ---
    logger.info("Loading datasets...")
    train_dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        precomputed_dir=args.precomputed_dir,
        max_samples=args.max_samples,
        domain_label_map=domain_label_map,
    )
    val_dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        precomputed_dir=args.precomputed_dir,
        max_samples=args.val_samples,
        domain_label_map=domain_label_map,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=trajectory_collate_fn,
        prefetch_factor=4, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, args.num_workers), pin_memory=True,
        collate_fn=trajectory_collate_fn,
    )
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # --- Build model ---
    model = TrafficPerceptionModel(config, stage=2, ablation=args.ablation).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Load pretrained FlowChain ---
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading FlowChain weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:
            flow_state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            flow_state = {k.replace("flow_chain.", ""): v
                          for k, v in ckpt["model_state_dict"].items()
                          if k.startswith("flow_chain")}
        elif "flowchain_state_dict" in ckpt:
            flow_state = ckpt["flowchain_state_dict"]
        else:
            flow_state = ckpt

        # Remove 'flow_chain.model.' prefix if present (Stage1 nested format)
        cleaned_state = {}
        for k, v in flow_state.items():
            if k.startswith("flow_chain.model."):
                cleaned_state[k[len("flow_chain.model."):]] = v
            elif k.startswith("model."):
                cleaned_state[k[len("model."):]] = v
            else:
                cleaned_state[k] = v

        # --- Handle architecture mismatch ---
        # Stage1: cond_label_size=64, encoder_input.weight=(64,82)
        # Stage2: cond_label_size=256, encoder_input.weight=(64,274)
        # → Copy obs+PE columns (0:18), zero-init condition columns (18:)
        enc_key = "encoder_input.weight"
        if enc_key in cleaned_state:
            old_w = cleaned_state[enc_key]  # (64, 82) from Stage1
            new_w = model.flow_chain.model.encoder_input.weight.data  # (64, 274)
            new_w.zero_()
            # Copy observation + positional encoding columns
            copy_cols = min(old_w.shape[1], 18)  # obs(2) + pe(16) = 18
            new_w[:, :copy_cols] = old_w[:, :copy_cols].to(device)
            logger.info(f"encoder_input: copied {copy_cols}/{old_w.shape[1]} cols from Stage1, "
                         f"zeroed {new_w.shape[1] - copy_cols} new cond cols")
            del cleaned_state[enc_key]

        if "encoder_input.bias" in cleaned_state:
            model.flow_chain.model.encoder_input.bias.data.copy_(
                cleaned_state["encoder_input.bias"].to(device))
            del cleaned_state["encoder_input.bias"]

        model.flow_chain.load_state_dict(cleaned_state, strict=False)
        logger.info("FlowChain weights loaded.")

        # Freeze FlowChain in Stage 2
        if args.stage == 2:
            for p in model.flow_chain.parameters():
                p.requires_grad_(False)
            logger.info("FlowChain frozen (Stage 2).")

    # --- Optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    use_amp = args.amp
    logger.info(f"AMP mixed precision: {'ON' if use_amp else 'OFF'}")

    # --- Training loop ---
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1,
                                 use_amp=use_amp)
        val_loss = validate(model, val_loader, device, use_amp=use_amp)
        scheduler.step()

        logger.info(f"Epoch {epoch+1:3d} | Train={train_loss:.4f} | "
                     f"Val={val_loss:.4f} | Time={time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.save_dir, "best_model.pt"))

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.save_dir, f"model_epoch{epoch+1:03d}.pt"))

    logger.info(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
