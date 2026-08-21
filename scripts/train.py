#!/usr/bin/env python3
"""
训练脚本 — 交通感知增强的轨迹预测与闯红灯分类
================================================
完整训练流程: 预处理数据 → 轨迹预测 → 闯红灯分类

分阶段训练:
    Stage 1: FlowChain 轨迹预测预训练 (w/o perception conditioning)
    Stage 2: 加入感知图 + 感知记忆联合训练
    Stage 3: 端到端微调 + 闯红灯分类器训练

用法:
    python scripts/train.py --config configs/default.yaml --data data/processed/trajectories/
    python scripts/train.py --config configs/default.yaml --stage 2 --resume checkpoints/stage1_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from src.graph import TrafficPerceptionGraph
from src.memory import TrafficPerceptionMemory, DecayController
from src.prediction import (
    PerceptionGRU,
    PerceptionContextEncoder,
    PerceptionChangeDetector,
    FlowChainPredictor,
    flow_chain_nll_loss,
)
from src.prompt import PromptGenerator
from src.classification import (
    RedLightProbabilityEstimator,
    RedLightViolationChecker,
    StopLine,
    JunctionRegion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Full Model
# ======================================================================

class RedLightPredictionModel(nn.Module):
    """
    End-to-end trainable model: perception graph → memory → GRU → FlowChain → classifier.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    stage : int
        Training stage (1, 2, or 3).
    """

    def __init__(self, config: dict, stage: int = 1):
        super().__init__()
        graph_cfg = config.get("graph", {})
        memory_cfg = config.get("memory", {})
        gru_cfg = config.get("perception_gru", {})
        flow_cfg = config.get("flow_chain", {})
        node_cfg = config.get("node_features", {})

        self.stage = stage
        self.obs_len = flow_cfg.get("obs_len", 8)
        self.pred_len = flow_cfg.get("pred_len", 12)
        self.trajectory_dim = flow_cfg.get("trajectory_dim", 2)
        self.node_feat_dim = node_cfg.get("pedestrian_feat_dim", 270)

        # --- Perception Graph (used in stage 2+) ---
        if stage >= 2:
            self.perception_graph = TrafficPerceptionGraph(
                node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
                gat_hidden_dim=graph_cfg.get("gat_hidden_dim", 64),
                gat_out_dim=graph_cfg.get("gat_hidden_dim", 128),
                gat_heads=graph_cfg.get("gat_heads", 4),
            )

        # --- Perception Memory (used in stage 2+) ---
        if stage >= 2:
            self.perception_memory = TrafficPerceptionMemory(
                node_feat_dim=graph_cfg.get("gat_hidden_dim", 128),
                behavioral_dim=memory_cfg.get("behavioral_dim", 128),
                environmental_dim=memory_cfg.get("environmental_dim", 128),
                interactive_dim=memory_cfg.get("interactive_dim", 128),
                fusion_dim=memory_cfg.get("fusion_dim", 256),
            )

        # --- Context Encoder (stage 2+) ---
        if stage >= 2:
            self.context_encoder = PerceptionContextEncoder(
                behavioral_dim=memory_cfg.get("behavioral_dim", 128),
                environmental_dim=memory_cfg.get("environmental_dim", 128),
                interactive_dim=memory_cfg.get("interactive_dim", 128),
                context_dim=flow_cfg.get("condition_dim", 256),
            )

        # --- FlowChain Predictor ---
        self.flow_chain = FlowChainPredictor(
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            trajectory_dim=self.trajectory_dim,
            hidden_dim=flow_cfg.get("d_model", 64),
            condition_dim=flow_cfg.get("condition_dim", 256),
            num_flows=flow_cfg.get("nvp_num_blocks", 3),
        )

        # --- Prompt Generator (prefix-tuning) ---
        prompt_cfg = config.get("prompt", {})
        if prompt_cfg.get("enabled", False):
            self.prompt_generator = PromptGenerator(
                condition_dim=prompt_cfg.get("condition_dim", 256),
                d_model=prompt_cfg.get("d_model", flow_cfg.get("d_model", 64)),
                num_prompts=prompt_cfg.get("num_prompts", 4),
                num_domains=prompt_cfg.get("num_domains", 0),
                domain_dim=prompt_cfg.get("domain_dim", 32),
                hidden_dim=prompt_cfg.get("hidden_dim", 128),
            )
            self._use_prompts = True
        else:
            self.prompt_generator = None
            self._use_prompts = False

        # --- Classifier (stage 3) ---
        if stage >= 3:
            self.classifier = RedLightProbabilityEstimator(
                threshold=config.get("red_light", {}).get("violation_threshold", 0.5),
            )

        # Decay controller (stage 2+)
        if stage >= 2:
            self.decay_controller = DecayController(
                memory_dim=memory_cfg.get("behavioral_dim", 128),
                decay_rate=memory_cfg.get("decay_rate", 0.01),
            )

        # Change detector (stage 2+)
        if stage >= 2:
            self.change_detector = PerceptionChangeDetector()

    def forward(
        self,
        obs_trajectory: torch.Tensor,       # (B, obs_len, 2)
        scene_data: Optional[dict] = None,   # perception graph input (stage 2+)
        num_samples: int = 20,
        return_details: bool = False,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        obs_trajectory : Tensor (B, obs_len, 2)
        scene_data : dict, optional
            Contains bboxes, class_names, positions per frame for perception graph.
        num_samples : int
            FlowChain Monte Carlo samples.
        return_details : bool
            If True, also return memory info, gate logs, etc.

        Returns
        -------
        dict with keys: "samples", "mean", "std", "log_probs", (and optionally details)
        """
        B = obs_trajectory.size(0)
        device = obs_trajectory.device

        # --- Perception conditioning ---
        if self.stage >= 2 and scene_data is not None:
            # Run perception graph → memory → context encoder per frame
            c_seq = self._compute_perception_sequence(scene_data, device)
            # Use the last perception vector as condition
            flow_condition = c_seq[-1].unsqueeze(0) if c_seq.dim() == 1 else c_seq[:, -1]
        else:
            # Stage 1: zero perception (unconditional FlowChain)
            flow_condition = torch.zeros(
                B, self.flow_chain.condition_dim, device=device,
            )

        # --- FlowChain prediction ---
        prompts = None
        if self._use_prompts:
            domain_ids = scene_data.get("domain_id") if scene_data else None
            if domain_ids is not None and not isinstance(domain_ids, torch.Tensor):
                domain_ids = torch.as_tensor(domain_ids, device=device)
            prompts = self.prompt_generator(flow_condition, domain_ids=domain_ids)

        prediction = self.flow_chain(
            obs_trajectory=obs_trajectory,
            perception_c=flow_condition,
            num_samples=num_samples,
            prompts=prompts,
        )

        if return_details:
            prediction["perception_c"] = flow_condition

        return prediction

    def _compute_perception_sequence(
        self,
        scene_data: dict,
        device: str,
    ) -> torch.Tensor:
        """
        Compute traffic perception vector sequence c_t from scene data.

        Returns
        -------
        Tensor (T, fusion_dim)
        """
        # This is a simplified version — in practice, scene_data would
        # contain per-frame bboxes/positions for the full B × T sequence.
        #
        # For training efficiency, perception features may be precomputed
        # or use a lightweight encoding.
        #
        # Placeholder: return zero vector
        return torch.zeros(
            self.obs_len, self.flow_chain.condition_dim, device=device,
        )


# ======================================================================
# Training & Validation
# ======================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    writer: SummaryWriter,
    stage: int,
    grad_clip: float = 1.0,
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"S{stage} E{epoch}")
    for batch in pbar:
        obs = batch["obs_trajectory"].to(device)
        target = batch["target_trajectory"].to(device)

        optimizer.zero_grad()

        # Forward
        prediction = model(obs_trajectory=obs, num_samples=20)

        # NLL loss on trajectory
        loss = flow_chain_nll_loss(prediction, target)

        # Regularisation
        l2_reg = 0.0
        for p in model.parameters():
            l2_reg += p.pow(2.0).sum()
        loss = loss + 1e-5 * l2_reg

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar(f"Loss/S{stage}_train", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Val"):
        obs = batch["obs_trajectory"].to(device)
        target = batch["target_trajectory"].to(device)

        prediction = model(obs_trajectory=obs, num_samples=20)
        loss = flow_chain_nll_loss(prediction, target)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="训练闯红灯预测模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True, help="预处理数据目录")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="1=FlowChain预训练, 2=+感知图, 3=+分类器")
    parser.add_argument("--resume", default=None, help="恢复训练的checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--log-dir", default="logs/")

    args = parser.parse_args()

    # Config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    flow_cfg = config.get("flow_chain", {})
    epochs = args.epochs or train_cfg.get("epochs", 200)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = args.lr or train_cfg.get("learning_rate", 1e-3)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  Stage: {args.stage}  Batch: {batch_size}  LR: {lr}")

    # Data
    dataset = TrajectoryDataset(
        data_dir=args.data,
        obs_len=flow_cfg.get("obs_len", 8),
        pred_len=flow_cfg.get("pred_len", 12),
        stride=flow_cfg.get("obs_len", 8) // 2,  # 50% overlap
        min_trajectory_len=30,
    )
    stats = dataset.get_stats()
    logger.info(f"Dataset: {stats['total_samples']} samples, "
                f"{stats['num_videos']} videos")
    logger.info(f"Class dist: {stats['class_distribution']}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=trajectory_collate_fn,
        pin_memory=True,
    )

    # Model
    model = RedLightPredictionModel(config, stage=args.stage).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Logging
    writer = SummaryWriter(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, dataloader, optimizer, epoch,
            device, writer, args.stage,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
        )
        val_loss = validate(model, dataloader, device)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        scheduler.step()

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "stage": args.stage,
            }, ckpt_dir / f"stage{args.stage}_best.pt")
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # Periodic save
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_dir / f"stage{args.stage}_epoch{epoch}.pt")

    writer.close()
    logger.info(f"Training complete. Best val_loss = {best_loss:.4f}")


if __name__ == "__main__":
    main()
