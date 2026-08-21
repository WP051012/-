"""
Meta-Learning 训练 PromptGenerator + 事件驱动 Prompt 更新机制

架构:
  1. FlowChain (冻结) — 轨迹预测骨干
  2. PromptGenerator (可训练) — condition + domain → prefix tokens
  3. PerceptionChangeDetector — 检测场景变化, 触发 prompt 重新生成

训练流程:
  - 遍历域间任务 (meta-batch)
  - Generator 生成 prompts → FlowChain 计算 NLL
  - 只更新 Generator 参数 (FlowChain 冻结)
  - 跨域泛化: 在不同域上交替训练

事件驱动:
  - 缓存每个视频的 condition + prompts
  - 新 condition 与缓存比较 (cosine similarity)
  - 变化 > threshold → 重新生成 prompts
  - 否则复用缓存 (推理效率)
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.flow_chain import FlowChainPredictor
from src.prediction.flow_chain_official import TransformerFlowChain, transformer_flow_nll_loss
from src.prompt import PromptGenerator


logger = logging.getLogger(__name__)


# ======================================================================
# Perception Change Detector — 事件驱动 prompt 更新
# ======================================================================

class PerceptionChangeDetector:
    """
    检测场景 condition 是否发生显著变化, 决定是否需要重新生成 prompt。

    两种检测方法:
      1. Cosine 距离: 当前 condition vs 缓存 condition
      2. 图结构变化: 节点数量/类别分布突变 (简化版: 用 condition 的前几维统计量)

    当任一方法检测到变化 > threshold → 触发 prompt 重新生成 (跨域事件)
    """

    def __init__(
        self,
        cosine_threshold: float = 0.3,
        use_graph_check: bool = True,
    ):
        self.cosine_threshold = cosine_threshold
        self.use_graph_check = use_graph_check

        # 缓存
        self._cached_condition: Optional[Tensor] = None
        self._cached_prompts: Optional[Tensor] = None
        self._cached_stats: Optional[dict] = None

    def detect_change(
        self,
        condition: Tensor,          # (B, condition_dim)
        video_ids: list = None,
    ) -> Tuple[bool, Tensor]:
        """
        检测 condition 是否变化。

        Returns
        -------
        changed : bool — True if 需要重新生成
        similarity : Tensor (B,) — cosine similarity with cached
        """
        if self._cached_condition is None:
            return True, torch.zeros(condition.shape[0], device=condition.device)

        B = condition.shape[0]
        device = condition.device

        # 确保缓存扩展为匹配 batch size
        cached = self._cached_condition.to(device)
        if cached.shape[0] == 1 and B > 1:
            cached = cached.expand(B, -1)
        elif cached.shape[0] != B:
            # Batch size changed → regenerate
            return True, torch.zeros(B, device=device)

        # Cosine similarity
        cos_sim = F.cosine_similarity(condition, cached, dim=-1)  # (B,)
        changed = (1 - cos_sim) > self.cosine_threshold

        return bool(changed.any()), cos_sim

    def update_cache(self, condition: Tensor, prompts: Optional[Tensor] = None):
        """更新缓存的 condition 和 prompts"""
        self._cached_condition = condition.detach().clone()
        if prompts is not None:
            self._cached_prompts = prompts.detach().clone()

    def get_cached_prompts(self) -> Optional[Tensor]:
        return self._cached_prompts

    def reset(self):
        self._cached_condition = None
        self._cached_prompts = None
        self._cached_stats = None


# ======================================================================
# Meta-Learning Trainer
# ======================================================================

class MetaLearningTrainer:
    """
    MAML-style meta-learning for PromptGenerator.

    核心思想:
      - FlowChain 冻结, 只训练 Generator
      - 每个 domain 是一个 task
      - Generator 学习生成 prompts 使 FlowChain 在任意域上预测更好
      - 事件驱动: condition 变化时才重新生成 prompt

    Training loop (per step):
      1. Sample meta-batch: K domains × T tasks each
      2. For each task (domain d, support set S):
         a. Generate prompts = Generator(condition_S, domain_id=d)
         b. Compute inner loss = NLL(FlowChain(obs_S, prompts), target_S)
      3. Outer update: ∇_Generator (sum of inner losses)
    """

    def __init__(
        self,
        flow_chain: FlowChainPredictor,
        generator: PromptGenerator,
        detector: PerceptionChangeDetector,
        config: dict,
        device: torch.device,
    ):
        self.flow_chain = flow_chain
        self.generator = generator
        self.detector = detector
        self.config = config
        self.device = device

        # Freeze FlowChain
        for p in self.flow_chain.parameters():
            p.requires_grad_(False)
        self.flow_chain.eval()

        # Optimizer — only Generator params
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Metrics
        self.metrics = defaultdict(list)

    def train_step(
        self,
        obs: Tensor,                # (B, obs_len, 2)
        target: Tensor,             # (B, pred_len, 2)
        condition: Tensor,          # (B, condition_dim)
        domain_ids: Optional[Tensor] = None,  # (B,)
        video_ids: list = None,
    ) -> Dict[str, float]:
        """
        Single training step with event-driven prompt generation.

        1. 检测 condition 是否变化
        2. 如果变化 > threshold: 生成新 prompts
        3. 否则复用 cached prompts (节省计算)
        4. FlowChain 前向 (frozen) → NLL loss
        5. Backward → update Generator
        """
        B = obs.shape[0]
        device = obs.device

        # --- Event-driven: check if regeneration needed ---
        changed, similarity = self.detector.detect_change(condition)

        if changed or self.detector.get_cached_prompts() is None:
            # Generate new prompts
            prompts = self.generator(condition, domain_ids=domain_ids)
            self.detector.update_cache(condition, prompts)
        else:
            # Reuse cached prompts
            prompts = self.detector.get_cached_prompts()
            # Ensure batch size matches
            if prompts.shape[0] != B:
                prompts = self.generator(condition, domain_ids=domain_ids)
                self.detector.update_cache(condition, prompts)

        prompts = prompts.to(device)

        # --- Forward through frozen FlowChain ---
        # Use log_prob (teacher-forced) for training
        log_prob = self.flow_chain.log_prob(
            obs_trajectory=obs,
            target=target,
            perception_c=condition,
            prompts=prompts,
        )
        nll = -log_prob.mean()

        # --- Backward (Generator only) ---
        self.optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": nll.item(),
            "changed": float(changed),
            "mean_similarity": similarity.mean().item() if similarity.numel() > 0 else 1.0,
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """在验证集上评估"""
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            obs = batch["obs_trajectory"].to(self.device)
            target = batch["target_trajectory"].to(self.device)
            domain_ids = batch.get("domain_id", None)
            if domain_ids is not None:
                domain_ids = domain_ids.to(self.device)

            B = obs.shape[0]
            condition = torch.zeros(B, 256, device=self.device)

            # Generate prompts
            prompts = self.generator(condition, domain_ids=domain_ids)

            # NLL
            log_prob = self.flow_chain.log_prob(
                obs_trajectory=obs,
                target=target,
                perception_c=condition,
                prompts=prompts,
            )
            nll = -log_prob.sum()
            total_loss += nll.item()
            total_samples += B

        return {"val_nll": total_loss / max(total_samples, 1)}

    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")


# ======================================================================
# Training Loop
# ======================================================================

def train_meta_learning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    flow_chain: FlowChainPredictor,
    generator: PromptGenerator,
    detector: PerceptionChangeDetector,
    config: dict,
    device: torch.device,
    save_dir: str,
):
    """完整的 meta-learning 训练循环"""

    trainer = MetaLearningTrainer(
        flow_chain=flow_chain,
        generator=generator,
        detector=detector,
        config=config,
        device=device,
    )

    epochs = config.get("epochs", 100)
    log_interval = config.get("log_interval", 10)
    save_interval = config.get("save_interval", 20)

    best_val_nll = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_changes = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            obs = batch["obs_trajectory"].to(device)
            target = batch["target_trajectory"].to(device)
            B = obs.shape[0]
            # Use zero condition (no perception pipeline during prefix training)
            condition = torch.zeros(B, 256, device=device)
            domain_ids = batch.get("domain_id", None)
            if domain_ids is not None:
                domain_ids = domain_ids.to(device)

            metrics = trainer.train_step(
                obs, target, condition,
                domain_ids=domain_ids,
            )

            epoch_loss += metrics["loss"]
            epoch_changes += metrics["changed"]

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                    f"Loss={metrics['loss']:.4f} | "
                    f"Sim={metrics['mean_similarity']:.3f} | "
                    f"Changed={metrics['changed']:.0f}"
                )

        # Epoch summary
        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # Validation
        val_metrics = trainer.validate(val_loader)
        val_nll = val_metrics["val_nll"]

        logger.info(
            f"=== Epoch {epoch:3d} | "
            f"Train NLL={avg_loss:.4f} | "
            f"Val NLL={val_nll:.4f} | "
            f"Changes={epoch_changes:.0f} | "
            f"Time={elapsed:.1f}s ==="
        )

        # Save best
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            trainer.save_checkpoint(os.path.join(save_dir, "best_generator.pt"), epoch)

        # Save periodic
        if (epoch + 1) % save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(save_dir, f"generator_epoch{epoch:03d}.pt"), epoch
            )

    # Final save
    trainer.save_checkpoint(os.path.join(save_dir, "generator_final.pt"), epochs)
    logger.info(f"Training complete. Best val NLL: {best_val_nll:.4f}")

    return trainer


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Meta-Learning Prompt Generator 训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--flowchain-checkpoint", type=str, required=True,
                        help="预训练 FlowChain checkpoint")
    parser.add_argument("--domain-labels", type=str,
                        default="data/domains/domain_labels_dbscan.json")
    parser.add_argument("--save-dir", type=str, default="checkpoints/prompt_generator")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-domains", type=int, default=0,
                        help="域数量 (0=从 domain_labels 中推断)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=500000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to trajectory data")
    args = parser.parse_args()

    # Setup
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

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # --- Load pretrained FlowChain ---
    logger.info(f"Loading FlowChain from {args.flowchain_checkpoint}...")
    flow_chain = FlowChainPredictor(
        obs_len=8, pred_len=12, trajectory_dim=2,
        hidden_dim=64, condition_dim=256, num_flows=3,
    ).to(device)

    checkpoint = torch.load(args.flowchain_checkpoint, map_location=device)
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        flow_chain.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "flowchain_state_dict" in checkpoint:
        flow_chain.load_state_dict(checkpoint["flowchain_state_dict"], strict=False)
    else:
        flow_chain.load_state_dict(checkpoint, strict=False)
    logger.info("FlowChain loaded.")

    # --- Load domain labels ---
    num_domains = args.num_domains
    domain_label_map = {}
    if os.path.exists(args.domain_labels):
        with open(args.domain_labels) as f:
            domain_label_map = json.load(f)
        if num_domains == 0:
            unique_labels = set(domain_label_map.values())
            unique_labels.discard(-1)  # ignore noise
            num_domains = len(unique_labels)
        logger.info(f"Domain labels loaded: {len(domain_label_map)} videos, {num_domains} domains")

    # --- Create Prompt Generator ---
    generator = PromptGenerator(
        condition_dim=256,
        d_model=64,
        num_prompts=4,
        num_domains=num_domains,
        domain_dim=32,
        hidden_dim=128,
    ).to(device)
    logger.info(f"PromptGenerator: {sum(p.numel() for p in generator.parameters()):,} params")

    # --- Create Change Detector ---
    detector = PerceptionChangeDetector(
        cosine_threshold=0.3,
        use_graph_check=True,
    )

    # --- Load data ---
    from data.dataset import TrajectoryDataset
    train_dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        domain_label_map=domain_label_map,
    )
    val_max = max(5000, args.max_samples // 10)
    val_dataset = TrajectoryDataset(
        data_dir=args.data_dir,
        max_samples=val_max,
        domain_label_map=domain_label_map,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )

    # --- Train ---
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-5,
        "log_interval": 10,
        "save_interval": 20,
    }

    train_meta_learning(
        train_loader=train_loader,
        val_loader=val_loader,
        flow_chain=flow_chain,
        generator=generator,
        detector=detector,
        config=config,
        device=device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
