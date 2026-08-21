"""
完整的交通感知轨迹预测模型 — 论文所有组件的串联 (REVISED)
==========================================================

Revised architecture (2026-08):
    1. 检测追踪数据 → 交通感知图构建
    2. Edge-aware GAT 图推理 → 节点嵌入 + 交通认知表示
    3. 三支感知记忆: Behavioral / Environmental / Interactive
    4. Memory Attention Fusion → 交通认知状态 c
    5. 认知状态检测: 图结构变化 + c变化 + Memory冲突
    6. Cognitive-Enhanced GRU (标准GRU + 丰富输入)
    7. FlowChain 采样 → 未来轨迹分布
    8. 轨迹违反检查 → 闯红灯概率

Key changes from old design:
    - Memory→GRU gate control REMOVED
    - Memory Attention Fusion ADDED (learnable attention weights)
    - PerceptionGRU → CognitiveEnhancedGRU (standard GRU, enriched input)
    - Edge-aware GAT (edge features in attention + message)
    - Cognitive state detector (c change + memory conflict)

Ablation variants:
    None           — 完整模型
    "no_graph"     — 去掉GAT, 用简单NodeFeatureEncoder
    "no_memory"    — 去掉三支记忆, 用直接投影
    "no_cogcontext" — GRU不含cognitive context (仅trajectory+GAT)
    "no_flowchain" — 确定性MLP替代FlowChain
    "no_change"    — 去掉变化检测+衰减控制器

References:
    Paper Sections 2-10 (revised)
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.graph import (
    TrafficPerceptionGraph,
    PerceptionGraphBuilder,
    SceneGraph,
    NodeFeatureEncoder,
)
from src.memory import (
    TrafficPerceptionMemory,
    DecayController,
)
from src.memory.memory_attention_fusion import MemoryAttentionFusion
from src.prediction import (
    PerceptionGRU,           # legacy — kept for checkpoint compat
    PerceptionContextEncoder,  # legacy — kept for ablation compat
    PerceptionChangeDetector,
    ChangeEvent,
    FlowChainPredictor,
    flow_chain_nll_loss,
)
from src.prediction.perception_gru import (
    CognitiveEnhancedGRU,
    CognitiveEnhancedGRUNoContext,
)
from src.prompt import PromptGenerator
from src.classification import (
    RedLightViolationChecker,
    RedLightProbabilityEstimator,
    StopLine,
    JunctionRegion,
)

logger = logging.getLogger(__name__)


class TrafficPerceptionModel(nn.Module):
    """
    完整的交通感知轨迹预测 + 闯红灯分类模型 (Revised)。

    Parameters
    ----------
    config : dict
        全局配置。
    stage : int
        训练阶段 (1/2/3)。
    ablation : str or None
        None (完整模型) | "no_graph" | "no_memory" | "no_cogcontext" |
        "no_flowchain" | "no_change"
    """

    def __init__(self, config: dict, stage: int = 1, ablation: Optional[str] = None):
        super().__init__()
        graph_cfg = config.get("graph", {})
        memory_cfg = config.get("memory", {})
        gru_cfg = config.get("perception_gru", {})
        flow_cfg = config.get("flow_chain", {})
        change_cfg = config.get("change_detection", {})

        self.stage = stage
        self.ablation = ablation
        self.obs_len = flow_cfg.get("obs_len", 8)
        self.pred_len = flow_cfg.get("pred_len", 12)
        self.trajectory_dim = flow_cfg.get("trajectory_dim", 2)
        self.node_feat_dim = graph_cfg.get("gat_hidden_dim", 128)
        self.behavioral_dim = memory_cfg.get("behavioral_dim", 128)
        self.environmental_dim = memory_cfg.get("environmental_dim", 128)
        self.interactive_dim = memory_cfg.get("interactive_dim", 128)
        self.fusion_dim = memory_cfg.get("fusion_dim", 256)
        self.condition_dim = flow_cfg.get("condition_dim", 256)
        self.gru_hidden_dim = gru_cfg.get("hidden_dim", 256)

        # ================================================================
        # Stage 2+ 模块 — 根据 ablation 条件化创建
        # ================================================================

        # ---- 1. 交通感知图 ----
        if stage >= 2:
            if ablation != "no_graph":
                self.perception_graph = TrafficPerceptionGraph(
                    node_feat_dim=self.node_feat_dim,
                    gat_hidden_dim=graph_cfg.get("gat_hidden_dim", 64),
                    gat_out_dim=self.node_feat_dim,
                    gat_heads=graph_cfg.get("gat_heads", 4),
                )
            else:
                self.simple_encoder = NodeFeatureEncoder(output_dim=self.node_feat_dim)
            self.graph_builder = PerceptionGraphBuilder()

        # ---- 2. 感知记忆 + Memory Attention Fusion ----
        if stage >= 2:
            if ablation != "no_memory":
                self.perception_memory = TrafficPerceptionMemory(
                    node_feat_dim=self.node_feat_dim,
                    behavioral_dim=self.behavioral_dim,
                    environmental_dim=self.environmental_dim,
                    interactive_dim=self.interactive_dim,
                    fusion_dim=self.fusion_dim,
                )
                # NEW: Memory Attention Fusion — learnable attention over 3 memories
                self.memory_fusion = MemoryAttentionFusion(
                    memory_dim=self.behavioral_dim,
                )
            else:
                # no_memory: 直接投影 node embedding → behavioral_dim
                self.direct_context = nn.Sequential(
                    nn.Linear(self.node_feat_dim, self.fusion_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fusion_dim, self.behavioral_dim),
                    nn.LayerNorm(self.behavioral_dim),
                )
                # Note: no condition projection needed — GRU h_final (256)
                # serves directly as flow_condition via self.condition_proj

        # ---- 3. 衰减控制器 ----
        if stage >= 2 and ablation not in ("no_memory", "no_change"):
            self.decay_controller = DecayController(
                memory_names=("behavioral", "environmental", "interactive"),
                memory_dim=self.behavioral_dim,
                decay_rate=memory_cfg.get("decay_rate", 0.01),
                forget_threshold=memory_cfg.get("confidence_threshold", 0.3),
            )

        # ---- 4. 认知状态变化检测器 (REVISED) ----
        if stage >= 2 and ablation != "no_change":
            self.change_detector = PerceptionChangeDetector(
                struct_config={
                    "agent_count_change_threshold": 3,
                },
                cognitive_config={
                    "threshold": change_cfg.get("cognitive_threshold", 0.3),
                    "metric": "cosine",
                },
                conflict_config={
                    "threshold": change_cfg.get("conflict_threshold", 0.5),
                    "metric": "cosine",
                },
                drift_config={
                    "window_size": change_cfg.get("drift_window", 30),
                    "drift_threshold": change_cfg.get("drift_threshold", 0.15),
                },
            )

        # ---- 5. Cognitive-Enhanced GRU (REVISED — replaces PerceptionGRU) ----
        if stage >= 2:
            if ablation != "no_cogcontext":
                # Full: GRU with trajectory + GAT + cognitive context
                self.perception_gru = CognitiveEnhancedGRU(
                    trajectory_dim=self.trajectory_dim,
                    gat_dim=self.node_feat_dim,
                    cognitive_dim=self.behavioral_dim,
                    hidden_dim=self.gru_hidden_dim,
                    num_layers=2,
                    dropout=0.1,
                )
            else:
                # no_cogcontext: GRU without cognitive context (only trajectory + GAT)
                self.perception_gru = CognitiveEnhancedGRUNoContext(
                    trajectory_dim=self.trajectory_dim,
                    gat_dim=self.node_feat_dim,
                    hidden_dim=self.gru_hidden_dim,
                    num_layers=2,
                    dropout=0.1,
                )

        # ---- 6. Condition projection: h_final → flow_condition ----
        # CRITICAL: zero-init so untrained perception doesn't corrupt frozen FlowChain.
        # Always use Linear (not Identity) so we can zero-init even when dims match.
        if stage >= 2:
            self.condition_proj = nn.Linear(
                self.gru_hidden_dim, self.condition_dim,
            )
            nn.init.zeros_(self.condition_proj.weight)
            nn.init.zeros_(self.condition_proj.bias)

        # ---- 7. 预测器: FlowChain 或 MLP ----
        if ablation != "no_flowchain":
            self.flow_chain = FlowChainPredictor(
                obs_len=self.obs_len,
                pred_len=self.pred_len,
                trajectory_dim=self.trajectory_dim,
                hidden_dim=flow_cfg.get("d_model", 64),
                condition_dim=self.condition_dim,
                num_flows=flow_cfg.get("nvp_num_blocks", 3),
                use_adapter=config.get("use_adapter", True),
            )
        else:
            mlp_input_dim = self.gru_hidden_dim + self.condition_dim
            self.mlp_decoder = nn.Sequential(
                nn.Linear(mlp_input_dim, self.gru_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.gru_hidden_dim // 2, self.pred_len * self.trajectory_dim),
            )

        # ---- 7.5. Prompt Generator (prefix-tuning) ----
        prompt_cfg = config.get("prompt", {})
        if prompt_cfg.get("enabled", False) and ablation != "no_flowchain":
            self.prompt_generator = PromptGenerator(
                condition_dim=prompt_cfg.get("condition_dim", self.condition_dim),
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

        # ---- 8. 闯红灯分类器 (Stage 3, 不受 ablation 影响) ----
        if stage >= 3:
            self.classifier = RedLightProbabilityEstimator(
                threshold=config.get("red_light", {}).get("violation_threshold", 0.5),
            )

        # ---- 内部状态 (跨帧追踪) ----
        self._gru_hidden: Optional[Tensor] = None
        self._prev_cognitive: Optional[Tensor] = None  # for c-state change detection
        self._prev_node_ids: set = set()
        self._change_events: deque[ChangeEvent] = deque(maxlen=500)  # bounded to prevent memory leak

    # ------------------------------------------------------------------
    # 感知上下文提取 (训练用 — perception pipeline → flow_condition)
    # ------------------------------------------------------------------

    def compute_perception_context(
        self,
        obs_trajectory: Tensor,                     # (obs_len, 2) or (B, obs_len, 2)
        scene_data: Optional[dict] = None,
    ) -> Tensor:
        """
        运行感知pipeline (图→记忆→融合→GRU)，返回 flow_condition。

        Revised flow:
            1. Per-frame: GAT → target_emb → memories → fusion → c
            2. CognitiveEnhancedGRU(trajectory, gat_seq, cognitive_seq)
            3. h_final → flow_condition

        Returns
        -------
        flow_condition : (B, condition_dim)
        """
        if obs_trajectory.dim() == 2:
            obs_trajectory = obs_trajectory.unsqueeze(0)
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        # Stage 1 / no scene data fallback
        if self.stage < 2 or scene_data is None:
            return torch.zeros(B, self.condition_dim, device=device)

        T = self.obs_len
        abl = self.ablation

        # Per-frame sequences
        gat_seq = []          # GAT target embeddings
        cognitive_seq = []    # MemoryAttentionFusion outputs
        prev_p = None         # previous frame positions for velocity

        for t in range(T):
            frame_data = self._get_frame_data(scene_data, t, B, device, prev_positions=prev_p)
            if frame_data is not None:
                prev_p = frame_data["positions"]
            if frame_data is None:
                gat_seq.append(torch.zeros(B, self.node_feat_dim, device=device))
                cognitive_seq.append(torch.zeros(B, self.behavioral_dim, device=device))
                continue

            # ---- 1. 感知图 / 简单编码 ----
            if abl != "no_graph":
                node_emb, target_emb = self.perception_graph(
                    bboxes=frame_data["bboxes"],
                    class_names=frame_data["class_names"],
                    positions=frame_data["positions"],
                    velocities=frame_data.get("velocities"),
                    target_idx=frame_data["target_idx"],
                )
            else:
                node_emb = self.simple_encoder(
                    bboxes=frame_data["bboxes"],
                    class_names=frame_data["class_names"],
                    positions=frame_data["positions"],
                    velocities=frame_data.get("velocities"),
                    device=str(device),
                )
                target_idx = frame_data.get("target_idx", 0)
                target_emb = self._safe_target_embed(node_emb, target_idx, device)

            # Store GAT target embedding (ensure batch dim)
            if target_emb.dim() == 1:
                target_emb = target_emb.unsqueeze(0)  # (D,) → (1, D)
            gat_seq.append(target_emb)

            # ---- 2. 感知记忆 + Memory Attention Fusion ----
            if abl != "no_memory":
                safe_target_idx = min(frame_data.get("target_idx", 0), node_emb.shape[0] - 1) if node_emb.shape[0] > 0 else 0
                _, node_types, edge_types_ = self.graph_builder.build(
                    positions=frame_data["positions"].cpu().numpy(),
                    class_names=frame_data["class_names"],
                    target_idx=safe_target_idx,
                )
                infra_idx = [i for i, t in enumerate(node_types) if t == 3]
                agent_idx = [i for i, t in enumerate(node_types)
                             if t in (0, 1) and i != safe_target_idx]
                c_vec, memory_info = self.perception_memory(
                    node_embeddings=node_emb,
                    target_idx=safe_target_idx,
                    infra_indices=infra_idx,
                    agent_indices=agent_idx,
                )
                # NEW: Memory Attention Fusion
                c_fused, _ = self.memory_fusion(
                    behavioral=memory_info["behavioral"],
                    environmental=memory_info["environmental"],
                    interactive=memory_info["interactive"],
                )
                b_vec = memory_info["behavioral"]
                e_vec = memory_info["environmental"]
                i_vec = memory_info["interactive"]
            else:
                # no_memory: 直接投影
                target_idx = frame_data.get("target_idx", 0)
                t_emb = self._safe_target_embed(node_emb, target_idx, device)
                c_fused = self.direct_context(t_emb)
                b_vec = c_fused
                e_vec = c_fused
                i_vec = c_fused

            if c_fused.dim() == 1:
                c_fused = c_fused.unsqueeze(0)
            cognitive_seq.append(c_fused)

            # ---- 3. 衰减控制器 ----
            if abl not in ("no_memory", "no_change"):
                # Ensure batch-safe: squeeze batch dim for decay controller
                b_decay = b_vec.squeeze(0) if b_vec.dim() == 2 else b_vec
                e_decay = e_vec.squeeze(0) if e_vec.dim() == 2 else e_vec
                i_decay = i_vec.squeeze(0) if i_vec.dim() == 2 else i_vec
                self.decay_controller.update(
                    behavioral=b_decay,
                    environmental=e_decay,
                    interactive=i_decay,
                )

            # ---- 4. 认知状态变化检测 (REVISED) ----
            if abl != "no_change":
                raw_ids = frame_data.get("track_ids", [])
                current_ids = set()
                for ids in raw_ids:
                    if isinstance(ids, (list, tuple)):
                        current_ids.update(ids)
                    else:
                        current_ids.add(ids)

                # Run all detectors
                events_by_type = self.change_detector.detect_all(
                    frame_id=t,
                    current_node_ids=current_ids,
                    c_current=c_fused,
                    behavioral=b_vec,
                    environmental=e_vec,
                    interactive=i_vec,
                    traffic_light_state=frame_data.get("traffic_light_state"),
                    agent_count=len(frame_data["class_names"]),
                )
                if self.change_detector.has_any_change(events_by_type):
                    self._handle_cognitive_change(events_by_type)

        # ---- 堆叠时序 ----
        gat_seq = torch.stack(gat_seq, dim=1)            # (B, T, D_gat)
        cognitive_seq = torch.stack(cognitive_seq, dim=1) # (B, T, D_cog)
        if gat_seq.dim() == 2:
            gat_seq = gat_seq.unsqueeze(0)
            cognitive_seq = cognitive_seq.unsqueeze(0)

        # ---- 5. Cognitive-Enhanced GRU 编码 ----
        if abl != "no_cogcontext":
            h_final, h_all = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
                cognitive_seq=cognitive_seq,
                h_0=self._gru_hidden,
            )
        else:
            h_final, h_all = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
                h_0=self._gru_hidden,
            )

        # Cache GRU hidden for no_flowchain training
        self._last_gru_hidden = h_final

        # ---- 6. flow_condition ----
        flow_condition = self.condition_proj(h_final)

        return flow_condition

    # ------------------------------------------------------------------
    # 核心 forward — 完整 pipeline (ablation-aware, REVISED)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_trajectory: Tensor,                     # (B, obs_len, 2)
        scene_data: Optional[dict] = None,
        num_samples: int = 20,
        return_details: bool = False,
        domain_ids: Optional[Tensor] = None,
    ) -> dict:
        """
        Returns
        -------
        dict with "mean", "samples", "log_probs", "std"
              and optionally "perception_c", "memory_info", "change_events"
        """
        if obs_trajectory.dim() == 2:
            obs_trajectory = obs_trajectory.unsqueeze(0)
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device
        abl = self.ablation

        # ----------------------------------------------------------------
        # Precomputed scene data path: per-sample list of frame dicts
        # ----------------------------------------------------------------
        if isinstance(scene_data, list):
            return self._forward_with_precomputed_scenes(
                obs_trajectory, scene_data, num_samples, return_details,
                domain_ids=domain_ids,
            )

        cond_dim = self.condition_dim

        # ================================================================
        # Stage 1 / no scene data: 无条件预测
        # ================================================================
        if self.stage < 2 or scene_data is None:
            c_zero = torch.zeros(B, cond_dim, device=device)

            # Generate prompts even for unconditional case (domain-aware)
            prompts = None
            if self._use_prompts:
                domain_ids = scene_data.get("domain_id") if scene_data else None
                if domain_ids is not None:
                    domain_ids = domain_ids.to(device) if isinstance(domain_ids, torch.Tensor) else torch.tensor(domain_ids, device=device)
                prompts = self.prompt_generator(c_zero, domain_ids=domain_ids)

            if abl != "no_flowchain":
                pred = self.flow_chain(obs_trajectory, c_zero, num_samples, prompts=prompts)
            else:
                h_final = self._encode_trajectory_simple(obs_trajectory, B, device)
                deltas = self.mlp_decoder(torch.cat([h_final, c_zero], dim=-1))
                deltas = deltas.view(B, self.pred_len, self.trajectory_dim)
                last_obs = obs_trajectory[:, -1:]
                mean_pred = last_obs + deltas
                pred = {
                    "mean": mean_pred,
                    "samples": mean_pred.unsqueeze(0).repeat(num_samples, 1, 1, 1),
                    "log_probs": torch.zeros(num_samples, B, device=device),
                    "std": torch.ones(B, self.pred_len, self.trajectory_dim, device=device),
                }

            if return_details:
                pred["perception_c"] = c_zero
                pred["change_events"] = []
            return pred

        # ================================================================
        # Stage 2+: 完整感知pipeline (REVISED)
        # ================================================================
        T = self.obs_len

        gat_seq = []
        cognitive_seq = []
        prev_p = None

        for t in range(T):
            frame_data = self._get_frame_data(scene_data, t, B, device, prev_positions=prev_p)
            if frame_data is not None:
                prev_p = frame_data["positions"]
            if frame_data is None:
                gat_seq.append(torch.zeros(B, self.node_feat_dim, device=device))
                cognitive_seq.append(torch.zeros(B, self.behavioral_dim, device=device))
                continue

            # --- 1. 感知图 / 简单编码 ---
            if abl != "no_graph":
                node_emb, target_emb = self.perception_graph(
                    bboxes=frame_data["bboxes"],
                    class_names=frame_data["class_names"],
                    positions=frame_data["positions"],
                    velocities=frame_data.get("velocities"),
                    target_idx=frame_data["target_idx"],
                )
            else:
                node_emb = self.simple_encoder(
                    bboxes=frame_data["bboxes"],
                    class_names=frame_data["class_names"],
                    positions=frame_data["positions"],
                    velocities=frame_data.get("velocities"),
                    device=str(device),
                )
                target_idx = frame_data.get("target_idx", 0)
                target_emb = self._safe_target_embed(node_emb, target_idx, device)

            if target_emb.dim() == 1:
                target_emb = target_emb.unsqueeze(0)
            gat_seq.append(target_emb)

            # --- 2. 感知记忆 + Memory Attention Fusion ---
            if abl != "no_memory":
                safe_target_idx = min(frame_data.get("target_idx", 0), node_emb.shape[0] - 1) if node_emb.shape[0] > 0 else 0
                _, node_types, edge_types_ = self.graph_builder.build(
                    positions=frame_data["positions"].cpu().numpy(),
                    class_names=frame_data["class_names"],
                    target_idx=safe_target_idx,
                )
                infra_idx = [i for i, t in enumerate(node_types) if t == 3]
                agent_idx = [i for i, t in enumerate(node_types)
                             if t in (0, 1) and i != safe_target_idx]
                c_vec, memory_info = self.perception_memory(
                    node_embeddings=node_emb,
                    target_idx=safe_target_idx,
                    infra_indices=infra_idx,
                    agent_indices=agent_idx,
                )
                c_fused, _ = self.memory_fusion(
                    behavioral=memory_info["behavioral"],
                    environmental=memory_info["environmental"],
                    interactive=memory_info["interactive"],
                )
                b_vec = memory_info["behavioral"]
                e_vec = memory_info["environmental"]
                i_vec = memory_info["interactive"]
            else:
                # no_memory: 直接投影
                target_idx = frame_data.get("target_idx", 0)
                t_emb = self._safe_target_embed(node_emb, target_idx, device)
                c_fused = self.direct_context(t_emb)
                b_vec = c_fused
                e_vec = c_fused
                i_vec = c_fused

            if c_fused.dim() == 1:
                c_fused = c_fused.unsqueeze(0)
            cognitive_seq.append(c_fused)

            # --- 3. 衰减控制器 ---
            if abl not in ("no_memory", "no_change"):
                b_decay = b_vec.squeeze(0) if b_vec.dim() == 2 else b_vec
                e_decay = e_vec.squeeze(0) if e_vec.dim() == 2 else e_vec
                i_decay = i_vec.squeeze(0) if i_vec.dim() == 2 else i_vec
                self.decay_controller.update(
                    behavioral=b_decay,
                    environmental=e_decay,
                    interactive=i_decay,
                )

            # --- 4. 认知状态检测 (REVISED) ---
            if abl != "no_change":
                raw_ids = frame_data.get("track_ids", [])
                current_ids = set()
                for ids in raw_ids:
                    if isinstance(ids, (list, tuple)):
                        current_ids.update(ids)
                    else:
                        current_ids.add(ids)

                events_by_type = self.change_detector.detect_all(
                    frame_id=t,
                    current_node_ids=current_ids,
                    c_current=c_fused,
                    behavioral=b_vec,
                    environmental=e_vec,
                    interactive=i_vec,
                    traffic_light_state=frame_data.get("traffic_light_state"),
                    agent_count=len(frame_data["class_names"]),
                )
                if self.change_detector.has_any_change(events_by_type):
                    self._handle_cognitive_change(events_by_type)

        # ---- 堆叠时序 ----
        gat_seq = torch.stack(gat_seq, dim=1)
        cognitive_seq = torch.stack(cognitive_seq, dim=1)
        if gat_seq.dim() == 2:
            gat_seq = gat_seq.unsqueeze(0)
            cognitive_seq = cognitive_seq.unsqueeze(0)

        # ---- 5. Cognitive-Enhanced GRU 编码 ----
        if abl != "no_cogcontext":
            h_final, h_all = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
                cognitive_seq=cognitive_seq,
                h_0=self._gru_hidden,
            )
        else:
            h_final, h_all = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
                h_0=self._gru_hidden,
            )

        self._last_gru_hidden = h_final

        # ---- 6. flow_condition ----
        flow_condition = self.condition_proj(h_final)

        # ---- 7. 预测 ----
        # Generate prompts from condition (prefix-tuning)
        prompts = None
        if self._use_prompts:
            domain_ids = scene_data.get("domain_id") if scene_data else None
            if domain_ids is not None and isinstance(domain_ids, (list, np.ndarray)):
                domain_ids = torch.as_tensor(domain_ids, device=device)
            prompts = self.prompt_generator(flow_condition, domain_ids=domain_ids)

        if abl != "no_flowchain":
            pred = self.flow_chain(
                obs_trajectory=obs_trajectory,
                perception_c=flow_condition,
                num_samples=num_samples,
                prompts=prompts,
            )
        else:
            h = h_final
            if h.dim() == 1:
                h = h.unsqueeze(0)
            deltas = self.mlp_decoder(
                torch.cat([h, flow_condition], dim=-1)
            )
            deltas = deltas.view(B, self.pred_len, self.trajectory_dim)
            last_obs = obs_trajectory[:, -1:]
            mean_pred = last_obs + deltas
            pred = {
                "mean": mean_pred,
                "samples": mean_pred.unsqueeze(0).repeat(num_samples, 1, 1, 1),
                "log_probs": torch.zeros(num_samples, B, device=device),
                "std": torch.ones(B, self.pred_len, self.trajectory_dim, device=device),
            }

        # ---- 8. 闯红灯分类 (Stage 3) ----
        if self.stage >= 3 and hasattr(self, 'classifier'):
            samples = pred["samples"].squeeze(1) if B == 1 else pred["samples"]
            is_viol, viol_prob = self.classifier.classify(samples)
            pred["is_violation"] = is_viol
            pred["violation_probability"] = viol_prob

        if return_details:
            pred["perception_c"] = flow_condition
            pred["cognitive_seq"] = cognitive_seq
            pred["gat_seq"] = gat_seq

        return pred

    # ------------------------------------------------------------------
    # Compute perception condition (no FlowChain — for fast training)
    # ------------------------------------------------------------------

    def compute_condition(
        self,
        obs_trajectory: Tensor,      # (B, obs_len, 2)
        scene_list: list,            # [B dicts] or None
        domain_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Run perception pipeline → flow_condition. No FlowChain.
        Use this during training with flow_chain.log_prob() for speed.

        Returns
        -------
        flow_condition : (B, condition_dim)
        prompts : (B, num_prompts, d_model) or None
        """
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device

        if self.stage < 2:
            flow_condition = torch.zeros(B, self.condition_dim, device=device)
        else:
            flow_condition = self._compute_perception_precomputed_batched(
                obs_trajectory, scene_list,
            )
            if torch.isnan(flow_condition).any():
                import logging
                _log = logging.getLogger(__name__)
                _log.warning("Perception pipeline produced NaN — falling back to zeros.")
                flow_condition = torch.where(
                    torch.isfinite(flow_condition), flow_condition,
                    torch.zeros_like(flow_condition),
                )

        prompts = None
        if self._use_prompts:
            prompts = self.prompt_generator(flow_condition, domain_ids=domain_ids)

        return flow_condition, prompts

    def _forward_with_precomputed_scenes(
        self,
        obs_trajectory: Tensor,
        scene_list: list,
        num_samples: int,
        return_details: bool,
        domain_ids: Optional[Tensor] = None,
    ) -> dict:
        """Handle precomputed scene format: batched perception pipeline with
        disjoint-union GAT for parallel processing across all B×T frames."""
        B = obs_trajectory.shape[0]
        device = obs_trajectory.device
        abl = self.ablation
        cond_dim = self.condition_dim

        # Stage 1 fallback
        if self.stage < 2:
            flow_condition = torch.zeros(B, cond_dim, device=device)
        else:
            flow_condition = self._compute_perception_precomputed_batched(
                obs_trajectory, scene_list,
            )

        # Generate prompts if enabled
        prompts = None
        if self._use_prompts:
            prompts = self.prompt_generator(flow_condition, domain_ids=domain_ids)

        # FlowChain prediction (batched)
        if abl != "no_flowchain":
            pred = self.flow_chain(
                obs_trajectory=obs_trajectory,
                perception_c=flow_condition,
                num_samples=num_samples,
                prompts=prompts,
            )
        else:
            pred = self._mlp_prediction(obs_trajectory, flow_condition, num_samples, B, device)

        if return_details:
            pred["perception_c"] = flow_condition

        return pred

    # ------------------------------------------------------------------
    # Batched perception pipeline (disjoint-union GAT for speed)
    # ------------------------------------------------------------------

    def _compute_perception_precomputed_batched(
        self,
        obs_trajectory: Tensor,      # (B, obs_len, 2)
        scene_list: list,            # [B dicts] or None per sample
    ) -> Tensor:
        """
        Batched perception pipeline using disjoint-union GAT.

        Instead of processing B×T frames sequentially (each with tiny
        2-5 node graphs triggering 256+ separate CUDA kernel launches),
        we concatenate all frames' nodes/edges into one big graph,
        run GAT ONCE, then split results back.

        Speedup: ~8-16× for the GAT portion (kernel launch overhead
        dominates when per-frame N < 10).
        """
        B = obs_trajectory.shape[0]
        T = self.obs_len
        device = obs_trajectory.device
        abl = self.ablation
        D_gat = self.node_feat_dim
        D_cog = self.behavioral_dim

        # ================================================================
        # Phase 1: Collect all per-frame data into flat lists
        # ================================================================

        # Per-frame data (some frames may be empty / padded)
        frame_data = []     # list of dicts: {b, t, bboxes, positions, velocities, class_names, N, valid}
        total_nodes = 0

        for b in range(B):
            scene = scene_list[b]
            if scene is None:
                for t in range(T):
                    frame_data.append({"b": b, "t": t, "valid": False, "N": 0})
                continue

            bboxes_list = scene.get("bboxes", [])
            positions_list = scene.get("positions", [])
            velocities_list = scene.get("velocities", [])
            class_names_list = scene.get("class_names", [])
            T_actual = min(T, len(bboxes_list))

            for t in range(T_actual):
                b_t = bboxes_list[t]
                if isinstance(b_t, np.ndarray):
                    b_t = torch.from_numpy(b_t).float()
                else:
                    b_t = torch.as_tensor(b_t, dtype=torch.float32)
                N = b_t.shape[0]

                p_t = positions_list[t]
                v_t = velocities_list[t]
                if isinstance(p_t, np.ndarray):
                    p_t = torch.from_numpy(p_t).float()
                if isinstance(v_t, np.ndarray):
                    v_t = torch.from_numpy(v_t).float()

                frame_data.append({
                    "b": b, "t": t, "valid": N > 0, "N": N,
                    "bboxes": b_t, "positions": p_t,
                    "velocities": v_t, "class_names": list(class_names_list[t]),
                })
                total_nodes += N

            # Pad remaining frames
            for t in range(T_actual, T):
                frame_data.append({"b": b, "t": t, "valid": False, "N": 0})

        # ================================================================
        # Phase 2: Node encoding + graph building (per-frame, CPU graphs)
        # ================================================================

        all_node_embs = []          # list of (N_f, D_gat) for each valid frame
        valid_frames = []           # parallel list of frame_data dicts
        all_edge_indices = []       # list of (2, E_f) for each valid frame
        all_edge_types_list = []    # list of (E_f,) for each valid frame
        all_node_types_list = []    # list of [int] for each valid frame
        frame_node_offset = []      # cumulative node offset per VALID frame
        frame_to_idx = {}           # (b, t) → valid_frame_idx
        cum_nodes = 0

        valid_idx = 0  # index within valid frames (not frame_data)
        for fd in frame_data:
            if not fd["valid"]:
                continue

            N = fd["N"]
            b_t = fd["bboxes"].to(device)
            p_t = fd["positions"].to(device)
            v_t = fd["velocities"].to(device)
            cn_t = fd["class_names"]

            # --- Node encoding (internal node_encoder, skip GAT for now) ---
            if abl != "no_graph":
                node_feats = self.perception_graph.node_encoder(
                    bboxes=b_t, class_names=cn_t,
                    positions=p_t, velocities=v_t, device=str(device),
                )  # (N, D_gat)
            else:
                node_feats = self.simple_encoder(
                    bboxes=b_t, class_names=cn_t,
                    positions=p_t, velocities=v_t, device=str(device),
                )
            all_node_embs.append(node_feats)

            # --- Build graph (CPU, numpy — fast for small N) ---
            pos_np = fd["positions"].numpy()
            ei, nt, et = self.graph_builder.build(
                positions=pos_np, class_names=cn_t, target_idx=0,
            )
            all_edge_indices.append(ei)
            all_node_types_list.append(nt)
            all_edge_types_list.append(et)

            valid_frames.append(fd)
            frame_node_offset.append(cum_nodes)
            frame_to_idx[(fd["b"], fd["t"])] = valid_idx
            valid_idx += 1
            cum_nodes += N

        n_valid = len(all_node_embs)  # number of non-empty frames

        if total_nodes == 0:
            return torch.zeros(B, self.condition_dim, device=device)

        # ================================================================
        # Phase 3: Disjoint-union GAT — run ONE forward for ALL frames
        # ================================================================

        all_target_embs = [None] * n_valid  # placeholder, filled below

        if n_valid > 0:
            # --- Build big node/position/bbox tensors ---
            big_node_feats = torch.cat(all_node_embs, dim=0).to(device)  # (total_N, D_gat)
            big_bboxes = torch.cat([fd["bboxes"] for fd in valid_frames], dim=0).to(device)
            big_positions = torch.cat([fd["positions"] for fd in valid_frames], dim=0).to(device)
            big_velocities = torch.cat([fd["velocities"] for fd in valid_frames], dim=0).to(device)

            # --- Build offset edge indices ---
            big_edge_parts = []
            big_edge_type_parts = []
            for vi in range(n_valid):
                ei = all_edge_indices[vi]
                et = all_edge_types_list[vi]
                if ei.numel() == 0:
                    continue
                ei_offset = ei + frame_node_offset[vi]
                big_edge_parts.append(ei_offset)
                big_edge_type_parts.append(et)

            if big_edge_parts and abl != "no_graph":
                big_edge_index = torch.cat(big_edge_parts, dim=1).to(device)
                big_edge_types_cat = torch.cat(big_edge_type_parts, dim=0).to(device)
            else:
                big_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
                big_edge_types_cat = torch.empty(0, dtype=torch.long, device=device)

            if abl != "no_graph" and big_edge_parts:
                big_src = big_edge_index[0]
                big_dst = big_edge_index[1]

                # STRR edge weights
                rel_spatial = self.perception_graph.rel_spatial_encoder(
                    bbox_src=big_bboxes[big_src],
                    bbox_dst=big_bboxes[big_dst],
                )  # (total_E, 8)
                big_edge_weight = self.perception_graph.edge_weight_calc(
                    h_src=big_node_feats[big_src],
                    h_dst=big_node_feats[big_dst],
                    spatial_src=rel_spatial,
                    spatial_dst=rel_spatial,
                )  # (total_E,)
                # Edge features
                big_edge_attr = self.perception_graph.edge_feat_encoder(
                    pos_src=big_positions[big_src],
                    pos_dst=big_positions[big_dst],
                    edge_types=big_edge_types_cat,
                    vel_src=big_velocities[big_src],
                    vel_dst=big_velocities[big_dst],
                )  # (total_E, 4)

                # --- ONE big GAT forward ---
                big_node_emb = self.perception_graph.gat(
                    x=big_node_feats,
                    edge_index=big_edge_index,
                    edge_weight=big_edge_weight,
                    edge_attr=big_edge_attr,
                )  # (total_N, D_gat)

                # Split back: update node embeddings per frame
                for vi in range(n_valid):
                    start = frame_node_offset[vi]
                    N_f = all_node_embs[vi].shape[0]
                    all_node_embs[vi] = big_node_emb[start:start + N_f]

                # Extract target embeddings PER SAMPLE in temporal order.
                # ped_gru accumulates state across frames of the same sample;
                # processing interleaved samples would corrupt that state.
                for b in range(B):
                    ped_hidden = None
                    for t in range(T):
                        key = (b, t)
                        if key not in frame_to_idx:
                            continue
                        vi = frame_to_idx[key]
                        start = frame_node_offset[vi]
                        t_emb = big_node_emb[start:start + 1]  # (1, D_gat)
                        if (self.perception_graph.use_ped_gru
                                and all_node_embs[vi].shape[0] > 0):
                            t2 = t_emb.unsqueeze(0)  # (1, 1, D_gat)
                            t2, ped_hidden = self.perception_graph.ped_gru(
                                t2, ped_hidden)
                            t_emb = t2.squeeze(0)  # (1, D_gat)
                        all_target_embs[vi] = t_emb.squeeze(0)  # (D_gat,)
            else:
                # no_graph or no edges: target from node embeddings directly
                for vi in range(n_valid):
                    t_emb = self._safe_target_embed(
                        all_node_embs[vi].to(device), 0, device)
                    all_target_embs[vi] = t_emb

        # ================================================================
        # Phase 4: Memory + Fusion (per-frame MLPs, fast)
        # ================================================================

        cognitive_out = {}  # (b, t) → (D_cog,) tensor
        gat_out = {}        # (b, t) → (D_gat,) tensor

        for vi in range(n_valid):
            fd = valid_frames[vi]
            b, t = fd["b"], fd["t"]
            node_emb = all_node_embs[vi]
            target_emb = all_target_embs[vi]

            if target_emb.dim() == 0:
                target_emb = target_emb.unsqueeze(0)
            if target_emb.dim() == 1:
                target_emb = target_emb.unsqueeze(0)
            gat_out[(b, t)] = target_emb

            if abl != "no_memory":
                N = node_emb.shape[0]
                safe_ti = min(0, N - 1) if N > 0 else 0
                nt = all_node_types_list[vi]
                infra_idx = [i for i, tp in enumerate(nt) if tp == 3]
                agent_idx = [i for i, tp in enumerate(nt)
                             if tp in (0, 1) and i != safe_ti]
                c_vec, mem_info = self.perception_memory(
                    node_embeddings=node_emb,
                    target_idx=safe_ti,
                    infra_indices=infra_idx,
                    agent_indices=agent_idx,
                )
                c_fused, _ = self.memory_fusion(
                    behavioral=mem_info["behavioral"],
                    environmental=mem_info["environmental"],
                    interactive=mem_info["interactive"],
                )
            else:
                c_fused = self.direct_context(target_emb)

            if c_fused.dim() == 1:
                c_fused = c_fused.unsqueeze(0)
            cognitive_out[(b, t)] = c_fused

        # ================================================================
        # Phase 5: Build (B, T, D) tensors for GRU
        # ================================================================

        gat_seq = torch.zeros(B, T, D_gat, device=device)
        cognitive_seq = torch.zeros(B, T, D_cog, device=device)

        for b in range(B):
            for t in range(T):
                key = (b, t)
                if key in gat_out:
                    g = gat_out[key]
                    c = cognitive_out[key]
                    if g.dim() == 1:
                        g = g.unsqueeze(0)
                    if c.dim() == 1:
                        c = c.unsqueeze(0)
                    gat_seq[b, t] = g.squeeze(0)
                    cognitive_seq[b, t] = c.squeeze(0)

        # ================================================================
        # Phase 6: GRU (already batched over B)
        # ================================================================

        if abl != "no_cogcontext":
            h_final, _ = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
                cognitive_seq=cognitive_seq,
            )
        else:
            h_final, _ = self.perception_gru.encode(
                trajectory=obs_trajectory,
                gat_seq=gat_seq,
            )

        flow_condition = self.condition_proj(h_final)
        return flow_condition

    def _mlp_prediction(self, obs_trajectory, flow_condition, num_samples, B, device):
        """Fallback MLP prediction for no_flowchain ablation."""
        if hasattr(self, '_last_gru_hidden') and self._last_gru_hidden is not None:
            h = self._last_gru_hidden
        else:
            h = flow_condition
        if h.dim() == 1:
            h = h.unsqueeze(0)
        deltas = self.mlp_decoder(torch.cat([h, flow_condition], dim=-1))
        deltas = deltas.view(B, self.pred_len, self.trajectory_dim)
        last_obs = obs_trajectory[:, -1:]
        mean_pred = last_obs + deltas
        return {
            "mean": mean_pred,
            "samples": mean_pred.unsqueeze(0).repeat(num_samples, 1, 1, 1),
            "log_probs": torch.zeros(num_samples, B, device=device),
            "std": torch.ones(B, self.pred_len, self.trajectory_dim, device=device),
        }

    # ------------------------------------------------------------------
    # 认知变化处理 (REVISED)
    # ------------------------------------------------------------------

    def _handle_cognitive_change(self, events_by_type: Dict[str, list]):
        """
        处理认知状态变化事件。

        检测流程:
            graph change → c state change → memory conflict
            → 清除记忆 / 重置GRU / 触发重计算

        Parameters
        ----------
        events_by_type : dict
            {"structural": [...], "cognitive": [...], "conflict": [...]}
        """
        structural = events_by_type.get("structural", [])
        cognitive = events_by_type.get("cognitive", [])
        conflict = events_by_type.get("conflict", [])

        if self.ablation in ("no_memory", "no_change") or not hasattr(self, 'decay_controller'):
            for evt_list in (structural, cognitive, conflict):
                self._change_events.extend(evt_list)
            return

        # 1. 图结构变化 → 清除记忆 + 重置GRU
        for evt in structural:
            affected = evt.affected_memory
            logger.debug(
                f"[Structural] {evt.change_type.value} "
                f"(severity={evt.severity:.2f}, affect={affected})"
            )
            if affected == "all":
                self.decay_controller.reset()
                self._gru_hidden = None
            elif affected == "environmental":
                self.decay_controller.clear_slot("environmental")
            elif affected == "interactive":
                self.decay_controller.clear_slot("interactive")
            elif affected == "behavioral":
                self.decay_controller.clear_slot("behavioral")

        # 2. 认知状态c变化 → 标记需要重计算Memory
        for evt in cognitive:
            logger.debug(
                f"[Cognitive] c-state changed: {evt.detail}"
            )
            # 下一帧会重新计算Memory + Fusion

        # 3. Memory冲突 → 降低冲突Memory置信度
        for evt in conflict:
            logger.debug(
                f"[Conflict] Memory conflict: {evt.detail}"
            )
            # 冲突触发Memory重计算 (下一帧)

        self._change_events.extend(structural)
        self._change_events.extend(cognitive)
        self._change_events.extend(conflict)

    # ------------------------------------------------------------------
    # 辅助: 安全获取 target embedding (防止空节点 IndexError)
    # ------------------------------------------------------------------

    def _safe_target_embed(
        self, node_emb: Tensor, target_idx: int, device
    ) -> Tensor:
        """安全获取 target embedding，node_emb 为空时返回零向量."""
        if node_emb.shape[0] == 0:
            t_emb = torch.zeros(self.node_feat_dim, device=device)
        else:
            safe_idx = min(target_idx, node_emb.shape[0] - 1)
            t_emb = node_emb[safe_idx]
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        return t_emb

    # ------------------------------------------------------------------
    # 辅助: 提取单帧场景数据
    # ------------------------------------------------------------------

    def _get_frame_data(
        self, scene_data: Optional[dict], t: int, B: int, device,
        prev_positions: Optional[torch.Tensor] = None,  # (N, 2) or None for frame t-1
    ) -> Optional[dict]:
        """从scene_data中提取第t帧的感知图输入，健壮处理各种维度."""
        if scene_data is None:
            return None

        try:
            bboxes = scene_data["bboxes"]
            class_names_all = scene_data["class_names"]
            positions = scene_data["positions"]

            ndim = bboxes.dim()

            # --- Dispatch by tensor dimensionality ---
            if ndim == 4:
                # Batch mode: (B, T, N, D)
                b_t = bboxes[0, t] if bboxes.shape[0] == 1 else bboxes[:, t]
                if positions.dim() == 4:
                    p_t = positions[0, t] if positions.shape[0] == 1 else positions[:, t]
                elif positions.dim() == 3:
                    p_t = positions[t]  # single-sample time-indexed
                else:
                    p_t = positions     # static positions (no time dim)

                if isinstance(class_names_all, list) and len(class_names_all) > 0 and isinstance(class_names_all[0], list):
                    # [B][T] or [T][N] format
                    # len <= B: [B][T] → class_names_all[0] = frame0_names, then [t] = agent name (WRONG)
                    # len >  B: [T][N] → class_names_all[t] = frame_t_names (list)
                    cn_t = class_names_all[0][t] if len(class_names_all) <= B else class_names_all[t]
                elif isinstance(class_names_all, list):
                    cn_t = class_names_all[t]  # [T] format shared across batch
                else:
                    cn_t = class_names_all

            elif ndim == 3:
                # Single-sample mode: (T, N, D)
                b_t = bboxes[t]
                if positions.dim() >= 3:
                    p_t = positions[t]
                else:
                    p_t = positions  # static (N, D)
                cn_t = class_names_all[t] if isinstance(class_names_all, list) else class_names_all

            elif ndim == 2:
                # Single frame, no time dim: (N, D)
                b_t = bboxes
                p_t = positions
                cn_t = class_names_all

            else:
                return None

            # --- Squeeze singleton batch dim for uniform downstream handling ---
            if b_t.dim() == 2 and b_t.shape[0] == 1:
                b_t = b_t.squeeze(0)
            if p_t.dim() == 3 and p_t.shape[0] == 1:
                p_t = p_t.squeeze(0)

            # --- Guard: empty nodes → return None so caller falls back to zeros ---
            if b_t.numel() == 0 or b_t.shape[-2] == 0:
                return None

            # Compute velocities from consecutive positions
            velocities = None
            if prev_positions is not None and p_t.shape == prev_positions.shape:
                velocities = p_t.to(device) - prev_positions.to(device)

            return {
                "bboxes": b_t.to(device),
                "class_names": cn_t,
                "positions": p_t.to(device),
                "velocities": velocities,
                "target_idx": scene_data.get("target_idx", 0),
                "track_ids": scene_data.get("track_ids", []),
                "traffic_light_state": scene_data.get("traffic_light_state"),
            }
        except (KeyError, IndexError):
            return None

    # ------------------------------------------------------------------
    # 辅助: 简单轨迹编码 (no_flowchain / 无条件 fallback 用)
    # ------------------------------------------------------------------

    def _encode_trajectory_simple(
        self, obs_trajectory: Tensor, B: int, device
    ) -> Tensor:
        """Encode trajectory using available GRU (zero perception)."""
        if hasattr(self, 'perception_gru'):
            # CognitiveEnhancedGRU with zeros
            z_gat = torch.zeros(B, self.obs_len, self.node_feat_dim, device=device)
            if self.ablation != "no_cogcontext":
                z_cog = torch.zeros(B, self.obs_len, self.behavioral_dim, device=device)
                h_final, _ = self.perception_gru.encode(
                    trajectory=obs_trajectory,
                    gat_seq=z_gat,
                    cognitive_seq=z_cog,
                )
            else:
                h_final, _ = self.perception_gru.encode(
                    trajectory=obs_trajectory,
                    gat_seq=z_gat,
                )
            if h_final.dim() == 1:
                h_final = h_final.unsqueeze(0)
            return h_final
        else:
            return torch.zeros(B, self.gru_hidden_dim, device=device)

    # ------------------------------------------------------------------
    # 状态重置 (新视频开始时调用)
    # ------------------------------------------------------------------

    def reset_state(self):
        """重置所有跨帧状态 (ablation-safe)."""
        self._gru_hidden = None
        self._last_gru_hidden = None
        self._prev_cognitive = None
        self._prev_node_ids = set()
        self._change_events.clear()
        if hasattr(self, 'decay_controller'):
            self.decay_controller.reset()
        if hasattr(self, 'change_detector'):
            self.change_detector.reset()
