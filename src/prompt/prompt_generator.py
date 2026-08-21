"""
Prompt Generator — 场景感知 → prefix tokens

输入: condition向量(256) + domain_id
输出: P个prompt token (P, d_model)，以prefix-tuning方式注入FlowChain Encoder

跨域事件: condition变化超阈值 → 重新生成prompt
静态场景: prompt复用，无需重复生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class PromptGenerator(nn.Module):
    """
    将场景condition + 域信息转化为prefix prompt tokens。

    Pipeline:
        1. Domain Embedding: domain_id → domain_emb(D_domain)
        2. Condition Projection: Linear(256 → d_model)
        3. Concat [domain_emb | condition_proj] → MLP → (P, d_model)
        4. Output: prefix tokens ready for Transformer Encoder

    Parameters
    ----------
    condition_dim : int    输入condition维度 (MemoryFusion输出, 256)
    d_model : int          FlowChain的d_model (64)
    num_prompts : int      生成的prompt token数量
    num_domains : int      域的总数 (聚类后确定，或设为0用continuous mode)
    domain_dim : int       域embedding维度
    hidden_dim : int       Generator内部MLP维度
    """

    def __init__(
        self,
        condition_dim: int = 256,
        d_model: int = 64,
        num_prompts: int = 4,
        num_domains: int = 0,
        domain_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.d_model = d_model
        self.num_prompts = num_prompts
        self.num_domains = num_domains

        # Condition projection: 256 → hidden
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # Domain embedding (if discrete domains exist)
        if num_domains > 0:
            self.domain_embed = nn.Embedding(num_domains, domain_dim)
            total_in = hidden_dim + domain_dim
        else:
            self.domain_embed = None
            total_in = hidden_dim

        # Prompt generation: concat → MLP → P tokens × d_model
        self.prompt_mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_prompts * d_model),
        )

        # LayerNorm per prompt token (stabilize prefix values)
        self.prompt_norm = nn.LayerNorm(d_model)

        # 跨域检测: 缓存上一次的condition，算cosine距离
        self.register_buffer("_cached_condition", None)
        self.register_buffer("_cached_prompts", None)
        self._change_threshold = 0.3

    def forward(
        self,
        condition: Tensor,                      # (B, condition_dim)
        domain_ids: Optional[Tensor] = None,    # (B,) int
        force_regenerate: bool = False,
    ) -> Tensor:
        """
        Generate prefix prompt tokens.

        Parameters
        ----------
        condition : (B, 256)  当前场景condition向量
        domain_ids : (B,)     域标签 (可选)
        force_regenerate : bool  强制重新生成 (跨域事件)

        Returns
        -------
        prompts : (B, num_prompts, d_model)  prefix tokens
        """
        B = condition.shape[0]
        device = condition.device

        # Condition encoding
        h = self.cond_proj(condition)  # (B, hidden_dim)

        # Domain injection
        if self.domain_embed is not None and domain_ids is not None:
            d_emb = self.domain_embed(domain_ids)  # (B, domain_dim)
            h = torch.cat([h, d_emb], dim=-1)       # (B, total_in)

        # Generate prompt tokens
        prompts_flat = self.prompt_mlp(h)  # (B, num_prompts * d_model)
        prompts = prompts_flat.view(B, self.num_prompts, self.d_model)

        # Normalize per token
        prompts = self.prompt_norm(prompts)

        return prompts

    def should_regenerate(
        self, condition: Tensor, threshold: Optional[float] = None
    ) -> bool:
        """
        检测condition是否发生显著变化 (跨域事件检测)。

        Returns
        -------
        True if condition变化超过阈值 (需要重新生成prompt)
        """
        if self._cached_condition is None:
            return True

        thresh = threshold or self._change_threshold
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            condition.detach(), self._cached_condition.detach(), dim=-1
        )
        changed = (1 - cos_sim) > thresh
        return bool(changed.any())

    def cache_condition(self, condition: Tensor):
        """缓存当前condition，用于后续跨域检测"""
        self._cached_condition = condition.detach().clone()

    @property
    def change_threshold(self) -> float:
        return self._change_threshold

    @change_threshold.setter
    def change_threshold(self, value: float):
        self._change_threshold = value
