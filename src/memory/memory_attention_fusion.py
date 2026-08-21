"""
Memory Attention Fusion Module
==============================

Replaces the old design where three memories directly control GRU gates
(Behavioral → candidate, Environmental → reset, Interactive → update).

New design: attention-based fusion that learns to weight the three memories
automatically, producing a unified traffic cognitive context **c**.

    score_i = W · M_i
    alpha_i = softmax(score_i)
    c = alpha_b * M_b + alpha_e * M_e + alpha_r * M_r

The attention weights are learned end-to-end via backpropagation.

References:
    Paper Section 3 (revised): 交通认知记忆融合
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MemoryAttentionFusion(nn.Module):
    """
    Attention-based fusion of behavioral, environmental, and interactive
    memories into a unified traffic cognitive context vector.

    Unlike the old MemoryFusion (simple concat + MLP with static confidence),
    this module learns dynamic, input-dependent weights for each memory.

    Parameters
    ----------
    memory_dim : int
        Dimension of each memory vector (all must be equal).
    attention_dim : int
        Hidden dimension for computing attention scores.
    """

    def __init__(
        self,
        memory_dim: int = 128,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.memory_dim = memory_dim

        # Score projection: M_i → scalar score
        self.score_proj = nn.Sequential(
            nn.Linear(memory_dim, attention_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_dim, 1),
        )

        # Optional: project fused output to a different dimension
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )

    def forward(
        self,
        behavioral: Tensor,      # (..., memory_dim)
        environmental: Tensor,   # (..., memory_dim)
        interactive: Tensor,     # (..., memory_dim)
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Fuse three memories via learned attention.

        Parameters
        ----------
        behavioral : Tensor (..., D)
        environmental : Tensor (..., D)
        interactive : Tensor (..., D)
        return_weights : bool
            If True, also return the attention weights.

        Returns
        -------
        c : Tensor (..., D)
            Traffic cognitive context vector.
        weights : Tensor (..., 3) or None
            Attention weights [alpha_b, alpha_e, alpha_r] if requested.
        """
        # Stack memories: (3, ..., D)
        memories = torch.stack([behavioral, environmental, interactive], dim=0)

        # Compute scores: (3, ..., 1)
        # Reshape to apply Linear: flatten batch dims, apply, then restore
        lead_shape = memories.shape[1:-1]  # (...) excluding the memory_dim
        memories_flat = memories.view(3, -1, self.memory_dim)  # (3, N, D)
        scores_flat = self.score_proj(memories_flat)            # (3, N, 1)

        # Softmax over the 3 memory types
        weights_flat = F.softmax(scores_flat, dim=0)            # (3, N, 1)

        # Weighted sum: (N, D)
        c_flat = (weights_flat * memories_flat).sum(dim=0)     # (N, D)

        # Restore batch dimensions
        c = c_flat.view(*lead_shape, self.memory_dim)
        c = self.output_proj(c)

        if return_weights:
            weights = weights_flat.squeeze(-1).transpose(0, 1)  # (N, 3)
            weights = weights.view(*lead_shape, 3)
            return c, weights

        return c, None


class MemoryAttentionFusionV2(nn.Module):
    """
    Enhanced variant with cross-memory interaction before fusion.

    Adds a lightweight self-attention over the 3 memory vectors before
    computing the final weighted sum. This allows memories to exchange
    information (e.g., behavioral intent adjusts based on environment).

    Pipeline:
        1. Stack [M_b, M_e, M_r] → (3, ..., D)
        2. Multi-head cross-attention over the 3-slot sequence
        3. Score projection → softmax → weighted sum
        4. Output projection + LayerNorm

    Parameters
    ----------
    memory_dim : int
    num_heads : int
        Number of attention heads for cross-memory interaction.
    """

    def __init__(
        self,
        memory_dim: int = 128,
        num_heads: int = 4,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.memory_dim = memory_dim

        # Cross-memory multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=False,  # seq-first: (3, B, D)
            dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(memory_dim)

        # Score projection
        self.score_proj = nn.Sequential(
            nn.Linear(memory_dim, attention_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_dim, 1),
        )

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )

    def forward(
        self,
        behavioral: Tensor,
        environmental: Tensor,
        interactive: Tensor,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns
        -------
        c : Tensor (..., memory_dim)
        weights : Tensor (..., 3) or None
        """
        lead_shape = behavioral.shape[:-1]
        D = self.memory_dim

        # Flatten batch dims: (..., D) → (N, D)
        b_flat = behavioral.reshape(-1, D)   # (N, D)
        e_flat = environmental.reshape(-1, D)
        i_flat = interactive.reshape(-1, D)

        # Stack as sequence: (3, N, D)
        mem_seq = torch.stack([b_flat, e_flat, i_flat], dim=0)  # (3, N, D)

        # Cross-memory attention
        attended, _ = self.cross_attn(mem_seq, mem_seq, mem_seq)  # (3, N, D)
        mem_seq = self.attn_norm(mem_seq + attended)               # residual

        # Score & softmax
        scores = self.score_proj(mem_seq)        # (3, N, 1)
        weights = F.softmax(scores, dim=0)       # (3, N, 1)

        # Weighted sum
        c_flat = (weights * mem_seq).sum(dim=0)  # (N, D)
        c = self.output_proj(c_flat)
        c = c.view(*lead_shape, D)

        if return_weights:
            w = weights.squeeze(-1).transpose(0, 1).view(*lead_shape, 3)
            return c, w

        return c, None
