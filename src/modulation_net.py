"""
Factorized hypernetwork for condition-guided parameter modulation.

Maps a 64-dim GAT scene embedding → parameter offsets (Δθ) for FlowChain's
trainable parameters, using a shared basis decomposition:

    Δθ = Σ_i c_i · basis_i    where c = MLP(cond)

This keeps the modulation net small (~18K MLP params) while the shared bases
(~3.3M params) capture common adaptation patterns across domains.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class ModulationNet(nn.Module):
    """Condition-to-parameter-offset hypernetwork with shared bases.

    Parameters
    ----------
    cond_dim : int        GAT embedding dimension (default 64)
    hidden_dim : int      MLP hidden dimension (default 128)
    n_bases : int         Number of shared basis vectors (default 64)
    param_shapes : list   List of (name, shape) tuples for trainable params
    """

    def __init__(
        self,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        n_bases: int = 64,
        param_shapes: List[tuple] = None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_bases = n_bases
        self.param_shapes = param_shapes or []

        # --- Total number of trainable parameters ---
        total_params = 0
        for _, s in self.param_shapes:
            n = 1
            for d in s:
                n *= d
            total_params += n
        self.total_params = total_params

        # --- Coefficient MLP: cond → basis mixing weights ---
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_bases),
        )

        # --- Shared basis vectors ---
        # Each basis is a full parameter-space vector, initialized near zero
        self.bases = nn.ParameterList([
            nn.Parameter(torch.zeros(total_params) * 0.01)
            for _ in range(n_bases)
        ])

        # --- Per-basis learnable scaling ---
        self.basis_scale = nn.Parameter(torch.ones(n_bases) * 0.01)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: (B, cond_dim) GAT embeddings

        Returns:
            delta: (B, total_params) parameter offsets in flat format
        """
        B = cond.shape[0]

        # Compute basis mixing coefficients
        coeffs = self.mlp(cond)  # (B, n_bases)

        # Weighted sum of bases: Δθ = Σ c_i · s_i · basis_i
        delta = torch.zeros(B, self.total_params, device=cond.device)
        for i in range(self.n_bases):
            scale = self.basis_scale[i]
            delta = delta + coeffs[:, i:i + 1] * scale * self.bases[i].unsqueeze(0)

        return delta

    def apply_delta(
        self,
        delta_flat: torch.Tensor,
        trainable_params: Dict[str, nn.Parameter],
        sign: float = 1.0,
    ):
        """Apply (or undo) parameter offsets to trainable params in-place.

        Args:
            delta_flat: (total_params,) or (1, total_params) parameter offsets
            trainable_params: {name: param} dict from get_trainable_params()
            sign: +1 to apply, -1 to undo
        """
        if delta_flat.dim() == 2:
            delta_flat = delta_flat[0]
        offset = 0
        for name, p in trainable_params.items():
            n = p.numel()
            block = delta_flat[offset:offset + n].view_as(p)
            # Detach the previous value so successive apply/undo calls do NOT
            # accumulate cross-forward autograd graphs (which causes
            # "backward through the graph a second time"). `block` stays
            # differentiable so mod_net still receives gradients in Phase A.
            p.data = p.data.detach() + sign * block
            offset += n
        assert offset == self.total_params, f"Size mismatch: {offset} vs {self.total_params}"

    def get_delta_dict(
        self,
        delta_flat: torch.Tensor,
        trainable_params: Dict[str, nn.Parameter],
    ) -> Dict[str, torch.Tensor]:
        """Convert flat delta to per-parameter dict (for saving/inspection)."""
        if delta_flat.dim() == 2:
            delta_flat = delta_flat[0]
        result = {}
        offset = 0
        for name, p in trainable_params.items():
            n = p.numel()
            result[name] = delta_flat[offset:offset + n].view_as(p).clone()
            offset += n
        return result
