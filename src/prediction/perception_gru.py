"""
Cognitive-state enhanced GRU for trajectory encoding.

Replaces the old design where three perception memories directly
controlled GRU gates (Behavioral → candidate, Environmental → reset,
Interactive → update).

New design:
    - Standard GRU (no internal gate modification)
    - Enriched input: concat(trajectory, GAT features, cognitive context)
    - GRU: h_t = GRU(x_t, h_{t-1})  — standard formulation
    - Returns all hidden states [h_1, ..., h_T] for downstream use

Old classes (PerceptionGRUCell, PerceptionGRU) kept for backward
compatibility with existing checkpoints.

References:
    Paper Section 5 (revised): 认知状态增强GRU
    Cho et al., 2014:  original GRU
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# 1. Cognitive-State Enhanced GRU (NEW — replaces PerceptionGRU)
# ======================================================================

class CognitiveEnhancedGRU(nn.Module):
    """
    Standard GRU with cognitively-enriched input.

    Input per frame t:
        x_t = concat(trajectory_t, gat_feature_t, cognitive_context_t)

    GRU (standard, no gate modification):
        h_t = GRU(x_t, h_{t-1})

    This replaces the old PerceptionGRU which injected memories into
    specific GRU gates.

    Parameters
    ----------
    trajectory_dim : int
        Trajectory coordinate dimension (2 for (x, y)).
    gat_dim : int
        GAT output feature dimension.
    cognitive_dim : int
        Memory Attention Fusion output dimension.
    hidden_dim : int
        GRU hidden state dimension.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
    """

    def __init__(
        self,
        trajectory_dim: int = 2,
        gat_dim: int = 128,
        cognitive_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.gat_dim = gat_dim
        self.cognitive_dim = cognitive_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        total_input_dim = trajectory_dim + gat_dim + cognitive_dim

        # Input projection: enrich and standardize
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Standard GRU (do NOT modify internal structure)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        trajectory: Tensor,            # (T, 2) or (B, T, 2)
        gat_seq: Tensor,               # (T, D_gat) or (B, T, D_gat)
        cognitive_seq: Tensor,         # (T, D_cog) or (B, T, D_cog)
        h_0: Optional[Tensor] = None,  # (num_layers, B, hidden_dim) or (num_layers, hidden_dim)
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode observation trajectory with cognitive context.

        Parameters
        ----------
        trajectory : Tensor
            Observed trajectory positions.
        gat_seq : Tensor
            GAT target node embeddings per frame (traffic perception features).
        cognitive_seq : Tensor
            Memory Attention Fusion outputs per frame (traffic cognitive context).

        Returns
        -------
        h_final : Tensor (hidden_dim,) or (B, hidden_dim)
            Final GRU hidden state (last layer).
        h_all : Tensor (T, hidden_dim) or (B, T, hidden_dim)
            All GRU hidden states, for downstream temporal modeling.
        """
        # Normalise to batch-first (B, T, D)
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)       # (1, T, 2)
            gat_seq = gat_seq.unsqueeze(0)             # (1, T, D_gat)
            cognitive_seq = cognitive_seq.unsqueeze(0) # (1, T, D_cog)
            squeeze = True
        else:
            squeeze = False

        B, T, _ = trajectory.shape
        device = trajectory.device

        # --- Build enriched input ---
        x = torch.cat([trajectory, gat_seq, cognitive_seq], dim=-1)  # (B, T, total_dim)
        x = self.input_proj(x)                                       # (B, T, hidden_dim)
        x = self.dropout(x)

        # --- Standard GRU ---
        gru_out, h_n = self.gru(x, h_0)  # gru_out: (B, T, H), h_n: (L, B, H)

        # Final hidden: last layer
        h_final = h_n[-1]  # (B, H)

        if squeeze:
            h_final = h_final.squeeze(0)
            gru_out = gru_out.squeeze(0)

        return h_final, gru_out

    def forward(
        self,
        trajectory: Tensor,
        gat_seq: Tensor,
        cognitive_seq: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Alias for encode()."""
        return self.encode(trajectory, gat_seq, cognitive_seq, h_0)


# ======================================================================
# 2. CognitiveEnhancedGRU — reduced-input variant (no cognitive context)
# ======================================================================

class CognitiveEnhancedGRUNoContext(nn.Module):
    """
    Variant of CognitiveEnhancedGRU without cognitive context input.

    Used for the "no_cogcontext" ablation: tests whether the cognitive context
    from Memory Attention Fusion actually helps trajectory prediction.

    Input per frame t:
        x_t = concat(trajectory_t, gat_feature_t)

    Same as CognitiveEnhancedGRU but without cognitive_seq.
    """

    def __init__(
        self,
        trajectory_dim: int = 2,
        gat_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        total_input_dim = trajectory_dim + gat_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        trajectory: Tensor,
        gat_seq: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Encode without cognitive context."""
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
            gat_seq = gat_seq.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = torch.cat([trajectory, gat_seq], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)

        gru_out, h_n = self.gru(x, h_0)
        h_final = h_n[-1]

        if squeeze:
            h_final = h_final.squeeze(0)
            gru_out = gru_out.squeeze(0)

        return h_final, gru_out


# ======================================================================
# 3. Perception Context Encoder (kept for backward compat)
# ======================================================================

class PerceptionContextEncoder(nn.Module):
    """
    Encodes the time-evolving traffic perception vector c_t from
    the three memory streams.

    At each time step t:
        c_t = GRU_context( [b_t || e_t || i_t], c_{t-1} )

    This captures the temporal dynamics of perception, not just
    instantaneous memory states.

    Note: With the new MemoryAttentionFusion design, this module is
    no longer the primary path for computing flow_condition. It is
    kept for backward compatibility and potential ablation use.

    Parameters
    ----------
    behavioral_dim, environmental_dim, interactive_dim : int
    context_dim : int
        Output dimension of perception vector c.
    """

    def __init__(
        self,
        behavioral_dim: int = 128,
        environmental_dim: int = 128,
        interactive_dim: int = 128,
        context_dim: int = 256,
    ):
        super().__init__()

        concat_dim = behavioral_dim + environmental_dim + interactive_dim
        self.context_gru = nn.GRUCell(
            input_size=concat_dim,
            hidden_size=context_dim,
        )
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        behavioral_seq: Tensor,         # (T, D) or (B, T, D)
        environmental_seq: Tensor,
        interactive_seq: Tensor,
    ) -> Tensor:
        """
        Returns
        -------
        c_seq : Tensor (T, context_dim) or (B, T, context_dim)
        """
        if behavioral_seq.dim() == 2:
            behavioral_seq = behavioral_seq.unsqueeze(0)      # (1, T, D)
            environmental_seq = environmental_seq.unsqueeze(0)
            interactive_seq = interactive_seq.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        B, T, _ = behavioral_seq.shape
        device = behavioral_seq.device

        concat = torch.cat(
            [behavioral_seq, environmental_seq, interactive_seq], dim=-1,
        )  # (B, T, 3*D)

        h = torch.zeros(B, self.context_gru.hidden_size, device=device)
        outputs = []
        for t in range(T):
            h = self.context_gru(concat[:, t, :], h)          # (B, concat_dim)
            outputs.append(self.output_norm(h))               # (B, context_dim)

        out = torch.stack(outputs, dim=1)  # (B, T, context_dim)
        return out.squeeze(0) if squeeze_out else out


# ======================================================================
# 4. Legacy Perception-GRU Cell (DEPRECATED — kept for checkpoint compat)
# ======================================================================

class PerceptionGRUCell(nn.Module):
    """
    [DEPRECATED] Single GRU cell with perception-memory injection.

    Old design:
        r = σ(... + U_r·e)       ← environmental
        z = σ(... + U_z·i)       ← interactive
        n = tanh(... + U_n·b)    ← behavioral

    This class is kept ONLY for loading old checkpoints. New code should
    use CognitiveEnhancedGRU instead.

    Parameters
    ----------
    input_dim : int
    hidden_dim : int
    behavioral_dim, environmental_dim, interactive_dim : int
    dropout : float
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        behavioral_dim: int = 128,
        environmental_dim: int = 128,
        interactive_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_ir = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_iz = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.b_ir = nn.Parameter(torch.zeros(hidden_dim))
        self.b_iz = nn.Parameter(torch.zeros(hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hr = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hz = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hn = nn.Parameter(torch.zeros(hidden_dim))

        self.U_r = nn.Linear(environmental_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(interactive_dim, hidden_dim, bias=False)
        self.U_n = nn.Linear(behavioral_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for name, param in self.named_parameters():
            if 'b_' in name:
                continue
            nn.init.uniform_(param, -std, std)

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        behavioral: Tensor,
        environmental: Tensor,
        interactive: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        r = self.W_ir(x) + self.b_ir + self.W_hr(h) + self.b_hr
        env_term = self.U_r(environmental)
        r = r + env_term
        r = torch.sigmoid(r)
        r = self.dropout(r)

        z = self.W_iz(x) + self.b_iz + self.W_hz(h) + self.b_hz
        interact_term = self.U_z(interactive)
        z = z + interact_term
        z = torch.sigmoid(z)
        z = self.dropout(z)

        n = self.W_in(x) + self.b_in + r * (self.W_hn(h) + self.b_hn)
        behavior_term = self.U_n(behavioral)
        n = n + behavior_term
        n = torch.tanh(n)

        h_new = (1 - z) * n + z * h

        gate_info = {
            "reset_gate": r.detach(),
            "update_gate": z.detach(),
            "env_influence": env_term.detach(),
            "interact_influence": interact_term.detach(),
            "behavioral_influence": behavior_term.detach(),
        }

        return h_new, gate_info


# ======================================================================
# 5. Legacy Perception-GRU (DEPRECATED — kept for checkpoint compat)
# ======================================================================

class PerceptionGRU(nn.Module):
    """
    [DEPRECATED] Perception-infused GRU encoder-decoder.

    Old design: injects three memories into specific GRU gates.

    This class is kept ONLY for loading old checkpoints. New code should
    use CognitiveEnhancedGRU instead.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        behavioral_dim: int = 128,
        environmental_dim: int = 128,
        interactive_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, input_dim)

        self.cells = nn.ModuleList([
            PerceptionGRUCell(
                input_dim=input_dim if layer == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                behavioral_dim=behavioral_dim,
                environmental_dim=environmental_dim,
                interactive_dim=interactive_dim,
                dropout=dropout,
            )
            for layer in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

    def encode(
        self,
        trajectory: Tensor,
        behavioral_seq: Tensor,
        environmental_seq: Tensor,
        interactive_seq: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict]]:
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
            behavioral_seq = behavioral_seq.unsqueeze(0)
            environmental_seq = environmental_seq.unsqueeze(0)
            interactive_seq = interactive_seq.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T, _ = trajectory.shape
        device = trajectory.device

        x = self.input_proj(trajectory)

        if h_0 is None:
            h = [torch.zeros(B, self.hidden_dim, device=device)
                 for _ in range(self.num_layers)]
        else:
            h = [h_0[i] for i in range(self.num_layers)]

        gate_log = []

        for t in range(T):
            x_t = x[:, t, :]
            b_t = behavioral_seq[:, t, :] if behavioral_seq.dim() == 3 else behavioral_seq
            e_t = environmental_seq[:, t, :] if environmental_seq.dim() == 3 else environmental_seq
            i_t = interactive_seq[:, t, :] if interactive_seq.dim() == 3 else interactive_seq

            step_gates = {}
            for layer in range(self.num_layers):
                h[layer], gates = self.cells[layer](
                    x=x_t if layer == 0 else h[layer - 1],
                    h=h[layer],
                    behavioral=b_t,
                    environmental=e_t,
                    interactive=i_t,
                )
                step_gates[f"layer_{layer}"] = gates

            gate_log.append(step_gates)

        h_final = h[-1]

        if squeeze:
            h_final = h_final.squeeze(0)

        return h_final, gate_log

    def decode(self, h_final: Tensor, pred_len: int = 12) -> Tensor:
        decoded = self.decoder(h_final)
        return decoded.unsqueeze(0).repeat(pred_len, 1)

    def forward(
        self,
        trajectory: Tensor,
        behavioral_seq: Tensor,
        environmental_seq: Tensor,
        interactive_seq: Tensor,
        pred_len: int = 12,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Dict]]:
        h_final, gate_log = self.encode(
            trajectory, behavioral_seq, environmental_seq, interactive_seq, h_0,
        )
        traj_pred = self.decode(h_final, pred_len)
        return h_final, traj_pred, gate_log
