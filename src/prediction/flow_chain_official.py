"""
Official FlowChain (ICCV 2023) implementation adapted for our project.

Core components ported from:
    https://github.com/meaten/FlowChain-ICCV2023
    File: src/models/TP/TFCondARFlow.py

Architecture:
    obs (B, T, 2) → PositionalEncoding → TransformerEncoder
        → Autoregressive TransformerDecoder
        → RealNVP (LinearMaskedCoupling + BatchNorm) → pred (B, T_pred, 2)

The flow uses MADE-style alternating masks (not half-split),
Tanh-activated scale networks, and MeanScaler for adaptive normalization.

References:
    FlowChain: Maeda et al., ICCV 2023
    RealNVP:   Dinh et al., ICLR 2017
    MADE:      Germain et al., ICML 2015
"""

import copy
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


# ======================================================================
# 1. MADE Mask Generation
# ======================================================================

def create_masks(
    input_size: int,
    hidden_size: int,
    n_hidden: int,
    input_order: str = "sequential",
    input_degrees: Optional[Tensor] = None,
):
    """
    MADE paper sec 4: degrees of connections between layers ensure
    at most in_degree - 1 connections.

    Returns:
        masks: list of Tensors, one per weight matrix
        first_degrees: first-layer input degrees
    """
    degrees = []

    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev, input_size, (hidden_size,))]
        min_prev = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


# ======================================================================
# 2. Flow Layer Primitives
# ======================================================================

class FlowSequential(nn.Sequential):
    """Container for normalizing flow layers — accumulates log-det-jacobian."""

    def forward(self, x, y):
        sum_log_abs_det = 0
        for module in self:
            x, log_det = module(x, y)
            sum_log_abs_det = sum_log_abs_det + log_det
        return x, torch.sum(sum_log_abs_det, dim=-1)

    def inverse(self, u, y):
        sum_log_abs_det = 0
        for module in reversed(self):
            u, log_det = module.inverse(u, y)
            sum_log_abs_det = sum_log_abs_det + log_det
        return u, torch.sum(sum_log_abs_det, dim=-1)


class BatchNorm(nn.Module):
    """RealNVP BatchNorm layer with running statistics + AdaBN interpolation.

    AdaBN (MetaHTR / Bhunia et al. CVPR 2021):
    When ``alpha < 1.0``, batch statistics are blended with running statistics:
        μ = (1-α)·μ_batch + α·μ_running
    This stabilises meta-learning inner-loop adaptation by preventing the
    BN mismatch that occurs when encoder weights are modified but running
    stats haven't caught up.
    """

    def __init__(self, input_size: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

        # AdaBN mixing factor: 0 = pure batch, 1 = pure running (default)
        self.register_buffer("ada_alpha", torch.tensor(1.0))

        # Cached post-interpolation stats for consistent inverse pass
        self._cached_mean = None
        self._cached_var = None

    def _compute_stats(self, x: torch.Tensor):
        """Return (mean, var) according to ada_alpha interpolation."""
        # Always compute per-batch statistics
        batch_mean = x.reshape(-1, x.shape[-1]).mean(0)
        batch_var = x.reshape(-1, x.shape[-1]).var(0, unbiased=False)

        # Standard momentum update when training
        if self.training:
            self.running_mean.mul_(self.momentum).add_(
                batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(
                batch_var.data * (1 - self.momentum))

        # AdaBN interpolation
        a = self.ada_alpha.item()
        if a < 1.0:
            mean = (1 - a) * batch_mean + a * self.running_mean
            var = (1 - a) * batch_var + a * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        return mean, var

    def forward(self, x, cond_y=None):
        mean, var = self._compute_stats(x)
        self._cached_mean = mean
        self._cached_var = var

        # Clamp log_gamma to prevent exp() overflow → INF → NaN downstream.
        # exp(10) ≈ 22026 is safely within float32 range.
        log_gamma = self.log_gamma.clamp(min=-10.0, max=10.0)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = log_gamma.exp() * x_hat + self.beta
        log_det = log_gamma - 0.5 * torch.log(var + self.eps)

        return y, log_det.expand_as(x)

    def inverse(self, y, cond_y=None):
        # Use cached stats from the most recent forward pass for consistency
        # between forward (log_prob) and inverse (sample) paths.
        if self._cached_mean is not None:
            mean = self._cached_mean
            var = self._cached_var
        else:
            mean = self.running_mean
            var = self.running_var

        log_gamma = self.log_gamma.clamp(min=-10.0, max=10.0)

        x_hat = (y - self.beta) * torch.exp(-log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_det = 0.5 * torch.log(var + self.eps) - log_gamma

        return x, log_det.expand_as(x)


class EncoderAdapter(nn.Module):
    """Lightweight bottleneck adapter inserted after frozen Transformer encoder.

    Architecture:  x → Linear(d_model, bottleneck) → ReLU → Linear(bottleneck, d_model) → +x → LayerNorm
    This limits the adaptation capacity so encoder outputs stay close to their
    pre-trained values, preventing downstream flow conditioning (dist_args) from
    drifting into INF territory.

    Total params: ~2K with d_model=64, bottleneck=16 (vs ~51K for full encoder).
    """

    def __init__(self, d_model: int = 64, bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._last_residual = None  # stored for feature-shift regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down(x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.up(h)
        self._last_residual = h  # (B, T, d_model) — adapter contribution before LayerNorm
        return self.norm(residual + h)

    def get_feature_shift(self) -> torch.Tensor:
        """Mean squared L2 norm of adapter residual, for regularization.
        Returns 0 if no forward pass has been done."""
        if self._last_residual is None:
            return torch.tensor(0.0)
        return (self._last_residual ** 2).mean()


class LinearMaskedCoupling(nn.Module):
    """
    Modified RealNVP Coupling Layer per the MAF paper.

    Uses MADE-style alternating masks (not half-split).
    Scale: Tanh-activated (for stability)
    Translation: ReLU-activated
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        mask: Tensor,
        cond_label_size: Optional[int] = None,
        t_clamp: float = 5.0,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("t_clamp", torch.tensor(t_clamp))

        # Scale network
        s_layers = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_layers += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_layers += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_layers)

        # Translation network (ReLU instead of Tanh, per MAF paper)
        self.t_net = copy.deepcopy(self.s_net)
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        mx = x.masked_fill(self.mask == 0, 0.0)
        inp = mx if y is None else torch.cat([y, mx], dim=-1)

        s = torch.clamp(self.s_net(inp), min=-20.0, max=20.0)
        t_raw = self.t_net(inp).masked_fill(self.mask.bool(), 0.0)

        # Bounded translation safeguard: T·tanh(t/T) caps |t| ≤ T
        # Prevents INF propagation when encoder adaptation shifts dist_args.
        t = self.t_clamp * torch.tanh(t_raw / self.t_clamp)

        # masked_fill (selection) instead of `* (1 - mask)`: at masked dims the
        # gradient is a hard 0, so an Inf gradient flowing back cannot form
        # Inf*0 = NaN (the "MulBackward0 returned nan" crash).
        log_s = torch.tanh(s).masked_fill(self.mask.bool(), 0.0)
        u = x * torch.exp(log_s) + t
        log_det = log_s

        return u, log_det

    def inverse(self, u, y=None):
        mu = u.masked_fill(self.mask == 0, 0.0)
        inp = mu if y is None else torch.cat([y, mu], dim=-1)

        s = torch.clamp(self.s_net(inp), min=-20.0, max=20.0)
        t_raw = self.t_net(inp).masked_fill(self.mask.bool(), 0.0)

        # Bounded translation safeguard — MUST match forward(), otherwise the
        # inverse (sampling) path can produce unbounded x → huge ADE → non-finite
        # gradients that corrupt log_gamma through the flow BatchNorm inverse.
        t = self.t_clamp * torch.tanh(t_raw / self.t_clamp)

        # masked_fill (selection) instead of `* (1 - mask)`: same Inf*0 = NaN
        # fix as forward(). See forward() for details.
        log_s = torch.tanh(s).masked_fill(self.mask.bool(), 0.0)
        x = (u - t) * torch.exp(-log_s)
        log_det = -log_s

        return x, log_det


# ======================================================================
# 3. Scalers
# ======================================================================

class Scaler(ABC, nn.Module):
    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    @abstractmethod
    def compute_scale(self, data: Tensor) -> Tensor:
        pass

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        scale = self.compute_scale(data)
        # dim=1 inserts after batch dim — correct for (B, T, D) batch_first format.
        # (The original official code used dim=0 for (T, B, D) time-first format.)
        dim = 1
        if self.keepdim:
            scale = scale.unsqueeze(dim=dim)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(dim=dim), scale


class MeanScaler(Scaler):
    """
    Computes per-item scale from average absolute value over time.
    Items with only zeros get the global average scale.
    """

    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("minimum_scale", torch.tensor(minimum_scale))

    def compute_scale(self, data: Tensor) -> Tensor:
        # data: (B, T, D) — compute scale per (B, D)
        dim = 1  # average over time dimension

        observed_indicator = torch.ones_like(data)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )
        return torch.max(scale, self.minimum_scale).detach()


class NOPScaler(Scaler):
    """No-op scaler — returns scale=1."""

    def compute_scale(self, data: Tensor) -> Tensor:
        dim = 1
        return torch.ones_like(data).mean(dim=dim)


# ======================================================================
# 4. Flow Base + RealNVP
# ======================================================================

class Flow(nn.Module):
    """Base class for normalizing flows."""

    def __init__(self, input_size: int):
        super().__init__()
        self.__scale = None
        self.net = None
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def _match_scale(self, x: Tensor) -> Optional[Tensor]:
        """Expand scale to match x's batch dim and squeeze time dim for flow ops."""
        if self.__scale is None:
            return None
        scale = self.__scale
        if x.shape[0] != scale.shape[0]:
            n = x.shape[0] // scale.shape[0]
            scale = scale.repeat_interleave(n, dim=0)
        # Squeeze the time dim — scale is (B, 1, D) from Scaler but
        # flow ops work on per-step (B, D) or (B*N, D) tensors.
        if scale.dim() == 3 and scale.shape[1] == 1:
            scale = scale.squeeze(1)  # (B, D)
        return scale

    def forward(self, x, cond):
        scale = self._match_scale(x)
        if scale is not None:
            x = x / scale
        u, log_det = self.net(x, cond)
        # Change-of-variables: scale contributes -sum(log|s|) to log-det
        if scale is not None:
            log_det = log_det - torch.log(torch.abs(scale)).reshape(
                log_det.shape[0], -1).sum(-1)
        return u, log_det

    def inverse(self, u, cond):
        x, log_det = self.net.inverse(u, cond)
        scale = self._match_scale(x)
        if scale is not None:
            x = x * scale
            # Change-of-variables: scale contributes +sum(log|s|) to log-det
            log_det = log_det + torch.log(torch.abs(scale)).reshape(
                log_det.shape[0], -1).sum(-1)
        return x, log_det

    def log_prob(self, x, cond):
        u, log_det = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u), dim=-1) + log_det

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape
        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample

    def sample_with_log_prob(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape
        u = self.base_dist.sample(shape)
        sample, log_det = self.inverse(u, cond)
        return sample, torch.sum(self.base_dist.log_prob(u), dim=-1) + log_det


class RealNVP(Flow):
    """
    RealNVP with MADE-style LinearMaskedCoupling + BatchNorm.

    Parameters
    ----------
    n_blocks : int          Number of coupling layers.
    input_size : int        Dimensionality of each data point (2 for 2D coords).
    hidden_size : int       Hidden layer size in coupling nets.
    n_hidden : int          Number of hidden layers in coupling nets.
    cond_label_size : int   Conditioning vector size.
    batch_norm : bool       Whether to insert BatchNorm between layers.
    """

    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: Optional[int] = None,
        batch_norm: bool = True,
    ):
        super().__init__(input_size)
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    input_size, hidden_size, n_hidden, mask, cond_label_size
                )
            ]
            mask = 1 - mask
            if batch_norm:
                modules += [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)


# ======================================================================
# 5. TransformerFlowChain — Full Model
# ======================================================================

class TransformerFlowChain(nn.Module):
    """
    FlowChain trajectory predictor using Transformer encoder-decoder + RealNVP.

    Adapts the official ARFlow architecture to our (B, T, D) batch format
    with optional traffic perception conditioning.

    Parameters
    ----------
    obs_len : int
        Number of observation frames.
    pred_len : int
        Number of prediction frames.
    feature_dim : int
        Coordinate dimension (2 for x,y).
    d_model : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    num_encoder_layers : int
    num_decoder_layers : int
    dim_feedforward_scale : int
        FFN scale factor (multiplier on d_model).
    dropout_rate : float
    n_blocks : int
        Number of RealNVP coupling blocks.
    n_hidden : int
        Number of hidden layers in coupling nets.
    flow_hidden_size : int
        Hidden size in coupling nets.
    conditioning_length : int
        Dimension of the conditioning vector for the flow.
    cond_label_size : int
        External condition dimension (perception vector c).
    scaling : bool
        Whether to use MeanScaler.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        feature_dim: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward_scale: int = 4,
        dropout_rate: float = 0.1,
        n_blocks: int = 3,
        n_hidden: int = 2,
        flow_hidden_size: int = 64,
        conditioning_length: int = 16,
        cond_label_size: Optional[int] = None,
        scaling: bool = True,
        use_adapter: bool = True,
        adapter_bottleneck: int = 16,
        cond_inject: str = "encoder",
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.cond_label_size = cond_label_size
        self.cond_inject = cond_inject

        # Positional encoding
        self.pe_dim = 16
        self.input_size = feature_dim + self.pe_dim

        # If external condition provided, expand input
        extra_dim = cond_label_size if cond_label_size else 0
        self.encoder_input = nn.Linear(self.input_size + extra_dim, d_model)
        self.decoder_input = nn.Linear(self.input_size, d_model)

        # Prefix-tuning: separate projection for trajectory-only input
        self.traj_proj = nn.Linear(self.input_size, d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * d_model,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,  # (B, T, D) format
        )

        # Flow
        self.flow = RealNVP(
            n_blocks=n_blocks,
            input_size=feature_dim,
            hidden_size=flow_hidden_size,
            n_hidden=n_hidden,
            cond_label_size=conditioning_length,
            batch_norm=True,
        )

        # Condition projection: d_model → conditioning_length
        self.dist_args_proj = nn.Linear(d_model, conditioning_length)

        # Native flow-level conditioning: project the external perception
        # vector directly onto the flow's conditioning_length (16), bypassing
        # the encoder-input concat. Used when cond_inject == "flow".
        # bias=False: a zero context must map to a zero condition, so the
        # "no-condition" ablation (condition_flow=False) stays truly zero and
        # the decoder's dist_args_proj bias owns the unconditional baseline.
        self.flow_cond_proj = (
            nn.Linear(cond_label_size, conditioning_length, bias=False)
            if cond_label_size is not None else None
        )

        # Scalers
        self.scaling = scaling
        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # Encoder adapter (lightweight, trained by FOMAML instead of full encoder)
        self.use_adapter = use_adapter
        if use_adapter:
            self.encoder_adapter = EncoderAdapter(
                d_model=d_model, bottleneck=adapter_bottleneck, dropout=0.0)

        # Positional encoding (sin/cos)
        position = torch.arange(obs_len + pred_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.pe_dim, 2).float()
            * (-math.log(10000.0) / self.pe_dim)
        )
        pe = torch.zeros(obs_len + pred_len, self.pe_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, pe_dim)

        # Causal mask for autoregressive decoding
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(pred_len),
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(
        self,
        obs: Tensor,                          # (B, obs_len, 2)
        cond: Optional[Tensor] = None,        # (B, cond_label_size) — old-style condition
        prompts: Optional[Tensor] = None,     # (B, num_prompts, d_model) — prefix prompts
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode observation trajectory with optional conditioning.

        Two modes:
        1. Prefix-tuning (prompts is not None):
           prompts prepended as extra tokens → cross-attend with trajectory
        2. Legacy (cond is not None):
           condition concatenated to input features (old behavior)

        Returns
        -------
        enc_out : Tensor (B, obs_len, d_model)
        scale : Tensor (B, 1, 2) — per-sample scale for denormalization
        """
        B = obs.shape[0]
        device = obs.device

        # Compute adaptive scale from observations
        _, scale = self.scaler(obs)  # (B, 1, 2)

        # Scale the flow's internal representation
        if self.scaling:
            self.flow.scale = scale

        # Add positional encoding
        enc_pe = self.pe[: self.obs_len]  # (obs_len, pe_dim)
        enc_pe_expanded = enc_pe.unsqueeze(0).expand(B, -1, -1)  # (B, obs_len, pe_dim)

        # ================================================================
        # Prefix-tuning mode
        # ================================================================
        if prompts is not None:
            # Trajectory tokens: only obs + PE (no condition concatenation)
            traj_input = torch.cat([obs, enc_pe_expanded], dim=-1)  # (B, obs_len, input_size)
            traj_tokens = self.traj_proj(traj_input)                 # (B, obs_len, d_model)

            # Prepend prompt tokens
            seq = torch.cat([prompts, traj_tokens], dim=1)           # (B, P+obs_len, d_model)

            # Full sequence through encoder
            enc_full = self.transformer.encoder(seq)                 # (B, P+obs_len, d_model)

            # Only take trajectory positions as output
            P = prompts.shape[1]
            enc_out = enc_full[:, P:, :]                             # (B, obs_len, d_model)
            if self.use_adapter:
                enc_out = self.encoder_adapter(enc_out)
            return enc_out, scale

        # ================================================================
        # Legacy mode: condition concatenated to input
        # ================================================================
        if cond is not None and self.cond_label_size and self.cond_inject != "flow":
            # Inject perception vector: broadcast to each timestep
            cond_expanded = cond.unsqueeze(1).expand(-1, self.obs_len, -1)  # (B, obs_len, cond_size)
            enc_input = torch.cat([obs, enc_pe_expanded, cond_expanded], dim=-1)
        else:
            enc_input = torch.cat([obs, enc_pe_expanded], dim=-1)
            # Pad with zeros to match encoder_input weight shape when cond_label_size > 0
            if self.cond_label_size:
                z = torch.zeros(B, self.obs_len, self.cond_label_size, device=device)
                enc_input = torch.cat([enc_input, z], dim=-1)

        enc_out = self.transformer.encoder(self.encoder_input(enc_input))
        if self.use_adapter:
            enc_out = self.encoder_adapter(enc_out)
        return enc_out, scale

    # ------------------------------------------------------------------
    # Native flow-level conditioning
    # ------------------------------------------------------------------

    def _project_flow_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        """Project the external perception vector onto the flow's native
        conditioning (conditioning_length). Returns None when unused."""
        if cond is not None and self.flow_cond_proj is not None:
            return self.flow_cond_proj(cond)  # (B, conditioning_length)
        return None

    # ------------------------------------------------------------------
    # Autoregressive Decoding
    # ------------------------------------------------------------------

    def _decode_autoregressive(
        self,
        enc_out: Tensor,      # (B, obs_len, d_model)
        obs: Tensor,          # (B, obs_len, 2)
        num_samples: int,
        flow_cond: Optional[Tensor] = None,  # (B, conditioning_length) native cond
    ) -> Tuple[Tensor, Tensor]:
        """
        Autoregressively decode pred_len steps.

        Returns
        -------
        preds : Tensor (B, num_samples, pred_len, 2)
        log_probs : Tensor (B, num_samples)
        """
        B = obs.shape[0]
        device = obs.device

        # Start from last observed position
        last_pos = obs[:, -1:]  # (B, 1, 2)

        all_samples = []
        all_log_probs = []

        current_pos = last_pos  # (B, 1, 2)
        current_dec_inputs = []

        for k in range(self.pred_len):
            # Positional encoding for step k
            dec_pe_k = self.pe[self.obs_len - 1 + k: self.obs_len + k]  # (1, pe_dim)
            dec_pe_k = dec_pe_k.unsqueeze(0).expand(B, -1, -1)  # (B, 1, pe_dim)

            dec_inp = torch.cat([current_pos, dec_pe_k], dim=-1)  # (B, 1, 2+16)
            current_dec_inputs.append(dec_inp)

            # Concatenate all decoder inputs so far
            dec_seq = torch.cat(current_dec_inputs, dim=1)  # (B, k+1, 2+16)
            dec_embedded = self.decoder_input(dec_seq)

            # Causal mask for k+1 steps
            tgt_mask = self.tgt_mask[:k + 1, :k + 1]

            dec_output = self.transformer.decoder(
                dec_embedded, enc_out, tgt_mask=tgt_mask,
            )  # (B, k+1, d_model)

            # Take last step's output
            last_dec_out = dec_output[:, -1:]  # (B, 1, d_model)

            # Project to flow conditioning
            dist_args = self.dist_args_proj(last_dec_out)  # (B, 1, cond_len)
            if flow_cond is not None:
                dist_args = dist_args + flow_cond.unsqueeze(1)  # + context bias

            # Sample from flow
            # Expand for multiple samples: (B, 1, cond_len) → (B, N, cond_len)
            dist_args_expanded = dist_args.expand(B, num_samples, -1)
            dist_args_flat = dist_args_expanded.reshape(B * num_samples, -1)

            pos_flat, log_prob_flat = self.flow.sample_with_log_prob(
                cond=dist_args_flat,
            )  # pos_flat: (B*N, 2), log_prob_flat: (B*N,)

            pos = pos_flat.view(B, num_samples, 2)       # (B, N, 2)
            log_prob = log_prob_flat.view(B, num_samples)  # (B, N)

            all_samples.append(pos)
            all_log_probs.append(log_prob)

            # For next step: use the most recent position (mean over samples
            # for stability, or last sample)
            # Detach the autoregressive feedback: prevents the gradient from
            # exploding through the recurrent decoder loop (pred_len × flow
            # Jacobian), which is the source of the Inf grads that then hit the
            # masked coupling layers as Inf*0 = NaN.
            current_pos = pos.mean(dim=1, keepdim=True).detach()  # (B, 1, 2)

        # Stack: (B, N, pred_len, 2)
        preds = torch.stack(all_samples, dim=2)
        # Sum log_probs across time: (B, N)
        total_log_probs = torch.stack(all_log_probs, dim=-1).sum(dim=-1)

        return preds, total_log_probs

    # ------------------------------------------------------------------
    # Forward Passes
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: Tensor,                          # (B, obs_len, 2)
        cond: Optional[Tensor] = None,        # (B, cond_label_size)
        prompts: Optional[Tensor] = None,     # (B, num_prompts, d_model)
        num_samples: int = 20,
    ) -> Dict[str, Tensor]:
        """
        Full forward: encode + autoregressive decode.

        Returns
        -------
        dict with:
            "samples":   (B, N, pred_len, 2)  sampled trajectories
            "log_probs": (B, N)                log-probabilities
            "mean":      (B, pred_len, 2)      mean prediction
            "std":       (B, pred_len, 2)      per-step std
        """
        # Encode
        enc_out, scale = self._encode(obs, cond, prompts=prompts)

        # Native flow-level condition (bypasses the encoder when
        # cond_inject == "flow"; the encoder stays trajectory-only).
        flow_cond = self._project_flow_cond(cond)

        # Autoregressive decode
        preds, log_probs = self._decode_autoregressive(
            enc_out, obs, num_samples, flow_cond=flow_cond,
        )

        # Compute mean and std
        mean = preds.mean(dim=1)  # (B, pred_len, 2)
        std = preds.std(dim=1)    # (B, pred_len, 2)

        return {
            "samples": preds,
            "log_probs": log_probs,
            "mean": mean,
            "std": std,
        }

    def log_prob(
        self,
        obs: Tensor,
        target: Tensor,                       # (B, pred_len, 2)
        cond: Optional[Tensor] = None,
        prompts: Optional[Tensor] = None,     # (B, num_prompts, d_model)
    ) -> Tensor:
        """
        Compute log-probability of a ground-truth trajectory.

        Uses teacher forcing: encodes full observed sequence, then computes
        per-step log-prob under the flow.
        """
        B = obs.shape[0]
        device = obs.device

        # Encode
        enc_out, scale = self._encode(obs, cond, prompts=prompts)

        # Build decoder inputs: shift right (last obs + all but last target)
        dec_inputs = torch.cat([obs[:, -1:], target[:, :-1]], dim=1)  # (B, pred_len, 2)

        dec_pe = self.pe[self.obs_len - 1: self.obs_len - 1 + self.pred_len]
        dec_pe = dec_pe.unsqueeze(0).expand(B, -1, -1)

        dec_embedded = self.decoder_input(
            torch.cat([dec_inputs, dec_pe], dim=-1)
        )  # (B, pred_len, d_model)

        dec_output = self.transformer.decoder(
            dec_embedded, enc_out, tgt_mask=self.tgt_mask,
        )  # (B, pred_len, d_model)

        # Project to conditioning per step
        dist_args = self.dist_args_proj(dec_output)  # (B, pred_len, cond_len)
        flow_cond = self._project_flow_cond(cond)
        if flow_cond is not None:
            dist_args = dist_args + flow_cond.unsqueeze(1)  # + context bias

        # Flatten for batched log_prob
        target_flat = target.reshape(B * self.pred_len, self.feature_dim)
        dist_args_flat = dist_args.reshape(B * self.pred_len, -1)

        # Per-step log-prob under conditional flow
        step_log_probs = self.flow.log_prob(target_flat, dist_args_flat)
        step_log_probs = step_log_probs.view(B, self.pred_len)

        # Sum across time steps
        total_log_prob = step_log_probs.sum(dim=-1)  # (B,)
        return total_log_prob

    def predict(
        self,
        obs: Tensor,
        cond: Optional[Tensor] = None,
        prompts: Optional[Tensor] = None,     # (B, num_prompts, d_model)
        num_samples: int = 20,
    ) -> Dict[str, Tensor]:
        """Alias for forward — used by training/evaluation scripts."""
        return self.forward(obs, cond, prompts=prompts, num_samples=num_samples)


# ======================================================================
# 6. Loss Function
# ======================================================================

def transformer_flow_nll_loss(
    model: TransformerFlowChain,
    obs: Tensor,              # (B, obs_len, 2)
    target: Tensor,           # (B, pred_len, 2)
    cond: Optional[Tensor] = None,
    prompts: Optional[Tensor] = None,  # (B, num_prompts, d_model)
) -> Tensor:
    """
    Compute NLL loss for training — negative log-likelihood of true
    trajectory under the flow model (teacher-forced).
    """
    log_prob = model.log_prob(obs, target, cond, prompts=prompts)
    return -log_prob.mean()


def joint_nll_mse_loss(
    pred: Dict[str, Tensor],
    target: Tensor,           # (B, pred_len, 2)
    mse_weight: float = 1.0,
) -> Tensor:
    """
    Joint NLL + MSE loss.

    NLL ensures the distribution is well-calibrated.
    MSE directly penalizes position error.

    Parameters
    ----------
    pred : dict from TransformerFlowChain.forward()
    target : ground-truth trajectory
    mse_weight : weight for MSE term (default 1.0)
    """
    # NLL: use best sample's log_prob
    log_probs = pred["log_probs"]  # (B, N)
    best_nll = -log_probs.max(dim=-1)[0].mean()  # best-of-N

    # MSE: mean prediction vs target
    mse = ((pred["mean"] - target) ** 2).mean()

    return best_nll + mse_weight * mse
