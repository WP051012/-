"""
FOMAML v2 — Condition-Guided Domain-Adaptive Trajectory Prediction
==================================================================
GAT scene embeddings (64-dim) → ModulationNet → θ_init = θ_meta + δ
Inner loop: K-step SGD from θ_init
Outer loop: Query loss → θ_meta, Support loss at θ_init → mod_net

Trainable (~51K params, <0.5%):
  - Encoder self-attention (Q/K/V/O)   — 3 layers
  - Encoder LayerNorm (norm1/norm2)    — 3 layers + final norm
  - Flow BatchNorm (log_gamma, beta)   — 4 blocks

Frozen: perception (GAT/Memory/GRU/condition_proj), encoder_input,
         FFN, decoder, NVP flow MLPs.
"""

import os, sys, json, time, argparse, logging, warnings
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

warnings.filterwarnings('ignore', message=r'std\(\): degrees of freedom is <= 0')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import TrajectoryDataset, trajectory_collate_fn, is_crossing_candidate
from src.perception_model import TrafficPerceptionModel
from src.modulation_net import ModulationNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ======================================================================
# Geometry loading (from annotation JSONs, same as eval_flowchain_domain.py)
# ======================================================================

def parse_geometry(annotations_dir="data/annotations"):
    """Load stop_line + junction_roi from annotation JSON files.
    junction_roi rectangle → 4-point polygon. No crosswalk_roi → use junction_roi.
    Returns (junction_roi, crosswalk_roi, stop_line).
    """
    annot_dir = Path(annotations_dir)
    if not annot_dir.exists():
        return None, None, None
    geo_a, geo_b = None, None
    for af in sorted(annot_dir.glob("*.json")):
        data = json.loads(af.read_text())
        video = data.get("video", af.stem)
        sl = data.get("stop_line", {})
        jr = data.get("junction_roi", {})
        sl_list = None
        jr_poly = None
        if sl and all(k in sl for k in ("x1", "y1", "x2", "y2")):
            sl_list = [float(sl["x1"]), float(sl["y1"]), float(sl["x2"]), float(sl["y2"])]
        if jr and all(k in jr for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = float(jr["x1"]), float(jr["y1"])
            x2, y2 = float(jr["x2"]), float(jr["y2"])
            jr_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        geo = {"stop_line": sl_list, "junction_roi": jr_poly}
        if "timing" in video:
            geo_a = geo
        else:
            geo_b = geo
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"] and geo["stop_line"]:
            return geo["junction_roi"], geo["junction_roi"], geo["stop_line"]
    for geo in [geo_a, geo_b]:
        if geo and geo["junction_roi"]:
            return geo["junction_roi"], geo["junction_roi"], geo.get("stop_line")
    return None, None, None


# ======================================================================
#  Data — domain-aware splitting
# ======================================================================

def build_domain_split(
    dataset: TrajectoryDataset,
    meta_train_domains: List[int],
    meta_val_domains: List[int],
    meta_test_domains: List[int],
    support_ratio: float = 0.7,
    seed: int = 42,
) -> dict:
    """Split dataset samples by domain at the video level."""
    rng = np.random.RandomState(seed)
    domain_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        did = sample.get("domain_id", -1)
        domain_to_indices[did].append(idx)

    for did in sorted(domain_to_indices.keys()):
        n_samples = len(domain_to_indices[did])
        videos = set()
        for idx in domain_to_indices[did]:
            s = dataset.samples[idx] if hasattr(dataset, 'samples') else {}
            v = s.get('video', '') if isinstance(s, dict) else getattr(s, 'video', '')
            if v:
                videos.add(v)
        logger.info(f"  Domain {did}: {n_samples} samples, ~{len(videos)} videos")

    domain_splits = {}
    for did, indices in domain_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n_support = max(1, int(len(indices) * support_ratio))
        domain_splits[did] = {
            "support": indices[:n_support].tolist(),
            "query": indices[n_support:].tolist(),
        }

    all_domains = set(domain_to_indices.keys())
    meta_train = [d for d in meta_train_domains if d in all_domains]
    meta_val = [d for d in meta_val_domains if d in all_domains]
    meta_test = [d for d in meta_test_domains if d in all_domains]

    logger.info(f"Meta-train domains: {meta_train}")
    logger.info(f"Meta-val   domains: {meta_val}")
    logger.info(f"Meta-test  domains: {meta_test}")

    return {
        "splits": domain_splits, "meta_train": meta_train,
        "meta_val": meta_val, "meta_test": meta_test,
    }


# ======================================================================
#  Dataset filtering helper
# ======================================================================

def filter_crossing_candidates(dataset, indices, junction_roi, stop_line,
                                crosswalk_roi, use_future_gt=True, name=""):
    """Filter indices to crossing-candidate samples."""
    kept = []
    for idx in indices:
        # Use raw samples, not dataset[idx]: __getitem__ returns "obs_trajectory"
        # (normalized/processed), while "obs_positions"/"target_positions" only
        # exist on the raw dataset.samples[idx] dict (same as eval filter_candidates).
        s = dataset.samples[idx]
        obs = s["obs_positions"]
        tgt = s.get("target_positions") if use_future_gt else None
        if is_crossing_candidate(obs, tgt, crosswalk_roi, stop_line, junction_roi):
            kept.append(idx)
    n_viol = sum(1 for i in kept if dataset.samples[i].get("is_violation", False))
    logger.info(f"  {name}: {len(kept)}/{len(indices)} samples, "
                f"{n_viol} violations ({100*n_viol/max(1,len(kept)):.1f}%)")
    return kept


# ======================================================================
#  Trainable parameter enumeration (shared)
# ======================================================================

def _enum_trainable_params(model: TrafficPerceptionModel):
    """Generator yielding (name, param) for all trainable FlowChain params
    in a fixed, deterministic order.

    Architecture: encoder (frozen) → encoder_adapter (trainable) → decoder → flow.
    Only adapter + flow BN are trained by FOMAML.
    """
    fc = model.flow_chain.model

    # Encoder adapter (trainable, ~2K params)
    if fc.use_adapter:
        for pname, p in fc.encoder_adapter.named_parameters():
            if p.requires_grad:
                yield f"enc_adapter.{pname}", p

    # Flow BatchNorm (trainable, ~16 params)
    for name, p in fc.flow.named_parameters():
        if ('log_gamma' in name or 'beta' in name) and p.requires_grad:
            yield f"flow.{name}", p


def get_trainable_params(model: TrafficPerceptionModel) -> OrderedDict:
    """Return {name: param} for all trainable parameters in deterministic order."""
    return OrderedDict(_enum_trainable_params(model))


def get_trainable_param_shapes(model: TrafficPerceptionModel) -> list:
    """Return [(name, tensor_shape)] for all trainable parameters."""
    return [(name, p.shape) for name, p in _enum_trainable_params(model)]


# ======================================================================
#  Model setup
# ======================================================================

def build_model(
    config: dict, perception_ckpt: str, flowchain_ckpt: str,
    device: torch.device,
) -> TrafficPerceptionModel:
    """Build TrafficPerceptionModel, load both checkpoints, set trainable params."""
    model = TrafficPerceptionModel(config, stage=2)
    model = model.to(device)
    model.eval()

    # Load perception checkpoint (skip flow_chain keys)
    logger.info(f"Loading perception from: {perception_ckpt}")
    p_ckpt = torch.load(perception_ckpt, map_location=device, weights_only=False)
    p_sd = p_ckpt.get("model_state") or p_ckpt.get("model") or p_ckpt
    p_keys_loaded = 0
    for k, v in p_sd.items():
        if k.startswith("flow_chain."):
            continue
        try:
            target = model
            parts = k.split(".")
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    break
            else:
                if hasattr(target, parts[-1]):
                    param = getattr(target, parts[-1])
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(v)
                        p_keys_loaded += 1
        except Exception:
            pass
    logger.info(f"  Loaded {p_keys_loaded} perception params by name")

    # Load FlowChain checkpoint
    logger.info(f"Loading FlowChain from: {flowchain_ckpt}")
    f_ckpt = torch.load(flowchain_ckpt, map_location=device, weights_only=False)
    f_sd = f_ckpt.get("model") or f_ckpt.get("model_state") or f_ckpt

    # Prefix remapping
    if any(k.startswith("flow_chain.") for k in f_sd.keys()):
        f_sd = {k.replace("flow_chain.", ""): v for k, v in f_sd.items()}
    if any(k.startswith("predictor.") for k in f_sd.keys()):
        f_sd = {k.replace("predictor.", ""): v for k, v in f_sd.items()}

    fc_missing, fc_unexpected = model.flow_chain.load_state_dict(f_sd, strict=False)
    logger.info(f"  FlowChain missing: {len(fc_missing)}, unexpected: {len(fc_unexpected)}")
    if fc_missing:
        logger.info(f"    First 5: {fc_missing[:5]}")

    # Freeze everything, then unfreeze selected groups
    for p in model.parameters():
        p.requires_grad_(False)

    fc_model = model.flow_chain.model

    use_adapter = config.get("use_adapter", True)

    n_adapter = 0
    # Unfreeze encoder adapter (lightweight, ~2K params)
    if use_adapter and fc_model.use_adapter:
        for p in fc_model.encoder_adapter.parameters():
            p.requires_grad_(True)
            n_adapter += 1

    n_bn = 0
    for name, p in fc_model.flow.named_parameters():
        if 'log_gamma' in name or 'beta' in name:
            p.requires_grad_(True)
            n_bn += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {n_trainable:,} / {n_total:,} "
                f"({100*n_trainable/n_total:.1f}%)")
    logger.info(f"    adapter={n_adapter}, bn={n_bn}")

    return model


# ======================================================================
#  FOMAML forward pass
# ======================================================================

def _log_nan_diag(model: TrafficPerceptionModel):
    """Log diagnostic ranges (log_gamma, dist_args weight) when NaN occurs."""
    try:
        fc_model = model.flow_chain.model
        log_gammas = []
        for m in fc_model.flow.net:
            if hasattr(m, 'log_gamma'):
                log_gammas.append(m.log_gamma.detach().cpu())
        if log_gammas:
            g = torch.cat(log_gammas)
            logger.warning(f"    [diag] log_gamma: min={g.min().item():.3f} "
                           f"max={g.max().item():.3f} mean={g.mean().item():.3f}")
        # dist_args projection weight magnitude
        w = fc_model.dist_args_proj.weight.detach()
        logger.warning(f"    [diag] dist_args_proj.weight: norm={w.norm().item():.3f} "
                       f"max={w.abs().max().item():.3f}")
    except Exception as e:
        logger.warning(f"    [diag] failed: {e}")


def compute_loss(
    model: TrafficPerceptionModel, obs: torch.Tensor, target: torch.Tensor,
    ade_weight: float = 1.0, lambda_feat: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """λ·ADE + λ_feat·||adapter_residual||² with zero perception conditioning.
    ADE-only objective: NLL (forward-flow log_prob) is intentionally dropped
    because it was found not to help. Returns NaN loss tensor on error (caller
    should check torch.isfinite). Logs which stage produced the NaN."""
    B = obs.shape[0]
    device = obs.device
    # Coordinate convention: the pretrained FlowChain encoder was trained on
    # [0,1]-normalized coords (eval_flowchain_domain.py divides by 3840/2160).
    # Normalize obs/target so the frozen encoder sees the same scale; ADE is
    # reported in pixels by denormalizing at the end.
    norm = torch.tensor([3840.0, 2160.0], device=device)
    obs_n = obs / norm
    target_n = target / norm
    zero_cond = torch.zeros(B, model.condition_dim, device=device)
    nan = torch.tensor(float('nan'), device=device, requires_grad=True)
    nan_metrics = {"loss": float('nan'), "ade": float('nan'),
                   "feat_shift": float('nan')}

    fc_model = model.flow_chain.model

    # ── Sampling (inverse flow) ──
    try:
        pred = model.flow_chain(obs_trajectory=obs_n, perception_c=zero_cond, num_samples=1)
        mean_pred = pred["mean"]
    except (ValueError, RuntimeError) as e:
        logger.warning(f"  [NaN] sampling (inverse flow) raised: {type(e).__name__}")
        return nan, nan_metrics
    if not torch.isfinite(mean_pred).all():
        logger.warning(f"  [NaN] sampling (inverse flow) non-finite: "
                       f"min={mean_pred.min().item():.3f} max={mean_pred.max().item():.3f}")
        return nan, nan_metrics
    diff_px = (mean_pred - target_n) * norm
    ade = torch.sqrt((diff_px ** 2).sum(dim=-1) + 1e-8).mean()

    # Feature-shift regularization: penalize adapter residual L2 norm.
    # MUST be read AFTER the forward above — get_feature_shift() returns the
    # residual cached by the *last* forward (EncoderAdapter._last_residual).
    # Reading it BEFORE forward made feat_shift reference the PREVIOUS forward's
    # graph, whose saved tensors were already freed → "backward second time".
    feat_shift = torch.tensor(0.0, device=device)
    if fc_model.use_adapter:
        feat_shift = fc_model.encoder_adapter.get_feature_shift()

    loss = ade_weight * ade + lambda_feat * feat_shift

    with torch.no_grad():
        metrics = {"loss": loss.item(), "ade": ade.item(),
                   "feat_shift": feat_shift.item()}
    return loss, metrics


# ======================================================================
#  FOMAML Trainer (condition-guided)
# ======================================================================

class FOMAMLTrainer:
    def __init__(
        self, model: TrafficPerceptionModel, domain_split: dict,
        dataset: TrajectoryDataset, config: dict, device: torch.device,
        mod_net: Optional[ModulationNet] = None,
        domain_conditions: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.model = model
        self.domain_split = domain_split
        self.dataset = dataset
        self.config = config
        self.device = device
        self.mod_net = mod_net
        self.domain_conditions = domain_conditions or {}

        # Hyperparams
        self.inner_lr = config.get("inner_lr", 0.01)
        self.inner_steps = config.get("inner_steps", 5)
        self.ade_weight = config.get("ade_weight", 1.0)
        self.batch_size = config.get("batch_size", 32)
        self.mod_lr = config.get("modulation_lr", config.get("outer_lr", 1e-3))

        # Regularization
        self.ada_alpha = config.get("ada_alpha", 0.3)
        self.lambda_feat = config.get("lambda_feat", 0.01)
        self.lambda_dist = config.get("lambda_dist", 0.01)
        self.max_delta_norm = config.get("max_delta_norm", 0.1)

        # Trainable meta-params (FlowChain)
        self.trainable_params = get_trainable_params(model)
        n_meta = sum(p.numel() for p in self.trainable_params.values())

        # Outer-loop optimizer (θ_meta)
        self.outer_optimizer = torch.optim.AdamW(
            list(self.trainable_params.values()),
            lr=config.get("outer_lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Modulation optimizer (separate, if mod_net present)
        self.mod_optimizer = None
        if self.mod_net is not None:
            n_mod = sum(p.numel() for p in self.mod_net.parameters())
            logger.info(f"ModulationNet: {n_mod:,} params  (meta: {n_meta:,})")
            self.mod_optimizer = torch.optim.AdamW(
                self.mod_net.parameters(), lr=self.mod_lr, weight_decay=1e-5)

        # Build dataloaders
        self.support_loaders = {}
        self.query_loaders = {}
        splits = domain_split["splits"]
        for did in domain_split["meta_train"] + domain_split["meta_val"]:
            s_indices = splits[did]["support"]
            q_indices = splits[did]["query"]
            if s_indices:
                self.support_loaders[did] = DataLoader(
                    Subset(dataset, s_indices), batch_size=min(self.batch_size, len(s_indices)),
                    shuffle=True, collate_fn=trajectory_collate_fn,
                    num_workers=4, pin_memory=True,
                )
            if q_indices:
                self.query_loaders[did] = DataLoader(
                    Subset(dataset, q_indices), batch_size=min(self.batch_size, len(q_indices)),
                    shuffle=True, collate_fn=trajectory_collate_fn,
                    num_workers=4, pin_memory=True,
                )

        self.best_val_loss = float("inf")
        self.epoch = 0

    # ------------------------------------------------------------------
    # Modulation helpers
    # ------------------------------------------------------------------

    def _get_domain_cond(self, domain_id: int) -> torch.Tensor:
        """Get GAT condition for a domain. Returns (64,) tensor."""
        cond = self.domain_conditions.get(domain_id)
        if cond is None:
            return torch.zeros(64, device=self.device)
        return cond.to(self.device)

    def _apply_modulation(self, cond: torch.Tensor, sign: float = 1.0):
        """Compute δ = mod_net(cond) and apply/undo to meta-params in-place."""
        if self.mod_net is None:
            return None
        cond_2d = cond.unsqueeze(0)  # (1, 64)
        delta_flat = self.mod_net(cond_2d)  # (1, total_params)
        self.mod_net.apply_delta(delta_flat, self.trainable_params, sign=sign)
        return delta_flat

    # ------------------------------------------------------------------
    # AdaBN alpha management
    # ------------------------------------------------------------------

    def _set_bn_alpha(self, alpha: float):
        """Set AdaBN mixing factor on all flow BatchNorm layers."""
        for m in self.model.flow_chain.model.flow.net:
            if hasattr(m, 'ada_alpha'):
                m.ada_alpha.fill_(alpha)

    def _get_bn_state(self) -> dict:
        """Save BN running stats + ada_alpha values."""
        state = {}
        for i, m in enumerate(self.model.flow_chain.model.flow.net):
            if hasattr(m, 'running_mean'):
                state[f'bn.{i}'] = {
                    'running_mean': m.running_mean.clone(),
                    'running_var': m.running_var.clone(),
                    'ada_alpha': m.ada_alpha.item(),
                }
        return state

    def _restore_bn_state(self, state: dict):
        """Restore BN running stats + ada_alpha values."""
        for i, m in enumerate(self.model.flow_chain.model.flow.net):
            k = f'bn.{i}'
            if k in state and hasattr(m, 'running_mean'):
                m.running_mean.copy_(state[k]['running_mean'])
                m.running_var.copy_(state[k]['running_var'])
                if 'ada_alpha' in state[k]:
                    m.ada_alpha.fill_(state[k]['ada_alpha'])

    def _save_state(self) -> dict:
        return {
            'params': {name: p.data.clone() for name, p in self.trainable_params.items()},
            'bn': self._get_bn_state(),
        }

    def _restore_state(self, saved: dict):
        for name, p in self.trainable_params.items():
            p.data.copy_(saved['params'][name])
            p.grad = None
        self._restore_bn_state(saved['bn'])

    def _sanitize_grads(self, tag: str = "") -> list:
        """Zero out NaN/Inf gradients to prevent param corruption.
        Returns the list of param names that had non-finite grads."""
        bad = []
        for name, p in self.trainable_params.items():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad.append(name)
                p.grad.zero_()
        if bad:
            # Diagnostic: are the *finite* grads huge (overflow) or the grads
            # genuinely NaN (0*Inf)? Report max abs finite grad magnitude.
            finite_max = 0.0
            finite_argmax = "none"
            for name, p in self.trainable_params.items():
                if p.grad is not None:
                    g = p.grad
                    if torch.isfinite(g).all():
                        m = g.abs().max().item()
                        if m > finite_max:
                            finite_max = m
                            finite_argmax = name
            logger.warning(
                f"  [grad-{tag}] zeroed {len(bad)} non-finite grads: {bad[:3]} "
                f"(max finite grad={finite_max:.3e} at {finite_argmax})")
        return bad

    def _sanitize_params(self, tag: str = "") -> list:
        """Detect (and report) NaN/Inf in current trainable param values."""
        bad = []
        for name, p in self.trainable_params.items():
            if not torch.isfinite(p.data).all():
                bad.append(name)
        if bad:
            logger.warning(f"  [param-{tag}] non-finite params: {bad[:3]}")
        return bad

    # ------------------------------------------------------------------
    # Inner loop (condition-guided)
    # ------------------------------------------------------------------

    def inner_loop(self, domain_id: int) -> float:
        """K-step inner-loop adaptation from modulated init.
        Assumes model params are already set to θ_init = θ_meta + δ.

        Includes:
        - Gradient clipping (max_norm=1.0)
        - Parameter trust-region clamping (||Δθ|| ≤ max_delta_norm)
        - Feature-shift regularization (λ_feat)
        """
        loader = self.support_loaders.get(domain_id)
        if loader is None:
            return 0.0

        # Save initial parameters for trust-region clamping
        p_init = {name: p.data.detach().clone() for name, p in self.trainable_params.items()}

        total_loss = 0.0
        n_steps = 0

        for k, batch in enumerate(loader):
            if k >= self.inner_steps:
                break
            obs = batch["obs_trajectory"].to(self.device)
            target = batch["target_trajectory"].to(self.device)

            loss, _ = compute_loss(self.model, obs, target,
                                   ade_weight=self.ade_weight,
                                   lambda_feat=self.lambda_feat)

            if torch.isfinite(loss):
                loss.backward()
                self._sanitize_grads("inner")
                for name, p in self.trainable_params.items():
                    if p.grad is not None:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                        # SGD step. Inner-loop SGD is a MANUAL update, so it
                        # must detach — otherwise the modulated (requires_grad)
                        # p.data keeps its grad_fn and the next step's backward
                        # re-traverses a freed graph ("backward second time").
                        p.data = (p.data - self.inner_lr * p.grad).detach()
                        p.grad = None

                        # Parameter trust-region clamping
                        delta = p.data - p_init[name]
                        delta_norm = delta.norm()
                        if delta_norm > self.max_delta_norm:
                            delta = delta * (self.max_delta_norm / delta_norm)
                            p.data = (p_init[name] + delta).detach()

                total_loss += loss.item()
                n_steps += 1
            else:
                logger.warning(f"  Inner loop step {k}: NaN loss, skipping")

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Outer step (condition-guided FOMAML)
    # ------------------------------------------------------------------

    def outer_step(self, domain_id: int) -> Dict[str, float]:
        """One FOMAML outer step with condition-guided initialization.

        Phase A: Modulated init → support loss → train mod_net
        Phase B: Modulated init → inner loop → query loss → train θ_meta
        """
        # ── Save meta-state ──
        meta_state = self._save_state()

        # ── Get domain condition ──
        cond = self._get_domain_cond(domain_id)

        # ── Phase A: Train mod_net via support loss at modulated init ──
        if self.mod_net is not None and self.mod_optimizer is not None:
            self._apply_modulation(cond, sign=+1)  # θ = θ_meta + δ

            # Compute support loss at modulated point
            s_loader = self.support_loaders.get(domain_id)
            if s_loader is not None:
                batch = next(iter(s_loader))
                obs = batch["obs_trajectory"].to(self.device)
                target = batch["target_trajectory"].to(self.device)
                mod_loss, _ = compute_loss(self.model, obs, target,
                                             ade_weight=self.ade_weight,
                                             lambda_feat=self.lambda_feat)
                self.mod_optimizer.zero_grad()
                mod_loss.backward()
                self.mod_optimizer.step()
                mod_loss_val = mod_loss.item()
            else:
                mod_loss_val = 0.0

            # Undo modulation (zero grads first). Restore the CLEAN θ_meta
            # saved in meta_state (NOT `apply_modulation(sign=-1)`, which would
            # re-forward mod_net and leave a graph on the params). This is the
            # fix for "backward through the graph a second time": Phase A's
            # mod_loss graph must not leak into Phase B's forward.
            for p in self.trainable_params.values():
                p.grad = None
            for name, p in self.trainable_params.items():
                p.data = meta_state['params'][name].detach()
        else:
            mod_loss_val = 0.0

        # Zero all param grads (might have residuals from mod_net backward)
        for p in self.trainable_params.values():
            p.grad = None

        # ── Phase B: Inner loop + query loss → θ_meta ──
        # Re-apply modulation (with potentially-updated mod_net)
        self._apply_modulation(cond, sign=+1)  # θ = θ_meta + δ

        # AdaBN: set alpha < 1 so BN blends batch + running stats during
        # inner loop AND query loss. Using pure running stats (alpha=1.0)
        # after adaptation causes NaN because the adapted encoder shifts
        # the flow's activation distribution. Keeping blended stats
        # ensures consistency between adaptation and evaluation.
        self._set_bn_alpha(self.ada_alpha)

        # Inner loop adaptation in eval mode (no dropout, BN uses blended stats)
        self.model.eval()
        inner_loss = self.inner_loop(domain_id)

        # Keep ada_alpha for query loss too — don't restore to 1.0.
        # At meta-test time we would use alpha=1.0, but during meta-training
        # consistency matters more than clean running stats.

        # Query loss on adapted params
        q_loader = self.query_loaders.get(domain_id)
        if q_loader is None:
            self._restore_state(meta_state)
            return {"inner_loss": inner_loss, "query_loss": 0.0, "mod_loss": mod_loss_val}

        total_q_loss = 0.0
        n_q = 0
        for batch in q_loader:
            obs = batch["obs_trajectory"].to(self.device)
            target = batch["target_trajectory"].to(self.device)
            try:
                q_loss, q_metrics = compute_loss(self.model, obs, target,
                                                   ade_weight=self.ade_weight,
                                                   lambda_feat=self.lambda_feat)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"  Query loss error for domain {domain_id}: {e}")
                continue
            if torch.isfinite(q_loss):
                q_loss = q_loss / max(len(q_loader), 1)
                q_loss.backward()
                self._sanitize_grads("query")
                # Clip query grads too (inner loop already clips; without this
                # the outer-loop query gradient can overflow AdamW's state).
                for name, p in self.trainable_params.items():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                total_q_loss += q_metrics["loss"]
                n_q += 1
            else:
                logger.warning(f"  Query loss NaN for domain {domain_id}, skipping backward")

        # Save adapted gradients
        adapted_grads = {}
        for name, p in self.trainable_params.items():
            if p.grad is not None:
                adapted_grads[name] = p.grad.clone()
            else:
                adapted_grads[name] = None
            p.grad = None

        # Restore meta-params + BN
        self._restore_state(meta_state)

        # Place adapted gradients on meta-params
        for name, p in self.trainable_params.items():
            if adapted_grads[name] is not None:
                p.grad = adapted_grads[name]

        return {
            "inner_loss": inner_loss,
            "query_loss": total_q_loss / max(n_q, 1),
            "mod_loss": mod_loss_val,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, float]:
        """Validate on meta-val after inner-loop adaptation from modulated init."""
        state = self._save_state()

        total_ade = 0.0
        total_feat = 0.0
        total_samples = 0

        for domain_id in self.domain_split["meta_val"]:
            cond = self._get_domain_cond(domain_id)
            self._apply_modulation(cond, sign=+1)

            # AdaBN: blend batch+running stats for inner loop + metrics
            self.model.eval()
            self._set_bn_alpha(self.ada_alpha)
            self.inner_loop(domain_id)
            # Keep ada_alpha for metrics too (consistency > clean stats)
            q_loader = self.query_loaders.get(domain_id)
            if q_loader is None:
                self._restore_state(state)
                continue

            for batch in q_loader:
                obs = batch["obs_trajectory"].to(self.device)
                target = batch["target_trajectory"].to(self.device)
                with torch.no_grad():
                    _, metrics = compute_loss(self.model, obs, target,
                                              ade_weight=self.ade_weight,
                                              lambda_feat=self.lambda_feat)
                    total_ade += metrics["ade"] * obs.shape[0]
                    total_feat += metrics.get("feat_shift", 0.0) * obs.shape[0]
                    total_samples += obs.shape[0]

            self._restore_state(state)

        self._restore_state(state)
        return {
            "val_ade": total_ade / max(total_samples, 1),
            "val_combined": self.ade_weight * total_ade / max(total_samples, 1),
            "val_feat": total_feat / max(total_samples, 1),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        ckpt = {
            "epoch": self.epoch,
            "trainable_params": {name: p.data.clone() for name, p in self.trainable_params.items()},
            "optimizer_state": self.outer_optimizer.state_dict(),
            "config": self.config,
        }
        if self.mod_net is not None:
            ckpt["modulation_net"] = self.mod_net.state_dict()
            ckpt["mod_optimizer_state"] = self.mod_optimizer.state_dict()
        if self.domain_conditions:
            ckpt["domain_conditions"] = {str(k): v.clone() for k, v in self.domain_conditions.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.epoch = ckpt["epoch"]
        for name, p in self.trainable_params.items():
            if name in ckpt["trainable_params"]:
                p.data.copy_(ckpt["trainable_params"][name])
        self.outer_optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.mod_net is not None and "modulation_net" in ckpt:
            self.mod_net.load_state_dict(ckpt["modulation_net"])
        if self.mod_optimizer is not None and "mod_optimizer_state" in ckpt:
            self.mod_optimizer.load_state_dict(ckpt["mod_optimizer_state"])
        if "domain_conditions" in ckpt:
            self.domain_conditions = {int(k): v for k, v in ckpt["domain_conditions"].items()}
            for k in self.domain_conditions:
                self.domain_conditions[k] = self.domain_conditions[k].to(self.device)
        logger.info(f"Checkpoint loaded: {path} (epoch {self.epoch})")


# ======================================================================
#  Main training loop
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FOMAML v2 condition-guided meta-learning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--domain-labels", default="data/domains/domain_labels_int.json")
    parser.add_argument("--perception-ckpt", default="checkpoints/stage1_best.pt")
    parser.add_argument("--flowchain-ckpt", default="checkpoints/flowchain_best_finetuned.pt")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--save-dir", default="checkpoints/fomaml_v2")
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--ade-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=500000)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--meta-train", nargs="+", type=int, default=[0, 1, 2, 4, 6])
    parser.add_argument("--meta-val", nargs="+", type=int, default=[3])
    parser.add_argument("--meta-test", nargs="+", type=int, default=[5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility)")

    # GAT condition + modulation
    parser.add_argument("--gat-conditions", default="data/gat_conditions.pt",
                        help="Path to precomputed GAT embeddings")
    parser.add_argument("--mod-bases", type=int, default=64, help="Number of shared bases")
    parser.add_argument("--mod-hidden", type=int, default=128, help="Modulation MLP hidden dim")
    parser.add_argument("--mod-lr", type=float, default=None, help="Separate LR for mod_net")
    parser.add_argument("--no-modulation", action="store_true", help="Disable GAT modulation (baseline)")

    # Crossing filter
    parser.add_argument("--filter-crossing", action="store_true",
                        help="Apply crossing-candidate filter")
    parser.add_argument("--annotations-dir", default="data/annotations")

    # Resume
    parser.add_argument("--resume", default="", help="Resume from checkpoint")

    # Stability / Regularization
    parser.add_argument("--no-adapter", action="store_true",
                        help="Disable encoder adapter (only flow BN trainable, ~16 params)")
    parser.add_argument("--ada-alpha", type=float, default=0.3,
                        help="AdaBN mixing factor: 0=pure batch, 1=pure running (default 0.3)")
    parser.add_argument("--lambda-feat", type=float, default=0.01,
                        help="Feature shift regularization weight (default 0.01)")
    parser.add_argument("--lambda-dist", type=float, default=0.01,
                        help="Distribution shift regularization weight (default 0.01)")
    parser.add_argument("--max-delta-norm", type=float, default=0.1,
                        help="Parameter delta trust-region max norm (default 0.1)")
    parser.add_argument("--anomaly", action="store_true",
                        help="Enable torch.autograd anomaly detection (traceback on NaN grad)")

    args = parser.parse_args()

    # Reproducibility: fix Python, numpy, and torch RNG (incl. CUDA) so the
    # freshly-initialized encoder adapter is deterministic across runs.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info("Anomaly detection ENABLED (will print traceback on first NaN grad)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    fomaml_config = {
        "inner_lr": args.inner_lr, "outer_lr": args.outer_lr,
        "inner_steps": args.inner_steps, "ade_weight": args.ade_weight,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "modulation_lr": args.mod_lr or args.outer_lr,
        "use_adapter": not args.no_adapter,
        "ada_alpha": args.ada_alpha,
        "lambda_feat": args.lambda_feat,
        "lambda_dist": args.lambda_dist,
        "max_delta_norm": args.max_delta_norm,
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Domain labels ──
    with open(args.domain_labels) as f:
        domain_label_map = json.load(f)

    # ── GAT conditions ──
    condition_map = None
    if not args.no_modulation and os.path.exists(args.gat_conditions):
        logger.info(f"Loading GAT conditions: {args.gat_conditions}")
        condition_map = torch.load(args.gat_conditions, map_location="cpu", weights_only=False)
        n_videos = len(condition_map)
        n_entries = sum(len(vd) for vd in condition_map.values())
        logger.info(f"  {n_videos} videos, {n_entries} entries")

    # ── Build dataset ──
    logger.info("Building dataset...")
    dataset = TrajectoryDataset(
        data_dir=args.data_dir, label_dir="labels",
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"], mode="trajectory_only",
        max_samples=args.max_samples, domain_label_map=domain_label_map,
        condition_map=condition_map,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    # ── Crossing-candidate filter (matching FlowChain baseline) ──
    if args.filter_crossing:
        junction_roi, crosswalk_roi, stop_line = parse_geometry(args.annotations_dir)
        if junction_roi is not None and stop_line is not None:
            logger.info("Applying crossing-candidate filter...")
            all_indices = list(range(len(dataset)))
            filtered = filter_crossing_candidates(
                dataset, all_indices, junction_roi, stop_line, crosswalk_roi,
                use_future_gt=True, name="all")
            # Replace dataset samples with filtered ones
            dataset.samples = [dataset.samples[i] for i in filtered]
            logger.info(f"Filtered dataset: {len(dataset)} samples (was {len(all_indices)})")

    # ── Domain split ──
    domain_split = build_domain_split(
        dataset, meta_train_domains=args.meta_train,
        meta_val_domains=args.meta_val, meta_test_domains=args.meta_test,
        support_ratio=0.7,
    )

    # ── Per-domain mean GAT conditions ──
    domain_conditions = {}
    if condition_map is not None:
        for did in domain_split["meta_train"] + domain_split["meta_val"] + domain_split["meta_test"]:
            embs = []
            for idx in domain_split["splits"].get(did, {}).get("support", []):
                s = dataset[idx]
                emb = s.get("cond_embedding")
                if emb is not None and emb.abs().sum() > 0:
                    embs.append(emb)
            if embs:
                domain_conditions[did] = torch.stack(embs).mean(dim=0)
            else:
                domain_conditions[did] = torch.zeros(64)
            logger.info(f"  Domain {did}: mean cond from {len(embs)}/{len(domain_split['splits'].get(did, {}).get('support', []))} samples, "
                        f"norm={domain_conditions[did].norm().item():.4f}")

    # ── Build model ──
    # Inject CLI flags into config for build_model
    config["use_adapter"] = not args.no_adapter
    model = build_model(config, args.perception_ckpt, args.flowchain_ckpt, device)

    # ── Modulation net ──
    mod_net = None
    if not args.no_modulation and condition_map is not None:
        param_shapes = get_trainable_param_shapes(model)
        mod_net = ModulationNet(
            cond_dim=64, hidden_dim=args.mod_hidden,
            n_bases=args.mod_bases, param_shapes=param_shapes,
        ).to(device)
        n_mod = sum(p.numel() for p in mod_net.parameters())
        logger.info(f"ModulationNet: {n_mod:,} params (bases={args.mod_bases}, hidden={args.mod_hidden})")

    # ── Trainer ──
    trainer = FOMAMLTrainer(
        model=model, domain_split=domain_split, dataset=dataset,
        config=fomaml_config, device=device,
        mod_net=mod_net, domain_conditions=domain_conditions,
    )

    # ── Resume ──
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Training loop ──
    meta_train_domains = domain_split["meta_train"]
    n_domains = len(meta_train_domains)
    logger.info(f"\n{'='*60}")
    logger.info(f"FOMAML v2 Training: {n_domains} meta-train domains")
    logger.info(f"  Inner steps: {args.inner_steps}, Inner LR: {args.inner_lr}")
    logger.info(f"  Outer LR: {args.outer_lr}, ADE weight: {args.ade_weight}")
    logger.info(f"  Adapter: {not args.no_adapter}, AdaBN α: {args.ada_alpha}")
    logger.info(f"  Regularization: λ_feat={args.lambda_feat}, λ_dist={args.lambda_dist}, "
                f"max_delta_norm={args.max_delta_norm}")
    logger.info(f"  Modulation: {mod_net is not None}, Crossing filter: {args.filter_crossing}")
    logger.info(f"{'='*60}\n")

    start_epoch = trainer.epoch
    for epoch in range(start_epoch, args.epochs):
        trainer.epoch = epoch
        t0 = time.time()

        trainer.outer_optimizer.zero_grad()

        epoch_inner_loss = 0.0
        epoch_query_loss = 0.0
        epoch_mod_loss = 0.0

        for domain_id in meta_train_domains:
            metrics = trainer.outer_step(domain_id)
            epoch_inner_loss += metrics["inner_loss"]
            epoch_query_loss += metrics["query_loss"]
            epoch_mod_loss += metrics.get("mod_loss", 0.0)

        trainer.outer_optimizer.step()

        # Safety net: detect any param that became non-finite after the step
        trainer._sanitize_params(f"post-step-{epoch}")

        avg_inner = epoch_inner_loss / n_domains
        avg_query = epoch_query_loss / n_domains
        avg_mod = epoch_mod_loss / n_domains
        elapsed = time.time() - t0

        log_msg = (f"Epoch {epoch:3d} | Inner={avg_inner:.4f} Query={avg_query:.4f} "
                   f"Mod={avg_mod:.4f} | Time={elapsed:.1f}s")

        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate()
            log_msg += (f" | Val ADE={val_metrics['val_ade']:.4f} "
                        f"Comb={val_metrics['val_combined']:.4f} "
                        f"Feat={val_metrics.get('val_feat', 0):.6f}")
            if val_metrics["val_combined"] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics["val_combined"]
                trainer.save_checkpoint(os.path.join(args.save_dir, "best_fomaml.pt"))
                log_msg += " *BEST*"

            # Modulation delta norms
            if mod_net is not None:
                for did in meta_train_domains[:2]:
                    cond = trainer._get_domain_cond(did)
                    delta = mod_net(cond.unsqueeze(0))
                    log_msg += f" |D{did}|={delta.norm().item():.4f}"

        logger.info(log_msg)

        if (epoch + 1) % args.save_interval == 0:
            trainer.save_checkpoint(os.path.join(args.save_dir, f"fomaml_epoch{epoch:03d}.pt"))

    trainer.save_checkpoint(os.path.join(args.save_dir, "fomaml_final.pt"))
    logger.info(f"Training complete. Best val combined: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
