"""
Memory decay mechanism for traffic perception memory.

Implements confidence-based memory decay inspired by MAGELLAN's
cognitive map decay mechanism:
    - Repeated observations increase confidence
    - Unobserved elements decay over time
    - When confidence drops below threshold, memory is "forgotten"

References:
    MAGELLAN: A cognitive map-based model of human wayfinding
    Paper Section 8: 引入记忆衰减机制的概率模型
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryDecay(nn.Module):
    """
    Confidence-based memory decay for traffic perception.

    Each memory slot maintains a confidence ∈ [0, 1]:
        - observed:    c_new = min(1.0, c_old + α_learn)
        - unobserved:  c_new = max(0.0, c_old - α_decay)

    When confidence falls below `forget_threshold`, the memory is
    zeroed out (corresponding to "forgetting").

    Parameters
    ----------
    memory_dim : int
        Dimension of each memory slot.
    num_slots : int
        Number of memory slots (e.g. 3 for behavioral/env/interactive).
    learn_rate : float
        Confidence increase rate on observation.
    decay_rate : float
        Confidence decay rate when unobserved.
    forget_threshold : float
        Minimum confidence before memory is cleared.
    """

    def __init__(
        self,
        memory_dim: int = 128,
        num_slots: int = 3,
        learn_rate: float = 0.1,
        decay_rate: float = 0.01,
        forget_threshold: float = 0.3,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold

        # Per-slot memory storage (stateful, updated externally)
        self.register_buffer(
            "memory_bank",
            torch.zeros(num_slots, memory_dim),
        )
        self.register_buffer(
            "confidence",
            torch.ones(num_slots) * 0.5,  # start at neutral confidence
        )

    # ------------------------------------------------------------------
    # Update API
    # ------------------------------------------------------------------

    def observe(
        self,
        slot_idx: int,
        memory_vector: torch.Tensor,
    ) -> None:
        """
        Update a memory slot on observation.

        Parameters
        ----------
        slot_idx : int
        memory_vector : Tensor (memory_dim,)
        """
        # Increase confidence
        self.confidence[slot_idx] = torch.clamp(
            self.confidence[slot_idx] + self.learn_rate, 0.0, 1.0,
        )
        # Exponential moving average for memory content
        alpha = self.learn_rate
        self.memory_bank[slot_idx] = (
            (1 - alpha) * self.memory_bank[slot_idx] + alpha * memory_vector
        )

    def decay_all(self, observed_slots: Optional[torch.Tensor] = None) -> None:
        """
        Apply decay to all unobserved slots.

        Parameters
        ----------
        observed_slots : Tensor (num_slots,) bool, optional
            Which slots were observed this step.
        """
        for slot in range(self.num_slots):
            if observed_slots is not None and observed_slots[slot]:
                continue
            self.confidence[slot] = torch.clamp(
                self.confidence[slot] - self.decay_rate, 0.0, 1.0,
            )
            # Forget if below threshold
            if self.confidence[slot] < self.forget_threshold:
                self.memory_bank[slot] = torch.zeros_like(self.memory_bank[slot])

    def read(self, slot_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read memory content and confidence for a slot."""
        return self.memory_bank[slot_idx], self.confidence[slot_idx]

    def read_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read all memory contents and confidences."""
        return self.memory_bank, self.confidence

    def reset(self) -> None:
        """Reset all memory and confidence."""
        self.memory_bank.zero_()
        self.confidence.fill_(0.5)


# ======================================================================
# Time-weighted confidence decay (advanced)
# ======================================================================

class TimeWeightedDecay(nn.Module):
    """
    Decay mechanism that weights observations by recency.

    confidence = Σ w_i · observation_indicator_i
    where w_i follows an exponential decay: w_i = exp(-λ · (t_now - t_i))

    This gives higher weight to recent observations, consistent with
    human memory recency effects.

    Parameters
    ----------
    max_window : int
        Maximum number of past observation slots to track.
    lambda_decay : float
        Exponential decay coefficient (larger = faster forgetting).
    """

    def __init__(
        self,
        memory_dim: int = 128,
        max_window: int = 50,
        lambda_decay: float = 0.05,
        forget_threshold: float = 0.3,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.max_window = max_window
        self.lambda_decay = lambda_decay
        self.forget_threshold = forget_threshold

        # Observation history: circular buffer
        self.register_buffer("tick", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "obs_history",
            torch.zeros(max_window, memory_dim),
        )
        self.register_buffer(
            "obs_mask",
            torch.zeros(max_window, dtype=torch.bool),
        )
        self.register_buffer(
            "timestamps",
            torch.zeros(max_window, dtype=torch.long),
        )

    def observe(self, memory_vector: torch.Tensor) -> None:
        """Record a new observation."""
        idx = self.tick % self.max_window
        self.obs_history[idx] = memory_vector
        self.obs_mask[idx] = True
        self.timestamps[idx] = self.tick
        self.tick += 1

    def get_weighted_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute recency-weighted memory content and confidence.

        Returns
        -------
        memory : Tensor (memory_dim,)
            Weighted average of past observations.
        confidence : Tensor (1,)
            Sum of weights = effective confidence.
        """
        if self.tick == 0 or not self.obs_mask.any():
            return torch.zeros(self.memory_dim), torch.tensor(0.0)

        # Time since each observation
        ages = self.tick - self.timestamps
        ages = ages.clamp(min=0)

        # Exponential decay weights
        weights = torch.exp(-self.lambda_decay * ages.float())
        weights = weights * self.obs_mask.float()

        total_weight = weights.sum()
        if total_weight < 1e-6:
            return torch.zeros(self.memory_dim), torch.tensor(0.0)

        weights_norm = weights / total_weight
        memory = (weights_norm.unsqueeze(-1) * self.obs_history).sum(dim=0)

        # Confidence = total weight (capped at 1)
        confidence = torch.clamp(total_weight / self.max_window, 0.0, 1.0)

        # Forget if below threshold
        if confidence < self.forget_threshold:
            memory = torch.zeros_like(memory)

        return memory, confidence

    def reset(self) -> None:
        self.tick.zero_()
        self.obs_history.zero_()
        self.obs_mask.zero_()
        self.timestamps.zero_()


# ======================================================================
# Per-slot decay controller (used by main model)
# ======================================================================

class DecayController(nn.Module):
    """
    Manages per-memory-slot decay for behavioral, environmental, and
    interactive memory.

    Integrates with the perception state change detector — when a
    change is detected, relevant memories can be explicitly cleared
    or re-weighted.

    Parameters
    ----------
    memory_names : list of str
        e.g. ["behavioral", "environmental", "interactive"]
    memory_dim : int
    decay_rate, learn_rate, forget_threshold : float
    """

    def __init__(
        self,
        memory_names=("behavioral", "environmental", "interactive"),
        memory_dim: int = 128,
        learn_rate: float = 0.1,
        decay_rate: float = 0.01,
        forget_threshold: float = 0.3,
    ):
        super().__init__()
        self.memory_names = list(memory_names)
        self.num_slots = len(memory_names)

        self.decay = MemoryDecay(
            memory_dim=memory_dim,
            num_slots=self.num_slots,
            learn_rate=learn_rate,
            decay_rate=decay_rate,
            forget_threshold=forget_threshold,
        )

        # Name → slot index
        self.name_to_slot = {name: i for i, name in enumerate(memory_names)}

    def update(
        self,
        behavioral: torch.Tensor,
        environmental: torch.Tensor,
        interactive: torch.Tensor,
        observed: Optional[Dict[str, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update all memory slots and return current memory state.

        Parameters
        ----------
        behavioral, environmental, interactive : Tensor (D,)
        observed : dict, optional
            Which memories were observed. Default = all observed.

        Returns
        -------
        Tuple of (behavioral, environmental, interactive) after decay.
        """
        # Observe
        self.decay.observe(self.name_to_slot["behavioral"], behavioral)
        self.decay.observe(self.name_to_slot["environmental"], environmental)
        self.decay.observe(self.name_to_slot["interactive"], interactive)

        # Decay
        if observed is None:
            observed_mask = torch.ones(self.num_slots, dtype=torch.bool)
        else:
            observed_mask = torch.tensor([
                observed.get(name, True) for name in self.memory_names
            ], dtype=torch.bool)

        self.decay.decay_all(observed_mask)

        # Read back
        b, _ = self.decay.read(self.name_to_slot["behavioral"])
        e, _ = self.decay.read(self.name_to_slot["environmental"])
        i, _ = self.decay.read(self.name_to_slot["interactive"])

        return b, e, i

    def clear_slot(self, name: str) -> None:
        """Explicitly clear one memory slot (triggered by change detection)."""
        slot = self.name_to_slot.get(name)
        if slot is not None:
            self.decay.memory_bank[slot].zero_()
            self.decay.confidence[slot] = 0.0

    def reset(self) -> None:
        self.decay.reset()

    def get_confidences(self) -> Dict[str, float]:
        return {
            name: float(self.decay.confidence[self.name_to_slot[name]])
            for name in self.memory_names
        }
