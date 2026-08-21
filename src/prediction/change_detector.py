"""
Perception state change detection.

Detects when the traffic perception state has changed significantly,
triggering memory re-initialisation in the FlowChain module.

Two complementary detection mechanisms:
    1. Structural change detection  — rule-based, explicit events
    2. Concept drift detection       — data-driven, monitors GRU gate statistics

References:
    Paper Section 7: 感知状态变化检测
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ======================================================================
# Change types
# ======================================================================

class ChangeType(Enum):
    NONE = "none"
    NODE_APPEAR = "node_appear"           # new traffic participant
    NODE_DISAPPEAR = "node_disappear"     # participant left scene
    TRAFFIC_LIGHT_CHANGE = "traffic_light_change"  # red↔green↔yellow
    LANE_CHANGE = "lane_change"           # target changed lane
    AGENT_COUNT_CHANGE = "agent_count_change"     # significant count change
    DRIFT_DETECTED = "drift_detected"     # concept drift in gate activations


@dataclass
class ChangeEvent:
    """Container for a detected perception state change."""
    change_type: ChangeType
    timestamp: int         # frame index
    detail: str            # human-readable description
    affected_memory: str   # which memory slot to clear: "behavioral" / "environmental" / "interactive" / "all"
    severity: float        # 0–1, how significant the change is


# ======================================================================
# 1. Structural Change Detection (rule-driven)
# ======================================================================

class StructuralChangeDetector:
    """
    Rule-based detection of explicit structural changes in the scene.

    Detects:
        - Nodes appearing / disappearing (new vehicles, pedestrians entering)
        - Traffic light state transitions
        - Lane changes by target pedestrian

    Parameters
    ----------
    iou_threshold : float
        IoU threshold for matching nodes across frames.
    traffic_light_change_threshold : int
        Minimum consecutive frames of changed light state to confirm.
    agent_count_change_threshold : int
        Absolute change in agent count to trigger.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        traffic_light_change_threshold: int = 3,
        agent_count_change_threshold: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.tl_change_threshold = traffic_light_change_threshold
        self.agent_count_threshold = agent_count_change_threshold

        # State tracking
        self.prev_node_ids: Set[int] = set()
        self.prev_traffic_light_state: Optional[int] = None  # 0=red,1=yellow,2=green
        self.tl_state_counter: int = 0
        self.prev_agent_count: int = 0

    def detect(
        self,
        frame_id: int,
        current_node_ids: Set[int],
        traffic_light_state: Optional[int] = None,
        agent_count: int = 0,
        target_lane_id: Optional[int] = None,
        prev_target_lane_id: Optional[int] = None,
    ) -> List[ChangeEvent]:
        """
        Detect structural changes between consecutive frames.

        Returns
        -------
        list of ChangeEvent
        """
        events: List[ChangeEvent] = []

        # --- Node appear / disappear ---
        appeared = current_node_ids - self.prev_node_ids
        disappeared = self.prev_node_ids - current_node_ids

        if len(appeared) > 0:
            events.append(ChangeEvent(
                change_type=ChangeType.NODE_APPEAR,
                timestamp=frame_id,
                detail=f"Nodes appeared: {appeared}",
                affected_memory="interactive",
                severity=min(1.0, len(appeared) / 5.0),
            ))

        if len(disappeared) > 0:
            events.append(ChangeEvent(
                change_type=ChangeType.NODE_DISAPPEAR,
                timestamp=frame_id,
                detail=f"Nodes disappeared: {disappeared}",
                affected_memory="interactive",
                severity=min(1.0, len(disappeared) / 5.0),
            ))

        # --- Traffic light change ---
        if traffic_light_state is not None and self.prev_traffic_light_state is not None:
            if traffic_light_state != self.prev_traffic_light_state:
                self.tl_state_counter += 1
                if self.tl_state_counter >= self.tl_change_threshold:
                    state_names = {0: "red", 1: "yellow", 2: "green"}
                    events.append(ChangeEvent(
                        change_type=ChangeType.TRAFFIC_LIGHT_CHANGE,
                        timestamp=frame_id,
                        detail=(
                            f"Traffic light: {state_names.get(self.prev_traffic_light_state, '?')}"
                            f" → {state_names.get(traffic_light_state, '?')}"
                        ),
                        affected_memory="environmental",
                        severity=0.8,
                    ))
                    self.tl_state_counter = 0
            else:
                self.tl_state_counter = 0

        # --- Agent count change ---
        if abs(agent_count - self.prev_agent_count) >= self.agent_count_threshold:
            events.append(ChangeEvent(
                change_type=ChangeType.AGENT_COUNT_CHANGE,
                timestamp=frame_id,
                detail=f"Agent count changed: {self.prev_agent_count} → {agent_count}",
                affected_memory="interactive",
                severity=min(1.0, abs(agent_count - self.prev_agent_count) / 10.0),
            ))

        # --- Lane change ---
        if (target_lane_id is not None and prev_target_lane_id is not None
                and target_lane_id != prev_target_lane_id):
            events.append(ChangeEvent(
                change_type=ChangeType.LANE_CHANGE,
                timestamp=frame_id,
                detail=f"Lane change: {prev_target_lane_id} → {target_lane_id}",
                affected_memory="environmental",
                severity=0.6,
            ))

        # --- Update state ---
        self.prev_node_ids = current_node_ids.copy()
        self.prev_traffic_light_state = traffic_light_state
        self.prev_agent_count = agent_count

        return events


# ======================================================================
# 2. Concept Drift Detection (data-driven)
# ======================================================================

class ConceptDriftDetector(nn.Module):
    """
    Detects concept drift by monitoring GRU gate activation statistics
    over a sliding window.

    When the distribution of gate activations shifts significantly
    (measured via KL divergence or Wasserstein distance between
    current and reference window), a drift event is triggered.

    Per paper Section 7.2:
        "在改进GRU的门控内部封装检测，当前后帧变化到一定程度时..."
        (Embed detection inside the GRU gates; trigger when
         frame-to-frame change exceeds threshold.)

    Parameters
    ----------
    window_size : int
        Sliding window size for computing statistics.
    drift_threshold : float
        KL-divergence / mean-change threshold for triggering drift.
    gate_names : list of str
        Which gates to monitor.
    """

    def __init__(
        self,
        window_size: int = 30,
        drift_threshold: float = 0.15,
        gate_names=("reset_gate", "update_gate"),
    ):
        super().__init__()
        self._window_size = window_size
        self.drift_threshold = drift_threshold
        self.gate_names = list(gate_names)

        # Sliding windows per gate (use list instead of deque for .to() compatibility)
        self._gate_buf: Dict[str, list] = {
            name: [] for name in gate_names
        }

        # Reference statistics (frozen after first window is filled)
        self._ref_mean: Optional[Dict[str, torch.Tensor]] = None
        self._ref_std: Optional[Dict[str, torch.Tensor]] = None
        self._ref_ready: bool = False

    def observe(self, gate_info: Dict[str, torch.Tensor]) -> None:
        """Feed one step of gate activations into the drift detector."""
        for name in self.gate_names:
            if name in gate_info:
                val = gate_info[name].detach().flatten().mean().unsqueeze(0)
                self._gate_buf[name].append(val)
                # Keep only window_size most recent entries
                if len(self._gate_buf[name]) > self._window_size:
                    self._gate_buf[name] = self._gate_buf[name][-self._window_size:]

        # Freeze reference after first full window
        if not self._ref_ready and all(
            len(buf) == self._window_size for buf in self._gate_buf.values()
        ):
            self._freeze_reference()

    def check_drift(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if current gate statistics have drifted from reference.

        Returns
        -------
        drift_detected : bool
        drift_scores : dict
            Per-gate drift scores.
        """
        if not self._ref_ready:
            return False, {}

        scores = {}
        for name in self.gate_names:
            buf = self._gate_buf[name]
            if len(buf) < self._window_size:
                continue

            current_vals = torch.stack(buf)  # (W, 1)
            current_mean = current_vals.mean()
            ref_mean = self._ref_mean[name]

            # Compute relative mean shift
            max_val = max(abs(ref_mean), abs(current_mean), 1e-6)
            score = abs(current_mean - ref_mean) / max_val
            scores[name] = float(score)

        if not scores:
            return False, scores

        max_score = max(scores.values())
        drift = max_score > self.drift_threshold

        return drift, scores

    def _freeze_reference(self):
        """Store current window statistics as reference."""
        self._ref_mean = {}
        self._ref_std = {}
        for name in self.gate_names:
            vals = torch.stack(self._gate_buf[name])
            self._ref_mean[name] = vals.mean()
            self._ref_std[name] = vals.std()
        self._ref_ready = True

    def reset_reference(self):
        """Reset reference (called after a drift event triggers re-init)."""
        self._ref_mean = None
        self._ref_std = None
        self._ref_ready = False
        for name in self.gate_names:
            self._gate_buf[name].clear()

    def reset(self):
        """Fully reset all buffers and references."""
        self.reset_reference()


# ======================================================================
# 3. Cognitive State Change Detector (NEW)
# ======================================================================

class CognitiveStateChangeDetector(nn.Module):
    """
    Detects changes in the fused cognitive state vector c over time.

    Two distance metrics:
        Euclidean:  D_euc = ||c_t - c_{t-1}||
        Cosine:     D_cos = 1 - cos(c_t, c_{t-1})

    When the distance exceeds a threshold, the traffic cognitive state
    is considered to have changed significantly.

    Parameters
    ----------
    threshold : float
        Distance threshold for triggering a change event.
    metric : str
        "euclidean" or "cosine".
    """

    def __init__(
        self,
        threshold: float = 0.3,
        metric: str = "cosine",
    ):
        super().__init__()
        self.threshold = threshold
        self.metric = metric

        # State tracking
        self._prev_c: Optional[torch.Tensor] = None

    def detect(
        self,
        c_current: Tensor,       # (..., D)  current cognitive state
        frame_id: int = 0,
    ) -> Tuple[bool, float, Optional[ChangeEvent]]:
        """
        Detect whether cognitive state c has changed significantly.

        Returns
        -------
        changed : bool
        distance : float
        event : ChangeEvent or None
        """
        if self._prev_c is None:
            self._prev_c = c_current.detach().clone()
            return False, 0.0, None

        c_prev = self._prev_c
        c_curr = c_current

        # Flatten for distance computation
        c_prev_flat = c_prev.reshape(-1)
        c_curr_flat = c_curr.reshape(-1)

        if self.metric == "euclidean":
            distance = torch.norm(c_curr_flat - c_prev_flat).item()
        elif self.metric == "cosine":
            cos_sim = F.cosine_similarity(
                c_curr_flat.unsqueeze(0), c_prev_flat.unsqueeze(0)
            ).item()
            distance = 1.0 - cos_sim
        else:
            distance = 0.0

        changed = distance > self.threshold

        event = None
        if changed:
            event = ChangeEvent(
                change_type=ChangeType.DRIFT_DETECTED,
                timestamp=frame_id,
                detail=f"Cognitive state changed: {self.metric} distance={distance:.4f} > {self.threshold}",
                affected_memory="all",
                severity=min(1.0, distance),
            )

        # Update state
        self._prev_c = c_current.detach().clone()

        return changed, distance, event

    def reset(self):
        self._prev_c = None


# ======================================================================
# 4. Cognitive Conflict Detector (NEW)
# ======================================================================

class CognitiveConflictDetector(nn.Module):
    """
    Detects conflicts between the three perception memories.

    When two memories diverge significantly (e.g., behavioral intent
    conflicts with environmental constraints), a conflict is flagged.

    Computes pairwise distances:
        D_be = 1 - cos(M_b, M_e)    Behavioral ↔ Environmental
        D_br = 1 - cos(M_b, M_r)    Behavioral ↔ Interactive
        D_er = 1 - cos(M_e, M_r)    Environmental ↔ Interactive

    When any pairwise distance exceeds the threshold, a conflict is
    detected, triggering memory recomputation.

    Parameters
    ----------
    threshold : float
        Cosine distance threshold for conflict detection.
    metric : str
        "cosine" or "euclidean".
    """

    def __init__(
        self,
        threshold: float = 0.5,
        metric: str = "cosine",
    ):
        super().__init__()
        self.threshold = threshold
        self.metric = metric

    def detect(
        self,
        behavioral: Tensor,      # (..., D)
        environmental: Tensor,   # (..., D)
        interactive: Tensor,     # (..., D)
        frame_id: int = 0,
    ) -> Tuple[bool, Dict[str, float], Optional[ChangeEvent]]:
        """
        Detect cognitive conflicts between memory pairs.

        Returns
        -------
        conflict : bool
            True if any memory pair exceeds threshold.
        distances : dict
            {"be": D_be, "br": D_br, "er": D_er}
        event : ChangeEvent or None
        """
        b_flat = behavioral.reshape(-1)
        e_flat = environmental.reshape(-1)
        i_flat = interactive.reshape(-1)

        distances = {}

        if self.metric == "cosine":
            d_be = 1.0 - F.cosine_similarity(b_flat.unsqueeze(0), e_flat.unsqueeze(0)).item()
            d_br = 1.0 - F.cosine_similarity(b_flat.unsqueeze(0), i_flat.unsqueeze(0)).item()
            d_er = 1.0 - F.cosine_similarity(e_flat.unsqueeze(0), i_flat.unsqueeze(0)).item()
        elif self.metric == "euclidean":
            d_be = torch.norm(b_flat - e_flat).item()
            d_br = torch.norm(b_flat - i_flat).item()
            d_er = torch.norm(e_flat - i_flat).item()
        else:
            d_be = d_br = d_er = 0.0

        distances = {"be": d_be, "br": d_br, "er": d_er}

        max_dist = max(distances.values())
        conflict = max_dist > self.threshold

        event = None
        if conflict:
            # Identify which pair triggered the conflict
            triggered = [k for k, v in distances.items() if v > self.threshold]
            pair_names = {
                "be": "Behavioral↔Environmental",
                "br": "Behavioral↔Interactive",
                "er": "Environmental↔Interactive",
            }
            detail_parts = [f"{pair_names[p]}: {distances[p]:.4f}" for p in triggered]
            detail = "Cognitive conflict: " + ", ".join(detail_parts)

            event = ChangeEvent(
                change_type=ChangeType.DRIFT_DETECTED,
                timestamp=frame_id,
                detail=detail,
                affected_memory="all",
                severity=min(1.0, max_dist),
            )

        return conflict, distances, event

    def reset(self):
        pass  # Stateless detector


# ======================================================================
# 5. Unified Change Detection Module (UPDATED)
# ======================================================================

class PerceptionChangeDetector(nn.Module):
    """
    Unified perception change detector combining four detection mechanisms:

    1. Structural change detection  — graph structure (nodes, traffic light,
                                       agent count, lane changes)
    2. Cognitive state change         — Drift in fused cognitive context c
    3. Cognitive conflict detection   — Divergence between memory pairs
    4. Concept drift detection        — (legacy) GRU gate statistics

    Detection flow:
        graph change → c state change → memory conflict
        → update flag → recompute memories → attention fusion → update c

    Parameters
    ----------
    struct_config : dict
        Config for StructuralChangeDetector.
    cognitive_config : dict
        Config for CognitiveStateChangeDetector.
    conflict_config : dict
        Config for CognitiveConflictDetector.
    drift_config : dict
        Config for ConceptDriftDetector (legacy).
    """

    def __init__(
        self,
        struct_config: Optional[dict] = None,
        cognitive_config: Optional[dict] = None,
        conflict_config: Optional[dict] = None,
        drift_config: Optional[dict] = None,
    ):
        super().__init__()

        # 1. Structural detector (stateless, not a nn.Module)
        self.structural = StructuralChangeDetector(
            **(struct_config or {})
        )

        # 2. Cognitive state change detector (stateful, nn.Module)
        self.cognitive = CognitiveStateChangeDetector(
            **(cognitive_config or {})
        )

        # 3. Cognitive conflict detector (stateless, nn.Module)
        self.conflict = CognitiveConflictDetector(
            **(conflict_config or {})
        )

        # 4. Legacy drift detector (stateful, nn.Module — kept for compat)
        self.drift = ConceptDriftDetector(
            **(drift_config or {})
        )

    # ------------------------------------------------------------------
    # Structural detection (unchanged interface)
    # ------------------------------------------------------------------

    def detect_structural(
        self,
        frame_id: int,
        current_node_ids: Set[int],
        traffic_light_state: Optional[int] = None,
        agent_count: int = 0,
        target_lane_id: Optional[int] = None,
        prev_target_lane_id: Optional[int] = None,
    ) -> List[ChangeEvent]:
        """Run structural (graph) change detection."""
        return self.structural.detect(
            frame_id=frame_id,
            current_node_ids=current_node_ids,
            traffic_light_state=traffic_light_state,
            agent_count=agent_count,
            target_lane_id=target_lane_id,
            prev_target_lane_id=prev_target_lane_id,
        )

    # ------------------------------------------------------------------
    # Cognitive state change detection (NEW)
    # ------------------------------------------------------------------

    def detect_cognitive_change(
        self,
        c_current: Tensor,
        frame_id: int = 0,
    ) -> Tuple[bool, float, Optional[ChangeEvent]]:
        """
        Detect whether the fused cognitive state c has changed significantly.

        Parameters
        ----------
        c_current : Tensor (..., D)
            Current fused cognitive context from MemoryAttentionFusion.

        Returns
        -------
        changed : bool
        distance : float
        event : ChangeEvent or None
        """
        return self.cognitive.detect(c_current, frame_id)

    # ------------------------------------------------------------------
    # Cognitive conflict detection (NEW)
    # ------------------------------------------------------------------

    def detect_conflict(
        self,
        behavioral: Tensor,
        environmental: Tensor,
        interactive: Tensor,
        frame_id: int = 0,
    ) -> Tuple[bool, Dict[str, float], Optional[ChangeEvent]]:
        """
        Detect conflicts between the three memory vectors.

        Returns
        -------
        conflict : bool
        distances : dict
        event : ChangeEvent or None
        """
        return self.conflict.detect(
            behavioral, environmental, interactive, frame_id,
        )

    # ------------------------------------------------------------------
    # Comprehensive detection: run all three detectors (NEW)
    # ------------------------------------------------------------------

    def detect_all(
        self,
        frame_id: int,
        current_node_ids: Set[int],
        c_current: Tensor,
        behavioral: Tensor,
        environmental: Tensor,
        interactive: Tensor,
        traffic_light_state: Optional[int] = None,
        agent_count: int = 0,
    ) -> Dict[str, List[ChangeEvent]]:
        """
        Run all detection mechanisms and return all events.

        Order: structural → cognitive state → cognitive conflict.

        Returns
        -------
        events_by_type : dict
            {"structural": [...], "cognitive": [...], "conflict": [...]}
        """
        results = {"structural": [], "cognitive": [], "conflict": []}

        # 1. Structural
        structural_events = self.detect_structural(
            frame_id=frame_id,
            current_node_ids=current_node_ids,
            traffic_light_state=traffic_light_state,
            agent_count=agent_count,
        )
        results["structural"] = structural_events

        # 2. Cognitive state change
        changed, _, cog_event = self.detect_cognitive_change(c_current, frame_id)
        if cog_event is not None:
            results["cognitive"].append(cog_event)

        # 3. Cognitive conflict
        conflict, _, conf_event = self.detect_conflict(
            behavioral, environmental, interactive, frame_id,
        )
        if conf_event is not None:
            results["conflict"].append(conf_event)

        return results

    def has_any_change(self, events_by_type: Dict[str, list]) -> bool:
        """Check if any detector found a change."""
        return any(len(v) > 0 for v in events_by_type.values())

    # ------------------------------------------------------------------
    # Legacy drift detection (kept for backward compat)
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        gate_info: Dict[str, torch.Tensor],
    ) -> Tuple[bool, Dict[str, float]]:
        """Run concept drift detection (legacy)."""
        self.drift.observe(gate_info)
        return self.drift.check_drift()

    def should_reinitialize(self) -> bool:
        """Check if any type of change requires memory re-initialisation."""
        drift, _ = self.drift.check_drift()
        cog_changed = self.cognitive._prev_c is not None
        return drift or cog_changed

    def reset(self):
        self.drift.reset()
        self.cognitive.reset()
        self.conflict.reset()
        self.structural = StructuralChangeDetector()
