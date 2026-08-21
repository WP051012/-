"""
Traffic Perception Memory Module
---------------------------------
Models traffic perception memory formation, fusion, and decay.

Classes:
    BehavioralMemory          — target pedestrian's own behavioral intent
    EnvironmentalMemory       — infrastructure constraints via attention
    InteractiveMemory         — surrounding agent influences
    MemoryFusion              — fuse 3 memories → perception vector c (legacy)
    TrafficPerceptionMemory   — complete perception memory module
    MemoryAttentionFusion     — NEW: attention-based memory fusion
    MemoryAttentionFusionV2   — NEW: cross-memory attention fusion
    MemoryDecay               — confidence-based memory decay (MAGELLAN-style)
    TimeWeightedDecay         — recency-weighted decay
    DecayController           — per-slot decay manager
"""

from .perception_memory import (
    BehavioralMemory,
    EnvironmentalMemory,
    InteractiveMemory,
    MemoryFusion,
    TrafficPerceptionMemory,
)
from .memory_attention_fusion import (
    MemoryAttentionFusion,
    MemoryAttentionFusionV2,
)
from .decay import (
    MemoryDecay,
    TimeWeightedDecay,
    DecayController,
)
