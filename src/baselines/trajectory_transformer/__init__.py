"""Trajectory-Transformer module.

Copied from https://github.com/FGiuliari/Trajectory-Transformer
(An OpenTrons implementation of "Attention is All You Need" tailored
for trajectory prediction.)

Core components:
    IndividualTF     — the full Transformer model
    subsequent_mask  — causal mask for autoregressive decoding
"""

from .functional import subsequent_mask

