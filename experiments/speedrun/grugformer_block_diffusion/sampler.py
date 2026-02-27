"""Compat wrapper for the block diffusion sampler.

Preferred import path:
  `experiments.grug.block_diffusion.sampler`
"""

from __future__ import annotations

from experiments.grug.block_diffusion.sampler import (
    SamplingConfig,
    sample_block_diffusion,
)

__all__ = [
    "SamplingConfig",
    "sample_block_diffusion",
]
