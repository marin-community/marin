"""Compat wrapper for the block diffusion objective.

Preferred import path:
  `experiments.grug.block_diffusion.objective`
"""

from __future__ import annotations

from experiments.grug.block_diffusion.objective import (
    BlockDiffusionObjectiveConfig,
    BlockDiffusionTransformer,
    corrupt_for_training,
    make_block_causal_mask,
)

__all__ = [
    "BlockDiffusionObjectiveConfig",
    "BlockDiffusionTransformer",
    "corrupt_for_training",
    "make_block_causal_mask",
]
