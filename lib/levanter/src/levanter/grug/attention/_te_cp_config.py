# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the opt-in Transformer Engine context-parallel attention backend.

Kept separate from ``_te_cp`` so ``_core.attention`` can name the config in its signature
without importing the Transformer Engine shim (``_te_cp`` imports ``AttentionMask`` from
``_core``).
"""

from dataclasses import dataclass
from enum import StrEnum

DEFAULT_CONTEXT_AXIS = "context"
DEFAULT_DATA_AXIS = "data"


class ContextParallelStrategy(StrEnum):
    """Transformer Engine's two context-parallel communication protocols.

    ``RING`` passes K/V around the context ring one block at a time; ``ALL_GATHER``
    materializes the full K/V on every context rank before the local kernel runs.
    """

    RING = "ring"
    ALL_GATHER = "all_gather"


@dataclass(frozen=True)
class TeContextParallelConfig:
    """Knobs for the ``gpu_te_cp`` attention backend.

    ``stripe_size`` is the striped causal load-balancing granularity: the sequence is cut into
    stripes of this many tokens, and stripe ``i`` goes to context rank ``i % cp_size`` so every
    rank holds a mix of early and late tokens. Transformer Engine's ring protocol implements
    only ``stripe_size=1``.

    ``max_segments_per_seq`` is part of the fused-attention kernel signature and therefore
    static: it must bound the number of packed documents in any sequence.
    """

    strategy: ContextParallelStrategy = ContextParallelStrategy.RING
    stripe_size: int = 1
    max_segments_per_seq: int = 1
    context_axis: str = DEFAULT_CONTEXT_AXIS
    data_axis: str = DEFAULT_DATA_AXIS

    def __post_init__(self) -> None:
        if self.stripe_size <= 0:
            raise ValueError(f"stripe_size must be positive, got {self.stripe_size}")
        if self.max_segments_per_seq <= 0:
            raise ValueError(f"max_segments_per_seq must be positive, got {self.max_segments_per_seq}")
        if self.strategy == ContextParallelStrategy.RING and self.stripe_size != 1:
            raise ValueError(
                "Transformer Engine ring context-parallel attention supports only stripe_size=1, got "
                f"{self.stripe_size}; use strategy=all_gather for wider stripes."
            )
