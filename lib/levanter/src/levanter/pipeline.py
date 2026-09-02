# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Small, backend-independent building blocks for pipeline parallelism."""

from typing import TypeVar

import jax
from jax import core


BatchT = TypeVar("BatchT")


def evenly_partition_layers(num_layers: int, num_stages: int) -> tuple[tuple[int, int], ...]:
    """Return contiguous, nearly even ``[start, end)`` layer ranges."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")
    if num_stages > num_layers:
        raise ValueError(f"num_stages ({num_stages}) cannot exceed num_layers ({num_layers})")

    layers_per_stage, stages_with_extra_layer = divmod(num_layers, num_stages)
    ranges = []
    start = 0
    for stage in range(num_stages):
        stage_layers = layers_per_stage + (stage < stages_with_extra_layer)
        end = start + stage_layers
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def split_batch_into_microbatches(batch: BatchT, num_microbatches: int) -> tuple[BatchT, ...]:
    """Split every non-scalar array leaf along its leading batch dimension."""
    if num_microbatches <= 0:
        raise ValueError(f"num_microbatches must be positive, got {num_microbatches}")

    def split_leaf(value):
        if not isinstance(value, jax.Array | core.Tracer) or value.ndim == 0:
            return tuple(value for _ in range(num_microbatches))
        if value.shape[0] % num_microbatches != 0:
            raise ValueError(
                f"batch axis size {value.shape[0]} must be divisible by num_microbatches={num_microbatches}"
            )
        microbatch_size = value.shape[0] // num_microbatches
        return tuple(
            jax.lax.slice_in_dim(value, index * microbatch_size, (index + 1) * microbatch_size, axis=0)
            for index in range(num_microbatches)
        )

    split_leaves = jax.tree.map(split_leaf, batch)
    is_split_leaf = lambda value: isinstance(value, tuple) and len(value) == num_microbatches
    return tuple(
        jax.tree.map(lambda leaves: leaves[index], split_leaves, is_leaf=is_split_leaf)
        for index in range(num_microbatches)
    )
