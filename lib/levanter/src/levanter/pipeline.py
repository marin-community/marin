# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Small, backend-independent building blocks for pipeline parallelism."""

from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import core

BatchT = TypeVar("BatchT")
type ArrayValue = jax.Array | core.Tracer


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


def reshape_array_into_microbatches(value: ArrayValue, num_microbatches: int) -> ArrayValue:
    """Split an array's leading batch axis into microbatch and example axes."""
    if num_microbatches <= 0:
        raise ValueError(f"num_microbatches must be positive, got {num_microbatches}")
    if value.ndim == 0:
        return value
    if value.shape[0] % num_microbatches != 0:
        raise ValueError(f"batch axis size {value.shape[0]} must be divisible by num_microbatches={num_microbatches}")
    microbatch_size = value.shape[0] // num_microbatches
    return jnp.reshape(value, (num_microbatches, microbatch_size, *value.shape[1:]))


def reshape_batch_into_microbatches(batch: BatchT, num_microbatches: int) -> BatchT:
    """Reshape every non-scalar array leaf to lead with a microbatch axis."""

    def reshape_leaf(value):
        if not isinstance(value, jax.Array | core.Tracer):
            return value
        return reshape_array_into_microbatches(value, num_microbatches)

    return jax.tree.map(reshape_leaf, batch)


def split_batch_into_microbatches(batch: BatchT, num_microbatches: int) -> tuple[BatchT, ...]:
    """Split every non-scalar array leaf along its leading batch dimension."""
    reshaped = reshape_batch_into_microbatches(batch, num_microbatches)

    def select_microbatch(value, index: int):
        if not isinstance(value, jax.Array | core.Tracer) or value.ndim == 0:
            return value
        return value[index]

    return tuple(
        jax.tree.map(lambda value: select_microbatch(value, index), reshaped) for index in range(num_microbatches)
    )
