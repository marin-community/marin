# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable reference for the batch-invariant weighted RMSNorm prototype."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

SUPPORTED_HIDDEN_DIMS = frozenset((128, 256, 2560))
MAX_FLATTENED_ROWS = 512
_HIDDEN_TILE_SIZE = 128
_ROW_TILE_SIZE = MAX_FLATTENED_ROWS


def _sum_hidden_tiles(x32: jax.Array) -> jax.Array:
    """Sum squares in fixed-width hidden tiles without changing the row layout."""
    tile_sums = jnp.zeros(x32.shape[:-1], dtype=jnp.float32)
    for start in range(0, x32.shape[-1], _HIDDEN_TILE_SIZE):
        tile = x32[..., start : start + _HIDDEN_TILE_SIZE]
        tile_sums = tile_sums + jnp.sum(jnp.square(tile), axis=-1)
    return tile_sums


def batch_invariant_rms_norm_reference(
    x: Float[Array, "... D"],
    weight: Float[Array, "D"],
    *,
    eps: float,
) -> jax.Array:
    """Apply weighted RMSNorm with a fixed per-row float32 arithmetic schedule."""
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got shape {x.shape}.")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight must have shape ({x.shape[-1]},), got {weight.shape}.")
    if x.shape[-1] not in SUPPORTED_HIDDEN_DIMS:
        raise ValueError(f"supported hidden dimensions are {sorted(SUPPORTED_HIDDEN_DIMS)}, got {x.shape[-1]}.")
    rows = math.prod(x.shape[:-1]) if x.ndim > 1 else 1
    if not 1 <= rows <= MAX_FLATTENED_ROWS:
        raise ValueError(f"batch-invariant rms_norm supports 1..{MAX_FLATTENED_ROWS} flattened rows, got {rows}.")
    input_dtype = x.dtype
    flat_x = x.reshape((rows, x.shape[-1])).astype(jnp.float32)
    padded_x = jnp.pad(flat_x, ((0, _ROW_TILE_SIZE - rows), (0, 0)))
    weight32 = weight.astype(jnp.float32)
    tile_sums = _sum_hidden_tiles(padded_x)
    mean_square = tile_sums[..., None] / padded_x.shape[-1]
    normalized = padded_x * jax.lax.rsqrt(mean_square + eps)
    return (normalized * weight32).astype(input_dtype)[:rows].reshape(x.shape)
