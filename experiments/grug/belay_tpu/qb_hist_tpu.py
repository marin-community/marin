# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact integer QB histograms using TPU matrix products.

Adapted from belay-base16-tpu-v4 aef63f29d. Inputs are device-local inside
the caller's shard_map. Coarse counts locate one 64-bin block per expert;
fine counts retain the original bins and interpolation, with smaller psums.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


def factorized_histogram(indices: Int[Array, "T E"], bins: int, group_size: int, radix: int) -> jax.Array:
    """Count bins exactly; negative indices contribute zero."""
    tokens, experts = indices.shape
    if experts % group_size:
        raise ValueError("Expert count must be divisible by the histogram group size")
    groups = indices.T.reshape(experts // group_size, group_size, tokens)
    high_bins = (bins + radix - 1) // radix

    def count(group):
        high = jax.nn.one_hot(group // radix, high_bins, dtype=jnp.int8)
        low = jax.nn.one_hot(group % radix, radix, dtype=jnp.int8)
        counts = jax.lax.dot_general(high, low, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        return counts.reshape(group_size, high_bins * radix)[:, :bins]

    return jax.lax.map(count, groups).reshape(experts, bins)


def _crossing(counts, rank, last_valid):
    cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
    crossing = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    return crossing, ct, height


def bincount_upper_quantile_tpu(
    s_local,
    *,
    num_experts: int,
    n_bins: int,
    lo,
    hi,
    target_rank,
    batch_axes: tuple[str, ...],
    group_size: int = 64,
):
    """Preserve the source quantile using exact coarse/fine integer counts."""
    if s_local.ndim != 2 or s_local.shape[1] != num_experts or n_bins <= 0:
        raise ValueError("QB histogram requires [tokens, experts] margins and positive bin count")
    bin_width = (hi - lo) / n_bins
    indices = jnp.clip(((s_local - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1)
    rank = jnp.broadcast_to(jnp.asarray(target_rank, jnp.float32), (num_experts,))
    block_bins = 64
    coarse_bins = (n_bins + block_bins - 1) // block_bins
    coarse_indices = indices // block_bins
    coarse = jax.lax.psum(factorized_histogram(coarse_indices, coarse_bins, group_size, 16), batch_axes).astype(
        jnp.float32
    )
    block, coarse_ct, coarse_height = _crossing(coarse, rank, coarse_bins - 1)
    selected = jnp.where(coarse_indices == block[None, :], indices % block_bins, -1)
    fine = jax.lax.psum(factorized_histogram(selected, block_bins, group_size, 8), batch_axes).astype(jnp.float32)
    above = coarse_ct - coarse_height
    cumulative = jnp.cumsum(fine[:, ::-1], axis=-1)[:, ::-1] + above[:, None]
    last_valid = jnp.minimum(block_bins - 1, n_bins - 1 - block * block_bins)
    crossing = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(fine, crossing[:, None], axis=-1)[:, 0]
    bin_index = block * block_bins + crossing
    lower = lo + bin_index.astype(jnp.float32) * bin_width
    return lower + bin_width * (ct - rank) / jnp.maximum(height, 1.0)
