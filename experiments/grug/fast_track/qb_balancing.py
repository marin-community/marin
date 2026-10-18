# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sharded QB routing-bias estimation: each expert's threshold from a histogram of routing margins."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

# Tuned on integer histograms at Belay d512's local routing shape; other
# devices/shapes retain the flat implementation. See the issue's raw grid:
# https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6153669097
_HISTOGRAM_MATMUL_TILES = {
    ("TPU v4", 32768, 512, 10000): (64, 64),
    ("TPU v4", 8192, 512, 10000): (64, 64),
}


def factorized_expert_histogram(indices: jax.Array, n_bins: int, group_size: int, radix: int) -> jax.Array:
    """Count clipped ``[tokens, experts]`` indices with binary matrix products.

    Each expert's binary one-hot matrices contract over tokens independently.
    Accumulation is int32 and exact while the token count fits; groups limit
    live memory. TPU XLA can fold the source int8 one-hots into predicates.
    """
    tokens, experts = indices.shape
    groups = indices.T.reshape(experts // group_size, group_size, tokens)
    high_bins = (n_bins + radix - 1) // radix

    def count_group(group):
        high = jax.nn.one_hot(group // radix, high_bins, dtype=jnp.int8)
        low = jax.nn.one_hot(group % radix, radix, dtype=jnp.int8)
        counts = jax.lax.dot_general(high, low, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        return counts.reshape(group_size, high_bins * radix)[:, :n_bins]

    return jax.lax.map(count_group, groups).reshape(experts, n_bins)


def local_expert_histogram(indices: jax.Array, n_bins: int, group_size: int | None = None) -> jax.Array:
    """Count already-clipped ``[tokens, experts]`` bin indices exactly.

    Grouping bounds each scatter's index count. It preserves integer counts
    while avoiding a large TPU sort at the measured Belay shape.
    """
    tokens, experts = indices.shape

    def flat_counts(idx):
        count = idx.shape[1]
        flat = (jnp.arange(count, dtype=jnp.int32)[None, :] * n_bins + idx).reshape(-1)
        return jnp.bincount(flat, length=count * n_bins).reshape(count, n_bins)

    if group_size is None:
        return flat_counts(indices)
    if experts % group_size:
        raise ValueError("Expert count must divide into equal histogram groups")
    groups = indices.T.reshape(experts // group_size, group_size, tokens)
    counts = jax.lax.map(lambda group: flat_counts(group.T), groups)
    return counts.reshape(experts, n_bins)


def hierarchical_histogram_crossing(counts, target_rank, *, block_bins):
    """Find a clipped suffix-count crossing using one selected block per expert."""
    counts = counts.astype(jnp.float32)
    experts, bins = counts.shape
    padding = (-bins) % block_bins
    blocks = jnp.pad(counts, ((0, 0), (0, padding))).reshape(experts, -1, block_bins)
    totals = blocks.sum(axis=-1)
    cumulative = jnp.cumsum(totals[:, ::-1], axis=-1)[:, ::-1]
    block = jnp.clip(jnp.sum(cumulative >= target_rank[:, None], axis=-1) - 1, 0, blocks.shape[1] - 1)
    selected = jnp.take_along_axis(blocks, block[:, None, None], axis=1)[:, 0, :]
    above = jnp.take_along_axis(cumulative - totals, block[:, None], axis=-1)
    local_cumulative = jnp.cumsum(selected[:, ::-1], axis=-1)[:, ::-1] + above
    last_valid = jnp.minimum(block_bins - 1, bins - 1 - block * block_bins)
    local_bin = jnp.clip(jnp.sum(local_cumulative >= target_rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(local_cumulative, local_bin[:, None], axis=-1)[:, 0]
    h = jnp.take_along_axis(selected, local_bin[:, None], axis=-1)[:, 0]
    return block * block_bins + local_bin, ct, h


def histogram_crossing(counts, rank, last_valid):
    """Return the crossing bin, suffix count and bin height for each expert."""
    cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
    crossing = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    return crossing, ct, height


def two_stage_histogram_crossing(indices, rank, *, batch_axes, n_bins, group_size):
    """Count coarse bins globally, then count only each expert's crossing block.

    Integer collectives preserve exact counts. A second histogram excludes
    unselected blocks with negative indices, whose high-digit one-hot is zero.
    """
    block_bins = 64
    coarse_bins = (n_bins + block_bins - 1) // block_bins
    coarse_indices = indices // block_bins
    coarse_counts = jax.lax.psum(
        factorized_expert_histogram(coarse_indices, coarse_bins, group_size, 16), batch_axes
    ).astype(jnp.float32)
    block, coarse_ct, coarse_height = histogram_crossing(coarse_counts, rank, coarse_bins - 1)
    above = coarse_ct - coarse_height
    selected = jnp.where(coarse_indices == block[None, :], indices % block_bins, -1)
    fine_counts = jax.lax.psum(factorized_expert_histogram(selected, block_bins, group_size, 8), batch_axes).astype(
        jnp.float32
    )
    cumulative = jnp.cumsum(fine_counts[:, ::-1], axis=-1)[:, ::-1] + above[:, None]
    last_valid = jnp.minimum(block_bins - 1, n_bins - 1 - block * block_bins)
    local_bin = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, local_bin[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(fine_counts, local_bin[:, None], axis=-1)[:, 0]
    return block * block_bins + local_bin, ct, height


def histogram_quantile(counts, target_rank, lo, bin_width, *, block_bins=None):
    """Interpolate the QB threshold; block extraction is an explicit experimental option."""
    if block_bins is None:
        cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
        crossing = jnp.clip(
            jnp.sum((cumulative >= target_rank[:, None]).astype(jnp.int32), axis=-1) - 1, 0, counts.shape[-1] - 1
        )
        ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
        h = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    else:
        crossing, ct, h = hierarchical_histogram_crossing(counts, target_rank, block_bins=block_bins)
    lower_edge = lo + crossing.astype(jnp.float32) * bin_width
    return lower_edge + bin_width * (ct - target_rank) / jnp.maximum(h, 1.0)


def bincount_upper_quantile(
    s_local: jax.Array,
    *,
    num_experts: int,
    n_bins: int,
    lo: jax.Array,
    hi: jax.Array,
    target_rank: float | jax.Array,
    batch_axes: tuple[str, ...],
    histogram_method: Literal["reference", "two-stage"] = "reference",
) -> jax.Array:
    """Per-expert (1-K/E) upper quantile of ``s_local`` via a histogram over ``[lo, hi]``.

    ``target_rank`` is the number of tokens at or above each expert's threshold: a scalar, or one per expert.

    Runs inside a ``shard_map``. Clip-to-edge bins build each expert's local histogram;
    integer reductions pool counts globally. Beta is interpolated in the crossing
    bin. The experimental two-stage method counts only a selected fine block at
    the tuned TPU v4 shape; other shapes retain the reference histogram.
    """
    bin_width = (hi - lo) / n_bins
    idx = jnp.clip(((s_local - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1)
    device_kind = jax.local_devices()[0].device_kind
    tile = _HISTOGRAM_MATMUL_TILES.get((device_kind, s_local.shape[0], num_experts, n_bins))
    target_rank = jnp.broadcast_to(jnp.asarray(target_rank, jnp.float32), (num_experts,))
    if histogram_method == "two-stage" and tile == (64, 64):
        crossing, ct, height = two_stage_histogram_crossing(
            idx, target_rank, batch_axes=batch_axes, n_bins=n_bins, group_size=tile[0]
        )
        lower = lo + crossing.astype(jnp.float32) * bin_width
        return lower + bin_width * (ct - target_rank) / jnp.maximum(height, 1.0)
    if tile is None:
        local_counts = local_expert_histogram(idx, n_bins)
    else:
        local_counts = factorized_expert_histogram(idx, n_bins, *tile)
    counts = jax.lax.psum(local_counts, axis_name=batch_axes).astype(jnp.float32)
    return histogram_quantile(counts, target_rank, lo, bin_width)


def qb_beta_hist(
    s_ma: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    *,
    target_share: float | jax.Array,
    num_experts: int,
    n_bins: int,
    batch_axes: tuple[str, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global (1-K/E)-quantile of the logit margins over the live ``[min, max]`` grid (this step's).

    A ``pmin``/``pmax`` sets the grid to the exact current range of the margins, then
    ``bincount_upper_quantile`` reads the per-expert threshold. Replaces the per-device ``top_k`` +
    ``pmean`` estimate with a smoother global quantile at the cost of the per-expert count reduction.
    ``target_share`` is the fraction of tokens each expert should take (``K/E``), or one per expert.

    Returns ``(beta, margin_min, margin_max)``: the per-expert threshold plus the live margin range
    (the grid ``lo``/``hi``), surfaced for logging.
    """
    # Tokens at/above beta per expert.
    target_rank = jnp.broadcast_to(jnp.asarray(float(s_ma.shape[0]) * target_share, jnp.float32), (num_experts,))

    def _fn(s_local: jax.Array, target_rank: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # pmin/pmax have no autodiff rule and the range is a control quantity, so detach their inputs;
        # the bincount path drops tangents at the integer bin cast, so it needs none downstream either.
        lo = jax.lax.pmin(jax.lax.stop_gradient(jnp.min(s_local)), axis_name=batch_axes)
        hi = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(s_local)), axis_name=batch_axes)
        hi_grid = jnp.maximum(hi, lo + 1e-6)  # guard a degenerate all-equal range
        beta = bincount_upper_quantile(
            s_local,
            num_experts=num_experts,
            n_bins=n_bins,
            lo=lo,
            hi=hi_grid,
            target_rank=target_rank,
            batch_axes=batch_axes,
        )
        return beta, lo, hi  # surface the live margin range for logging

    return shard_map(_fn, mesh=mesh, in_specs=(P(batch_axes, None), P()), out_specs=(P(), P(), P()))(
        s_ma, reshard(target_rank, P())
    )
