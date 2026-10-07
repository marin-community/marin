# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sharded QB routing-bias estimation: each expert's threshold from a histogram of routing margins."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map


def bincount_upper_quantile(
    s_local: jax.Array,
    *,
    num_experts: int,
    n_bins: int,
    lo: jax.Array,
    hi: jax.Array,
    target_rank: float | jax.Array,
    batch_axes: tuple[str, ...],
) -> jax.Array:
    """Per-expert (1-K/E) upper quantile of ``s_local`` via one fused bincount over ``[lo, hi]``.

    ``target_rank`` is the number of tokens at or above each expert's threshold: a scalar, or one per expert.

    Runs inside a ``shard_map``: a single ``jnp.bincount`` over an expert-major flat index
    (``expert*n_bins + bin``, clip-to-edge) builds the local per-expert histogram, one integer ``psum``
    pools it globally, and beta is read from the top-cumulative counts, interpolated in the crossing bin.
    """
    bin_width = (hi - lo) / n_bins
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)[None, :]
    idx = jnp.clip(((s_local - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1)
    flat = (expert_ids * n_bins + idx).reshape(-1)
    local_counts = jnp.bincount(flat, length=num_experts * n_bins).reshape(num_experts, n_bins)
    counts = jax.lax.psum(local_counts, axis_name=batch_axes).astype(jnp.float32)
    cum_from_top = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]  # #{margins in bins >= b}
    target_rank = jnp.broadcast_to(jnp.asarray(target_rank, jnp.float32), (num_experts,))
    bstar = jnp.clip(jnp.sum((cum_from_top >= target_rank[:, None]).astype(jnp.int32), axis=-1) - 1, 0, n_bins - 1)
    ct_b = jnp.take_along_axis(cum_from_top, bstar[:, None], axis=-1)[:, 0]
    h_b = jnp.take_along_axis(counts, bstar[:, None], axis=-1)[:, 0]
    lower_edge = lo + bstar.astype(jnp.float32) * bin_width
    return lower_edge + bin_width * (ct_b - target_rank) / jnp.maximum(h_b, 1.0)


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
