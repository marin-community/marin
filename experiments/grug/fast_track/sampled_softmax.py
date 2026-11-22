# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``sampled_softmax_shortlist_frac``: a debiased sampled-softmax training loss for the lm_head.

Every position scores a cheap loss over a shortlist of the vocabulary plus its own target token. One position per
block of ``SAMPLE_BLOCK`` is drawn with probability ``1 / block length``; there the full-vocabulary loss is computed
and ``(full - shortlist) / p`` is added. In expectation over the draw this is the full-vocabulary loss (z-loss
included), so its gradient is the full gradient, output weights included. The shortlist is the batch's most frequent
labels (stop-gradient), and the draw is a hash of each block's index and tokens."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

SAMPLE_BLOCK = 16


def sampled_positions(token_ids: Int[Array, "B S"], block: int) -> tuple[Int[Array, "B N"], Float[Array, " N"]]:
    """One position per block of ``block`` positions (the last block may be shorter) and each block's length, the
    inverse of its positions' sampling probability. The offset is a uint32 hash of the row, the block index, the
    block's first token and the sequence's first token, so it changes with the batch."""
    batch, seq = token_ids.shape
    starts = jnp.arange(0, seq, block, dtype=jnp.int32)
    lengths = jnp.minimum(block, seq - starts)
    tokens = token_ids.astype(jnp.uint32)
    row = jnp.arange(batch, dtype=jnp.uint32)[:, None] * jnp.uint32(starts.shape[0])
    h = (row + jnp.arange(starts.shape[0], dtype=jnp.uint32)[None, :]) * jnp.uint32(3266489917)
    h = h ^ (tokens[:, ::block] * jnp.uint32(2654435761)) ^ (tokens[:, :1] * jnp.uint32(2246822519))
    h = (h ^ (h >> 15)) * jnp.uint32(2246822519)
    h = (h ^ (h >> 13)) * jnp.uint32(3266489917)
    h = h ^ (h >> 16)
    offsets = (h % lengths.astype(jnp.uint32)).astype(jnp.int32)
    return starts[None, :] + offsets, lengths.astype(jnp.float32)


def batch_shortlist(
    labels: Int[Array, "B S"], loss_weight: Float[Array, "B S"], vocab_size: int, size: int
) -> Int[Array, " K"]:
    """The ``size`` most frequent labels of the batch by loss weight, replicated."""
    everywhere = P(None, None)
    labels = reshard(labels, everywhere).reshape(-1)
    weight = reshard(jax.lax.stop_gradient(loss_weight).astype(jnp.float32), everywhere).reshape(-1)
    counts = jnp.zeros((vocab_size,), jnp.float32).at[labels].add(weight)
    return jax.lax.top_k(counts, size)[1].astype(jnp.int32)


def _lookup(table: jax.Array, ids: jax.Array) -> jax.Array:
    return table[ids]


def shortlist_positions(
    shortlist: Int[Array, " K"], labels: Int[Array, "B S"], vocab_size: int, batch_axes
) -> jax.Array:
    """Each label's column in ``shortlist``, or ``K`` when the shortlist lacks it."""
    size = shortlist.shape[0]
    table = jnp.full((vocab_size,), size, jnp.int32).at[shortlist].set(jnp.arange(size, dtype=jnp.int32))
    spec = P(batch_axes, None)
    return jax.shard_map(_lookup, mesh=get_abstract_mesh(), in_specs=(P(None), spec), out_specs=spec)(
        table, reshard(labels, spec)
    )


def shortlist_loss(
    shortlist_ce: Float[Array, "B S"],
    pivot_logit: Float[Array, "B S"],
    target_logit: Float[Array, "B S"],
    in_shortlist: jax.Array,
    logsumexp_weight: float | None,
) -> Float[Array, "B S"]:
    """Per-position loss over the shortlist plus the target. ``shortlist_ce`` is the shortlist cross-entropy at
    each label's shortlist column (column 0 when the shortlist lacks it), ``pivot_logit`` that column's logit; adding
    them back gives the shortlist's log-partition, which the target joins when it is not in the shortlist."""
    log_partition = shortlist_ce + pivot_logit
    outside = jnp.where(in_shortlist, -jnp.inf, target_logit)
    log_partition = jnp.logaddexp(log_partition, outside)
    loss = log_partition - target_logit
    if logsumexp_weight:
        loss = loss + logsumexp_weight * jnp.square(log_partition)
    return loss


def debiased_loss(
    short: Float[Array, "B S"],
    full_at_sampled: Float[Array, "B N"],
    positions: Int[Array, "B N"],
    inverse_probability: Float[Array, " N"],
    loss_weight: Float[Array, "B S"],
) -> tuple[jax.Array, Float[Array, "B N"]]:
    """The loss-weighted mean of ``short`` plus the importance-weighted corrections at the sampled positions, and
    those positions' ``full - short`` gaps."""
    weight = loss_weight.astype(jnp.float32)
    gap = full_at_sampled - jnp.take_along_axis(short, positions, axis=1)
    sampled_weight = jnp.take_along_axis(weight, positions, axis=1)
    total = jnp.sum(weight * short) + jnp.sum(sampled_weight * gap * inverse_probability[None, :])
    denom = jnp.sum(weight)
    return jnp.where(denom != 0, total / denom, jnp.zeros_like(denom)), gap
