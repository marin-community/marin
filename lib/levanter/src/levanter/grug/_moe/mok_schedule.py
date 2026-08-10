# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Destination-order dispatch schedule for the MoK expert-parallel backend.

The schedule turns routing into a dense address table: for every row of this
rank's expert-sorted receive buffer it records which ``(peer_rank, token)``
fills that row. Dispatch then issues one remote copy per row with no runtime
offset exchange, which is what makes the transport deterministic.

Within an expert's segment, assignments interleave round-robin across source
ranks so that a destination tile draws from many peers at once.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

EXPERT_PADDING = 256


def build_dispatch_schedule(
    topk_all: Int[Array, "P T K"],
    *,
    num_local_experts: int,
    rank: int,
    schedule_capacity: int,
) -> tuple[Int[Array, "C"], Int[Array, "C"], Int[Array, ""], Int[Array, "E"]]:
    """Builds this rank's destination-order dispatch table.

    Args:
        topk_all: Every rank's selected global expert ids, all-gathered over the
            expert axis.
        num_local_experts: Experts owned by this rank.
        rank: This rank's index in the expert axis.
        schedule_capacity: Static row count of the receive buffer. Rows beyond
            the padded token count keep ``peer_rank = -1`` and are zero-filled
            by dispatch.

    Returns:
        ``peer_rank`` and ``peer_token_idx`` tables of length
        ``schedule_capacity``, the padded total token count, and the padded
        per-expert token counts that feed the grouped GEMM.
    """
    world_size, num_local_tokens, topk = topk_all.shape
    stride = num_local_tokens * topk

    flat = topk_all.reshape(world_size, stride) - rank * num_local_experts
    mine = (flat >= 0) & (flat < num_local_experts)
    local_expert = jnp.where(mine, flat, 0)

    peer_idx = jnp.broadcast_to(jnp.arange(world_size, dtype=jnp.int32)[:, None], (world_size, stride))
    counts = (
        jnp.zeros((num_local_experts, world_size), jnp.int32).at[local_expert, peer_idx].add(mine.astype(jnp.int32))
    )

    tokens_per_expert = -(-counts.sum(axis=1) // EXPERT_PADDING) * EXPERT_PADDING
    num_tokens = tokens_per_expert.sum()
    expert_base = jnp.cumsum(tokens_per_expert) - tokens_per_expert

    one_hot = mine[:, :, None] & (local_expert[:, :, None] == jnp.arange(num_local_experts))
    rank_within_group = jnp.take_along_axis(jnp.cumsum(one_hot, axis=1) - one_hot, local_expert[:, :, None], axis=2)[
        ..., 0
    ]

    def accumulate_source(source, offset):
        source_counts = counts[:, source][local_expert]
        ahead = (source < peer_idx) & (source_counts > rank_within_group)
        return offset + jnp.minimum(source_counts, rank_within_group) + ahead.astype(jnp.int32)

    offset = jax.lax.fori_loop(0, world_size, accumulate_source, jnp.zeros_like(rank_within_group))
    destination = expert_base[local_expert] + offset

    scatter_at = jnp.where(mine, destination, schedule_capacity)
    peer_rank = jnp.full((schedule_capacity,), -1, jnp.int32).at[scatter_at].set(peer_idx, mode="drop")
    token_idx = jnp.broadcast_to(jnp.arange(stride, dtype=jnp.int32)[None, :], (world_size, stride))
    peer_token_idx = jnp.zeros((schedule_capacity,), jnp.int32).at[scatter_at].set(token_idx, mode="drop")

    return peer_rank, peer_token_idx, num_tokens, tokens_per_expert
