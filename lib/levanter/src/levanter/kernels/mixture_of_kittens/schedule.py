# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic destination-side schedule for Marin's MoK-like backend."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from levanter.kernels.mixture_of_kittens.config import MokLikeConfig


_TILE_ROWS = 256


class MokLikeSchedule(NamedTuple):
    """Padded local-expert schedule for one expert-parallel rank."""

    peer_rank: Int[Array, " capacity"]
    peer_token_idx: Int[Array, " capacity"]
    num_tokens: Int[Array, ""]
    tokens_per_expert: Int[Array, " local_experts"]
    dropped_assignments: Int[Array, ""]
    overflow: Bool[Array, ""]


def schedule_capacity(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    config: MokLikeConfig,
) -> int:
    """Return the static padded route capacity for one expert rank."""

    if num_local_experts < 1:
        raise ValueError("num_local_experts must be at least one")
    assignments = math.ceil(num_tokens * top_k * config.schedule_capacity_factor)
    expert_padding = num_local_experts * (_TILE_ROWS - 1)
    unaligned = assignments + expert_padding
    return (unaligned + config.minibatch_size - 1) // config.minibatch_size * config.minibatch_size


def build_schedule(
    top_experts: Int[Array, "world tokens top_k"],
    *,
    num_local_experts: int,
    schedule_capacity: int,
    rank: Int[Array, ""],
    expert_padding: int = 256,
) -> MokLikeSchedule:
    """Build the padded pull schedule for one destination rank.

    Each expert segment interleaves source ranks by assignment ordinal. Padding
    rows contain `-1` in both schedule arrays.

    Args:
        top_experts: Global expert indexes from all source ranks.
        num_local_experts: Number of experts owned by the destination rank.
        schedule_capacity: Static size of each output schedule array.
        rank: Destination rank as a scalar JAX array.
        expert_padding: Row alignment for each local expert segment.

    Returns:
        The padded schedule and a data-dependent capacity overflow indicator.
    """
    if top_experts.ndim != 3:
        raise ValueError(f"top_experts must have shape [world, tokens, top_k], got {top_experts.shape}")
    if not jnp.issubdtype(top_experts.dtype, jnp.integer):
        raise TypeError(f"top_experts must have an integer dtype, got {top_experts.dtype}")
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    if expert_padding <= 0:
        raise ValueError(f"expert_padding must be positive, got {expert_padding}")
    if schedule_capacity <= 0 or schedule_capacity % expert_padding != 0:
        raise ValueError(
            f"schedule_capacity must be positive and divisible by expert_padding, got {schedule_capacity}"
        )

    world_size, num_local_tokens, top_k = top_experts.shape
    assignments_per_peer = num_local_tokens * top_k
    routes = top_experts.astype(jnp.int32).reshape(world_size, assignments_per_peer)

    source_ranks = jnp.broadcast_to(
        jnp.arange(world_size, dtype=jnp.int32)[:, None],
        routes.shape,
    )
    source_token_indices = jnp.broadcast_to(
        jnp.arange(assignments_per_peer, dtype=jnp.int32)[None, :],
        routes.shape,
    )

    first_expert = rank.astype(jnp.int32) * num_local_experts
    local_experts = routes - first_expert
    valid = (local_experts >= 0) & (local_experts < num_local_experts)
    safe_local_experts = jnp.clip(local_experts, 0, num_local_experts - 1)

    assignment_experts = jax.nn.one_hot(safe_local_experts, num_local_experts, dtype=jnp.int32)
    assignment_experts *= valid[..., None]
    tokens_per_expert_and_peer = jnp.sum(assignment_experts, axis=1, dtype=jnp.int32).T
    real_tokens_per_expert = jnp.sum(tokens_per_expert_and_peer, axis=1, dtype=jnp.int32)
    tokens_per_expert = ((real_tokens_per_expert + expert_padding - 1) // expert_padding * expert_padding).astype(
        jnp.int32
    )
    expert_offsets = jnp.cumsum(tokens_per_expert, dtype=jnp.int32) - tokens_per_expert

    assignment_ordinals_by_expert = jnp.cumsum(assignment_experts, axis=1, dtype=jnp.int32) - 1
    assignment_ordinals = jnp.take_along_axis(
        assignment_ordinals_by_expert,
        safe_local_experts[..., None],
        axis=2,
    )[..., 0]
    assignment_ordinals = jnp.where(valid, assignment_ordinals, 0)

    round_indices = jnp.arange(assignments_per_peer, dtype=jnp.int32)
    peer_counts_by_round = tokens_per_expert_and_peer[:, None, :]
    round_offsets = jnp.sum(
        jnp.minimum(peer_counts_by_round, round_indices[None, :, None]),
        axis=2,
        dtype=jnp.int32,
    )
    peers_active_by_round = peer_counts_by_round > round_indices[None, :, None]
    active_peer_prefix = jnp.cumsum(peers_active_by_round, axis=2, dtype=jnp.int32)
    active_peer_prefix = jnp.concatenate(
        [jnp.zeros((*active_peer_prefix.shape[:2], 1), dtype=jnp.int32), active_peer_prefix[:, :, :-1]],
        axis=2,
    )

    flat_local_experts = safe_local_experts.reshape(-1)
    flat_ordinals = assignment_ordinals.reshape(-1)
    flat_source_ranks = source_ranks.reshape(-1)
    destinations = expert_offsets[flat_local_experts]
    destinations += round_offsets[flat_local_experts, flat_ordinals]
    destinations += active_peer_prefix[flat_local_experts, flat_ordinals, flat_source_ranks]

    flat_valid = valid.reshape(-1)
    within_capacity = flat_valid & (destinations < schedule_capacity)
    scatter_indices = jnp.where(within_capacity, destinations, schedule_capacity)

    peer_rank = jnp.full((schedule_capacity,), -1, dtype=jnp.int32)
    peer_rank = peer_rank.at[scatter_indices].set(
        jnp.where(within_capacity, flat_source_ranks, -1),
        mode="drop",
    )
    peer_token_idx = jnp.full((schedule_capacity,), -1, dtype=jnp.int32)
    peer_token_idx = peer_token_idx.at[scatter_indices].set(
        jnp.where(within_capacity, source_token_indices.reshape(-1), -1),
        mode="drop",
    )

    total_num_tokens = jnp.sum(tokens_per_expert, dtype=jnp.int32)
    available_per_expert = jnp.maximum(schedule_capacity - expert_offsets, 0)
    clipped_tokens_per_expert = jnp.minimum(tokens_per_expert, available_per_expert)
    num_tokens = jnp.minimum(total_num_tokens, schedule_capacity)
    dropped_assignments = jnp.sum(flat_valid, dtype=jnp.int32) - jnp.sum(within_capacity, dtype=jnp.int32)
    return MokLikeSchedule(
        peer_rank=peer_rank,
        peer_token_idx=peer_token_idx,
        num_tokens=num_tokens,
        tokens_per_expert=clipped_tokens_per_expert,
        dropped_assignments=dropped_assignments,
        overflow=dropped_assignments > 0,
    )
