# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert exact Ring routing into NCCL UB-X dispatch and combine maps."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


class UbXRoutingMaps(NamedTuple):
    dispatch_topk_expert: Int[Array, "Tlocal K"]
    dispatch_topk_slot: Int[Array, "Tlocal K"]
    inverse_map: Int[Array, "S 4"]
    topk_idx: Int[Array, "Tlocal K"]
    compact_slots: Int[Array, "C"]
    compact_valid: Bool[Array, "C"]
    group_sizes: Int[Array, "Elocal"]
    accepted_local: Bool[Array, "Tlocal K"]
    accepted_counts: Int[Array, "E"]
    drops_by_expert_rank: Int[Array, "R"]


def build_ubx_routing_maps(
    selected_experts_global: Int[Array, "T K"],
    accepted_assignment_indices_by_rank: Int[Array, "R C"],
    accepted_valid_by_rank: Bool[Array, "R C"],
    *,
    rank: int | Int[Array, ""],
    local_experts: int,
) -> UbXRoutingMaps:
    """Build UB-X maps from the exact accepted assignment list used by Ring.

    `accepted_assignment_indices_by_rank` is the all-gathered result of
    `_ring_routing_prepass`: each row is expert-major and source-major within
    each expert, padded to the fixed per-rank capacity `C`.
    """
    if selected_experts_global.ndim != 2:
        raise ValueError("selected_experts_global must have shape [tokens, topk]")
    if accepted_assignment_indices_by_rank.ndim != 2:
        raise ValueError("accepted_assignment_indices_by_rank must have shape [ranks, capacity]")
    if accepted_valid_by_rank.shape != accepted_assignment_indices_by_rank.shape:
        raise ValueError("accepted assignment indices and validity must have the same shape")
    if local_experts <= 0:
        raise ValueError(f"local_experts must be positive, got {local_experts}")

    num_ranks, capacity = accepted_assignment_indices_by_rank.shape
    global_tokens, topk = selected_experts_global.shape
    num_experts = num_ranks * local_experts
    assignments = global_tokens * topk
    if global_tokens % num_ranks:
        raise ValueError(f"global token count {global_tokens} must be divisible by rank count {num_ranks}")
    if capacity <= 0:
        raise ValueError(f"accepted assignment capacity must be positive, got {capacity}")

    selected = jnp.asarray(selected_experts_global, dtype=jnp.int32)
    accepted_indices = jnp.asarray(accepted_assignment_indices_by_rank, dtype=jnp.int32)
    accepted_valid = jnp.asarray(accepted_valid_by_rank, dtype=jnp.bool_)

    # Ring's per-rank accepted lists partition the accepted assignment set.
    # Scatter through a sentinel row so padded entries cannot alias assignment 0.
    accepted_safe = jnp.where(accepted_valid, accepted_indices, assignments)
    accepted_flat = (
        jnp.zeros(assignments + 1, dtype=jnp.int32)
        .at[accepted_safe.reshape(-1)]
        .add(accepted_valid.reshape(-1).astype(jnp.int32))[:assignments]
        > 0
    )
    accepted = accepted_flat.reshape(global_tokens, topk)

    token_rows = jnp.repeat(jnp.arange(global_tokens, dtype=jnp.int32), topk)
    routing = (
        jnp.zeros((global_tokens, num_experts), dtype=jnp.int32)
        .at[token_rows, selected.reshape(-1)]
        .add(accepted_flat.astype(jnp.int32))
        > 0
    )
    accepted_counts = jnp.sum(routing, axis=0, dtype=jnp.int32)
    prefix = jnp.cumsum(routing, axis=0, dtype=jnp.int32) - routing

    # UB-X scans expert ids in ascending order. Sorting the accepted top-k
    # reproduces its dispatch map and combine k-index independently of the
    # router's original top-k ordering.
    sentinel_expert = jnp.asarray(num_experts, dtype=jnp.int32)
    sorted_experts = jnp.sort(jnp.where(accepted, selected, sentinel_expert), axis=1)
    sorted_valid = sorted_experts < num_experts
    sorted_safe = jnp.minimum(sorted_experts, num_experts - 1)
    sorted_prefix = jnp.take_along_axis(prefix, sorted_safe, axis=1)
    sorted_slots = (sorted_safe % local_experts) * capacity + sorted_prefix
    sorted_experts = jnp.where(sorted_valid, sorted_experts, -1)
    sorted_slots = jnp.where(sorted_valid, sorted_slots, -1)

    local_tokens = global_tokens // num_ranks
    local_start = jnp.asarray(rank, dtype=jnp.int32) * local_tokens
    dispatch_topk_expert = jax.lax.dynamic_slice_in_dim(sorted_experts, local_start, local_tokens, axis=0)
    dispatch_topk_slot = jax.lax.dynamic_slice_in_dim(sorted_slots, local_start, local_tokens, axis=0)
    accepted_local = jax.lax.dynamic_slice_in_dim(accepted, local_start, local_tokens, axis=0)

    assignment_prefix = jnp.take_along_axis(prefix, selected, axis=1)
    assignment_slots = (selected % local_experts) * capacity + assignment_prefix
    lower_accepted = accepted[:, None, :] & (selected[:, None, :] < selected[:, :, None])
    combine_k = jnp.sum(lower_accepted, axis=2, dtype=jnp.int32)
    destination_rank = selected // local_experts
    owned = accepted & (destination_rank == jnp.asarray(rank, dtype=jnp.int32))

    source_token = jnp.arange(global_tokens, dtype=jnp.int32)
    source_rank = source_token // local_tokens
    source_local_token = source_token % local_tokens
    inverse_values = jnp.stack(
        (
            jnp.broadcast_to(source_rank[:, None], selected.shape),
            jnp.broadcast_to(source_local_token[:, None], selected.shape),
            combine_k,
            jnp.ones_like(selected),
        ),
        axis=-1,
    )
    max_slots_per_rank = local_experts * capacity
    inverse_safe = jnp.where(owned, assignment_slots, max_slots_per_rank)
    inverse_map = (
        jnp.zeros((max_slots_per_rank + 1, 4), dtype=jnp.int32)
        .at[inverse_safe.reshape(-1)]
        .add((inverse_values * owned[:, :, None]).reshape(-1, 4))[:max_slots_per_rank]
    )

    expert_start = jnp.asarray(rank, dtype=jnp.int32) * local_experts
    local_counts = jax.lax.dynamic_slice_in_dim(accepted_counts, expert_start, local_experts, axis=0)
    accepted_local_total = jnp.sum(local_counts, dtype=jnp.int32)
    compact_slots = jnp.nonzero(inverse_map[:, 3], size=capacity, fill_value=0)[0].astype(jnp.int32)
    compact_valid = jnp.arange(capacity, dtype=jnp.int32) < accepted_local_total
    group_sizes = local_counts.at[-1].add(capacity - jnp.sum(local_counts, dtype=jnp.int32))

    original_counts = jnp.bincount(selected.reshape(-1), length=num_experts).astype(jnp.int32)
    drops_by_expert_rank = original_counts.reshape(num_ranks, local_experts).sum(axis=1) - accepted_counts.reshape(
        num_ranks, local_experts
    ).sum(axis=1)
    return UbXRoutingMaps(
        dispatch_topk_expert=dispatch_topk_expert,
        dispatch_topk_slot=dispatch_topk_slot,
        inverse_map=inverse_map,
        topk_idx=dispatch_topk_expert,
        compact_slots=compact_slots,
        compact_valid=compact_valid,
        group_sizes=group_sizes,
        accepted_local=accepted_local,
        accepted_counts=accepted_counts,
        drops_by_expert_rank=drops_by_expert_rank,
    )
