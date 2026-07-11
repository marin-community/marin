# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Streamed collective-permute expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug.sharding import _batch_axes


def _ring_permutation(size: int, distance: int) -> list[tuple[int, int]]:
    return [(source, (source + distance) % size) for source in range(size)]


def _accepted_counts_for_source(
    counts_by_source: Int[Array, "S Elocal"],
    accepted_totals: Int[Array, "Elocal"],
    source_index: Int[Array, ""],
) -> Int[Array, "Elocal"]:
    """Allocate receiver capacity in source-major order, matching bulk ring."""
    source_ids = jnp.arange(counts_by_source.shape[0], dtype=jnp.int32)
    prior_counts = jnp.sum(
        jnp.where(source_ids[:, None] < source_index, counts_by_source, 0),
        axis=0,
        dtype=jnp.int32,
    )
    source_counts = jax.lax.dynamic_index_in_dim(counts_by_source, source_index, axis=0, keepdims=False)
    remaining = jnp.maximum(accepted_totals - prior_counts, 0)
    return jnp.minimum(source_counts, remaining)


def _compute_source_for_local_experts(
    x_source: Float[Array, "Tlocal H"],
    selected_experts_source: Int[Array, "Tlocal K"],
    combine_weights_source: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    accepted_counts: Int[Array, "Elocal"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_start: Int[Array, ""],
) -> Float[Array, "Tlocal H"]:
    """Run this receiver's experts for one source shard and return source-local rows."""
    tokens, topk = selected_experts_source.shape
    assignments = tokens * topk
    local_experts = moe_w13_local.shape[0]

    expert_flat = selected_experts_source.reshape(assignments)
    local_expert = expert_flat - expert_start
    local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
    local_expert = jnp.where(local_mask, local_expert, 0)

    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    assignment_expert_mask = jnp.logical_and(local_mask[:, None], local_expert[:, None] == expert_ids[None, :])
    assignment_rank_by_expert = jnp.cumsum(assignment_expert_mask.astype(jnp.int32), axis=0) - 1
    assignment_rank = jnp.sum(jnp.where(assignment_expert_mask, assignment_rank_by_expert, 0), axis=1, dtype=jnp.int32)
    accepted_mask = jnp.logical_and(local_mask, assignment_rank < accepted_counts[local_expert])
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)

    # Pack accepted rows in (expert, source assignment) order. The fixed-size
    # tail remains unused by XLA ragged-dot, so skew does not inflate GEMM work.
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(accepted_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, assignments)
    valid = flat_pos < accepted_total

    token_idx = jnp.floor_divide(local_idx, topk)
    x_take = jnp.take(x_source, token_idx, axis=0)
    x_dispatch = jnp.where(valid[:, None], x_take, 0)
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    weight_dispatch = jnp.where(
        valid,
        jnp.take(combine_weights_source.reshape(assignments), local_idx, axis=0).astype(x_source.dtype),
        0,
    )

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, accepted_counts, implementation="xla"),
            _CHECKPOINT_EXPERT_HIDDEN,
        )
        intermediate_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, accepted_counts, implementation="xla"),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    weighted = jnp.where(valid[:, None], out_dispatch * weight_dispatch[:, None], 0)
    return jnp.zeros_like(x_source).at[token_idx].add(weighted, mode="drop")


def _moe_mlp_ep_ring_ppermute_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Rotate source shards through experts and return partial outputs directly."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    ep_size = num_experts // local_experts

    assignments_per_source = selected_experts_local.size
    global_assignments = assignments_per_source * ep_size
    local_capacity = max(local_experts, int(math.ceil(capacity_factor * global_assignments / ep_size)))

    local_route_counts = jnp.bincount(selected_experts_local.reshape(-1), length=num_experts).astype(jnp.int32)
    counts_by_source_global = jax.lax.all_gather(local_route_counts, "expert", axis=0, tiled=False)
    expert_axis = jax.lax.axis_index("expert")
    expert_start = expert_axis * local_experts
    counts_by_source = jax.lax.dynamic_slice_in_dim(
        counts_by_source_global,
        start_index=expert_start,
        slice_size=local_experts,
        axis=1,
    )
    total_counts = jnp.sum(counts_by_source, axis=0, dtype=jnp.int32)
    accepted_totals = _prefix_cap_counts(total_counts, capacity=local_capacity)
    dropped_local = jnp.sum(total_counts, dtype=jnp.int32) - jnp.sum(accepted_totals, dtype=jnp.int32)

    packet = (x_local, selected_experts_local, combine_weights_local)
    out_local = jnp.zeros_like(x_local)
    forward_permutation = _ring_permutation(ep_size, 1)

    # Defining the next packet before local compute leaves the collective
    # independent in HLO, allowing XLA to schedule communication ahead.
    for step in range(ep_size):
        if step + 1 < ep_size:
            next_packet = jax.lax.ppermute(packet, "expert", forward_permutation)

        source_index = jnp.mod(expert_axis - step, ep_size).astype(jnp.int32)
        accepted_counts = _accepted_counts_for_source(counts_by_source, accepted_totals, source_index)
        partial_out = _compute_source_for_local_experts(
            packet[0],
            packet[1],
            packet[2],
            moe_w13_local,
            moe_w2_local,
            accepted_counts,
            activation_fn=activation_fn,
            expert_start=expert_start,
        )
        if step:
            partial_out = jax.lax.ppermute(partial_out, "expert", _ring_permutation(ep_size, -step))
        out_local = out_local + partial_out

        if step + 1 < ep_size:
            packet = next_packet

    dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
