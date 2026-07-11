# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug._moe.sonic_quack import quack_mlp_varlen
from levanter.grug.sharding import _batch_axes


class _RingRouting(NamedTuple):
    assignment_indices: Int[Array, "C"]
    valid: Bool[Array, "C"]
    accepted_counts: Int[Array, "Elocal"]
    local_expert: Int[Array, "A"]
    dropped_local: Int[Array, ""]
    local_capacity: int
    tokens_per_shard: int
    topk: int


def _ring_routing_prepass(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> _RingRouting:
    """Select the exact source-major assignment prefix used by bulk ring."""
    selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
    tokens = selected_experts_global.shape[0]
    tokens_per_shard, topk = selected_experts_local.shape
    assignments = tokens * topk
    expert_flat = selected_experts_global.reshape(assignments)

    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    ep_size = num_experts // local_experts
    local_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments / ep_size)))
    expert_start = jax.lax.axis_index("expert") * local_experts
    local_expert: jax.Array = expert_flat - expert_start
    local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
    local_expert = jnp.where(local_mask, local_expert, 0)

    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    local_mask_i32 = local_mask.astype(jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
        axis=0,
        dtype=jnp.int32,
    )
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    # Select by (local expert, source-major assignment position), exactly as
    # the original bulk implementation. The resulting accepted prefix remains
    # grouped by expert and source-major within each group.
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, assignment_indices = jax.lax.top_k(selection_key, local_capacity)

    return _RingRouting(
        assignment_indices=assignment_indices,
        valid=valid,
        accepted_counts=accepted_counts,
        local_expert=local_expert,
        dropped_local=jnp.sum(counts, dtype=jnp.int32) - accepted_total,
        local_capacity=local_capacity,
        tokens_per_shard=tokens_per_shard,
        topk=topk,
    )


def _group_sizes_with_padding(accepted_counts: Int[Array, "Elocal"], capacity: int) -> Int[Array, "Elocal"]:
    return accepted_counts.at[-1].add(capacity - jnp.sum(accepted_counts, dtype=jnp.int32))


def _bulk_ring_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        weight_flat = combine_weights_global.reshape(-1)
        token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
        weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(x_local.dtype)
        x_take = jnp.take(x_global, token_global, axis=0)
        x_dispatch = jnp.where(routing.valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(routing.valid, weight, jnp.zeros_like(weight))

    group_sizes = _group_sizes_with_padding(routing.accepted_counts, routing.local_capacity)
    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out_global = (
            jnp.zeros_like(x_global).at[token_global].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        )
        return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)


def _validate_quack_bulk_ring_contract(
    x_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> None:
    if activation_fn is not jax.nn.silu:
        raise ValueError("EP-local QuACK bulk ring only supports SiLU/SwiGLU activation")
    if x_local.dtype != jnp.bfloat16 or moe_w13_local.dtype != jnp.bfloat16 or moe_w2_local.dtype != jnp.bfloat16:
        raise TypeError("EP-local QuACK bulk ring requires bfloat16 activations and weights")
    if x_local.ndim != 2 or moe_w13_local.ndim != 3 or moe_w2_local.ndim != 3:
        raise ValueError("EP-local QuACK inputs must have shapes [C,H], [Elocal,H,2I], and [Elocal,I,H]")
    if moe_w13_local.shape[0] != moe_w2_local.shape[0]:
        raise ValueError("EP-local QuACK W13 and W2 must have the same local expert count")
    if moe_w13_local.shape[1] != x_local.shape[1] or moe_w2_local.shape[2] != x_local.shape[1]:
        raise ValueError("EP-local QuACK W13/W2 hidden dimensions must match the activation hidden dimension")
    if moe_w13_local.shape[2] != 2 * moe_w2_local.shape[1]:
        raise ValueError("EP-local QuACK W13 output must be twice the W2 intermediate dimension")


def _bulk_ring_quack_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    _validate_quack_bulk_ring_contract(x_local, moe_w13_local, moe_w2_local, activation_fn=activation_fn)

    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        weight_flat = combine_weights_global.reshape(-1)
        token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
        weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(x_local.dtype)
        x_take = jnp.take(x_global, token_global, axis=0)
        x_dispatch = jnp.where(routing.valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(routing.valid, weight, jnp.zeros_like(weight))

    group_sizes = _group_sizes_with_padding(routing.accepted_counts, routing.local_capacity)
    with jax.named_scope("moe_up_down"):
        out_dispatch = tree_checkpoint_name(
            quack_mlp_varlen(x_dispatch, moe_w13_local, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out_global = (
            jnp.zeros_like(x_global).at[token_global].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        )
        return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)


def _chunk_routing(
    routing: _RingRouting,
    *,
    chunk_start: int,
    chunk_tokens: int,
    chunk_capacity: int,
) -> tuple[Int[Array, "Cchunk"], Bool[Array, "Cchunk"], Int[Array, "Elocal"], Int[Array, "Cchunk"]]:
    token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
    source_token = token_global % routing.tokens_per_shard
    in_chunk = jnp.logical_and(source_token >= chunk_start, source_token < chunk_start + chunk_tokens)
    keep = jnp.logical_and(routing.valid, in_chunk)

    # Filtering the already expert-grouped accepted list preserves its order
    # without another sort. The fixed size keeps both branch shapes static.
    selected_positions = jnp.nonzero(keep, size=chunk_capacity, fill_value=0)[0]
    assignment_indices = jnp.take(routing.assignment_indices, selected_positions, axis=0)
    selected_expert = jnp.take(routing.local_expert, assignment_indices, axis=0)
    chunk_total = jnp.sum(keep.astype(jnp.int32), dtype=jnp.int32)
    chunk_valid = jnp.arange(chunk_capacity, dtype=jnp.int32) < chunk_total

    expert_ids = jnp.arange(routing.accepted_counts.shape[0], dtype=jnp.int32)
    chunk_counts = jnp.sum(
        (selected_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * chunk_valid[:, None].astype(jnp.int32),
        axis=0,
        dtype=jnp.int32,
    )

    selected_token_global = jnp.floor_divide(assignment_indices, routing.topk)
    selected_source_shard = jnp.floor_divide(selected_token_global, routing.tokens_per_shard)
    selected_source_token = selected_token_global % routing.tokens_per_shard
    chunk_token_indices = selected_source_shard * chunk_tokens + selected_source_token - chunk_start
    chunk_token_indices = jnp.where(chunk_valid, chunk_token_indices, 0)
    return assignment_indices, chunk_valid, chunk_counts, chunk_token_indices


def _two_chunk_fast_path_decision(routing: _RingRouting) -> Bool[Array, ""]:
    first_tokens = (routing.tokens_per_shard + 1) // 2
    first_capacity = (routing.local_capacity + 1) // 2
    first_source_token = jnp.floor_divide(routing.assignment_indices, routing.topk) % routing.tokens_per_shard
    first_count = jnp.sum(
        jnp.logical_and(routing.valid, first_source_token < first_tokens).astype(jnp.int32), dtype=jnp.int32
    )
    second_count = jnp.sum(routing.valid.astype(jnp.int32), dtype=jnp.int32) - first_count
    local_fast = jnp.logical_and(
        first_count <= first_capacity, second_count <= routing.local_capacity - first_capacity
    )
    return jax.lax.pmin(local_fast.astype(jnp.int32), "expert") != 0


def _two_chunk_ring_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    first_tokens = (routing.tokens_per_shard + 1) // 2
    second_tokens = routing.tokens_per_shard - first_tokens
    first_capacity = (routing.local_capacity + 1) // 2
    second_capacity = routing.local_capacity - first_capacity

    first_indices, first_valid, first_counts, first_token_indices = _chunk_routing(
        routing, chunk_start=0, chunk_tokens=first_tokens, chunk_capacity=first_capacity
    )
    second_indices, second_valid, second_counts, second_token_indices = _chunk_routing(
        routing, chunk_start=first_tokens, chunk_tokens=second_tokens, chunk_capacity=second_capacity
    )

    # Keep both gathers ahead of the first expert compute. This gives the
    # latency-hiding scheduler an AG1 -> compute0-independent edge.
    with jax.named_scope("gather_chunk_0"):
        x_global_0 = jax.lax.all_gather(x_local[:first_tokens], "expert", tiled=True)
        weights_global_0 = jax.lax.all_gather(combine_weights_local[:first_tokens], "expert", tiled=True)
        weights_flat_0 = weights_global_0.reshape(-1)
        weight_index_0 = (first_token_indices * routing.topk) + (first_indices % routing.topk)
        x_take_0 = jnp.take(x_global_0, first_token_indices, axis=0)
        x_dispatch_0 = jnp.where(first_valid[:, None], x_take_0, jnp.zeros_like(x_take_0))
        weight_0 = jnp.take(weights_flat_0, weight_index_0, axis=0).astype(x_local.dtype)
        weight_dispatch_0 = jnp.where(first_valid, weight_0, jnp.zeros_like(weight_0))

    with jax.named_scope("gather_chunk_1"):
        x_global_1 = jax.lax.all_gather(x_local[first_tokens:], "expert", tiled=True)
        weights_global_1 = jax.lax.all_gather(combine_weights_local[first_tokens:], "expert", tiled=True)
        weights_flat_1 = weights_global_1.reshape(-1)
        weight_index_1 = (second_token_indices * routing.topk) + (second_indices % routing.topk)
        x_take_1 = jnp.take(x_global_1, second_token_indices, axis=0)
        x_dispatch_1 = jnp.where(
            second_valid[:, None],
            x_take_1,
            jnp.zeros_like(x_take_1),
        )
        weight_1 = jnp.take(weights_flat_1, weight_index_1, axis=0).astype(x_local.dtype)
        weight_dispatch_1 = jnp.where(second_valid, weight_1, jnp.zeros_like(weight_1))

    x_dispatch_0 = tree_checkpoint_name(x_dispatch_0, _CHECKPOINT_DISPATCH_INPUT)
    group_sizes_0 = _group_sizes_with_padding(first_counts, first_capacity)
    with jax.named_scope("moe_up_down_chunk_0"):
        w13_out_0 = tree_checkpoint_name(
            ragged_dot(x_dispatch_0, moe_w13_local, group_sizes_0), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate_0, up_0 = jnp.split(w13_out_0, [moe_dim], axis=-1)
        out_dispatch_0 = tree_checkpoint_name(
            ragged_dot(activation_fn(gate_0) * up_0, moe_w2_local, group_sizes_0),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )
    with jax.named_scope("scatter_chunk_0"):
        out_global_0 = (
            jnp.zeros_like(x_global_0)
            .at[first_token_indices]
            .add(out_dispatch_0 * weight_dispatch_0[:, None], mode="drop")
        )
        out_local_0 = jax.lax.psum_scatter(out_global_0, "expert", scatter_dimension=0, tiled=True)

    x_dispatch_1 = tree_checkpoint_name(x_dispatch_1, _CHECKPOINT_DISPATCH_INPUT)
    group_sizes_1 = _group_sizes_with_padding(second_counts, second_capacity)
    with jax.named_scope("moe_up_down_chunk_1"):
        w13_out_1 = tree_checkpoint_name(
            ragged_dot(x_dispatch_1, moe_w13_local, group_sizes_1), _CHECKPOINT_EXPERT_HIDDEN
        )
        gate_1, up_1 = jnp.split(w13_out_1, [moe_dim], axis=-1)
        out_dispatch_1 = tree_checkpoint_name(
            ragged_dot(activation_fn(gate_1) * up_1, moe_w2_local, group_sizes_1),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )
    with jax.named_scope("scatter_chunk_1"):
        out_global_1 = (
            jnp.zeros_like(x_global_1)
            .at[second_token_indices]
            .add(out_dispatch_1 * weight_dispatch_1[:, None], mode="drop")
        )
        out_local_1 = jax.lax.psum_scatter(out_global_1, "expert", scatter_dimension=0, tiled=True)

    return jnp.concatenate((out_local_0, out_local_1), axis=0)


def _moe_mlp_ep_ring_local(
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
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local = _bulk_ring_from_routing(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        routing,
        activation_fn=activation_fn,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_quack_local(
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
    """Benchmark-only bulk-ring path using the EP-local QuACK expert MLP."""
    _validate_quack_bulk_ring_contract(x_local, moe_w13_local, moe_w2_local, activation_fn=activation_fn)
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local = _bulk_ring_quack_from_routing(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        routing,
        activation_fn=activation_fn,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_two_chunk_local(
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
    """Benchmark-only exact two-chunk bulk-ring prototype."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    use_two_chunk = _two_chunk_fast_path_decision(routing)
    out_local = jax.lax.cond(
        use_two_chunk,
        lambda _: _two_chunk_ring_from_routing(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            routing,
            activation_fn=activation_fn,
        ),
        lambda _: _bulk_ring_from_routing(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            routing,
            activation_fn=activation_fn,
        ),
        operand=None,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _ep_ring_two_chunk_fast_path_local(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> Bool[Array, ""]:
    """Return the globally consistent two-chunk gate for benchmark reporting."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=local_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    return _two_chunk_fast_path_decision(routing)
