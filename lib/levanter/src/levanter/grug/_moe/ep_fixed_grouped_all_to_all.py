# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Grouped fixed-capacity all-to-all expert-parallel Grug MoE backend.

``fixed_all_to_all`` gives every (sender shard, global expert) pair its own fixed cell, so an idle
expert cannot lend rows to a hot one even when both live on the same device. This backend pools
that capacity across the experts sharing a destination: cells become (sender shard, destination
shard), which is ``local_experts`` times fewer and ``local_experts`` times deeper. **The total
buffer is unchanged**, so routing imbalance within a device is absorbed rather than dropped, at no
memory cost. On a 64-way expert axis with 256 experts that is 4,096 cells instead of 16,384.

The receive side pays for it. Rows arriving at a device now belong to several experts with
data-dependent counts, so the per-expert dense GEMM of ``fixed_all_to_all`` cannot be used: the
rows are permuted into expert-sorted order and run through a ragged dot. Sorting by global expert
id is already (destination, local expert) lexicographic, so each destination's segment leaves the
sender expert-sorted and the receiver only needs the per-expert counts, which ride along in a
second small all-to-all.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug.sharding import _batch_axes


def _moe_mlp_ep_fixed_grouped_a2a_local(
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
    """Run grouped fixed-capacity all-to-all dispatch, expert MLPs, and combine.

    ``capacity_factor`` scales each (sender shard, destination shard) cell as
    ``ceil(factor * local assignments / destination shards)``. Because the cell spans every expert
    on the destination, this factor is not comparable to the per-expert factor in
    ``fixed_all_to_all``: the same value here buys the same bytes but drops strictly less.
    """
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    # Pooled over the destination's experts, so this matches `fixed_all_to_all`'s total buffer:
    # `expert_shards * group_capacity == num_experts * per_expert_capacity`.
    group_capacity = max(int(math.ceil(capacity_factor * assignments_per_shard / expert_shards)), 1)
    send_size = expert_shards * group_capacity

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)

    # Rank within the destination group, not within the expert. Sorting on the global expert id
    # keeps each destination's segment ordered by local expert, which is what lets the receiver
    # rebuild expert-sorted order from counts alone.
    order = jnp.argsort(flat_experts, stable=True)
    inverse_order = jnp.argsort(order)
    dest_counts = jnp.bincount(destination_shards, length=expert_shards).astype(jnp.int32)
    dest_start = jnp.cumsum(dest_counts) - dest_counts
    sorted_rank = jnp.arange(assignments_per_shard, dtype=jnp.int32) - dest_start[destination_shards[order]]
    slot = sorted_rank[inverse_order]
    keep = slot < group_capacity
    linear_indices = jnp.where(keep, destination_shards * group_capacity + slot, send_size)

    with jax.named_scope("dispatch"):
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[linear_indices]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )
        token_sources = jnp.where(
            assignment_sources < assignments_per_shard,
            assignment_sources // topk,
            tokens_per_shard,
        )
        padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
        send_x = padded_x[token_sources].reshape(expert_shards, group_capacity, hidden_dim)

        # Per-(destination, local expert) kept counts. The receiver needs these to recover which
        # expert each arriving row belongs to; they cost `local_experts` int32 per destination.
        kept_pair = jnp.where(keep, flat_experts, num_experts)
        send_counts = (
            jnp.bincount(kept_pair, length=num_experts + 1)[:num_experts]
            .reshape(expert_shards, local_experts)
            .astype(jnp.int32)
        )

        received = jax.lax.all_to_all(send_x, "expert", split_axis=0, concat_axis=0, tiled=True)
        received = tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)
        recv_counts = jax.lax.all_to_all(send_counts, "expert", split_axis=0, concat_axis=0, tiled=True)

    with jax.named_scope("regroup"):
        # Row (sender s, slot j) belongs to the expert whose count interval contains j. Rows past
        # every interval are the cell's unused tail.
        slot_index = jnp.arange(group_capacity, dtype=jnp.int32)[None, :]
        interval_end = jnp.cumsum(recv_counts, axis=1)
        interval_start = interval_end - recv_counts
        expert_of = jnp.sum(slot_index[..., None] >= interval_end[:, None, :], axis=-1).astype(jnp.int32)
        occupied = expert_of < local_experts
        expert_safe = jnp.minimum(expert_of, local_experts - 1)

        group_sizes = jnp.sum(recv_counts, axis=0).astype(jnp.int32)
        expert_base = jnp.cumsum(group_sizes) - group_sizes
        sender_base = jnp.cumsum(recv_counts, axis=0) - recv_counts
        sender_index = jnp.arange(expert_shards, dtype=jnp.int32)[:, None]
        target = (
            expert_base[expert_safe]
            + sender_base[sender_index, expert_safe]
            + (slot_index - interval_start[sender_index, expert_safe])
        )
        target = jnp.where(occupied, target, send_size).reshape(-1)

        # Invert once and gather, rather than scatter the rows themselves: the backward pass of a
        # gather is a scatter-add over indices, not over hidden-sized rows.
        source = (
            jnp.full((send_size + 1,), send_size, dtype=jnp.int32)
            .at[target]
            .set(jnp.arange(send_size, dtype=jnp.int32))[:send_size]
        )
        padded_recv = jnp.concatenate(
            [received.reshape(send_size, hidden_dim), jnp.zeros((1, hidden_dim), received.dtype)], axis=0
        )
        expert_sorted = padded_recv[source]

    with jax.named_scope("moe_up_down"):
        moe_dim = moe_w2_local.shape[1]
        hidden = ragged_dot(expert_sorted, moe_w13_local, group_sizes)
        gate, up = jnp.split(hidden, [moe_dim], axis=-1)
        expert_output = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("combine"):
        padded_output = jnp.concatenate([expert_output, jnp.zeros((1, hidden_dim), expert_output.dtype)], axis=0)
        unsorted = padded_output[target].reshape(expert_shards, group_capacity, hidden_dim)
        returned = jax.lax.all_to_all(unsorted, "expert", split_axis=0, concat_axis=0, tiled=True)
        send_output = tree_checkpoint_name(returned, _CHECKPOINT_MOE_OUTPUT).reshape(send_size, hidden_dim)
        gathered = jnp.where(keep[:, None], send_output[jnp.minimum(linear_indices, send_size - 1)], 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
