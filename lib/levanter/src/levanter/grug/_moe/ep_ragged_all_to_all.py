# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import logging
import math
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug.sharding import _batch_axes

logger = logging.getLogger(__name__)


def _dispatch_rows(
    x_local: Float[Array, "Tlocal H"],
    linear_indices: Int[Array, "A"],
    send_size: int,
    topk: int,
) -> Float[Array, "S H"]:
    """Build the flat fixed-capacity dispatch buffer from local activations.

    With SCALE_A2A_GATHER_DISPATCH=1, scatters int32 assignment indices and gathers
    activation rows from ``x_local`` (avoids materializing the topk-repeated activation
    and the full-row random scatter). Otherwise scatters repeated bf16 rows directly.
    """
    tokens_per_shard, hidden_dim = x_local.shape
    assignments_per_shard = linear_indices.shape[0]
    if os.environ.get("SCALE_A2A_GATHER_DISPATCH") == "1":
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
        padded_x = jnp.concatenate(
            [x_local, jnp.zeros((1, hidden_dim), dtype=x_local.dtype)],
            axis=0,
        )
        return padded_x[token_sources]
    repeated_x = jnp.repeat(x_local, topk, axis=0)
    return jnp.zeros((send_size, hidden_dim), x_local.dtype).at[linear_indices].set(repeated_x, mode="drop")


def _moe_mlp_ep_fixed_a2a_local(
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
    """Expert-parallel MoE with fixed-capacity all-to-all dispatch and combine."""
    chunks = max(int(os.environ.get("SCALE_A2A_CHUNKS", "1")), 1)
    tokens_per_shard = x_local.shape[0]
    if tokens_per_shard % chunks != 0:
        raise ValueError(f"tokens_per_shard={tokens_per_shard} must be divisible by SCALE_A2A_CHUNKS={chunks}")
    if chunks == 1:
        return _fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    tokens_per_chunk = tokens_per_shard // chunks
    chunk_outputs = []
    chunk_dropped = []
    for chunk_index in range(chunks):
        chunk_start = chunk_index * tokens_per_chunk
        chunk_end = (chunk_index + 1) * tokens_per_chunk
        output, dropped = _fixed_a2a_core(
            x_local[chunk_start:chunk_end],
            selected_experts_local[chunk_start:chunk_end],
            combine_weights_local[chunk_start:chunk_end],
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        chunk_outputs.append(output)
        chunk_dropped.append(dropped)
    return jnp.concatenate(chunk_outputs, axis=0), jnp.sum(jnp.stack(chunk_dropped))


def _fixed_a2a_core(
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
    """Run one fixed-capacity all-to-all dispatch, expert MLP, and combine."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = max(int(math.ceil(capacity_factor * assignments_per_shard / num_experts)), 1)

    # Keep XLA's auto-rematerializer from materializing flash-attention scores when
    # estimating memory pressure for the full attention-plus-MoE scan body.
    use_barrier = os.environ.get("SCALE_A2A_NO_BARRIER") != "1"
    if use_barrier:
        x_local, combine_weights_local = jax.lax.optimization_barrier((x_local, combine_weights_local))

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)

    order = jnp.argsort(flat_experts, stable=True)
    inverse_order = jnp.argsort(order)
    expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    segment_start = jnp.cumsum(expert_counts) - expert_counts
    sorted_rank = jnp.arange(assignments_per_shard, dtype=jnp.int32) - segment_start[flat_experts[order]]
    slot = sorted_rank[inverse_order]
    keep = slot < capacity
    local_expert_indices = (flat_experts % local_experts).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * capacity
    send_size = local_experts * bucket_size

    rotate_groups = int(os.environ.get("SCALE_A2A_ROTATE", "0") or "0")
    if rotate_groups > 0:
        return _fixed_a2a_rotation(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            slot=slot,
            keep=keep,
            local_expert_indices=local_expert_indices,
            destination_shards=destination_shards,
            activation_fn=activation_fn,
            capacity=capacity,
            expert_shards=expert_shards,
            groups=rotate_groups,
            use_barrier=use_barrier,
        )

    linear_indices = jnp.where(
        keep,
        local_expert_indices * bucket_size + destination_shards * capacity + slot,
        send_size,
    )

    moe_dim = moe_w2_local.shape[1]
    with jax.named_scope("dispatch"):
        send_x = _dispatch_rows(x_local, linear_indices, send_size, topk)
        send_x = send_x.reshape(local_experts, expert_shards, capacity, hidden_dim)

    output_parts = []
    for local_expert_index in range(local_experts):
        with jax.named_scope("dispatch"):
            received = jax.lax.all_to_all(
                send_x[local_expert_index],
                "expert",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
            received = tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)
        with jax.named_scope("moe_up_down"):
            expert_input = received.reshape(bucket_size, hidden_dim)
            hidden = expert_input @ moe_w13_local[local_expert_index]
            gate, up = jnp.split(hidden, [moe_dim], axis=-1)
            expert_output = (activation_fn(gate) * up) @ moe_w2_local[local_expert_index]
        with jax.named_scope("combine"):
            returned = jax.lax.all_to_all(
                expert_output.reshape(expert_shards, capacity, hidden_dim),
                "expert",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
            output_parts.append(returned)

    with jax.named_scope("combine"):
        send_output = jnp.stack(output_parts, axis=0)
        send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
        send_output = send_output.reshape(send_size, hidden_dim)
        gathered = send_output[jnp.minimum(linear_indices, send_size - 1)]
        gathered = jnp.where(keep[:, None], gathered, 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, dropped_total


def _fixed_a2a_rotation(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    slot: Int[Array, "A"],
    keep: Array,
    local_expert_indices: Int[Array, "A"],
    destination_shards: Int[Array, "A"],
    activation_fn: Callable[[jax.Array], jax.Array],
    capacity: int,
    expert_shards: int,
    groups: int,
    use_barrier: bool,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Fixed-capacity dispatch/combine decomposed into round-robin ppermute rounds.

    The dispatch buffer is laid out by peer *offset* r = (destination - self) mod P
    instead of destination shard id, so round r is a compile-time slice exchanged with
    ppermute perm i -> (i+r) mod P. Rounds are processed in ``groups`` groups: the
    group g+1 permutes are issued before group g's expert GEMM, so the permutes for the
    next group have no data dependency on the current group's compute and can overlap
    it. Combine returns each processed block with the inverted permutation. Bucket
    granularity, capacity, and the overflow policy are identical to the monolithic
    path; only the buffer layout and the collective decomposition differ.
    """
    if expert_shards % groups != 0:
        raise ValueError(f"expert_shards={expert_shards} must be divisible by SCALE_A2A_ROTATE={groups}")
    group_size = expert_shards // groups
    local_experts = moe_w13_local.shape[0]
    tokens_per_shard, hidden_dim = x_local.shape
    topk = combine_weights_local.shape[1]
    moe_dim = moe_w2_local.shape[1]
    block_rows = local_experts * capacity
    send_size = expert_shards * block_rows
    logger.info(
        "fixed-a2a rotation active: groups=%d group_size=%d expert_shards=%d capacity=%d",
        groups,
        group_size,
        expert_shards,
        capacity,
    )

    my_index = jax.lax.axis_index("expert")
    offsets = jnp.mod(destination_shards - my_index, expert_shards).astype(jnp.int32)
    linear_indices = jnp.where(
        keep,
        offsets * block_rows + local_expert_indices * capacity + slot,
        send_size,
    )

    with jax.named_scope("dispatch"):
        send_x = _dispatch_rows(x_local, linear_indices, send_size, topk)
        send_x = send_x.reshape(expert_shards, local_experts, capacity, hidden_dim)

    def permute_group(g: int) -> Float[Array, "S Elocal C H"]:
        parts = []
        for s in range(group_size):
            r = g * group_size + s
            if r == 0:
                parts.append(send_x[0])
                continue
            perm = [(i, (i + r) % expert_shards) for i in range(expert_shards)]
            parts.append(jax.lax.ppermute(send_x[r], "expert", perm))
        return tree_checkpoint_name(jnp.stack(parts, axis=0), _CHECKPOINT_DISPATCH_INPUT)

    def gemm_group(arrived: Float[Array, "S Elocal C H"]) -> Float[Array, "S Elocal C H"]:
        expert_input = arrived.transpose(1, 0, 2, 3).reshape(local_experts, group_size * capacity, hidden_dim)
        hidden = jnp.einsum("eth,ehi->eti", expert_input, moe_w13_local)
        gate, up = jnp.split(hidden, [moe_dim], axis=-1)
        expert_output = jnp.einsum("eti,eih->eth", activation_fn(gate) * up, moe_w2_local)
        return expert_output.reshape(local_experts, group_size, capacity, hidden_dim).transpose(1, 0, 2, 3)

    def combine_group(expert_output: Float[Array, "S Elocal C H"], g: int) -> list[jax.Array]:
        parts = []
        for s in range(group_size):
            r = g * group_size + s
            if r == 0:
                parts.append(expert_output[0])
                continue
            perm = [(i, (i - r) % expert_shards) for i in range(expert_shards)]
            parts.append(jax.lax.ppermute(expert_output[s], "expert", perm))
        return parts

    output_parts: list[jax.Array] = []
    with jax.named_scope("dispatch"):
        arrived = permute_group(0)
    for g in range(groups):
        if g + 1 < groups:
            with jax.named_scope("dispatch"):
                next_arrived = permute_group(g + 1)
        with jax.named_scope("moe_up_down"):
            expert_output = gemm_group(arrived)
        with jax.named_scope("combine"):
            output_parts.extend(combine_group(expert_output, g))
        if g + 1 < groups:
            arrived = next_arrived

    with jax.named_scope("combine"):
        send_output = jnp.stack(output_parts, axis=0)
        send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
        send_output = send_output.reshape(send_size, hidden_dim)
        gathered = send_output[jnp.minimum(linear_indices, send_size - 1)]
        gathered = jnp.where(keep[:, None], gathered, 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        assignments_per_shard = tokens_per_shard * topk
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, dropped_total


def _moe_mlp_ep_ragged_a2a_local(
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
    if os.environ.get("SCALE_A2A_FIXED") == "1":
        return _moe_mlp_ep_fixed_a2a_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    local_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    local_capacity = max(local_experts, local_capacity)
    recv_capacity = local_capacity

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
        clipped_group_sizes = _clip_receiver_group_sizes(
            all_group_sizes,
            local_expert_size=local_experts,
            receiver_capacity=local_capacity,
        )
        sender_group_sizes = clipped_group_sizes[shard_id]
        keep_mask = _expert_prefix_keep_mask(
            group_sizes.astype(jnp.int32),
            sender_group_sizes,
            total_size=assignments_per_shard,
        )
        sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)

        all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        x_dispatched = jax.lax.ragged_all_to_all(
            sorted_x,
            dispatch_out_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
            x_dispatched,
            clipped_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        local_output = _sort_activations(out_dispatch, jnp.argsort(local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            all_shard_counts.T, shard_id
        )
        returned = jax.lax.ragged_all_to_all(
            local_output,
            return_out_shape,
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        returned = _expand_from_keep_mask(returned, keep_mask)
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
