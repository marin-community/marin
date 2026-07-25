# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import math
import os
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _compact_by_keep_mask_to_size,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug._moe.sonic import (
    sonic_dispatch_gather,
    sonic_expert_local_rank,
    sonic_gather_sum,
    sonic_gather_sum_bf16_accum,
    sonic_slot_weighted_grad,
    sonic_unpermute_i32,
)
from levanter.grug.sharding import _batch_axes


_DEFAULT_RECEIVER_SENDER_CAPACITY_FACTOR = 1.125


def _round_robin_ppermute_all_to_all(
    inputs: jax.Array,
    *,
    axis_name: str,
    peer_axis: int,
) -> jax.Array:
    """Decompose an all-to-all into direct round-robin collective permutes."""
    peer_count = inputs.shape[peer_axis]
    peer_index = jax.lax.axis_index(axis_name)
    received_by_round = []
    for round_index in range(peer_count):
        destination = (peer_index + round_index) % peer_count
        payload = jax.lax.dynamic_index_in_dim(
            inputs,
            destination,
            axis=peer_axis,
            keepdims=False,
        )
        if round_index == 0:
            received = payload
        else:
            permutation = tuple((source, (source + round_index) % peer_count) for source in range(peer_count))
            received = jax.lax.ppermute(
                payload,
                axis_name,
                permutation,
            )
        received_by_round.append(received)

    received = jnp.stack(received_by_round, axis=peer_axis)
    source_order = (peer_index - jnp.arange(peer_count, dtype=jnp.int32)) % peer_count
    return jnp.take(received, source_order, axis=peer_axis)


def _fixed_dense_expert_mlp(
    expert_inputs: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    moe_dim = moe_w2.shape[-2]
    if os.environ.get("SCALE_A2A_SPLIT_W13_GEMMS") == "1":
        gate = jnp.matmul(expert_inputs, moe_w13[..., :moe_dim])
        up = jnp.matmul(expert_inputs, moe_w13[..., moe_dim:])
    else:
        hidden = jnp.matmul(expert_inputs, moe_w13)
        gate, up = jnp.split(hidden, [moe_dim], axis=-1)
    return jnp.matmul(activation_fn(gate) * up, moe_w2)


def _moe_mlp_ep_fixed_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    dispatch_slots_local: Int[Array, "Tlocal K"] | None = None,
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
    if os.environ.get("SCALE_A2A_SAME_EXPERT_CLONES") == "1":
        if dispatch_slots_local is not None:
            raise ValueError("precomputed dispatch slots cannot be combined with same-expert clones")
        if chunks != 1:
            raise ValueError("SCALE_A2A_SAME_EXPERT_CLONES=1 currently requires SCALE_A2A_CHUNKS=1")
        return _same_expert_cloned_fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
    if os.environ.get("SCALE_A2A_RECEIVER_CLIP") == "1":
        if dispatch_slots_local is not None:
            raise ValueError("precomputed dispatch slots cannot be combined with receiver clipping")
        if chunks != 1:
            raise ValueError("SCALE_A2A_RECEIVER_CLIP=1 currently requires SCALE_A2A_CHUNKS=1")
        sender_capacity_factor = float(
            os.environ.get(
                "SCALE_A2A_RECEIVER_SENDER_CAPACITY_FACTOR",
                str(_DEFAULT_RECEIVER_SENDER_CAPACITY_FACTOR),
            )
        )
        return _receiver_clipped_fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            sender_capacity_factor=sender_capacity_factor,
        )
    if chunks == 1:
        return _fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            dispatch_slots_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    if dispatch_slots_local is not None:
        raise ValueError("precomputed dispatch slots currently require SCALE_A2A_CHUNKS=1")

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


def _same_expert_clone_dispatch_metadata(
    flat_experts: Int[Array, "TK"],
    all_group_sizes: Int[Array, "S E"],
    sender_index: Int[Array, ""],
    *,
    sender_destination_capacity: int,
    receiver_capacity: int,
) -> tuple[Int[Array, "TK"], Int[Array, "TK"], Int[Array, "S E"], Int[Array, ""]]:
    """Assign every routed expert to a balanced receiver without changing the expert.

    Global assignments for each expert are striped over all expert shards. Rotating
    the stripe phase by expert keeps receiver load within one assignment per expert
    of the mean. Each sender's traffic to a receiver has the same bound, which gives
    the fixed all-to-all a small, deterministic static envelope.
    """
    expert_shards, num_experts = all_group_sizes.shape
    local_rank = _stable_expert_local_rank(flat_experts, num_experts=num_experts)
    preceding_sender = jnp.arange(expert_shards, dtype=jnp.int32) < sender_index
    sender_prefix = jnp.sum(
        jnp.where(preceding_sender[:, None], all_group_sizes, 0),
        axis=0,
        dtype=jnp.int32,
    )
    global_rank = sender_prefix[flat_experts] + local_rank

    expert_phase = jnp.arange(num_experts, dtype=jnp.int32) % expert_shards
    destination = (global_rank + expert_phase[flat_experts]) % expert_shards
    first_rank = (destination - expert_phase[flat_experts]) % expert_shards
    receiver_expert_rank = (global_rank - first_rank) // expert_shards

    global_group_sizes = jnp.sum(all_group_sizes, axis=0, dtype=jnp.int32)
    receiver_indices = jnp.arange(expert_shards, dtype=jnp.int32)[:, None]
    first_rank_by_receiver = (receiver_indices - expert_phase[None, :]) % expert_shards
    receiver_group_sizes = jnp.maximum(
        (global_group_sizes[None, :] + expert_shards - 1 - first_rank_by_receiver) // expert_shards,
        0,
    )
    receiver_group_offsets = jnp.cumsum(receiver_group_sizes, axis=1, dtype=jnp.int32) - receiver_group_sizes
    receiver_slot = receiver_group_offsets[destination, flat_experts] + receiver_expert_rank

    destination_rank = _stable_expert_local_rank(destination, num_experts=expert_shards)
    transport_size = expert_shards * sender_destination_capacity
    within_capacity = jnp.logical_and(
        destination_rank < sender_destination_capacity,
        receiver_slot < receiver_capacity,
    )
    transport_position = jnp.where(
        within_capacity,
        destination * sender_destination_capacity + destination_rank,
        transport_size,
    )
    overflow = jnp.sum(jnp.logical_not(within_capacity), dtype=jnp.int32)
    return transport_position, receiver_slot, receiver_group_sizes, overflow


def _same_expert_pooled_dispatch_metadata(
    flat_experts: Int[Array, "TK"],
    all_group_sizes: Int[Array, "S E"],
    sender_index: Int[Array, ""],
    *,
    sender_destination_capacity: int,
    receiver_capacity: int,
) -> tuple[Int[Array, "TK"], Int[Array, "TK"], Int[Array, "S E"], Int[Array, ""]]:
    """Pack an exact global expert stream into fixed receiver-capacity bins.

    Assignments within each expert are interleaved by sender before the stream is
    split across receivers. Interleaving avoids concentrating a contiguous sender
    prefix in one receiver while retaining the ragged path's stable sender order.
    """
    expert_shards, num_experts = all_group_sizes.shape
    local_rank = _stable_expert_local_rank(flat_experts, num_experts=num_experts)
    assignment_group_sizes = all_group_sizes[:, flat_experts]
    completed_round_assignments = jnp.sum(
        jnp.minimum(assignment_group_sizes, local_rank[None, :]),
        axis=0,
        dtype=jnp.int32,
    )
    sender_indices = jnp.arange(expert_shards, dtype=jnp.int32)
    preceding_sender_in_round = jnp.sum(
        jnp.logical_and(
            assignment_group_sizes > local_rank[None, :],
            sender_indices[:, None] < sender_index,
        ),
        axis=0,
        dtype=jnp.int32,
    )
    interleaved_expert_rank = completed_round_assignments + preceding_sender_in_round

    global_group_sizes = jnp.sum(all_group_sizes, axis=0, dtype=jnp.int32)
    global_group_offsets = jnp.cumsum(global_group_sizes, dtype=jnp.int32) - global_group_sizes
    global_stream_position = global_group_offsets[flat_experts] + interleaved_expert_rank
    destination = global_stream_position // receiver_capacity
    receiver_slot = global_stream_position % receiver_capacity

    receiver_start = jnp.arange(expert_shards, dtype=jnp.int32)[:, None] * receiver_capacity
    receiver_end = receiver_start + receiver_capacity
    expert_start = global_group_offsets[None, :]
    expert_end = expert_start + global_group_sizes[None, :]
    receiver_group_sizes = jnp.maximum(
        jnp.minimum(receiver_end, expert_end) - jnp.maximum(receiver_start, expert_start),
        0,
    )

    destination_rank = _stable_expert_local_rank(destination, num_experts=expert_shards)
    transport_size = expert_shards * sender_destination_capacity
    within_capacity = destination_rank < sender_destination_capacity
    transport_position = jnp.where(
        within_capacity,
        destination * sender_destination_capacity + destination_rank,
        transport_size,
    )
    overflow = jnp.sum(jnp.logical_not(within_capacity), dtype=jnp.int32)
    return transport_position, receiver_slot, receiver_group_sizes, overflow


def _sparse_clone_weight_metadata(
    receiver_group_sizes: Int[Array, "S E"],
    receiver_index: Int[Array, ""],
    *,
    local_experts: int,
    topk: int,
) -> tuple[
    Int[Array, "Msend"],
    Int[Array, "S"],
    Int[Array, "S"],
    Int[Array, "S"],
    Int[Array, "S"],
    Int[Array, "E"],
    Int[Array, ""],
]:
    """Build ragged expert-weight exchange metadata for receiver-pooled clones."""
    expert_shards, num_experts = receiver_group_sizes.shape
    needed = receiver_group_sizes > 0
    local_expert_start = receiver_index * local_experts
    local_needed = jax.lax.dynamic_slice_in_dim(
        needed,
        start_index=local_expert_start,
        slice_size=local_experts,
        axis=1,
    )
    send_sizes = jnp.sum(local_needed, axis=1, dtype=jnp.int32)
    input_offsets = jnp.cumsum(send_sizes, dtype=jnp.int32) - send_sizes

    # Top-k has distinct experts, so one expert receives at most one assignment
    # per global token and can cross at most ceil(S / K) + 1 receiver bins.
    max_receiver_spans = min(expert_shards, int(math.ceil(expert_shards / topk)) + 1)
    max_send_segments = local_experts * max_receiver_spans
    flat_needed = local_needed.reshape(-1)
    compact_position = jnp.cumsum(flat_needed.astype(jnp.int32), dtype=jnp.int32) - 1
    compact_position = jnp.where(flat_needed, compact_position, max_send_segments)
    local_expert_indices = jnp.broadcast_to(
        jnp.arange(local_experts, dtype=jnp.int32)[None, :],
        local_needed.shape,
    ).reshape(-1)
    packed_local_experts = (
        jnp.full((max_send_segments,), local_experts, dtype=jnp.int32)
        .at[compact_position]
        .set(local_expert_indices, mode="drop")
    )

    receiver_needed = needed[receiver_index].reshape(expert_shards, local_experts)
    recv_sizes = jnp.sum(receiver_needed, axis=1, dtype=jnp.int32)
    output_offsets = jnp.cumsum(recv_sizes, dtype=jnp.int32) - recv_sizes

    receiver_groups = receiver_group_sizes[receiver_index]
    receiver_group_position = jnp.cumsum((receiver_groups > 0).astype(jnp.int32), dtype=jnp.int32) - 1
    receiver_group_position = jnp.where(receiver_groups > 0, receiver_group_position, num_experts)
    compact_group_sizes = (
        jnp.zeros((num_experts,), dtype=jnp.int32).at[receiver_group_position].set(receiver_groups, mode="drop")
    )
    send_overflow = jnp.maximum(jnp.sum(send_sizes, dtype=jnp.int32) - max_send_segments, 0)
    return (
        packed_local_experts,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        compact_group_sizes,
        send_overflow,
    )


def _sparse_clone_weight_exchange(
    local_weights: jax.Array,
    packed_local_experts: Int[Array, "Msend"],
    input_offsets: Int[Array, "S"],
    send_sizes: Int[Array, "S"],
    output_offsets: Int[Array, "S"],
    recv_sizes: Int[Array, "S"],
    *,
    num_experts: int,
) -> jax.Array:
    """Move only expert weights needed by this receiver's clone segments."""
    padded_local_weights = jnp.concatenate(
        [local_weights, jnp.zeros((1, *local_weights.shape[1:]), dtype=local_weights.dtype)],
        axis=0,
    )
    send_weights = padded_local_weights[packed_local_experts]
    output_shape = jnp.zeros((num_experts, *local_weights.shape[1:]), dtype=local_weights.dtype)
    return jax.lax.ragged_all_to_all(
        send_weights,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )


def _same_expert_cloned_fixed_a2a_core(
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
    """Run a dropless fixed A2A by executing the same logical expert on clone shards.

    This first correctness-oriented implementation stripes every expert over every
    receiver and gathers the expert weights. It establishes exact routing and a
    balanced fixed token transport; sparse clone-weight movement is the subsequent
    performance step.
    """
    if capacity_factor < 1.0:
        raise ValueError(f"same-expert clones require capacity_factor >= 1.0, got {capacity_factor}")
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard, hidden_dim = x_local.shape
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    expert_shards = num_experts // local_experts
    pooled_dispatch = os.environ.get("SCALE_A2A_CLONE_POOLED") == "1"
    if pooled_dispatch:
        receiver_capacity = assignments_per_shard
    else:
        receiver_capacity = max(
            int(math.ceil(capacity_factor * assignments_per_shard)) + num_experts,
            num_experts,
        )
    sender_destination_capacity = int(math.ceil(assignments_per_shard / expert_shards)) + num_experts
    send_size = expert_shards * sender_destination_capacity

    use_barrier = os.environ.get("SCALE_A2A_NO_BARRIER") != "1"
    if use_barrier:
        x_local, combine_weights_local = jax.lax.optimization_barrier((x_local, combine_weights_local))

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    local_group_sizes = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    all_group_sizes = jax.lax.all_gather(local_group_sizes, "expert")
    receiver_index = jax.lax.axis_index("expert")
    if pooled_dispatch:
        transport_position, receiver_slot, receiver_group_sizes, overflow = _same_expert_pooled_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            receiver_index,
            sender_destination_capacity=sender_destination_capacity,
            receiver_capacity=receiver_capacity,
        )
    else:
        transport_position, receiver_slot, receiver_group_sizes, overflow = _same_expert_clone_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            receiver_index,
            sender_destination_capacity=sender_destination_capacity,
            receiver_capacity=receiver_capacity,
        )
    keep = transport_position < send_size
    dispatch_positions = transport_position.reshape(tokens_per_shard, topk)
    keep_by_token = keep.reshape(tokens_per_shard, topk)

    with jax.named_scope("dispatch"):
        if os.environ.get("SCALE_A2A_SONIC_DISPATCH") == "1":
            send_x = _fixed_dispatch_gather_sonic(
                x_local,
                dispatch_positions,
                keep_by_token,
                send_size,
            )
        elif os.environ.get("SCALE_A2A_SONIC_DISPATCH_GRAD") == "1":
            send_x = _fixed_dispatch_gather_sonic_grad(
                x_local,
                dispatch_positions,
                keep_by_token,
                send_size,
            )
        else:
            send_x = _fixed_dispatch_gather_reference(
                x_local,
                dispatch_positions,
                send_size=send_size,
            )
        send_x = send_x.reshape(expert_shards, sender_destination_capacity, hidden_dim)
        send_receiver_slot = (
            jnp.full((send_size,), receiver_capacity, dtype=jnp.int32)
            .at[transport_position]
            .set(receiver_slot, mode="drop")
            .reshape(expert_shards, sender_destination_capacity)
        )
        received_x = jax.lax.all_to_all(
            send_x,
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        received_slot = jax.lax.all_to_all(
            send_receiver_slot,
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        received_x = tree_checkpoint_name(received_x, _CHECKPOINT_DISPATCH_INPUT)
        received_x_flat = received_x.reshape(send_size, hidden_dim)
        received_slot_flat = received_slot.reshape(send_size)
        receiver_sources = (
            jnp.full((receiver_capacity,), send_size, dtype=jnp.int32)
            .at[received_slot_flat]
            .set(jnp.arange(send_size, dtype=jnp.int32), mode="drop")
        )
        padded_received_x = jnp.concatenate(
            [received_x_flat, jnp.zeros((1, hidden_dim), dtype=x_local.dtype)],
            axis=0,
        )
        expert_inputs = padded_received_x[receiver_sources]

    with jax.named_scope("clone_weights"):
        if pooled_dispatch and os.environ.get("SCALE_A2A_CLONE_SPARSE_WEIGHTS") == "1":
            (
                packed_local_experts,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                group_sizes,
                weight_envelope_overflow,
            ) = _sparse_clone_weight_metadata(
                receiver_group_sizes,
                receiver_index,
                local_experts=local_experts,
                topk=topk,
            )
            global_w13 = _sparse_clone_weight_exchange(
                moe_w13_local,
                packed_local_experts,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                num_experts=num_experts,
            )
            global_w2 = _sparse_clone_weight_exchange(
                moe_w2_local,
                packed_local_experts,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                num_experts=num_experts,
            )
            overflow = overflow + weight_envelope_overflow
        else:
            global_w13 = jax.lax.all_gather(
                moe_w13_local,
                "expert",
                axis=0,
                tiled=True,
            )
            global_w2 = jax.lax.all_gather(
                moe_w2_local,
                "expert",
                axis=0,
                tiled=True,
            )
            group_sizes = receiver_group_sizes[receiver_index]

    with jax.named_scope("moe_up_down"):
        valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)
        group_sizes = group_sizes.at[-1].add(receiver_capacity - valid_rows)
        moe_dim = global_w2.shape[1]
        if os.environ.get("SCALE_A2A_CLONE_SONIC_CUTE") == "1":
            # QuACK/CuTeDSL is an optional Blackwell-only dependency.
            from levanter.grug._moe.sonic_cute import _expert_mlp, _interleave_gate_up  # noqa: PLC0415

            interleaved_w13 = _interleave_gate_up(global_w13, moe_dim)
            cumulative_group_sizes = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
            )
            expert_outputs = _expert_mlp(
                expert_inputs,
                interleaved_w13,
                global_w2,
                group_sizes,
                cumulative_group_sizes,
            )
        else:
            hidden = ragged_dot(expert_inputs, global_w13, group_sizes)
            gate, up = jnp.split(hidden, [moe_dim], axis=-1)
            expert_outputs = ragged_dot(activation_fn(gate) * up, global_w2, group_sizes)

    with jax.named_scope("combine"):
        received_valid = received_slot_flat < receiver_capacity
        returned_x = expert_outputs[jnp.minimum(received_slot_flat, receiver_capacity - 1)]
        returned_x = jnp.where(received_valid[:, None], returned_x, 0)
        returned_x = returned_x.reshape(expert_shards, sender_destination_capacity, hidden_dim)
        returned_x = jax.lax.all_to_all(
            returned_x,
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        returned_x = tree_checkpoint_name(returned_x, _CHECKPOINT_MOE_OUTPUT).reshape(send_size, hidden_dim)
        if os.environ.get("SCALE_A2A_SONIC_COMBINE") == "1":
            masked_combine_weights = jnp.where(
                keep_by_token,
                combine_weights_local,
                0,
            )
            out_local = sonic_gather_sum(
                returned_x,
                dispatch_positions,
                masked_combine_weights,
            ).astype(x_local.dtype)
        else:
            gathered = returned_x[jnp.minimum(dispatch_positions, send_size - 1)]
            gathered = jnp.where(keep_by_token[:, :, None], gathered, 0)
            out_local = jnp.einsum(
                "tkh,tk->th",
                gathered,
                combine_weights_local.astype(gathered.dtype),
                preferred_element_type=jnp.float32,
            ).astype(x_local.dtype)
        overflow_total = jax.lax.psum(
            overflow,
            _batch_axes(jax.sharding.get_abstract_mesh()),
        )
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, overflow_total


def _fixed_dispatch_assignment_sources(
    dispatch_positions: Int[Array, "Tlocal K"],
    *,
    send_size: int,
) -> Int[Array, "M"]:
    assignments_per_shard = dispatch_positions.size
    return (
        jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
        .at[dispatch_positions.reshape(-1)]
        .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
    )


def _fixed_dispatch_token_sources(
    dispatch_positions: Int[Array, "Tlocal K"],
    *,
    send_size: int,
) -> Int[Array, "M"]:
    tokens_per_shard, topk = dispatch_positions.shape
    assignments_per_shard = tokens_per_shard * topk
    assignment_sources = _fixed_dispatch_assignment_sources(dispatch_positions, send_size=send_size)
    token_sources = jnp.where(
        assignment_sources < assignments_per_shard,
        assignment_sources // topk,
        tokens_per_shard,
    )
    return token_sources


def _fixed_dispatch_gather_reference(
    x_local: Float[Array, "Tlocal H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    *,
    send_size: int,
) -> Float[Array, "M H"]:
    token_sources = _fixed_dispatch_token_sources(dispatch_positions, send_size=send_size)
    padded_x = jnp.concatenate(
        [x_local, jnp.zeros((1, x_local.shape[1]), dtype=x_local.dtype)],
        axis=0,
    )
    return padded_x[token_sources]


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fixed_dispatch_gather_sonic_grad(
    x_local: Float[Array, "Tlocal H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    send_size: int,
) -> Float[Array, "M H"]:
    """Use a token-major gather-reduce kernel for the dispatch gather's adjoint."""
    return _fixed_dispatch_gather_reference(x_local, dispatch_positions, send_size=send_size)


def _fixed_dispatch_gather_sonic_grad_fwd(
    x_local: Float[Array, "Tlocal H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    send_size: int,
) -> tuple[Float[Array, "M H"], tuple[Int[Array, "Tlocal K"], jax.Array]]:
    send_x = _fixed_dispatch_gather_reference(x_local, dispatch_positions, send_size=send_size)
    position_order = jnp.argsort(dispatch_positions, axis=-1)
    sorted_positions = jnp.take_along_axis(dispatch_positions, position_order, axis=-1)
    sorted_keep = jnp.take_along_axis(keep, position_order, axis=-1)
    return send_x, (sorted_positions, sorted_keep)


def _fixed_dispatch_gather_sonic_grad_bwd(
    send_size: int,
    residuals: tuple[Int[Array, "Tlocal K"], jax.Array],
    send_x_grad: Float[Array, "M H"],
) -> tuple[Float[Array, "Tlocal H"], None, None]:
    del send_size
    dispatch_positions, keep = residuals
    x_grad = sonic_gather_sum_bf16_accum(
        send_x_grad,
        dispatch_positions,
        keep.astype(send_x_grad.dtype),
    )
    return x_grad, None, None


_fixed_dispatch_gather_sonic_grad.defvjp(
    _fixed_dispatch_gather_sonic_grad_fwd,
    _fixed_dispatch_gather_sonic_grad_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fixed_dispatch_gather_sonic(
    x_local: Float[Array, "Tlocal H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    send_size: int,
) -> Float[Array, "M H"]:
    """Use Triton for fixed-capacity dispatch materialization and its adjoint."""
    del keep
    token_sources = _fixed_dispatch_token_sources(dispatch_positions, send_size=send_size)
    return sonic_dispatch_gather(x_local, token_sources)


def _fixed_dispatch_gather_sonic_fwd(
    x_local: Float[Array, "Tlocal H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    send_size: int,
) -> tuple[Float[Array, "M H"], tuple[Int[Array, "Tlocal K"], jax.Array]]:
    token_sources = _fixed_dispatch_token_sources(dispatch_positions, send_size=send_size)
    send_x = sonic_dispatch_gather(x_local, token_sources)
    position_order = jnp.argsort(dispatch_positions, axis=-1)
    sorted_positions = jnp.take_along_axis(dispatch_positions, position_order, axis=-1)
    sorted_keep = jnp.take_along_axis(keep, position_order, axis=-1)
    return send_x, (sorted_positions, sorted_keep)


_fixed_dispatch_gather_sonic.defvjp(
    _fixed_dispatch_gather_sonic_fwd,
    _fixed_dispatch_gather_sonic_grad_bwd,
)


def _fixed_distributed_combine_impl(
    expert_output: Float[Array, "Elocal S C H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    combine_weights: Float[Array, "Tlocal K"],
) -> Float[Array, "Tlocal H"]:
    returned = jax.lax.all_to_all(
        expert_output,
        "expert",
        split_axis=1,
        concat_axis=1,
        tiled=True,
    )
    returned = tree_checkpoint_name(returned, _CHECKPOINT_MOE_OUTPUT)
    returned = returned.reshape(-1, returned.shape[-1])
    if os.environ.get("SCALE_A2A_SONIC_COMBINE") == "1":
        masked_combine_weights = jnp.where(keep, combine_weights, 0)
        return sonic_gather_sum(returned, dispatch_positions, masked_combine_weights)

    gathered = returned[jnp.minimum(dispatch_positions, returned.shape[0] - 1)]
    gathered = jnp.where(keep[:, :, None], gathered, 0)
    return jnp.einsum(
        "tkh,tk->th",
        gathered,
        combine_weights.astype(gathered.dtype),
        preferred_element_type=jnp.float32,
    ).astype(returned.dtype)


@jax.custom_vjp
def _fixed_distributed_combine(
    expert_output: Float[Array, "Elocal S C H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    combine_weights: Float[Array, "Tlocal K"],
) -> Float[Array, "Tlocal H"]:
    """Combine fixed slots while keeping the rematerialization residual before the A2A."""
    return _fixed_distributed_combine_impl(expert_output, dispatch_positions, keep, combine_weights)


def _fixed_distributed_combine_fwd(
    expert_output: Float[Array, "Elocal S C H"],
    dispatch_positions: Int[Array, "Tlocal K"],
    keep: jax.Array,
    combine_weights: Float[Array, "Tlocal K"],
) -> tuple[
    Float[Array, "Tlocal H"],
    tuple[Float[Array, "Elocal S C H"], Int[Array, "Tlocal K"], jax.Array, Float[Array, "Tlocal K"]],
]:
    output = _fixed_distributed_combine_impl(expert_output, dispatch_positions, keep, combine_weights)
    return output, (expert_output, dispatch_positions, keep, combine_weights)


def _fixed_distributed_combine_bwd(
    residuals: tuple[
        Float[Array, "Elocal S C H"],
        Int[Array, "Tlocal K"],
        jax.Array,
        Float[Array, "Tlocal K"],
    ],
    output_cotangent: Float[Array, "Tlocal H"],
) -> tuple[Float[Array, "Elocal S C H"], None, None, Float[Array, "Tlocal K"]]:
    expert_output, dispatch_positions, keep, combine_weights = residuals
    tokens_per_shard, topk = dispatch_positions.shape
    assignments_per_shard = tokens_per_shard * topk
    send_size = math.prod(expert_output.shape[:-1])
    assignment_sources = _fixed_dispatch_assignment_sources(dispatch_positions, send_size=send_size)
    token_sources = jnp.where(
        assignment_sources < assignments_per_shard,
        assignment_sources // topk,
        tokens_per_shard,
    )

    padded_cotangent = jnp.concatenate(
        [output_cotangent, jnp.zeros((1, output_cotangent.shape[1]), dtype=output_cotangent.dtype)],
        axis=0,
    )
    if os.environ.get("SCALE_A2A_SONIC_COMBINE_COTANGENT") == "1":
        slot_cotangent = sonic_dispatch_gather(padded_cotangent, token_sources).reshape(expert_output.shape)
    else:
        slot_cotangent = padded_cotangent[token_sources].reshape(expert_output.shape)

    masked_combine_weights = jnp.where(keep, combine_weights, 0).reshape(-1)
    padded_combine_weights = jnp.concatenate(
        [masked_combine_weights, jnp.zeros((1,), dtype=masked_combine_weights.dtype)],
        axis=0,
    )
    slot_weights = padded_combine_weights[jnp.minimum(assignment_sources, assignments_per_shard)].reshape(
        expert_output.shape[:-1]
    )

    expert_cotangent = jax.lax.all_to_all(
        slot_cotangent,
        "expert",
        split_axis=1,
        concat_axis=1,
        tiled=True,
    )
    expert_weights = jax.lax.all_to_all(
        slot_weights,
        "expert",
        split_axis=1,
        concat_axis=1,
        tiled=True,
    )

    flat_expert_cotangent = expert_cotangent.reshape(-1, expert_cotangent.shape[-1])
    flat_expert_output = expert_output.reshape(-1, expert_output.shape[-1])
    flat_expert_weights = expert_weights.reshape(-1)
    if os.environ.get("SCALE_A2A_SONIC_DISTRIBUTED_COMBINE_GRAD") == "1":
        flat_expert_output_cotangent, flat_slot_weight_cotangent = sonic_slot_weighted_grad(
            flat_expert_cotangent,
            flat_expert_output,
            flat_expert_weights,
        )
    else:
        flat_expert_output_cotangent = (
            flat_expert_cotangent.astype(jnp.float32) * flat_expert_weights[:, None].astype(jnp.float32)
        ).astype(expert_output.dtype)
        flat_slot_weight_cotangent = jnp.sum(
            flat_expert_cotangent.astype(jnp.float32) * flat_expert_output.astype(jnp.float32),
            axis=-1,
        )

    slot_weight_cotangent = jax.lax.all_to_all(
        flat_slot_weight_cotangent.reshape(expert_output.shape[:-1]),
        "expert",
        split_axis=1,
        concat_axis=1,
        tiled=True,
    ).reshape(-1)
    padded_slot_weight_cotangent = jnp.concatenate(
        [slot_weight_cotangent, jnp.zeros((1,), dtype=slot_weight_cotangent.dtype)],
        axis=0,
    )
    combine_weights_cotangent = padded_slot_weight_cotangent[jnp.minimum(dispatch_positions, send_size)]
    combine_weights_cotangent = jnp.where(keep, combine_weights_cotangent, 0).astype(combine_weights.dtype)
    return (
        flat_expert_output_cotangent.reshape(expert_output.shape),
        None,
        None,
        combine_weights_cotangent,
    )


_fixed_distributed_combine.defvjp(
    _fixed_distributed_combine_fwd,
    _fixed_distributed_combine_bwd,
)


def _stable_expert_local_rank(
    flat_experts: Int[Array, "TK"],
    *,
    num_experts: int,
) -> Int[Array, "TK"]:
    """Return each assignment's stable zero-based rank within its expert."""
    if os.environ.get("SCALE_A2A_SONIC_EXPERT_RANK") == "1":
        return sonic_expert_local_rank(flat_experts, num_experts=num_experts)

    assignments_per_shard = flat_experts.shape[0]
    order = jnp.argsort(flat_experts, stable=True)
    expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    segment_start = jnp.cumsum(expert_counts) - expert_counts
    sorted_rank = jnp.arange(assignments_per_shard, dtype=jnp.int32) - segment_start[flat_experts[order]]
    if os.environ.get("SCALE_A2A_SONIC_SLOT_UNPERMUTE") == "1":
        return sonic_unpermute_i32(sorted_rank, order.astype(jnp.int32))
    inverse_order = jnp.argsort(order)
    return sorted_rank[inverse_order]


def _receiver_clipped_dispatch_metadata(
    flat_experts: Int[Array, "TK"],
    all_group_sizes: Int[Array, "S E"],
    sender_index: Int[Array, ""],
    *,
    local_experts: int,
    receiver_capacity: int,
    sender_expert_capacity: int,
) -> tuple[jax.Array, jax.Array, Int[Array, "S E"], Int[Array, ""], Int[Array, ""]]:
    """Build the ragged receiver-clipped mask plus a fixed-envelope transport mask.

    ``receiver_keep`` is identical to the ragged backend's first-sender-wins
    receiver clipping. ``transport_keep`` additionally applies the fixed
    sender/expert envelope; ``envelope_overflow`` must be zero for the fixed
    transport to preserve the ragged result.
    """
    num_experts = all_group_sizes.shape[1]
    clipped_group_sizes = _clip_receiver_group_sizes(
        all_group_sizes,
        local_expert_size=local_experts,
        receiver_capacity=receiver_capacity,
    )
    accepted_group_sizes = clipped_group_sizes[sender_index]
    local_rank = _stable_expert_local_rank(flat_experts, num_experts=num_experts)
    receiver_keep = local_rank < accepted_group_sizes[flat_experts]
    transport_keep = jnp.logical_and(receiver_keep, local_rank < sender_expert_capacity)
    envelope_overflow = jnp.sum(
        jnp.logical_and(receiver_keep, local_rank >= sender_expert_capacity),
        dtype=jnp.int32,
    )
    receiver_dropped = flat_experts.shape[0] - jnp.sum(receiver_keep, dtype=jnp.int32)
    return receiver_keep, transport_keep, clipped_group_sizes, receiver_dropped, envelope_overflow


def _receiver_clipped_fixed_a2a_core(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    sender_capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Fixed-shape A2A using the ragged backend's receiver-pooled clipping rule.

    The fixed transport reserves modest headroom per sender/expert. If that
    envelope overflows, the layer falls back to the ragged transport rather
    than dropping assignments beyond the receiver-pooled capacity rule.
    """
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if sender_capacity_factor < 1.0:
        raise ValueError(f"sender_capacity_factor must be at least 1.0, got {sender_capacity_factor}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    receiver_capacity = max(int(math.ceil(capacity_factor * assignments_per_shard)), local_experts)
    mean_sender_expert_capacity = capacity_factor * assignments_per_shard / num_experts
    sender_expert_capacity = max(int(math.ceil(sender_capacity_factor * mean_sender_expert_capacity)), 1)

    use_barrier = os.environ.get("SCALE_A2A_NO_BARRIER") != "1"
    if use_barrier:
        x_local, combine_weights_local = jax.lax.optimization_barrier((x_local, combine_weights_local))

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    group_sizes = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    all_group_sizes = jax.lax.all_gather(group_sizes, "expert")
    sender_index = jax.lax.axis_index("expert")
    _, transport_keep, clipped_group_sizes, receiver_dropped, envelope_overflow = _receiver_clipped_dispatch_metadata(
        flat_experts,
        all_group_sizes,
        sender_index,
        local_experts=local_experts,
        receiver_capacity=receiver_capacity,
        sender_expert_capacity=sender_expert_capacity,
    )
    local_rank = _stable_expert_local_rank(flat_experts, num_experts=num_experts)
    local_expert_indices = (flat_experts % local_experts).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * sender_expert_capacity
    send_size = local_experts * bucket_size
    linear_indices = jnp.where(
        transport_keep,
        local_expert_indices * bucket_size + destination_shards * sender_expert_capacity + local_rank,
        send_size,
    )
    dispatch_positions = linear_indices.reshape(tokens_per_shard, topk)
    transport_keep_by_token = transport_keep.reshape(tokens_per_shard, topk)

    with jax.named_scope("dispatch"):
        if os.environ.get("SCALE_A2A_SONIC_DISPATCH") == "1":
            send_x = _fixed_dispatch_gather_sonic(
                x_local,
                dispatch_positions,
                transport_keep_by_token,
                send_size,
            )
        elif os.environ.get("SCALE_A2A_SONIC_DISPATCH_GRAD") == "1":
            send_x = _fixed_dispatch_gather_sonic_grad(
                x_local,
                dispatch_positions,
                transport_keep_by_token,
                send_size,
            )
        else:
            send_x = _fixed_dispatch_gather_reference(
                x_local,
                dispatch_positions,
                send_size=send_size,
            )
        send_x = send_x.reshape(local_experts, expert_shards, sender_expert_capacity, hidden_dim)
        received_by_expert = jax.lax.all_to_all(
            send_x,
            "expert",
            split_axis=1,
            concat_axis=1,
            tiled=True,
        )

    with jax.named_scope("moe_up_down"):
        local_clipped_group_sizes = jax.lax.dynamic_slice_in_dim(
            clipped_group_sizes,
            start_index=sender_index * local_experts,
            slice_size=local_experts,
            axis=1,
        )
        received_keep = (
            jnp.arange(sender_expert_capacity, dtype=jnp.int32)[None, None, :]
            < local_clipped_group_sizes.T[:, :, None]
        )
        transport_size = local_experts * bucket_size
        expert_inputs = tree_checkpoint_name(received_by_expert, _CHECKPOINT_DISPATCH_INPUT)
        if os.environ.get("SCALE_A2A_RECEIVER_DENSE_EXPERTS") == "1":
            # Computing the fixed envelope and masking rejected rows is value/gradient-equivalent
            # to compacting the keep mask, but preserves dense batched GEMMs.
            dense_inputs = expert_inputs.reshape(local_experts, bucket_size, hidden_dim)
            dense_outputs = _fixed_dense_expert_mlp(
                dense_inputs,
                moe_w13_local,
                moe_w2_local,
                activation_fn=activation_fn,
            ).reshape(
                local_experts,
                expert_shards,
                sender_expert_capacity,
                hidden_dim,
            )
            expert_outputs = jnp.where(received_keep[:, :, :, None], dense_outputs, 0)
        else:
            compact_expert_inputs = _compact_by_keep_mask_to_size(
                expert_inputs.reshape(transport_size, hidden_dim),
                received_keep.reshape(transport_size),
                output_size=receiver_capacity,
            )
            local_group_sizes = jnp.sum(local_clipped_group_sizes, axis=0, dtype=jnp.int32)
            valid_received = jnp.sum(local_group_sizes, dtype=jnp.int32)
            local_group_sizes = local_group_sizes.at[-1].add(receiver_capacity - valid_received)
            moe_dim = moe_w2_local.shape[1]
            if os.environ.get("SCALE_A2A_RECEIVER_SONIC_CUTE") == "1":
                # QuACK/CuTeDSL is an optional Blackwell-only dependency.
                from levanter.grug._moe.sonic_cute import _expert_mlp, _interleave_gate_up  # noqa: PLC0415

                interleaved_w13 = _interleave_gate_up(moe_w13_local, moe_dim)
                cumulative_group_sizes = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.cumsum(local_group_sizes, dtype=jnp.int32),
                    ]
                )
                compact_expert_outputs = _expert_mlp(
                    compact_expert_inputs,
                    interleaved_w13,
                    moe_w2_local,
                    local_group_sizes,
                    cumulative_group_sizes,
                )
            else:
                hidden = ragged_dot(compact_expert_inputs, moe_w13_local, local_group_sizes)
                gate, up = jnp.split(hidden, [moe_dim], axis=-1)
                compact_expert_outputs = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)
            expert_outputs = _expand_from_keep_mask(
                compact_expert_outputs,
                received_keep.reshape(transport_size),
            ).reshape(
                local_experts,
                expert_shards,
                sender_expert_capacity,
                hidden_dim,
            )

    with jax.named_scope("combine"):
        if os.environ.get("SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE") == "1":
            out_local = _fixed_distributed_combine(
                expert_outputs,
                dispatch_positions,
                transport_keep_by_token,
                combine_weights_local,
            ).astype(x_local.dtype)
        else:
            returned = jax.lax.all_to_all(
                expert_outputs,
                "expert",
                split_axis=1,
                concat_axis=1,
                tiled=True,
            )
            returned = tree_checkpoint_name(returned, _CHECKPOINT_MOE_OUTPUT).reshape(send_size, hidden_dim)
            if os.environ.get("SCALE_A2A_SONIC_COMBINE") == "1":
                masked_combine_weights = jnp.where(
                    transport_keep_by_token,
                    combine_weights_local,
                    0,
                )
                out_local = sonic_gather_sum(
                    returned,
                    dispatch_positions,
                    masked_combine_weights,
                ).astype(x_local.dtype)
            else:
                gathered = returned[jnp.minimum(dispatch_positions, send_size - 1)]
                gathered = jnp.where(transport_keep_by_token[:, :, None], gathered, 0)
                out_local = jnp.einsum(
                    "tkh,tk->th",
                    gathered,
                    combine_weights_local.astype(gathered.dtype),
                    preferred_element_type=jnp.float32,
                ).astype(x_local.dtype)

        receiver_dropped_total = jax.lax.psum(
            receiver_dropped,
            _batch_axes(jax.sharding.get_abstract_mesh()),
        )
        envelope_overflow_total = jax.lax.psum(
            envelope_overflow,
            _batch_axes(jax.sharding.get_abstract_mesh()),
        )
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)

    def fixed_result(_):
        return out_local, receiver_dropped_total

    def ragged_fallback(_):
        return _moe_mlp_ep_ragged_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    if os.environ.get("SCALE_A2A_RECEIVER_RAGGED_FALLBACK", "1") != "1":
        dropped_or_overflow = jnp.where(
            envelope_overflow_total == 0,
            receiver_dropped_total,
            -envelope_overflow_total,
        )
        return out_local, dropped_or_overflow

    return jax.lax.cond(
        envelope_overflow_total == 0,
        fixed_result,
        ragged_fallback,
        operand=None,
    )


def _fixed_a2a_core(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    dispatch_slots_local: Int[Array, "Tlocal K"] | None = None,
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
    pack_dispatch = os.environ.get("SCALE_A2A_PACK_DISPATCH") == "1"
    pack_combine = os.environ.get("SCALE_A2A_PACK_COMBINE") == "1"
    batch_expert_gemms = os.environ.get("SCALE_A2A_BATCH_EXPERT_GEMMS") == "1"
    custom_distributed_combine = os.environ.get("SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE") == "1"
    if batch_expert_gemms and not pack_dispatch:
        raise ValueError("SCALE_A2A_BATCH_EXPERT_GEMMS=1 requires SCALE_A2A_PACK_DISPATCH=1")
    if batch_expert_gemms and not pack_combine:
        raise ValueError("SCALE_A2A_BATCH_EXPERT_GEMMS=1 requires SCALE_A2A_PACK_COMBINE=1")
    if custom_distributed_combine and not pack_combine:
        raise ValueError("SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE=1 requires SCALE_A2A_PACK_COMBINE=1")

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)

    slot = (
        dispatch_slots_local.reshape(-1).astype(jnp.int32)
        if dispatch_slots_local is not None
        else _stable_expert_local_rank(flat_experts, num_experts=num_experts)
    )
    keep = slot < capacity
    local_expert_indices = (flat_experts % local_experts).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * capacity
    send_size = local_experts * bucket_size
    linear_indices = jnp.where(
        keep,
        local_expert_indices * bucket_size + destination_shards * capacity + slot,
        send_size,
    )

    with jax.named_scope("dispatch"):
        dispatch_positions = linear_indices.reshape(tokens_per_shard, topk)
        keep_by_token = keep.reshape(tokens_per_shard, topk)
        if os.environ.get("SCALE_A2A_SONIC_DISPATCH") == "1":
            send_x = _fixed_dispatch_gather_sonic(
                x_local,
                dispatch_positions,
                keep_by_token,
                send_size,
            )
        elif os.environ.get("SCALE_A2A_SONIC_DISPATCH_GRAD") == "1":
            send_x = _fixed_dispatch_gather_sonic_grad(
                x_local,
                dispatch_positions,
                keep_by_token,
                send_size,
            )
        elif os.environ.get("SCALE_A2A_GATHER_DISPATCH") == "1":
            send_x = _fixed_dispatch_gather_reference(
                x_local,
                dispatch_positions,
                send_size=send_size,
            )
        else:
            repeated_x = jnp.repeat(x_local, topk, axis=0)
            send_x = jnp.zeros((send_size, hidden_dim), x_local.dtype).at[linear_indices].set(repeated_x, mode="drop")
        send_x = send_x.reshape(local_experts, expert_shards, capacity, hidden_dim)

        if pack_dispatch:
            received_by_expert = jax.lax.all_to_all(
                send_x,
                "expert",
                split_axis=1,
                concat_axis=1,
                tiled=True,
            )

    if batch_expert_gemms:
        with jax.named_scope("moe_up_down"):
            expert_inputs = tree_checkpoint_name(received_by_expert, _CHECKPOINT_DISPATCH_INPUT)
            expert_inputs = expert_inputs.reshape(local_experts, bucket_size, hidden_dim)
            expert_outputs = _fixed_dense_expert_mlp(
                expert_inputs,
                moe_w13_local,
                moe_w2_local,
                activation_fn=activation_fn,
            )
            expert_outputs = expert_outputs.reshape(local_experts, expert_shards, capacity, hidden_dim)
    else:
        output_parts = []
        for local_expert_index in range(local_experts):
            with jax.named_scope("dispatch"):
                if pack_dispatch:
                    received = received_by_expert[local_expert_index]
                else:
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
                expert_output = _fixed_dense_expert_mlp(
                    expert_input,
                    moe_w13_local[local_expert_index],
                    moe_w2_local[local_expert_index],
                    activation_fn=activation_fn,
                )
            with jax.named_scope("combine"):
                expert_output = expert_output.reshape(expert_shards, capacity, hidden_dim)
                if pack_combine:
                    output_parts.append(expert_output)
                else:
                    returned = jax.lax.all_to_all(
                        expert_output,
                        "expert",
                        split_axis=0,
                        concat_axis=0,
                        tiled=True,
                    )
                    output_parts.append(returned)

    with jax.named_scope("combine"):
        if batch_expert_gemms:
            send_output = expert_outputs
        else:
            send_output = jnp.stack(output_parts, axis=0)
        if custom_distributed_combine:
            out_local = _fixed_distributed_combine(
                send_output,
                dispatch_positions,
                keep_by_token,
                combine_weights_local,
            ).astype(x_local.dtype)
        elif pack_combine:
            send_output = jax.lax.all_to_all(
                send_output,
                "expert",
                split_axis=1,
                concat_axis=1,
                tiled=True,
            )
        if not custom_distributed_combine:
            send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
            send_output = send_output.reshape(send_size, hidden_dim)
            if os.environ.get("SCALE_A2A_SONIC_COMBINE") == "1":
                masked_combine_weights = jnp.where(keep_by_token, combine_weights_local, 0)
                out_local = sonic_gather_sum(
                    send_output,
                    dispatch_positions,
                    masked_combine_weights,
                ).astype(x_local.dtype)
            else:
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


def _moe_mlp_ep_ragged_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    dispatch_slots_local: Int[Array, "Tlocal K"] | None = None,
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
            dispatch_slots_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    if dispatch_slots_local is not None:
        raise ValueError("precomputed dispatch slots require SCALE_A2A_FIXED=1")

    return _moe_mlp_ep_ragged_a2a_core(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_ep_ragged_a2a_core(
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
