# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Receiver-balanced expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug._moe.ep_common import _shard_a2a_params
from levanter.grug._moe.ep_fixed_all_to_all import _combine_gather, _dispatch_gather
from levanter.grug.sharding import _batch_axes

_MAX_RECEIVER_EXPERTS = 16
_TOKEN_PADDING_EXPERTS = 1


def _stable_segment_rank(bucket: Int[Array, " n"], num_buckets: int) -> Int[Array, " n"]:
    """Return each element's stable rank in its bucket."""
    total = bucket.shape[0]
    order = jnp.argsort(bucket, stable=True)
    counts = jnp.bincount(bucket, length=num_buckets).astype(jnp.int32)
    starts = jnp.cumsum(counts) - counts
    ranks_sorted = jnp.arange(total, dtype=jnp.int32) - starts[bucket[order]]
    return ranks_sorted[jnp.argsort(order)]


def _echo_dispatch_metadata(
    flat_experts: Int[Array, " assignments"],
    all_group_sizes: Int[Array, "shards experts"],
    sender_index: Int[Array, ""],
    *,
    receiver_capacity: int,
    max_receiver_experts: int,
) -> tuple[Int[Array, " assignments"], Int[Array, " assignments"], Int[Array, "shards experts"], Int[Array, ""]]:
    """Move home-rank overflow to receiver ranks without a change to the selected expert."""
    expert_shards, num_experts = all_group_sizes.shape
    if num_experts % expert_shards != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by expert_shards={expert_shards}")
    local_experts = num_experts // expert_shards

    global_group_sizes = jnp.sum(all_group_sizes, axis=0, dtype=jnp.int32)
    home_group_sizes = global_group_sizes.reshape(expert_shards, local_experts)
    home_loads = jnp.sum(home_group_sizes, axis=1, dtype=jnp.int32)

    retained_target = jnp.minimum(home_loads, receiver_capacity)
    retention_scale = retained_target.astype(jnp.float32) / jnp.maximum(home_loads, 1).astype(jnp.float32)
    retained_home = jnp.floor(home_group_sizes.astype(jnp.float32) * retention_scale[:, None]).astype(jnp.int32)
    retained_remainder = retained_target - jnp.sum(retained_home, axis=1, dtype=jnp.int32)
    can_increment = retained_home < home_group_sizes
    increment_rank = jnp.cumsum(can_increment.astype(jnp.int32), axis=1, dtype=jnp.int32) - 1
    retained_home = retained_home + jnp.logical_and(
        can_increment,
        increment_rank < retained_remainder[:, None],
    ).astype(jnp.int32)
    retained_group_sizes = retained_home.reshape(num_experts)

    overflow_group_sizes = global_group_sizes - retained_group_sizes
    overflow_group_offsets = jnp.cumsum(overflow_group_sizes, dtype=jnp.int32) - overflow_group_sizes
    receiver_spare = receiver_capacity - jnp.sum(retained_home, axis=1, dtype=jnp.int32)
    receiver_spare_offsets = jnp.cumsum(receiver_spare, dtype=jnp.int32) - receiver_spare

    overflow_start = overflow_group_offsets[None, :]
    overflow_end = overflow_start + overflow_group_sizes[None, :]
    receiver_start = receiver_spare_offsets[:, None]
    receiver_end = receiver_start + receiver_spare[:, None]
    cloned_group_sizes = jnp.maximum(
        jnp.minimum(receiver_end, overflow_end) - jnp.maximum(receiver_start, overflow_start),
        0,
    )
    receiver_indices = jnp.arange(expert_shards, dtype=jnp.int32)[:, None]
    expert_home = jnp.arange(num_experts, dtype=jnp.int32)[None, :] // local_experts
    receiver_group_sizes = cloned_group_sizes + jnp.where(
        receiver_indices == expert_home,
        retained_group_sizes[None, :],
        0,
    )

    receiver_group_position = jnp.cumsum((receiver_group_sizes > 0).astype(jnp.int32), axis=1, dtype=jnp.int32) - 1
    retained_group = jnp.logical_and(
        receiver_group_sizes > 0,
        receiver_group_position < max_receiver_experts,
    )
    receiver_group_sizes = jnp.where(retained_group, receiver_group_sizes, 0)

    local_rank = _stable_segment_rank(flat_experts, num_buckets=num_experts)
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
    global_expert_rank = completed_round_assignments + preceding_sender_in_round

    retained = retained_group_sizes[flat_experts]
    stays_home = global_expert_rank < retained
    overflow_position = overflow_group_offsets[flat_experts] + global_expert_rank - retained
    receiver_spare_ends = receiver_spare_offsets + receiver_spare
    overflow_destination = jnp.searchsorted(receiver_spare_ends, overflow_position, side="right").astype(jnp.int32)
    home_destination = flat_experts // local_experts
    destination = jnp.where(stays_home, home_destination, overflow_destination)
    safe_destination = jnp.minimum(destination, expert_shards - 1)

    receiver_group_offsets = jnp.cumsum(receiver_group_sizes, axis=1, dtype=jnp.int32) - receiver_group_sizes
    overflow_segment_start = jnp.maximum(
        overflow_group_offsets[flat_experts],
        receiver_spare_offsets[safe_destination],
    )
    rank_in_receiver_group = jnp.where(
        stays_home,
        global_expert_rank,
        overflow_position - overflow_segment_start,
    )
    receiver_slot = receiver_group_offsets[safe_destination, flat_experts] + rank_in_receiver_group

    keep = jnp.logical_and(destination < expert_shards, receiver_slot < receiver_capacity)
    keep = jnp.logical_and(keep, retained_group[safe_destination, flat_experts])
    destination = jnp.where(keep, destination, expert_shards)
    receiver_slot = jnp.where(keep, receiver_slot, receiver_capacity)
    overflow = jnp.sum(jnp.logical_not(keep), dtype=jnp.int32)
    return destination, receiver_slot, receiver_group_sizes, overflow


def _echo_fixed_transport_metadata(
    destination: Int[Array, " assignments"],
    *,
    expert_shards: int,
    sender_destination_capacity: int,
) -> tuple[Int[Array, " assignments"], Array, Int[Array, ""]]:
    """Pack receiver destinations into a fixed sender-to-receiver envelope."""
    destination_rank = _stable_segment_rank(destination, num_buckets=expert_shards + 1)
    destination_valid = destination < expert_shards
    keep = jnp.logical_and(destination_valid, destination_rank < sender_destination_capacity)
    send_size = expert_shards * sender_destination_capacity
    transport_position = jnp.where(
        keep,
        destination * sender_destination_capacity + destination_rank,
        send_size,
    )
    overflow = jnp.sum(jnp.logical_and(destination_valid, jnp.logical_not(keep)), dtype=jnp.int32)
    return transport_position, keep, overflow


def _clone_weight_metadata(
    receiver_group_sizes: Int[Array, "shards experts"],
    receiver_index: Int[Array, ""],
    *,
    local_experts: int,
    max_receiver_experts: int,
    topk: int,
) -> tuple[
    Int[Array, " send_experts"],
    Int[Array, " shards"],
    Int[Array, " shards"],
    Int[Array, " shards"],
    Int[Array, " shards"],
    Int[Array, " receiver_experts"],
    Int[Array, ""],
]:
    """Build sparse expert-weight exchange parameters for receiver copies."""
    expert_shards, num_experts = receiver_group_sizes.shape
    needed = receiver_group_sizes > 0
    local_expert_start = receiver_index * local_experts
    local_needed = jax.lax.dynamic_slice_in_dim(
        needed,
        start_index=local_expert_start,
        slice_size=local_experts,
        axis=1,
    )
    send_matrix = jnp.sum(
        needed.reshape(expert_shards, expert_shards, local_experts),
        axis=2,
        dtype=jnp.int32,
    ).T
    input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(send_matrix, receiver_index)

    max_receiver_spans = min(expert_shards, int(math.ceil(expert_shards / topk)) + 1)
    max_send_experts = local_experts * max_receiver_spans
    flat_needed = local_needed.reshape(-1)
    compact_position = jnp.cumsum(flat_needed.astype(jnp.int32), dtype=jnp.int32) - 1
    compact_position = jnp.where(flat_needed, compact_position, max_send_experts)
    local_expert_indices = jnp.broadcast_to(
        jnp.arange(local_experts, dtype=jnp.int32)[None, :],
        local_needed.shape,
    ).reshape(-1)
    packed_local_experts = (
        jnp.full((max_send_experts,), local_experts, dtype=jnp.int32)
        .at[compact_position]
        .set(local_expert_indices, mode="drop")
    )

    receiver_groups = receiver_group_sizes[receiver_index]
    receiver_group_position = jnp.cumsum((receiver_groups > 0).astype(jnp.int32), dtype=jnp.int32) - 1
    receiver_group_position = jnp.where(receiver_groups > 0, receiver_group_position, num_experts)
    compact_group_sizes = (
        jnp.zeros((max_receiver_experts,), dtype=jnp.int32)
        .at[receiver_group_position]
        .set(receiver_groups, mode="drop")
    )
    weight_overflow = jnp.maximum(jnp.sum(send_sizes, dtype=jnp.int32) - max_send_experts, 0)
    return (
        packed_local_experts,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        compact_group_sizes,
        weight_overflow,
    )


def _exchange_clone_weights(
    local_weights: jax.Array,
    packed_local_experts: Int[Array, " send_experts"],
    input_offsets: Int[Array, " shards"],
    send_sizes: Int[Array, " shards"],
    output_offsets: Int[Array, " shards"],
    recv_sizes: Int[Array, " shards"],
    *,
    max_receiver_experts: int,
) -> jax.Array:
    """Move only the expert weights that each receiver executes."""
    padded_weights = jnp.concatenate(
        [local_weights, jnp.zeros((1, *local_weights.shape[1:]), dtype=local_weights.dtype)],
        axis=0,
    )
    send_weights = padded_weights[packed_local_experts]
    receiver_weights = jnp.zeros((max_receiver_experts, *local_weights.shape[1:]), dtype=local_weights.dtype)
    return jax.lax.ragged_all_to_all(
        send_weights,
        receiver_weights,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )


@jax.custom_vjp
def _receiver_reorder(
    values: Float[Array, "send H"],
    receiver_sources: Int[Array, " receiver"],
    received_slots: Int[Array, " send"],
) -> Float[Array, "receiver H"]:
    """Reorder received rows into receiver expert groups."""
    del received_slots
    padded_values = jnp.concatenate([values, jnp.zeros((1, values.shape[1]), dtype=values.dtype)], axis=0)
    return padded_values[receiver_sources]


def _receiver_reorder_fwd(values, receiver_sources, received_slots):
    result = _receiver_reorder(values, receiver_sources, received_slots)
    return result, (received_slots, receiver_sources.shape[0])


def _receiver_reorder_bwd(residual, cotangent):
    received_slots, receiver_capacity = residual
    valid = received_slots < receiver_capacity
    slots = jnp.minimum(received_slots, receiver_capacity - 1)
    values_cotangent = jnp.where(valid[:, None], cotangent[slots], 0).astype(cotangent.dtype)
    return values_cotangent, None, None


_receiver_reorder.defvjp(_receiver_reorder_fwd, _receiver_reorder_bwd)


@jax.custom_vjp
def _receiver_unreorder(
    expert_outputs: Float[Array, "receiver H"],
    received_slots: Int[Array, " send"],
    receiver_sources: Int[Array, " receiver"],
) -> Float[Array, "send H"]:
    """Move receiver expert outputs back into sender transport order."""
    del receiver_sources
    padded_outputs = jnp.concatenate(
        [expert_outputs, jnp.zeros((1, expert_outputs.shape[1]), dtype=expert_outputs.dtype)],
        axis=0,
    )
    return padded_outputs[received_slots]


def _receiver_unreorder_fwd(expert_outputs, received_slots, receiver_sources):
    result = _receiver_unreorder(expert_outputs, received_slots, receiver_sources)
    return result, (receiver_sources, received_slots.shape[0])


def _receiver_unreorder_bwd(residual, cotangent):
    receiver_sources, send_size = residual
    valid = receiver_sources < send_size
    sources = jnp.minimum(receiver_sources, send_size - 1)
    outputs_cotangent = jnp.where(valid[:, None], cotangent[sources], 0).astype(cotangent.dtype)
    return outputs_cotangent, None, None


_receiver_unreorder.defvjp(_receiver_unreorder_fwd, _receiver_unreorder_bwd)


def _ragged_expert_mlp(
    expert_inputs: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    group_sizes: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    moe_dim = moe_w2.shape[1]
    hidden = ragged_dot(expert_inputs, moe_w13, group_sizes)
    gate, up = jnp.split(hidden, [moe_dim], axis=-1)
    return ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes)


def _quack_expert_mlp(
    expert_inputs: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    group_sizes: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    if activation_fn is not jax.nn.silu:
        raise ValueError("echo_receiver_cute requires SiLU because its QuACK kernel fuses SwiGLU")
    from levanter.grug._moe.sonic_cute import _expert_mlp, _interleave_gate_up  # noqa: PLC0415

    moe_dim = moe_w2.shape[1]
    interleaved_w13 = _interleave_gate_up(moe_w13, moe_dim)
    cumulative_group_sizes = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
    )
    return _expert_mlp(expert_inputs, interleaved_w13, moe_w2, group_sizes, cumulative_group_sizes)


def _moe_mlp_ep_echo_receiver_core(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_mlp: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, Callable[[jax.Array], jax.Array]], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Execute selected experts on receiver-balanced sparse expert copies."""
    if capacity_factor < 1.0:
        raise ValueError(f"echo receiver requires capacity_factor >= 1.0, got {capacity_factor}")
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard, hidden_dim = x_local.shape
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    expert_shards = num_experts // local_experts
    receiver_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    sender_destination_capacity = (
        int(math.ceil(assignments_per_shard / expert_shards)) + _TOKEN_PADDING_EXPERTS * num_experts
    )
    send_size = expert_shards * sender_destination_capacity

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    local_group_sizes = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    all_group_sizes = jax.lax.all_gather(local_group_sizes, "expert")
    receiver_index = jax.lax.axis_index("expert")
    destination, receiver_slot, receiver_group_sizes, overflow = _echo_dispatch_metadata(
        flat_experts,
        all_group_sizes,
        receiver_index,
        receiver_capacity=receiver_capacity,
        max_receiver_experts=_MAX_RECEIVER_EXPERTS,
    )
    transport_position, keep, envelope_overflow = _echo_fixed_transport_metadata(
        destination,
        expert_shards=expert_shards,
        sender_destination_capacity=sender_destination_capacity,
    )
    overflow += envelope_overflow

    with jax.named_scope("dispatch"):
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[transport_position]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )
        token_sources = jnp.where(
            assignment_sources < assignments_per_shard,
            assignment_sources // topk,
            tokens_per_shard,
        )
        send_x = _dispatch_gather(x_local, token_sources, transport_position, keep)
        send_receiver_slot = (
            jnp.full((send_size,), receiver_capacity, dtype=jnp.int32)
            .at[transport_position]
            .set(receiver_slot, mode="drop")
        )
        received_x = jax.lax.all_to_all(
            send_x.reshape(expert_shards, sender_destination_capacity, hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        received_slot = jax.lax.all_to_all(
            send_receiver_slot.reshape(expert_shards, sender_destination_capacity),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        received_x = tree_checkpoint_name(received_x, _CHECKPOINT_DISPATCH_INPUT).reshape(send_size, hidden_dim)
        received_slot = received_slot.reshape(send_size)
        receiver_sources = (
            jnp.full((receiver_capacity,), send_size, dtype=jnp.int32)
            .at[received_slot]
            .set(jnp.arange(send_size, dtype=jnp.int32), mode="drop")
        )
        expert_inputs = _receiver_reorder(received_x, receiver_sources, received_slot)

    with jax.named_scope("clone_weights"):
        (
            packed_local_experts,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            group_sizes,
            weight_overflow,
        ) = _clone_weight_metadata(
            receiver_group_sizes,
            receiver_index,
            local_experts=local_experts,
            max_receiver_experts=_MAX_RECEIVER_EXPERTS,
            topk=topk,
        )
        receiver_w13 = _exchange_clone_weights(
            moe_w13_local,
            packed_local_experts,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            max_receiver_experts=_MAX_RECEIVER_EXPERTS,
        )
        receiver_w2 = _exchange_clone_weights(
            moe_w2_local,
            packed_local_experts,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            max_receiver_experts=_MAX_RECEIVER_EXPERTS,
        )
        overflow += weight_overflow

    with jax.named_scope("moe_up_down"):
        valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)
        group_sizes = group_sizes.at[-1].add(receiver_capacity - valid_rows)
        expert_outputs = expert_mlp(expert_inputs, receiver_w13, receiver_w2, group_sizes, activation_fn)

    with jax.named_scope("combine"):
        returned_x = _receiver_unreorder(expert_outputs, received_slot, receiver_sources)
        returned_x = jax.lax.all_to_all(
            returned_x.reshape(expert_shards, sender_destination_capacity, hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        returned_x = tree_checkpoint_name(returned_x, _CHECKPOINT_MOE_OUTPUT).reshape(send_size, hidden_dim)
        gather_indices = jnp.minimum(transport_position, send_size - 1)
        gathered = _combine_gather(returned_x, gather_indices, keep, assignment_sources)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        overflow_total = jax.lax.psum(overflow, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, overflow_total


def _moe_mlp_ep_echo_receiver_local(
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
    """Run portable receiver-ECHO expert parallelism."""
    return _moe_mlp_ep_echo_receiver_core(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        expert_mlp=_ragged_expert_mlp,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_ep_echo_receiver_cute_local(
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
    """Run receiver-ECHO expert parallelism with QuACK expert kernels."""
    return _moe_mlp_ep_echo_receiver_core(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        expert_mlp=_quack_expert_mlp,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
