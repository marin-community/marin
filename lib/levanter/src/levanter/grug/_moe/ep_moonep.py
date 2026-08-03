# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable JAX implementation of MoonEP expert planning and transport.

The allocation policy follows the MIT-licensed MoonEP planning reference at
https://github.com/moonshotAI/moonep/tree/0f385f038fc33bec22e3bcf5a07a8a22693e754c.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
    MoonEPGroupedGemm,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import _shard_a2a_params
from levanter.grug._moe.ep_fixed_all_to_all import _combine_gather, _dispatch_gather
from levanter.grug.sharding import _batch_axes

_QUACK_ALIGNMENT = 128
_EXPERT_ID_DTYPE = jnp.dtype(jnp.int32)


class MoonEPPlan(NamedTuple):
    """Static-shape MoonEP allocation and receiver layout."""

    allocation: Int[Array, "E R"]
    cumulative_allocation: Int[Array, "E R"]
    experts_to_copy: Int[Array, "R B"]
    group_experts: Int[Array, "R 2B"]
    group_sizes: Int[Array, "R 2B"]
    padded_group_sizes: Int[Array, "R 2B"]
    group_offsets: Int[Array, "R 2B"]
    rank_loads: Int[Array, " R"]
    remote_expert_counts: Int[Array, " R"]
    weight_copy_counts: Int[Array, " R"]
    violations: Int[Array, ""]


def _balance_owner_groups(
    allocation: Int[Array, "E R"],
    group_counts: Int[Array, " R"],
    *,
    experts_per_rank: int,
    assignments_per_rank: Int[Array, ""],
) -> Int[Array, "E R"]:
    num_ranks = group_counts.shape[0]
    balance = group_counts - assignments_per_rank
    migration = jnp.zeros((num_ranks, num_ranks), dtype=jnp.int32)

    def _select_receiver(_iteration, state):
        current_balance, current_migration = state
        owner = jnp.argmax(current_balance)
        receiver = jnp.argmin(current_balance)
        active = current_balance[owner] > 0
        move = jnp.where(active, -current_balance[receiver], 0)
        current_migration = current_migration.at[owner, receiver].add(move)
        current_balance = current_balance.at[owner].add(-move)
        receiver_balance = jnp.where(active, 0, current_balance[receiver])
        current_balance = current_balance.at[receiver].set(receiver_balance)
        return current_balance, current_migration

    _, migration = jax.lax.fori_loop(0, num_ranks - 1, _select_receiver, (balance, migration))

    def _allocate_owner(owner, current_allocation):
        expert_start = owner * experts_per_rank
        remaining = jax.lax.dynamic_slice_in_dim(
            jnp.sum(current_allocation, axis=1, dtype=jnp.int32),
            expert_start,
            experts_per_rank,
            axis=0,
        )
        quotas = migration[owner]

        def _move_expert(_iteration, state):
            allocation_state, remaining_state, quota_state = state
            receiver = jnp.argmax(quota_state)
            local_expert = jnp.argmax(remaining_state)
            quota = quota_state[receiver]
            expert_count = remaining_state[local_expert]
            active = jnp.logical_and(quota > 0, expert_count > 0)
            take = jnp.where(active, jnp.minimum(quota, expert_count), 0)
            expert = expert_start + local_expert
            allocation_state = allocation_state.at[expert, receiver].add(take)
            allocation_state = allocation_state.at[expert, owner].add(-take)
            remaining_state = remaining_state.at[local_expert].add(-take)
            quota_state = quota_state.at[receiver].add(-take)
            return allocation_state, remaining_state, quota_state

        max_moves = num_ranks + experts_per_rank - 2
        current_allocation, _, _ = jax.lax.fori_loop(
            0,
            max_moves,
            _move_expert,
            (current_allocation, remaining, quotas),
        )
        return current_allocation

    return jax.lax.fori_loop(0, num_ranks, _allocate_owner, allocation)


def moon_ep_plan(
    tokens_per_expert: Int[Array, "R E"],
    *,
    token_padding: int,
) -> MoonEPPlan:
    """Create the deterministic MoonEP allocation for equal-size source ranks."""
    if tokens_per_expert.ndim != 2:
        raise ValueError(f"tokens_per_expert must have shape [ranks, experts], got {tokens_per_expert.shape}")
    num_ranks, num_experts = tokens_per_expert.shape
    if num_experts % num_ranks != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}")
    if token_padding <= 0:
        raise ValueError(f"token_padding must be positive, got {token_padding}")

    tokens_per_expert = tokens_per_expert.astype(jnp.int32)
    experts_per_rank = num_experts // num_ranks
    source_loads = jnp.sum(tokens_per_expert, axis=1, dtype=jnp.int32)
    assignments_per_rank = source_loads[0]
    global_counts = jnp.sum(tokens_per_expert, axis=0, dtype=jnp.int32)
    group_counts = jnp.sum(global_counts.reshape(num_ranks, experts_per_rank), axis=1, dtype=jnp.int32)

    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    home_ranks = expert_ids // experts_per_rank
    allocation = jnp.zeros((num_experts, num_ranks), dtype=jnp.int32)
    allocation = allocation.at[expert_ids, home_ranks].set(global_counts)
    allocation = _balance_owner_groups(
        allocation,
        group_counts,
        experts_per_rank=experts_per_rank,
        assignments_per_rank=assignments_per_rank,
    )

    destination_counts = allocation.T
    destination_ids = jnp.arange(num_ranks, dtype=jnp.int32)[:, None]
    remote_mask = jnp.logical_and(
        destination_counts > 0,
        home_ranks[None, :] != destination_ids,
    )
    remote_expert_counts = jnp.sum(remote_mask, axis=1, dtype=jnp.int32)
    remote_keys = jnp.where(
        remote_mask,
        destination_counts * (num_experts + 1) + expert_ids[None, :],
        -1,
    )
    remote_values, experts_to_copy = jax.lax.top_k(remote_keys, experts_per_rank)
    experts_to_copy = jnp.where(remote_values >= 0, experts_to_copy, -1).astype(jnp.int32)

    local_experts = destination_ids * experts_per_rank + jnp.arange(experts_per_rank, dtype=jnp.int32)
    local_group_sizes = jnp.take_along_axis(destination_counts, local_experts, axis=1)
    safe_remote_experts = jnp.maximum(experts_to_copy, 0)
    remote_group_sizes = jnp.take_along_axis(destination_counts, safe_remote_experts, axis=1)
    remote_group_sizes = jnp.where(experts_to_copy >= 0, remote_group_sizes, 0)
    group_experts = jnp.concatenate((local_experts, experts_to_copy), axis=1)
    group_sizes = jnp.concatenate((local_group_sizes, remote_group_sizes), axis=1)
    padded_group_sizes = jnp.maximum(
        ((group_sizes + token_padding - 1) // token_padding) * token_padding,
        token_padding,
    )
    group_ends = jnp.cumsum(padded_group_sizes, axis=1, dtype=jnp.int32)
    group_offsets = group_ends - padded_group_sizes

    rank_loads = jnp.sum(allocation, axis=0, dtype=jnp.int32)
    weight_copy_counts = jnp.sum(
        remote_mask.reshape(num_ranks, num_ranks, experts_per_rank),
        axis=(0, 2),
        dtype=jnp.int32,
    )
    conservation_error = jnp.sum(
        jnp.abs(jnp.sum(allocation, axis=1, dtype=jnp.int32) - global_counts),
        dtype=jnp.int32,
    )
    rank_load_error = jnp.sum(jnp.abs(rank_loads - assignments_per_rank), dtype=jnp.int32)
    source_load_error = jnp.sum(jnp.abs(source_loads - assignments_per_rank), dtype=jnp.int32)
    remote_expert_error = jnp.sum(jnp.maximum(remote_expert_counts - experts_per_rank, 0), dtype=jnp.int32)
    max_weight_copies = num_ranks + experts_per_rank - 2
    weight_copy_error = jnp.sum(jnp.maximum(weight_copy_counts - max_weight_copies, 0), dtype=jnp.int32)
    violations = conservation_error + rank_load_error + source_load_error + remote_expert_error + weight_copy_error

    return MoonEPPlan(
        allocation=allocation,
        cumulative_allocation=jnp.cumsum(allocation, axis=1, dtype=jnp.int32),
        experts_to_copy=experts_to_copy,
        group_experts=group_experts,
        group_sizes=group_sizes,
        padded_group_sizes=padded_group_sizes,
        group_offsets=group_offsets,
        rank_loads=rank_loads,
        remote_expert_counts=remote_expert_counts,
        weight_copy_counts=weight_copy_counts,
        violations=violations,
    )


def _stable_segment_rank(bucket: Int[Array, " N"], *, num_buckets: int) -> Int[Array, " N"]:
    order = jnp.argsort(bucket, stable=True)
    counts = jnp.bincount(bucket, length=num_buckets).astype(jnp.int32)
    starts = jnp.cumsum(counts, dtype=jnp.int32) - counts
    ranks_sorted = jnp.arange(bucket.shape[0], dtype=jnp.int32) - starts[bucket[order]]
    return ranks_sorted[jnp.argsort(order)]


def _assignment_destinations(
    flat_experts: Int[Array, " N"],
    all_expert_counts: Int[Array, "R E"],
    plan: MoonEPPlan,
    rank: Int[Array, ""],
) -> tuple[Int[Array, " N"], Int[Array, ""]]:
    num_ranks, num_experts = all_expert_counts.shape
    local_rank = _stable_segment_rank(flat_experts, num_buckets=num_experts)
    preceding_sources = jnp.arange(num_ranks, dtype=jnp.int32) < rank
    source_prefix = jnp.sum(jnp.where(preceding_sources[:, None], all_expert_counts, 0), axis=0)
    global_rank = source_prefix[flat_experts] + local_rank
    cumulative = plan.cumulative_allocation[flat_experts]
    destinations = jnp.sum(global_rank[:, None] >= cumulative, axis=1, dtype=jnp.int32)
    errors = jnp.sum(destinations >= num_ranks, dtype=jnp.int32)
    return jnp.minimum(destinations, num_ranks - 1), errors


def _pack_expert_ids(expert_ids: Int[Array, " N"], token_dtype: jnp.dtype) -> jax.Array:
    token_dtype = jnp.dtype(token_dtype)
    if _EXPERT_ID_DTYPE.itemsize % token_dtype.itemsize != 0:
        raise ValueError(f"Token dtype {token_dtype} cannot carry int32 expert IDs")
    metadata_width = _EXPERT_ID_DTYPE.itemsize // token_dtype.itemsize
    return jax.lax.bitcast_convert_type(expert_ids, token_dtype).reshape(expert_ids.shape[0], metadata_width)


def _unpack_expert_ids(packed_expert_ids: jax.Array) -> Int[Array, " N"]:
    return jax.lax.bitcast_convert_type(packed_expert_ids, _EXPERT_ID_DTYPE).reshape(packed_expert_ids.shape[0])


def _exchange_remote_weights_impl(
    local_weights: jax.Array,
    experts_to_copy: Int[Array, "R B"],
    rank: Int[Array, ""],
) -> jax.Array:
    num_ranks, experts_per_rank = experts_to_copy.shape
    valid = experts_to_copy >= 0
    safe_experts = jnp.maximum(experts_to_copy, 0)
    owners = safe_experts // experts_per_rank
    needed_from_rank = jnp.logical_and(valid, owners == rank)
    input_offsets = (safe_experts % experts_per_rank).reshape(-1)
    send_sizes = needed_from_rank.reshape(-1).astype(jnp.int32)
    output_offsets = jnp.broadcast_to(
        jnp.arange(experts_per_rank, dtype=jnp.int32)[None, :], experts_to_copy.shape
    ).reshape(-1)
    receiver_owners = owners[rank]
    recv_sizes = (
        jnp.logical_and(
            valid[rank][None, :],
            receiver_owners[None, :] == jnp.arange(num_ranks, dtype=jnp.int32)[:, None],
        )
        .reshape(-1)
        .astype(jnp.int32)
    )
    receiver_weights = jnp.zeros((experts_per_rank, *local_weights.shape[1:]), dtype=local_weights.dtype)
    return jax.lax.ragged_all_to_all(
        local_weights,
        receiver_weights,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )


def _return_remote_weight_gradients(
    remote_gradients: jax.Array,
    experts_to_copy: Int[Array, "R B"],
    rank: Int[Array, ""],
) -> jax.Array:
    num_ranks, experts_per_rank = experts_to_copy.shape
    max_recv_copies = max(num_ranks + experts_per_rank - 2, 1)
    valid = experts_to_copy >= 0
    safe_experts = jnp.maximum(experts_to_copy, 0)
    owners = safe_experts // experts_per_rank
    owner_ids = jnp.arange(num_ranks, dtype=jnp.int32)[:, None, None]
    copies_by_owner = jnp.logical_and(valid[None, :, :], owners[None, :, :] == owner_ids)
    compact_positions = jnp.cumsum(copies_by_owner.reshape(num_ranks, -1), axis=1, dtype=jnp.int32) - 1
    compact_positions = compact_positions.reshape(copies_by_owner.shape)

    sender_copies = copies_by_owner[:, rank, :]
    input_offsets = jnp.broadcast_to(
        jnp.arange(experts_per_rank, dtype=jnp.int32)[None, :], sender_copies.shape
    ).reshape(-1)
    send_sizes = sender_copies.reshape(-1).astype(jnp.int32)
    output_offsets = compact_positions[:, rank, :].reshape(-1)
    recv_sizes = copies_by_owner[rank].reshape(-1).astype(jnp.int32)
    packed_gradients = jax.lax.ragged_all_to_all(
        remote_gradients,
        jnp.zeros((max_recv_copies, *remote_gradients.shape[1:]), dtype=remote_gradients.dtype),
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )

    local_indices = (safe_experts % experts_per_rank).reshape(-1)
    receiver_copies = copies_by_owner[rank].reshape(-1)
    receiver_positions = compact_positions[rank].reshape(-1)
    packed_local_indices = (
        jnp.full((max_recv_copies,), experts_per_rank, dtype=jnp.int32)
        .at[jnp.where(receiver_copies, receiver_positions, max_recv_copies)]
        .set(local_indices, mode="drop")
    )
    return (
        jnp.zeros((experts_per_rank, *remote_gradients.shape[1:]), dtype=remote_gradients.dtype)
        .at[packed_local_indices]
        .add(packed_gradients, mode="drop")
    )


@jax.custom_vjp
def _exchange_remote_weights(
    local_weights: jax.Array,
    experts_to_copy: Int[Array, "R B"],
    rank: Int[Array, ""],
) -> jax.Array:
    return _exchange_remote_weights_impl(local_weights, experts_to_copy, rank)


def _exchange_remote_weights_fwd(
    local_weights: jax.Array,
    experts_to_copy: Int[Array, "R B"],
    rank: Int[Array, ""],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return _exchange_remote_weights_impl(local_weights, experts_to_copy, rank), (experts_to_copy, rank)


def _exchange_remote_weights_bwd(
    residuals: tuple[jax.Array, jax.Array],
    remote_gradients: jax.Array,
) -> tuple[jax.Array, None, None]:
    experts_to_copy, rank = residuals
    return _return_remote_weight_gradients(remote_gradients, experts_to_copy, rank), None, None


_exchange_remote_weights.defvjp(_exchange_remote_weights_fwd, _exchange_remote_weights_bwd)


def _moe_mlp_ep_moonep_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_padding: int,
    grouped_gemm: MoonEPGroupedGemm,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run portable MoonEP with sparse expert copies and exact receiver loads."""
    if capacity_factor != 1.0:
        raise ValueError(f"MoonEP requires capacity_factor=1.0, got {capacity_factor}")
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local experts={local_experts}")

    rank = jax.lax.axis_index("expert")
    num_ranks = num_experts // local_experts
    tokens_per_rank, hidden_dim = x_local.shape
    top_k = selected_experts_local.shape[1]
    assignments_per_rank = tokens_per_rank * top_k
    receiver_capacity = assignments_per_rank + 2 * local_experts * token_padding
    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    local_expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    all_expert_counts = jax.lax.all_gather(local_expert_counts, "expert")
    plan = moon_ep_plan(all_expert_counts, token_padding=token_padding)
    destinations, destination_errors = _assignment_destinations(flat_experts, all_expert_counts, plan, rank)

    with jax.named_scope("moonep_dispatch"):
        send_order = jnp.argsort(destinations, stable=True)
        inverse_send_order = jnp.argsort(send_order)
        send_token_sources = send_order // top_k
        send_x = _dispatch_gather(
            x_local,
            send_token_sources,
            inverse_send_order,
            jnp.ones((assignments_per_rank,), dtype=jnp.bool_),
        )
        send_experts = flat_experts[send_order]
        packed_experts = _pack_expert_ids(send_experts, send_x.dtype)
        send_payload = jnp.concatenate((send_x, packed_experts), axis=1)
        local_send_sizes = jnp.bincount(destinations, length=num_ranks).astype(jnp.int32)
        send_matrix = jax.lax.all_gather(local_send_sizes, "expert")
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(send_matrix, rank)
        received_payload = jax.lax.ragged_all_to_all(
            send_payload,
            jnp.zeros((assignments_per_rank, send_payload.shape[1]), dtype=x_local.dtype),
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        received_x = received_payload[:, :hidden_dim]
        received_experts = _unpack_expert_ids(received_payload[:, hidden_dim:])

        local_start = rank * local_experts
        is_local = jnp.logical_and(received_experts >= local_start, received_experts < local_start + local_experts)
        local_group = received_experts - local_start
        remote_matches = received_experts[:, None] == plan.experts_to_copy[rank][None, :]
        has_remote_group = jnp.any(remote_matches, axis=1)
        remote_group = jnp.argmax(remote_matches, axis=1).astype(jnp.int32)
        group_ids = jnp.where(is_local, local_group, local_experts + remote_group)
        mapping_errors = jnp.sum(jnp.logical_not(jnp.logical_or(is_local, has_remote_group)), dtype=jnp.int32)
        rank_in_group = _stable_segment_rank(group_ids, num_buckets=2 * local_experts)
        receiver_positions = plan.group_offsets[rank, group_ids] + rank_in_group
        expert_inputs = jnp.zeros((receiver_capacity, hidden_dim), dtype=x_local.dtype)
        expert_inputs = expert_inputs.at[receiver_positions].set(received_x)
        expert_inputs = tree_checkpoint_name(expert_inputs, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moonep_weight_exchange"):
        remote_w13 = _exchange_remote_weights(moe_w13_local, plan.experts_to_copy, rank)
        remote_w2 = _exchange_remote_weights(moe_w2_local, plan.experts_to_copy, rank)
        group_w13 = jnp.concatenate((moe_w13_local, remote_w13), axis=0)
        group_w2 = jnp.concatenate((moe_w2_local, remote_w2), axis=0)

    with jax.named_scope("moonep_grouped_gemm"):
        group_sizes = plan.padded_group_sizes[rank]
        layout_size = jnp.sum(group_sizes, dtype=jnp.int32)
        group_sizes = group_sizes.at[-1].add(receiver_capacity - layout_size)
        if grouped_gemm == MoonEPGroupedGemm.QUACK:
            if activation_fn is not jax.nn.silu:
                raise ValueError("MoonEP QuACK grouped GEMM requires SiLU")
            if hidden_dim % _QUACK_ALIGNMENT != 0 or group_w2.shape[1] % _QUACK_ALIGNMENT != 0:
                raise ValueError(
                    f"MoonEP QuACK dimensions must be multiples of {_QUACK_ALIGNMENT}, "
                    f"got hidden={hidden_dim} and intermediate={group_w2.shape[1]}"
                )
            from levanter.grug._moe.sonic_cute import _expert_mlp, _interleave_gate_up  # noqa: PLC0415

            cumulative_group_sizes = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
            )
            interleaved_w13 = _interleave_gate_up(group_w13, group_w2.shape[1])
            expert_outputs = _expert_mlp(
                expert_inputs,
                interleaved_w13,
                group_w2,
                group_sizes,
                cumulative_group_sizes,
            )
        else:
            w13_out = tree_checkpoint_name(
                ragged_dot(expert_inputs, group_w13, group_sizes, implementation="xla"),
                _CHECKPOINT_EXPERT_HIDDEN,
            )
            gate, up = split_moe_w13_output(
                w13_out,
                intermediate_dim=group_w2.shape[1],
                interleaved=False,
            )
            expert_outputs = ragged_dot(
                activation_fn(gate) * up,
                group_w2,
                group_sizes,
                implementation="xla",
            )

    with jax.named_scope("moonep_combine"):
        outputs_in_receive_order = expert_outputs[receiver_positions]
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            send_matrix.T,
            rank,
        )
        returned = jax.lax.ragged_all_to_all(
            outputs_in_receive_order,
            jnp.zeros((assignments_per_rank, hidden_dim), dtype=expert_outputs.dtype),
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        returned = tree_checkpoint_name(returned, _CHECKPOINT_MOE_OUTPUT)
        restored = _combine_gather(
            returned,
            inverse_send_order,
            jnp.ones((assignments_per_rank,), dtype=jnp.bool_),
            send_order,
        )
        restored = restored.reshape(tokens_per_rank, top_k, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            restored,
            combine_weights_local.astype(restored.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        local_errors = destination_errors + mapping_errors
        errors = jax.lax.psum(local_errors, _batch_axes(jax.sharding.get_abstract_mesh())) + plan.violations
    return out_local, errors
