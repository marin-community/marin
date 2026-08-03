# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable JAX implementation of MoonEP expert planning and transport.

The allocation policy follows the MIT-licensed MoonEP planning reference at
https://github.com/moonshotAI/moonep/tree/0f385f038fc33bec22e3bcf5a07a8a22693e754c.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
    MoonEPBucketSchedule,
    MoonEPGroupedGemm,
    MoonEPMode,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import _shard_a2a_params
from levanter.grug._moe.ep_fixed_all_to_all import (
    _combine_gather,
    _dispatch_gather,
    _moe_mlp_ep_fixed_a2a_local,
)
from levanter.grug._moe.sonic import sonic_row_gather
from levanter.grug.sharding import _batch_axes

_QUACK_ALIGNMENT = 128
_MOONEP_DISPATCH_COMPUTE_GROUP_BASE = 789_000
_MOONEP_COMBINE_COMPUTE_GROUP_BASE = 790_000


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


class _MoonEPDispatchedBucket(NamedTuple):
    received_x: Float[Array, "Tb H"]
    received_experts: Int[Array, " Tb"]
    received_valid: jax.Array
    send_matrix: Int[Array, "R R"]
    starts: Int[Array, "R R"]


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

    owner_ids = jnp.arange(num_ranks, dtype=jnp.int32)
    remaining = jnp.sum(allocation, axis=1, dtype=jnp.int32).reshape(num_ranks, experts_per_rank)

    def _move_experts(_iteration, state):
        allocation_state, remaining_state, quota_state = state
        receivers = jnp.argmax(quota_state, axis=1)
        local_experts = jnp.argmax(remaining_state, axis=1)
        quotas = quota_state[owner_ids, receivers]
        expert_counts = remaining_state[owner_ids, local_experts]
        active = jnp.logical_and(quotas > 0, expert_counts > 0)
        take = jnp.where(active, jnp.minimum(quotas, expert_counts), 0)
        experts = owner_ids * experts_per_rank + local_experts
        allocation_state = allocation_state.at[experts, receivers].add(take)
        allocation_state = allocation_state.at[experts, owner_ids].add(-take)
        remaining_state = remaining_state.at[owner_ids, local_experts].add(-take)
        quota_state = quota_state.at[owner_ids, receivers].add(-take)
        return allocation_state, remaining_state, quota_state

    max_moves = num_ranks + experts_per_rank - 2
    allocation, _, _ = jax.lax.fori_loop(
        0,
        max_moves,
        _move_experts,
        (allocation, remaining, migration),
    )
    return allocation


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


def _received_expert_ids(
    all_expert_counts: Int[Array, "R E"],
    plan: MoonEPPlan,
    rank: Int[Array, ""],
    source_slice_starts: Int[Array, " R"],
    source_slice_ends: Int[Array, " R"],
    *,
    output_capacity: int,
) -> tuple[Int[Array, " N"], Array, Int[Array, "R E"]]:
    source_ends = jnp.cumsum(all_expert_counts, axis=0, dtype=jnp.int32)
    source_starts = source_ends - all_expert_counts
    destination_ends = plan.cumulative_allocation[:, rank]
    destination_starts = destination_ends - plan.allocation[:, rank]
    # Each count is the overlap of one source interval and one destination interval.
    counts_by_source_expert = jnp.maximum(
        jnp.minimum(source_ends, destination_ends[None, :]) - jnp.maximum(source_starts, destination_starts[None, :]),
        0,
    )
    message_expert_ends = jnp.cumsum(counts_by_source_expert, axis=1, dtype=jnp.int32)
    message_expert_starts = message_expert_ends - counts_by_source_expert
    sliced_counts = jnp.maximum(
        jnp.minimum(message_expert_ends, source_slice_ends[:, None])
        - jnp.maximum(message_expert_starts, source_slice_starts[:, None]),
        0,
    )
    expert_ids = jnp.tile(
        jnp.arange(all_expert_counts.shape[1], dtype=jnp.int32),
        all_expert_counts.shape[0],
    )
    received_experts = jnp.repeat(
        expert_ids,
        sliced_counts.reshape(-1),
        total_repeat_length=output_capacity,
    )
    received_count = jnp.sum(sliced_counts, dtype=jnp.int32)
    valid = jnp.arange(output_capacity, dtype=jnp.int32) < received_count
    return received_experts, valid, sliced_counts


def _token_bucket_bounds(
    send_matrix: Int[Array, "R R"],
    *,
    bucket: int,
    num_buckets: int,
) -> tuple[Int[Array, "R R"], Int[Array, "R R"]]:
    starts = send_matrix * bucket // num_buckets
    ends = send_matrix * (bucket + 1) // num_buckets
    return starts, ends


def _ordered_after(dependency: jax.Array, value: jax.Array) -> jax.Array:
    """Keep ``value`` unchanged, but schedule its consumer after ``dependency``."""
    _, ordered_value = jax.lax.optimization_barrier((dependency, value))
    return ordered_value


def _scheduled_ragged_all_to_all_impl(
    operand: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    output_rows: int,
    axis_name: str,
    scheduling_group_id: int,
) -> jax.Array:
    output = jnp.zeros((output_rows, *operand.shape[1:]), dtype=operand.dtype)
    with set_xla_metadata(_scheduling_group_id=scheduling_group_id):
        return jax.lax.ragged_all_to_all(
            operand,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=axis_name,
        )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _scheduled_ragged_all_to_all(
    operand: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    output_rows: int,
    axis_name: str,
    scheduling_group_id: int,
) -> jax.Array:
    """Run an annotated forward collective with an unannotated transpose."""
    return _scheduled_ragged_all_to_all_impl(
        operand,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        output_rows=output_rows,
        axis_name=axis_name,
        scheduling_group_id=scheduling_group_id,
    )


def _scheduled_ragged_all_to_all_fwd(
    operand,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    output_rows,
    axis_name,
    scheduling_group_id,
):
    result = _scheduled_ragged_all_to_all_impl(
        operand,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        output_rows=output_rows,
        axis_name=axis_name,
        scheduling_group_id=scheduling_group_id,
    )
    residuals = (input_offsets, send_sizes, output_offsets, recv_sizes, operand.shape[0])
    return result, residuals


def _scheduled_ragged_all_to_all_bwd(
    _output_rows,
    axis_name,
    _scheduling_group_id,
    residuals,
    output_gradient,
):
    input_offsets, send_sizes, output_offsets, recv_sizes, input_rows = residuals
    transpose_output_offsets = jax.lax.all_to_all(output_offsets, axis_name, 0, 0, tiled=True)
    transpose_input_offsets = jax.lax.all_to_all(input_offsets, axis_name, 0, 0, tiled=True)
    operand_gradient = jax.lax.ragged_all_to_all(
        output_gradient,
        jnp.zeros((input_rows, *output_gradient.shape[1:]), dtype=output_gradient.dtype),
        transpose_output_offsets,
        recv_sizes,
        transpose_input_offsets,
        send_sizes,
        axis_name=axis_name,
    )
    return operand_gradient, None, None, None, None


_scheduled_ragged_all_to_all.defvjp(_scheduled_ragged_all_to_all_fwd, _scheduled_ragged_all_to_all_bwd)


def _bucket_assignment_ids(
    send_order: Int[Array, " N"],
    send_sizes: Int[Array, " R"],
    bucket_starts: Int[Array, " R"],
    bucket_sizes: Int[Array, " R"],
    *,
    output_capacity: int,
) -> Int[Array, " C"]:
    assignments = send_order.shape[0]
    send_offsets = jnp.cumsum(send_sizes, dtype=jnp.int32) - send_sizes
    bucket_offsets = jnp.cumsum(bucket_sizes, dtype=jnp.int32) - bucket_sizes
    bucket_ends = bucket_offsets + bucket_sizes
    positions = jnp.arange(output_capacity, dtype=jnp.int32)
    destinations = jnp.searchsorted(bucket_ends, positions, side="right")
    safe_destinations = jnp.minimum(destinations, send_sizes.shape[0] - 1)
    positions_in_message = positions - bucket_offsets[safe_destinations]
    sorted_positions = send_offsets[safe_destinations] + bucket_starts[safe_destinations] + positions_in_message
    valid = positions < jnp.sum(bucket_sizes, dtype=jnp.int32)
    safe_sorted_positions = jnp.minimum(sorted_positions, assignments - 1)
    return jnp.where(valid, send_order[safe_sorted_positions], assignments)


def _moonep_grouped_mlp(
    expert_inputs: jax.Array,
    group_w13: jax.Array,
    group_w2: jax.Array,
    group_sizes: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    grouped_gemm: MoonEPGroupedGemm,
    scheduling_group_id: int | None,
) -> jax.Array:
    hidden_dim = expert_inputs.shape[1]
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
        return _expert_mlp(
            expert_inputs,
            interleaved_w13,
            group_w2,
            group_sizes,
            cumulative_group_sizes,
            scheduling_group_id,
        )

    w13_out = tree_checkpoint_name(
        ragged_dot(expert_inputs, group_w13, group_sizes, implementation="xla"),
        _CHECKPOINT_EXPERT_HIDDEN,
    )
    gate, up = split_moe_w13_output(
        w13_out,
        intermediate_dim=group_w2.shape[1],
        interleaved=False,
    )
    return ragged_dot(
        activation_fn(gate) * up,
        group_w2,
        group_sizes,
        implementation="xla",
    )


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


def _moe_mlp_ep_moonep_exact_local(
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
    token_buckets: int,
    bucket_schedule: MoonEPBucketSchedule,
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
    num_groups = 2 * local_experts
    bucket_capacity = (
        assignments_per_rank
        if token_buckets == 1
        else (assignments_per_rank + token_buckets - 1) // token_buckets + num_ranks
    )
    receiver_capacity = bucket_capacity + num_groups * token_padding
    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    local_expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    all_expert_counts = jax.lax.all_gather(local_expert_counts, "expert")
    plan = moon_ep_plan(all_expert_counts, token_padding=token_padding)
    destinations, destination_errors = _assignment_destinations(flat_experts, all_expert_counts, plan, rank)

    send_order = jnp.argsort(destinations * num_experts + flat_experts, stable=True)
    inverse_send_order = jnp.argsort(send_order)
    send_token_sources = send_order // top_k
    send_x = _dispatch_gather(
        x_local,
        send_token_sources,
        inverse_send_order,
        jnp.ones((assignments_per_rank,), dtype=jnp.bool_),
    )
    local_send_sizes = jnp.bincount(destinations, length=num_ranks).astype(jnp.int32)
    send_matrix = jax.lax.all_gather(local_send_sizes, "expert")
    send_offsets = jnp.cumsum(local_send_sizes, dtype=jnp.int32) - local_send_sizes

    with jax.named_scope("moonep_weight_exchange"):
        remote_w13 = _exchange_remote_weights(moe_w13_local, plan.experts_to_copy, rank)
        remote_w2 = _exchange_remote_weights(moe_w2_local, plan.experts_to_copy, rank)
        group_w13 = jnp.concatenate((moe_w13_local, remote_w13), axis=0)
        group_w2 = jnp.concatenate((moe_w2_local, remote_w2), axis=0)

    # In the overlap schedule, each dispatch after the first waits for the
    # prior dispatch output. The next dispatch and prior bucket GEMM then become
    # ready together.
    dispatched_buckets = []
    for bucket in range(token_buckets):
        bucket_starts, bucket_ends = _token_bucket_bounds(
            send_matrix,
            bucket=bucket,
            num_buckets=token_buckets,
        )
        bucket_send_matrix = bucket_ends - bucket_starts

        with jax.named_scope(f"moonep_dispatch_bucket_{bucket}"):
            _, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(bucket_send_matrix, rank)
            input_offsets = send_offsets + bucket_starts[rank]
            if bucket_schedule == MoonEPBucketSchedule.COMPUTE_OVERLAP and dispatched_buckets:
                input_offsets = _ordered_after(dispatched_buckets[-1].received_x[0, 0], input_offsets)

            if bucket_schedule == MoonEPBucketSchedule.COMPUTE_OVERLAP and bucket > 0:
                received_x = _scheduled_ragged_all_to_all(
                    send_x,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    bucket_capacity,
                    "expert",
                    _MOONEP_DISPATCH_COMPUTE_GROUP_BASE + bucket - 1,
                )
            else:
                received_x = jax.lax.ragged_all_to_all(
                    send_x,
                    jnp.zeros((bucket_capacity, hidden_dim), dtype=x_local.dtype),
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name="expert",
                )
            received_experts, received_valid, _ = _received_expert_ids(
                all_expert_counts,
                plan,
                rank,
                bucket_starts[:, rank],
                bucket_ends[:, rank],
                output_capacity=bucket_capacity,
            )

        dispatched_buckets.append(
            _MoonEPDispatchedBucket(
                received_x=received_x,
                received_experts=received_experts,
                received_valid=received_valid,
                send_matrix=bucket_send_matrix,
                starts=bucket_starts,
            )
        )

    returned_buckets = []
    assignment_id_buckets = []
    mapping_errors = jnp.array(0, dtype=jnp.int32)
    for bucket, dispatched in enumerate(dispatched_buckets):
        received_x = dispatched.received_x
        received_experts = dispatched.received_experts
        received_valid = dispatched.received_valid
        bucket_send_matrix = dispatched.send_matrix
        bucket_starts = dispatched.starts

        with jax.named_scope(f"moonep_layout_bucket_{bucket}"):
            local_start = rank * local_experts
            is_local = jnp.logical_and(
                received_experts >= local_start,
                received_experts < local_start + local_experts,
            )
            local_group = received_experts - local_start
            remote_matches = received_experts[:, None] == plan.experts_to_copy[rank][None, :]
            has_remote_group = jnp.any(remote_matches, axis=1)
            remote_group = jnp.argmax(remote_matches, axis=1).astype(jnp.int32)
            mapped_group_ids = jnp.where(is_local, local_group, local_experts + remote_group)
            group_ids = jnp.where(received_valid, mapped_group_ids, num_groups - 1)
            mapping_errors = mapping_errors + jnp.sum(
                jnp.logical_and(received_valid, jnp.logical_not(jnp.logical_or(is_local, has_remote_group))),
                dtype=jnp.int32,
            )
            rank_in_group = _stable_segment_rank(group_ids, num_buckets=num_groups)
            group_sizes = jnp.bincount(
                group_ids,
                weights=received_valid.astype(jnp.int32),
                length=num_groups,
            ).astype(jnp.int32)
            padded_group_sizes = jnp.maximum(
                ((group_sizes + token_padding - 1) // token_padding) * token_padding,
                token_padding,
            )
            group_ends = jnp.cumsum(padded_group_sizes, dtype=jnp.int32)
            group_offsets = group_ends - padded_group_sizes
            receiver_positions = group_offsets[group_ids] + rank_in_group
            safe_receiver_positions = jnp.minimum(receiver_positions, receiver_capacity - 1)
            expert_inputs = jnp.zeros((receiver_capacity, hidden_dim), dtype=x_local.dtype)
            expert_inputs = expert_inputs.at[
                jnp.where(received_valid, safe_receiver_positions, receiver_capacity)
            ].set(received_x, mode="drop")
            expert_inputs = tree_checkpoint_name(expert_inputs, _CHECKPOINT_DISPATCH_INPUT)

        with jax.named_scope(f"moonep_grouped_gemm_bucket_{bucket}"):
            layout_size = jnp.sum(padded_group_sizes, dtype=jnp.int32)
            compute_group_sizes = padded_group_sizes.at[-1].add(receiver_capacity - layout_size)
            # Each compute bucket overlaps one collective. Earlier buckets hide
            # the next dispatch; the final bucket hides the prior combine.
            scheduling_group_id = None
            if bucket_schedule == MoonEPBucketSchedule.COMPUTE_OVERLAP:
                if bucket + 1 < token_buckets:
                    scheduling_group_id = _MOONEP_DISPATCH_COMPUTE_GROUP_BASE + bucket
                elif bucket > 0:
                    scheduling_group_id = _MOONEP_COMBINE_COMPUTE_GROUP_BASE + bucket - 1
            expert_outputs = _moonep_grouped_mlp(
                expert_inputs,
                group_w13,
                group_w2,
                compute_group_sizes,
                activation_fn=activation_fn,
                grouped_gemm=grouped_gemm,
                scheduling_group_id=scheduling_group_id,
            )

        with jax.named_scope(f"moonep_combine_bucket_{bucket}"):
            if grouped_gemm == MoonEPGroupedGemm.QUACK and bucket_schedule == MoonEPBucketSchedule.COMPUTE_OVERLAP:
                outputs_in_receive_order = sonic_row_gather(
                    expert_outputs,
                    safe_receiver_positions,
                    received_valid,
                )
            else:
                outputs_in_receive_order = jnp.where(
                    received_valid[:, None],
                    expert_outputs[safe_receiver_positions],
                    0,
                )
            return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
                bucket_send_matrix.T,
                rank,
            )

            if bucket_schedule == MoonEPBucketSchedule.COMPUTE_OVERLAP and bucket + 2 == token_buckets:
                returned = _scheduled_ragged_all_to_all(
                    outputs_in_receive_order,
                    return_input_offsets,
                    return_send_sizes,
                    return_output_offsets,
                    return_recv_sizes,
                    bucket_capacity,
                    "expert",
                    _MOONEP_COMBINE_COMPUTE_GROUP_BASE + bucket,
                )
            else:
                returned = jax.lax.ragged_all_to_all(
                    outputs_in_receive_order,
                    jnp.zeros((bucket_capacity, hidden_dim), dtype=expert_outputs.dtype),
                    return_input_offsets,
                    return_send_sizes,
                    return_output_offsets,
                    return_recv_sizes,
                    axis_name="expert",
                )
            returned_buckets.append(tree_checkpoint_name(returned, _CHECKPOINT_MOE_OUTPUT))
            assignment_id_buckets.append(
                _bucket_assignment_ids(
                    send_order,
                    local_send_sizes,
                    bucket_starts[rank],
                    bucket_send_matrix[rank],
                    output_capacity=bucket_capacity,
                )
            )

    with jax.named_scope("moonep_combine"):
        returned = jnp.concatenate(returned_buckets, axis=0)
        assignment_sources = jnp.concatenate(assignment_id_buckets, axis=0)
        returned_capacity = returned.shape[0]
        assignment_positions = (
            jnp.full((assignments_per_rank,), returned_capacity, dtype=jnp.int32)
            .at[assignment_sources]
            .set(jnp.arange(returned_capacity, dtype=jnp.int32), mode="drop")
        )
        keep = assignment_positions < returned_capacity
        restored = _combine_gather(
            returned,
            jnp.minimum(assignment_positions, returned_capacity - 1),
            keep,
            assignment_sources,
        )
        restored = restored.reshape(tokens_per_rank, top_k, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            restored,
            combine_weights_local.astype(restored.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        local_errors = destination_errors + mapping_errors + jnp.sum(jnp.logical_not(keep), dtype=jnp.int32)
        errors = jax.lax.psum(local_errors, _batch_axes(jax.sharding.get_abstract_mesh())) + plan.violations
    return out_local, errors


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
    token_buckets: int,
    bucket_schedule: MoonEPBucketSchedule,
    grouped_gemm: MoonEPGroupedGemm,
    mode: MoonEPMode,
    fixed_capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run the selected static MoonEP schedule."""
    if fixed_capacity_factor < 1.0:
        raise ValueError(f"fixed_capacity_factor must be at least 1.0, got {fixed_capacity_factor}")
    if mode == MoonEPMode.QB_FIXED:
        return _moe_mlp_ep_fixed_a2a_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=fixed_capacity_factor,
        )
    if mode == MoonEPMode.EXACT:
        return _moe_mlp_ep_moonep_exact_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            token_padding=token_padding,
            token_buckets=token_buckets,
            bucket_schedule=bucket_schedule,
            grouped_gemm=grouped_gemm,
        )
    raise AssertionError(f"Unhandled MoonEP mode {mode!r}")
