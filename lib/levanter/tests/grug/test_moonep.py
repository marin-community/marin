# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from levanter.grug._moe.ep_moonep import (
    _assignment_destinations,
    _expert_order_bucket_layout,
    _token_bucket_bounds,
    moon_ep_plan,
)


def _reference_allocation(tokens_per_expert: np.ndarray) -> np.ndarray:
    num_ranks, num_experts = tokens_per_expert.shape
    experts_per_rank = num_experts // num_ranks
    capacity = int(tokens_per_expert[0].sum())
    expert_counts = tokens_per_expert.sum(axis=0)
    group_counts = expert_counts.reshape(num_ranks, experts_per_rank).sum(axis=1)
    balance = group_counts - capacity
    migration = np.zeros((num_ranks, num_ranks), dtype=np.int32)

    while balance.max() > 0:
        owner = int(balance.argmax())
        receiver = int(balance.argmin())
        move = int(-balance[receiver])
        migration[owner, receiver] = move
        balance[owner] -= move
        balance[receiver] = 0

    allocation = np.zeros((num_experts, num_ranks), dtype=np.int32)
    allocation[np.arange(num_experts), np.arange(num_experts) // experts_per_rank] = expert_counts
    for owner in range(num_ranks):
        start = owner * experts_per_rank
        remaining = expert_counts[start : start + experts_per_rank].copy()
        quotas = migration[owner].copy()
        while quotas.max() > 0:
            receiver = int(quotas.argmax())
            local_expert = int(remaining.argmax())
            take = min(int(remaining[local_expert]), int(quotas[receiver]))
            allocation[start + local_expert, receiver] += take
            allocation[start + local_expert, owner] -= take
            remaining[local_expert] -= take
            quotas[receiver] -= take
    return allocation


@pytest.mark.parametrize(
    "tokens_per_expert",
    [
        np.array(
            [
                [8, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 8, 0],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [8, 0, 0, 0, 0, 0, 0, 0],
                [4, 4, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 8, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [8, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    ],
)
def test_moonep_plan_matches_reference_and_balances_every_rank(tokens_per_expert: np.ndarray):
    plan = jax.jit(lambda counts: moon_ep_plan(counts, token_padding=4))(jnp.asarray(tokens_per_expert))
    expected = _reference_allocation(tokens_per_expert)

    np.testing.assert_array_equal(np.asarray(plan.allocation), expected)
    np.testing.assert_array_equal(np.asarray(plan.rank_loads), np.full((4,), 8, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(plan.group_sizes).sum(axis=1), np.full((4,), 8, dtype=np.int32))
    assert int(plan.violations) == 0


def test_moonep_plan_bounds_remote_experts_and_sender_weight_copies_under_full_skew():
    tokens_per_expert = jnp.zeros((4, 8), dtype=jnp.int32).at[:, 0].set(8)
    plan = moon_ep_plan(tokens_per_expert, token_padding=4)

    assert int(jnp.max(plan.remote_expert_counts)) == 1
    assert int(jnp.max(plan.weight_copy_counts)) == 3
    assert int(jnp.min(plan.padded_group_sizes)) == 4
    assert int(jnp.max(jnp.sum(plan.padded_group_sizes, axis=1))) <= 8 + 2 * 2 * 4
    assert int(plan.violations) == 0


def test_moonep_plan_remote_experts_come_from_one_owner():
    tokens_per_expert = jnp.array(
        [
            [4, 4, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 4, 0, 0],
        ],
        dtype=jnp.int32,
    )
    plan = moon_ep_plan(tokens_per_expert, token_padding=1)
    experts_per_rank = 2

    for receiver, experts in enumerate(np.asarray(plan.experts_to_copy)):
        remote = experts[experts >= 0]
        if remote.size:
            owners = remote // experts_per_rank
            assert np.unique(owners).size == 1, (receiver, remote)


def test_expert_order_bucket_layout_round_trips_each_message_slice():
    counts = jnp.array(
        [
            [3, 1, 0, 0],
            [3, 1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    plan = moon_ep_plan(counts, token_padding=1)
    flat_experts = jnp.array([0, 0, 0, 1], dtype=jnp.int32)
    send_payloads = []
    local_send_sizes = []
    for source in range(2):
        destinations, errors = _assignment_destinations(flat_experts, counts, plan, jnp.array(source))
        assert int(errors) == 0
        order = jnp.argsort(destinations * counts.shape[1] + flat_experts, stable=True)
        send_payloads.append(np.asarray((1 + source * 10 + jnp.arange(4))[order]))
        local_send_sizes.append(np.bincount(np.asarray(destinations), minlength=2))

    send_matrix = jnp.asarray(np.stack(local_send_sizes), dtype=jnp.int32)
    bucket_capacity = 4
    receiver_capacity = bucket_capacity + plan.group_experts.shape[1]

    def emulate_ragged(operands, layouts, *, use_return: bool, output_size: int):
        outputs = [np.zeros((output_size,), dtype=np.int32) for _ in range(2)]
        for source, (operand, layout) in enumerate(zip(operands, layouts, strict=True)):
            if use_return:
                input_offsets = np.asarray(layout.return_input_offsets)
                send_sizes = np.asarray(layout.return_send_sizes)
                output_offsets = np.asarray(layout.return_output_offsets)
            else:
                input_offsets = np.asarray(layout.input_offsets)
                send_sizes = np.asarray(layout.send_sizes)
                output_offsets = np.asarray(layout.output_offsets)
            slices_per_destination = send_sizes.size // 2
            for update, size in enumerate(send_sizes):
                destination = update // slices_per_destination
                input_start = input_offsets[update]
                output_start = output_offsets[update]
                outputs[destination][output_start : output_start + size] = operand[input_start : input_start + size]
        return outputs

    for bucket in range(2):
        layouts = []
        for rank in range(2):
            sizes = send_matrix[rank]
            send_offsets = jnp.cumsum(sizes, dtype=jnp.int32) - sizes
            layout = _expert_order_bucket_layout(
                counts,
                plan,
                send_offsets,
                send_matrix,
                jnp.array(rank),
                bucket=bucket,
                num_buckets=2,
                token_padding=1,
                receiver_capacity=receiver_capacity,
            )
            assert int(layout.mapping_errors) == 0
            layouts.append(layout)

        expert_order_payloads = emulate_ragged(
            send_payloads,
            layouts,
            use_return=False,
            output_size=receiver_capacity,
        )
        returned = emulate_ragged(
            expert_order_payloads,
            layouts,
            use_return=True,
            output_size=bucket_capacity,
        )

        bucket_starts, bucket_ends = _token_bucket_bounds(send_matrix, bucket=bucket, num_buckets=2)
        for source in range(2):
            expected_parts = []
            full_offset = 0
            for destination in range(2):
                start = full_offset + int(bucket_starts[source, destination])
                end = full_offset + int(bucket_ends[source, destination])
                expected_parts.append(send_payloads[source][start:end])
                full_offset += int(send_matrix[source, destination])
            expected = np.concatenate(expected_parts)
            np.testing.assert_array_equal(returned[source][: expected.size], expected)
            np.testing.assert_array_equal(returned[source][expected.size :], 0)
