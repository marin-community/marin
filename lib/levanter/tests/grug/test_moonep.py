# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from levanter.grug._moe.ep_moonep import moon_ep_plan


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
