# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.capability_pool.allocation import (
    InfeasiblePolicy,
    InfeasibleTargetError,
    Supply,
    allocate,
)

ARMS = {"control": 0.0, "taskforge30": 0.3}


def test_arms_share_node_quotas_and_split_taskforge_per_node():
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    supply = {node: Supply(tasktrove=1000, taskforge=100) for node in weights}

    allocation = allocate(weights, supply, 101, ARMS, InfeasiblePolicy.ERROR)

    assert allocation.quotas == {"a": 51, "b": 30, "c": 20}
    assert all(count == 0 for count in allocation.taskforge["control"].values())
    treatment = allocation.taskforge["taskforge30"]
    assert sum(treatment.values()) == 30
    for node, quota in allocation.quotas.items():
        assert abs(treatment[node] - 0.3 * quota) < 1
        assert allocation.tasktrove("taskforge30", node) == quota - treatment[node]


def test_remainder_units_go_to_nodes_with_supply():
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    supply = {"a": Supply(3, 3), "b": Supply(10, 10), "c": Supply(10, 10)}

    allocation = allocate(weights, supply, 10, {"all_taskforge": 1.0}, InfeasiblePolicy.ERROR)

    assert allocation.quotas["a"] == 3
    assert sum(allocation.quotas.values()) == 10
    assert allocation.taskforge["all_taskforge"] == allocation.quotas


def test_missing_taskforge_supply_fails_with_the_largest_feasible_pool():
    weights = {"a": 0.5, "b": 0.5}
    supply = {"a": Supply(tasktrove=500, taskforge=50), "b": Supply(tasktrove=500, taskforge=0)}

    with pytest.raises(InfeasibleTargetError) as raised:
        allocate(weights, supply, 100, ARMS, InfeasiblePolicy.ERROR)

    assert [(item.node_id, item.source, item.arm) for item in raised.value.shortfalls] == [
        ("b", "taskforge", "taskforge30")
    ]
    largest = raised.value.max_pool_size
    assert largest > 0
    assert allocate(weights, supply, largest, ARMS, InfeasiblePolicy.ERROR).taskforge["taskforge30"]["b"] == 0


def test_drop_policy_removes_unfillable_nodes_and_renormalizes():
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}
    supply = {"a": Supply(500, 100), "b": Supply(500, 0), "c": Supply(10, 100)}

    allocation = allocate(weights, supply, 100, ARMS, InfeasiblePolicy.DROP_NODES)

    assert allocation.dropped == ("b", "c")
    assert allocation.quotas == {"a": 100}
    assert allocation.weights == {"a": 1.0}
    assert allocation.taskforge["taskforge30"] == {"a": 30}
