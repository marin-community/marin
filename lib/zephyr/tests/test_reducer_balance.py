# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from zephyr.reducer_balance import ReducerBalancePolicy, ReduceTarget, plan_reduce_targets


def test_plan_reduce_targets_splits_only_oversized_targets_in_order():
    policy = ReducerBalancePolicy(skew_factor=2, max_slices=4, min_split_bytes=0)

    targets = plan_reduce_targets([10, 10, 35, 0], policy)

    assert targets == [
        ReduceTarget(0),
        ReduceTarget(1),
        ReduceTarget(2, 0, 4),
        ReduceTarget(2, 1, 4),
        ReduceTarget(2, 2, 4),
        ReduceTarget(2, 3, 4),
        ReduceTarget(3),
    ]


def test_plan_reduce_targets_ignores_empty_targets_for_median():
    policy = ReducerBalancePolicy(skew_factor=2, max_slices=16, min_split_bytes=0)

    targets = plan_reduce_targets([0, 10, 0, 31], policy)

    assert targets == [ReduceTarget(0), ReduceTarget(1), ReduceTarget(2), ReduceTarget(3)]


def test_plan_reduce_targets_respects_minimum_split_size_and_disabled_policy():
    policy = ReducerBalancePolicy(skew_factor=1, max_slices=16, min_split_bytes=100)

    assert plan_reduce_targets([10, 40], policy) == [ReduceTarget(0), ReduceTarget(1)]
    assert plan_reduce_targets([10, 1000], None) == [ReduceTarget(0), ReduceTarget(1)]
