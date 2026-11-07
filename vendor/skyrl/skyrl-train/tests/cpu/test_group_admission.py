# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.rollouts.buffer import RolloutGroup


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([1.0, 1.0, 1.0, 0.0], GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD),
        ([1.0, 0.0, 1.0, 1.0], GroupSelectionResult.KEEP),
    ],
)
def test_dynamic_filter_reads_only_rows_in_the_group_baseline(rewards, expected, rollout_group_batch_factory):
    group = RolloutGroup(
        rollout_group_batch_factory(rewards, exclude_from_baseline=[False, False, False, True]), "g", 1, {}
    )
    assert GroupSelectionPolicy(DynamicSamplingType.FILTER).evaluate(group) is expected
