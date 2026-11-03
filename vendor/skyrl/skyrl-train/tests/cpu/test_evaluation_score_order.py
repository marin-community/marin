# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations

import pytest

from skyrl_train.trajectory_runners.trajectory_processing import get_metrics_from_trajectory_batch


@pytest.mark.parametrize("per_token", [False, True])
def test_evaluation_mean_survives_cancellation_and_completion_order(per_token):
    means = set()
    for values in permutations((1e16, 1.0, -1e16, 2.0)):
        rewards = [list(values)] if per_token else list(values)
        batch = {"rewards": rewards, "unshaped_rewards": [1.0] * len(rewards)}
        mean, passed = get_metrics_from_trajectory_batch(batch, [str(i) for i in range(len(rewards))])
        means.add(mean)
        assert passed == 1.0
    assert means == ({3.0} if per_token else {0.75})
