# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import statistics

import pytest

from .review_terminal_bench import Aggregate, failed_verifier_setup, remove_rewards


def test_verifier_failures_are_removed_from_reward_mean_and_sem():
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    original = Aggregate(len(rewards), statistics.mean(rewards), statistics.stdev(rewards) / math.sqrt(len(rewards)))
    result = remove_rewards(original, [0.0, 0.0, 0.0])
    valid = rewards[:-3]
    assert result.count == len(valid)
    assert result.mean == pytest.approx(statistics.mean(valid), abs=1e-12)
    assert result.sem == pytest.approx(statistics.stdev(valid) / math.sqrt(len(valid)), abs=1e-12)


def test_verifier_failure_does_not_turn_a_genuine_test_failure_into_infrastructure():
    bootstrap_error = "404 Not Found\ncurl: command not found\nuvx: command not found"
    assert failed_verifier_setup(bootstrap_error, "")
    assert not failed_verifier_setup(bootstrap_error + "\n1 failed, 5 passed", "")
    assert not failed_verifier_setup("1 failed, 5 passed", "")
