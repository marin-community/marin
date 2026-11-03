# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evaluation.serving_parity.nightly import classify


@pytest.mark.parametrize(
    "states,score_exit,expected",
    [
        (["succeeded", "succeeded"], 0, "pass"),
        (["succeeded", "succeeded"], 1, "gap"),
        (["succeeded", "succeeded"], 2, "crash"),
        (["succeeded", "no_capacity"], None, "no_capacity"),
        (["worker_failed", "no_capacity"], None, "crash"),
        (["succeeded", "unschedulable"], None, "crash"),
    ],
)
def test_nightly_outcome(states, score_exit, expected):
    assert classify(states, score_exit) == expected
