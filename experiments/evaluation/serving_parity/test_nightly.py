# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly outcomes distinguish capacity loss from a failed score check."""

import copy

import pytest

from experiments.evaluation.serving_parity.nightly import Outcome, classify


def test_outcomes_prioritize_crashes_and_require_complete_paired_scores():
    arms = [
        dict(
            hardware=hardware,
            state="succeeded",
            started=True,
            scheduling_timeout=False,
            result=dict(
                hardware=hardware,
                model="step12",
                revision="fixed",
                score=0.95,
                items=[dict(id=i, sha256=str(i), correct=int(i < 95)) for i in range(100)],
            ),
        )
        for hardware in ("h100", "v4")
    ]
    assert classify(arms) == Outcome.PASS
    gap = copy.deepcopy(arms)
    for item in gap[1]["result"]["items"][89:95]:
        item["correct"] = 0
    assert classify(gap) == Outcome.GAP
    capacity = copy.deepcopy(arms)
    capacity[1] = dict(hardware="v4", state="unschedulable", started=False, scheduling_timeout=True)
    assert classify(capacity) == Outcome.NO_CAPACITY
    capacity[1]["started"] = True
    assert classify(capacity) == Outcome.CRASH
    capacity[1]["started"] = False
    capacity[0]["state"] = "worker_failed"
    assert classify(capacity) == Outcome.CRASH
    for broken in ([], arms[:1]):
        assert classify(broken) == Outcome.CRASH
    partial = copy.deepcopy(arms)
    partial[0]["result"]["items"].pop()
    assert classify(partial) == Outcome.CRASH
    missing = copy.deepcopy(arms)
    del missing[1]["result"]
    assert classify(missing) == Outcome.CRASH


@pytest.mark.parametrize("state", ["failed", "error", "killed", "worker_failed"])
def test_non_capacity_failures_never_skip(state):
    assert (
        classify(
            [
                dict(hardware="h100", state=state, started=False, scheduling_timeout=False),
                dict(hardware="v4", state="unschedulable", started=False, scheduling_timeout=True),
            ]
        )
        == Outcome.CRASH
    )
