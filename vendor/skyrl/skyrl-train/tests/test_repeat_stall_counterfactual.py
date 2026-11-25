# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline repeat-stall counterfactual over saved trial rows."""

import json

from vendor.skyrl.scripts.repeat_stall_counterfactual import StallSimulation, simulate_trial, verdict_passed


def assistant(command):
    return {"role": "assistant", "content": json.dumps({"commands": [{"keystrokes": command, "duration": 1.0}]})}


def user(text):
    return {"role": "user", "content": text}


def row(messages, *, passed=None, stop_reason=None):
    verification = {} if passed is None else {"status": "verified", "passed": passed}
    return {"task_id": "t", "messages": messages, "verification": verification, "stop_reason": stop_reason}


def test_counterfactual_fires_at_the_third_identical_pair_and_cuts_the_rest():
    messages = []
    for index in range(6):
        messages += [assistant("cat a.txt"), user("same")]
    trial, suppressed = simulate_trial(row(messages, passed=False), 3)
    assert (trial["fired"], trial["fire_index"]) == (True, 2)
    # Turns 3..5 and their observations would not have run.
    assert trial["cut_turns"] == 3
    assert suppressed == 0


def test_counterfactual_counts_wait_suppressed_windows():
    messages = [assistant(""), user("same")] * 3
    trial, suppressed = simulate_trial(row(messages), 3)
    assert trial["fired"] is False
    assert suppressed == 1


def test_counterfactual_does_not_fire_on_changed_observations():
    messages = []
    for index in range(5):
        messages += [assistant("cat a.txt"), user(f"out {index}")]
    trial, suppressed = simulate_trial(row(messages, passed=True), 3)
    assert (trial["fired"], trial["cut_turns"]) == (False, 0)
    assert verdict_passed(row([assistant("x")], passed=True)) is True
    assert verdict_passed(row([assistant("x")])) is None
