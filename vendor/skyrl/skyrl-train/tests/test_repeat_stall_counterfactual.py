# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline repeat-stall counterfactual over saved trial rows."""

import json

from vendor.skyrl.scripts.repeat_stall_counterfactual import simulate_trial, summarize, verdict_passed


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


def test_counterfactual_counts_native_full_credit_without_optional_pass_flag():
    saved = row([assistant("cat a.txt"), user("same")] * 6)
    saved["verification"] = {"status": "verified", "score": 1.0, "passed": None}
    trial, suppressed = simulate_trial(saved, 3)
    report = summarize("tb2", [trial], suppressed, 3, 30)
    assert report["stopped_trials_with_passing_verdict"] == 1
    assert report["at_risk_passes_cut_before_their_last_turn"] == 1


def test_hand_check_sample_identifies_the_selected_repetition_of_a_task():
    identity = {"uid": "task-group", "attempt_id": "eval-attempt"}
    short = {**row([assistant("cat a.txt"), user("same")] * 3), **identity, "ordinal": 0}
    long = {**row([assistant("cat a.txt"), user("same")] * 6), **identity, "ordinal": 1}
    simulations = [simulate_trial(saved, 3)[0] for saved in (short, long)]

    report = summarize("tb2", simulations, 0, 3, 1)

    selected = report["hand_check_sample"][0]
    assert (selected["attempt_id"], selected["uid"], selected["ordinal"]) == ("eval-attempt", "task-group", 1)
    assert report["hand_check_sample"][0]["cut_turns"] == 3


def test_stopped_pass_rate_excludes_unknown_verifier_outcomes():
    messages = [assistant("cat a.txt"), user("same")] * 3
    trials = [simulate_trial(row(messages, passed=passed), 3)[0] for passed in (True, False, None)]
    for exception in ("TaskMachineError", "AgentTimeoutError"):
        saved = {**row(messages, passed=True), "exception_type": exception}
        trials.append(simulate_trial(saved, 3)[0])

    report = summarize("tb2", trials, 0, 3, 30)

    assert report["fired_trials"] == 5
    assert report["stopped_trials_with_passing_verdict"] == 2
    assert report["stopped_trials_with_unknown_verdict"] == 2
    assert report["stopped_trials_pass_rate"] == 2 / 3
