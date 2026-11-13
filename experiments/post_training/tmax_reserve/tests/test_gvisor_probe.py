# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.tmax_reserve.gvisor_probe import summarize


def _row(index, start, **overrides):
    row = {"index": index, "job_id": f"/p/s{index}", "start_seconds": start, "first_exec_seconds": 0.1}
    row.update(setsid=True, tmux=True, pytest=True, egress_blocked=True)
    row.update(overrides)
    return row


BUILDS = [{"task": "t", "succeeded": True}]


def test_probe_passes_only_when_every_rule_holds():
    rounds = [[_row(0, 10.0), _row(1, 20.0)]]
    terminal = {"/p/s0": True, "/p/s1": True}
    assert summarize(rounds, terminal, BUILDS, max_p90=120)["passed"]

    failed_tmux = summarize([[_row(0, 10.0), _row(1, 20.0, tmux=False)]], terminal, BUILDS, max_p90=120)
    assert not failed_tmux["passed"] and failed_tmux["checks_passed"]["tmux"] == 1

    slow = summarize([[_row(0, 10.0), _row(1, 300.0)]], terminal, BUILDS, max_p90=120)
    assert not slow["passed"] and slow["latency_seconds"][0]["max"] == 300.0

    leaked = summarize(rounds, {"/p/s0": True, "/p/s1": False}, BUILDS, max_p90=120)
    assert not leaked["passed"] and leaked["jobs_not_terminal_after_close"] == ["/p/s1"]

    broken_build = summarize(rounds, terminal, [{"task": "t", "succeeded": False}], max_p90=120)
    assert not broken_build["passed"]
