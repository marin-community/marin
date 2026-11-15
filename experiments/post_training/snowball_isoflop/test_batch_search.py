# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import unittest.mock
from dataclasses import asdict
from types import SimpleNamespace

from iris.client.client import JobFailedError
from iris.cluster.types import JobName

from experiments.post_training.snowball_isoflop import optimizers, run_probes
from experiments.post_training.snowball_isoflop.probes import Layout, Probe, batch_probes


def test_cp8_batch_search_brackets_accumulation_after_microbatch_oom(monkeypatch):
    observed = []
    rates = iter((100.0, 200.0, 201.0, 202.0))

    def measurement(probe: Probe) -> dict:
        observed.append((probe.microbatch, probe.accumulation))
        if probe.microbatch == 4:
            return {"probe": asdict(probe), "status": "memory_limit"}
        return {
            "probe": asdict(probe),
            "status": "complete",
            "measurement": {
                "minimum_memory_headroom_fraction": 0.2,
                "inclusive_positions_per_second_per_gpu": next(rates),
                "sender_dropped": 0,
                "receiver_dropped": 0,
            },
        }

    monkeypatch.setattr(run_probes, "run_probe", measurement)
    candidates = batch_probes(512, Layout(context=8, data=1))[1:]
    result = run_probes.saturation_search(512, candidates)

    assert observed == [(1, 8), (2, 8), (4, 8), (2, 16), (2, 32)]
    assert result["reason"] == "two_consecutive_gains_below_5_percent"
    assert result["accumulation_growth_used"]
    assert result["selected"]["microbatch"] == 2
    assert result["selected"]["accumulation"] == 8
    assert result["selected"]["layout"]["context"] == 8


def test_cp8_batch_search_brackets_accumulation_after_first_update_stall(monkeypatch):
    observed = []
    rates = iter((100.0, 180.0, 181.0, 182.0))

    def measurement(probe: Probe) -> dict:
        observed.append((probe.microbatch, probe.accumulation))
        if probe.microbatch == 2:
            return {"probe": asdict(probe), "status": "stall_limit", "error": "watchdog"}
        return {
            "probe": asdict(probe),
            "status": "complete",
            "measurement": {
                "minimum_memory_headroom_fraction": 0.2,
                "inclusive_positions_per_second_per_gpu": next(rates),
                "sender_dropped": 0,
                "receiver_dropped": 0,
            },
        }

    monkeypatch.setattr(run_probes, "run_probe", measurement)
    candidates = batch_probes(512, Layout(context=8, data=1))[1:]
    result = run_probes.saturation_search(512, candidates)

    assert observed == [(1, 8), (2, 8), (1, 16), (1, 32), (1, 64)]
    assert result["reason"] == "two_consecutive_gains_below_5_percent"
    assert result["accumulation_growth_used"]
    assert result["selected"]["microbatch"] == 1
    assert result["selected"]["accumulation"] == 16
    assert result["selected"]["layout"]["context"] == 8


def test_layout_throughput_override_runs_degenerate_finalist_set(monkeypatch):
    selected = Probe("B", 512, 1, 8, layout=Layout(context=8, data=1))
    monkeypatch.setattr(
        run_probes,
        "read_json",
        lambda path: {"result": {"selected": asdict(selected)}},
    )
    rates = iter((100.0, 120.0))

    def measurement(probe: Probe) -> dict:
        return {
            "probe": asdict(probe),
            "status": "complete",
            "measurement": {
                "minimum_memory_headroom_fraction": 0.2,
                "inclusive_positions_per_second_per_gpu": next(rates),
                "sender_dropped": 0,
                "receiver_dropped": 0,
            },
        }

    monkeypatch.setattr(run_probes, "run_probe", measurement)
    result = run_probes.layout_throughput(
        512,
        {512: (Layout(context=8, data=1), Layout(context=8, data=1, dispatch="ring_gather_combine"))},
    )

    assert [probe["layout"]["dispatch"] for probe in result["finalists"]] == ["sonic", "ring_gather_combine"]
    assert result["degenerate_finalists"]
    assert result["layout_override_used"]
    assert result["finalists"][0]["accumulation"] == 8


def test_layout_throughput_override_rejects_undivided_batch(monkeypatch):
    selected = Probe("B", 512, 2, 4, layout=Layout(context=8, data=1))
    monkeypatch.setattr(run_probes, "read_json", lambda path: {"result": {"selected": asdict(selected)}})
    try:
        run_probes.layout_throughput(512, {512: (Layout(context=1, data=8),)})
    except ValueError as error:
        assert "does not divide" in str(error)
    else:
        raise AssertionError("undivided override layout was accepted")


def test_failure_classification_reads_aggregate_status_without_task_detail():
    """GetJobStatus payloads omit task/attempt detail; the watchdog signature
    must be classified from the terminal error text instead."""
    # Shape observed live for a stalled D child: aggregate status with no
    # task detail and a zero aggregate exit code, but the watchdog text in
    # the terminal error message.
    status = SimpleNamespace(
        state=SimpleNamespace(name="FAILED"),
        tasks=(),
        error_message=(
            "Cumulative failed task attempts exceeded max_task_failures: failures=1, limit=0; "
            "levanter.callbacks.progress_watchdog No progress after process_started for 1800.0 "
            "seconds (timeout 1800.0); terminating with exit code 124"
        ),
        status_message="",
        exit_code=0,
    )
    error = JobFailedError(JobName.from_wire("/benfeuer/snowball-isoflop-D-r5-v10/child"), status)
    signals, watchdog_exit = run_probes._failure_signals(error)
    assert watchdog_exit is False
    assert run_probes.classify_probe_failure(signals, "D-r5", watchdog_exit) == "stall_limit"
    # OOM text still classifies as a memory limit, and unrelated failures
    # remain unclassified so they propagate.
    oom = SimpleNamespace(
        state=SimpleNamespace(name="FAILED"),
        tasks=(),
        error_message="RESOURCE_EXHAUSTED: out of memory",
        status_message="",
        exit_code=1,
    )
    signals, _ = run_probes._failure_signals(JobFailedError(JobName.from_wire("/benfeuer/j"), oom))
    assert run_probes.classify_probe_failure(signals, "D-r5", False) == "memory_limit"
    other = SimpleNamespace(
        state=SimpleNamespace(name="FAILED"),
        tasks=(),
        error_message="segfault at pc 0xdeadbeef",
        status_message="",
        exit_code=139,
    )
    signals, _ = run_probes._failure_signals(JobFailedError(JobName.from_wire("/benfeuer/j"), other))
    assert run_probes.classify_probe_failure(signals, "D-r5", False) is None


def _stall_report(probe):
    return {"status": "stall_limit", "error": "No progress after process_started", "probe": asdict(probe)}


def _complete_report(probe):
    return {"status": "complete", "error": None, "probe": asdict(probe)}


def _memory_report(probe):
    return {"status": "memory_limit", "error": "RESOURCE_EXHAUSTED", "probe": asdict(probe)}


def test_lr_trials_retry_stalls_under_fresh_identities(monkeypatch):
    """A stalled LR candidate retries up to two more times under fresh
    identities and completes; the completed report is kept."""

    selected = Probe("D", 512, 1, 128, layout=Layout(context=8, data=1), optimizer="adam", learning_rate=3.84615e-06)
    outcomes = iter([_stall_report, _stall_report, _complete_report])
    phases = []

    def fake_run_probe(probe):
        phases.append(probe.cell.phase)
        return next(outcomes)(probe)

    monkeypatch.setattr(optimizers, "run_probe", fake_run_probe)
    monkeypatch.setattr(optimizers, "lr_candidates", lambda size, optimizer: (3.84615e-06,))
    reports, exclusions = optimizers.run_lr_trials(selected, "adam", "D-r7")
    assert len(reports) == 1 and reports[0]["status"] == "complete"
    assert exclusions == []
    # Each retry gets a fresh cell identity under its attempt-scoped phase.
    assert [
        phase.startswith(prefix) for phase, prefix in zip(phases, ("D-r7-", "D-r7-retry1-", "D-r7-retry2-"), strict=True)
    ] == [
        True,
        True,
        True,
    ]
    assert len(set(phases)) == 3
    assert optimizers.trial_attempt_phase("D-r7", 2) == "D-r7-retry2"


def test_lr_trials_excludes_candidate_after_three_stalls(monkeypatch):
    """Three consecutive stalls exclude the candidate with all attempts
    recorded instead of killing the sweep."""

    selected = Probe(
        "D",
        1024,
        4,
        8,
        layout=Layout(context=8, data=1, dispatch="ring_gather_combine"),
        optimizer="adam",
        learning_rate=3.84615e-06,
    )
    phases = []

    def fake_run_probe(probe):
        phases.append(probe.cell.phase)
        return _stall_report(probe)

    monkeypatch.setattr(optimizers, "run_probe", fake_run_probe)
    monkeypatch.setattr(optimizers, "lr_candidates", lambda size, optimizer: (3.84615e-06,))
    reports, exclusions = optimizers.run_lr_trials(selected, "adam", "D-r7")
    assert reports == []
    assert len(exclusions) == 1
    record = exclusions[0]
    assert record["optimizer"] == "adam" and record["learning_rate"] == 3.84615e-06
    assert [
        a["phase"].startswith(prefix)
        for a, prefix in zip(record["attempts"], ("D-r7-", "D-r7-retry1-", "D-r7-retry2-"), strict=True)
    ] == [
        True,
        True,
        True,
    ]
    assert all(a["status"] == "stall_limit" for a in record["attempts"])


def test_lr_trials_memory_limit_excludes_immediately(monkeypatch):
    """A deterministic memory limit does not burn stall retries."""

    selected = Probe("D", 768, 8, 8, layout=Layout(context=8, data=1), optimizer="adam", learning_rate=1.15385e-05)

    def fake_run_probe(probe):
        return _memory_report(probe)

    monkeypatch.setattr(optimizers, "run_probe", fake_run_probe)
    monkeypatch.setattr(optimizers, "lr_candidates", lambda size, optimizer: (1.15385e-05,))
    reports, exclusions = optimizers.run_lr_trials(selected, "adam", "D-r7")
    assert reports == [] and len(exclusions) == 1
    assert len(exclusions[0]["attempts"]) == 1
    assert exclusions[0]["attempts"][0]["status"] == "memory_limit"


def test_q5_accounting_accepts_recorded_exclusions():
    """Q5's runner accepts a sweep whose non-complete candidates are recorded
    three-stall exclusions, and still flags the degenerate tuning."""

    selected = {
        "probe": {
            "phase": "D-r7",
            "size": 512,
            "microbatch": 1,
            "accumulation": 16,
            "layout": {"context": 8, "data": 1, "expert": 1, "replica": 1, "dispatch": "sonic", "offload": True},
            "steps": 23,
            "seed": 0,
            "optimizer": "grug_moe_muonh_v1",
            "learning_rate": 5e-05,
            "host_memory": "512GB",
        }
    }
    muonh_trial = {
        **selected,
        "status": "complete",
        "loss_curve": [],
    }
    adam_trial = {
        **selected,
        "status": "complete",
        "loss_curve": [],
    }
    adam_trial["probe"] = {**selected["probe"], "optimizer": "adam"}
    excluded = [
        {
            "optimizer": "adam",
            "learning_rate": 3.84615e-06,
            "attempts": [{"phase": "D-r7", "status": "stall_limit", "error": "x"}] * 3,
        }
    ]
    result = {
        "size": 512,
        "trials": [muonh_trial, muonh_trial, adam_trial],
        "excluded": excluded,
        "degenerate_lr_tuning": True,
        "winners": {
            "grug_moe_muonh_v1": {"probe": selected["probe"], "final_loss": 1.7},
            "adam": {"probe": selected["probe"], "final_loss": 1.8},
        },
    }
    payload = json.dumps({"result": result})

    with (
        unittest.mock.patch.object(optimizers, "read_json", return_value=json.loads(payload)),
        unittest.mock.patch.object(optimizers, "lr_candidates", return_value=(5e-05,)),
    ):
        probes = optimizers.q5_probes()
    # The mock returns the d512 payload for every size: 3 sizes x 6 paired arms.
    assert len(probes) == 18
    assert {p.seed for p in probes} == {0, 1, 2}
