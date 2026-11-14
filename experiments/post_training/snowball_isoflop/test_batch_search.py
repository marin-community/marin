# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

from experiments.post_training.snowball_isoflop import run_probes
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
