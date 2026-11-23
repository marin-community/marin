# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable guardian phase timing using real marker files and readiness pipes."""

import json
import os
import select
from pathlib import Path

import pytest
from marin.rl.native_shutdown.observer_guardian import safety, wait_for_debugger


@pytest.fixture
def observer_wait(tmp_path):
    read, write = os.pipe()
    config = {"nonce": "c" * 32, "ready_path": str(tmp_path / "ready.json"), "ack_path": str(tmp_path / "ack.json")}
    expected = safety.Identity(pid=41, starttime=73, pgrp=41)
    try:
        yield config, expected, read, write
    finally:
        os.close(read)
        os.close(write)


def advance_on_poll(monkeypatch, times, publish=None):
    clock = [0]
    ticks = iter(times)
    real_select = select.select

    def poll(readers, writers, errors, timeout):
        if timeout:
            clock[0] = next(ticks)
            if publish is not None:
                publish(clock[0])
        return real_select(readers, writers, errors, 0)

    monkeypatch.setattr(select, "select", poll)
    return clock


def publish_marker(config, expected, kind, monotonic):
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    path = Path(config[f"{kind}_path"])
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(dict(marker, running=True) if kind == "ready" else dict(marker, monotonic_ns=int(monotonic * 1e9)))
    )
    if kind == "ready":
        os.replace(pending, path)
    else:
        os.link(pending, path)
        pending.unlink()


def test_cold_readiness_leaves_a_new_ack_window_and_keeps_post_ack_deadline(observer_wait, monkeypatch):
    config, expected, descriptor, finished = observer_wait

    def publish(value):
        if value == 8.25:
            publish_marker(config, expected, "ready", value)
        elif value == 8.5:
            publish_marker(config, expected, "ack", value)
        elif value == 10.75:
            os.write(finished, b"exited")

    clock = advance_on_poll(monkeypatch, [7.75, 8.25, 8.5, 10.75], publish)
    result = {}
    wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result == {
        "deadline_phase": "post_ack",
        "startup_deadline_monotonic": 9,
        "ready_observed_monotonic": 8.25,
        "ack_wait_deadline_monotonic": 18.25,
        "ack_observed_monotonic": 8.5,
        "post_ack_deadline_monotonic": 11.45,
    }


@pytest.mark.parametrize(
    "phase,times", [("startup_readiness", [9.1]), ("await_ack", [1, 11.1]), ("post_ack", [1, 3.96])]
)
def test_each_guardian_phase_still_rejects_its_expired_budget(observer_wait, phase, times, monkeypatch):
    config, expected, descriptor, _ = observer_wait
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    if phase != "startup_readiness":
        Path(config["ready_path"]).write_text(json.dumps(dict(marker, running=True)))
    if phase == "post_ack":
        Path(config["ack_path"]).write_text(json.dumps(dict(marker, monotonic_ns=1_000_000_000)))
    clock = advance_on_poll(monkeypatch, times)
    result = {}
    with pytest.raises(TimeoutError, match="Persistent tracer deadline exceeded"):
        wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["deadline_phase"] == phase


def test_simultaneously_observed_ready_and_ack_keep_the_actual_ack_deadline(observer_wait, monkeypatch):
    config, expected, descriptor, finished = observer_wait

    def publish(value):
        if value == 8.5:
            publish_marker(config, expected, "ready", value)
            publish_marker(config, expected, "ack", value)
        else:
            os.write(finished, b"exited")

    clock = advance_on_poll(monkeypatch, [8.5, 10.75], publish)
    result = {}
    wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["ready_observed_monotonic"] == result["ack_observed_monotonic"] == 8.5
    assert result["post_ack_deadline_monotonic"] == 11.45


@pytest.mark.parametrize(
    "kind,times,read_finishes_at,phase",
    [
        ("ready", [9.1], None, "startup_readiness"),
        ("ack", [1, 11.1], None, "await_ack"),
        ("both", [9.1], None, "startup_readiness"),
        ("ready", [8.9], 9.1, "startup_readiness"),
        ("ack", [1, 10.9], 11.1, "await_ack"),
    ],
)
def test_late_markers_cannot_rearm_an_expired_phase(observer_wait, monkeypatch, kind, times, read_finishes_at, phase):
    config, expected, descriptor, _ = observer_wait

    def publish(value):
        if kind == "ack" and value == 1:
            publish_marker(config, expected, "ready", value)
        elif kind == "both":
            publish_marker(config, expected, "ready", value)
            publish_marker(config, expected, "ack", value)
        else:
            publish_marker(config, expected, kind, value)

    clock = advance_on_poll(monkeypatch, times, publish)
    if read_finishes_at is not None:
        marker_path = Path(config[f"{kind}_path"])
        real_read = Path.read_text

        def delayed_read(path, *args, **kwargs):
            value = real_read(path, *args, **kwargs)
            if path == marker_path:
                clock[0] = read_finishes_at
            return value

        monkeypatch.setattr(Path, "read_text", delayed_read)
    result = {}
    with pytest.raises(TimeoutError, match="Persistent tracer deadline exceeded"):
        wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["deadline_phase"] == phase
    assert "post_ack_deadline_monotonic" not in result


@pytest.mark.parametrize("field", ["pid", "starttime", "nonce"])
@pytest.mark.parametrize("marker_kind", ["ready", "ack"])
def test_guardian_rejects_markers_bound_to_a_different_target(observer_wait, field, marker_kind):
    config, expected, descriptor, _ = observer_wait
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    changed = dict(marker)
    changed[field] = "different" if field == "nonce" else int(changed[field]) + 1
    Path(config["ready_path"]).write_text(json.dumps(dict(changed if marker_kind == "ready" else marker, running=True)))
    if marker_kind == "ack":
        Path(config["ack_path"]).write_text(json.dumps(dict(changed, monotonic_ns=1_000_000_000)))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        wait_for_debugger(config, expected, descriptor, None, 0, {}, now=lambda: 1)
