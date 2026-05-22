# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ProbeRunner basics. Each probe is a plain callable returning ProbeResult."""

from __future__ import annotations

import threading
import time

import pytest
from probes import ProbeResult, ProbeRunner


def _runner_with(probes, on_result):
    runner = ProbeRunner(on_result=on_result)
    for spec in probes:
        runner.add_probe(**spec)
    return runner


def _run_briefly(runner, duration=0.3):
    """Start the runner in a background thread; stop after `duration`."""
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    time.sleep(duration)
    runner.stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "runner thread did not stop cleanly"
    return t


def test_success_probe_emits_ok_result():
    seen: list[tuple[str, ProbeResult]] = []

    def ok():
        return ProbeResult(is_success=True)

    runner = _runner_with(
        [{"name": "ok", "fn": ok, "timeout": 1.0, "cadence": 0.05}],
        on_result=lambda n, r: seen.append((n, r)),
    )
    _run_briefly(runner, duration=0.15)
    assert seen, "expected at least one probe result"
    name, result = seen[0]
    assert name == "ok"
    assert result.is_success is True
    assert result.wall_time is not None and result.wall_time >= 0


def test_raising_probe_records_failure():
    seen: list[ProbeResult] = []

    def boom():
        raise RuntimeError("nope")

    runner = _runner_with(
        [{"name": "boom", "fn": boom, "timeout": 1.0, "cadence": 0.05}],
        on_result=lambda _n, r: seen.append(r),
    )
    _run_briefly(runner, duration=0.15)
    assert seen
    assert all(r.is_success is False for r in seen)
    assert all(r.wall_time is not None for r in seen)


def test_timeout_probe_records_failure():
    seen: list[ProbeResult] = []

    def slow():
        time.sleep(1.0)
        return ProbeResult(is_success=True)

    runner = _runner_with(
        [{"name": "slow", "fn": slow, "timeout": 0.05, "cadence": 0.05}],
        on_result=lambda _n, r: seen.append(r),
    )
    _run_briefly(runner, duration=0.25)
    assert seen
    assert seen[0].is_success is False
    assert seen[0].wall_time is not None and seen[0].wall_time >= 0.05


def test_run_with_no_probes_raises():
    runner = ProbeRunner()
    with pytest.raises(ValueError, match="no probes registered"):
        runner.run()


def test_multiple_probes_run_independently():
    seen: list[str] = []

    runner = ProbeRunner(on_result=lambda n, _r: seen.append(n))
    runner.add_probe("a", lambda: ProbeResult(is_success=True), timeout=1.0, cadence=0.05)
    runner.add_probe("b", lambda: ProbeResult(is_success=True), timeout=1.0, cadence=0.05)
    _run_briefly(runner, duration=0.2)
    names = set(seen)
    assert names == {"a", "b"}, f"expected both probes to fire, got {names}"
