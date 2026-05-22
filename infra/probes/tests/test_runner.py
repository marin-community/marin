# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ProbeRunner basics. The runner has no shutdown; tests bound its duration
with asyncio.wait_for and treat the resulting CancelledError as completion."""

from __future__ import annotations

import asyncio
import time

import pytest
from marin_infra_probes import ProbeResult, ProbeRunner


def _run_briefly(runner, duration=0.15):
    """Run the runner for ``duration`` seconds and return. asyncio.wait_for
    cancels the gather, which propagates CancelledError; we swallow it."""
    try:
        asyncio.run(asyncio.wait_for(runner._run_async(), timeout=duration))
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def test_success_probe_emits_ok_result():
    seen: list[tuple[str, ProbeResult]] = []
    runner = ProbeRunner(on_result=lambda n, r: seen.append((n, r)))
    runner.add_probe("ok", lambda: ProbeResult(is_success=True), timeout=1.0, cadence=0.05)
    _run_briefly(runner)
    assert seen, "expected at least one probe result"
    name, result = seen[0]
    assert name == "ok"
    assert result.is_success is True
    assert result.wall_time is not None and result.wall_time >= 0


def test_raising_probe_records_failure():
    seen: list[ProbeResult] = []

    def boom():
        raise RuntimeError("nope")

    runner = ProbeRunner(on_result=lambda _n, r: seen.append(r))
    runner.add_probe("boom", boom, timeout=1.0, cadence=0.05)
    _run_briefly(runner)
    assert seen
    assert all(r.is_success is False for r in seen)
    assert all(r.wall_time is not None for r in seen)


def test_timeout_probe_records_failure():
    seen: list[ProbeResult] = []

    def slow():
        time.sleep(1.0)
        return ProbeResult(is_success=True)

    runner = ProbeRunner(on_result=lambda _n, r: seen.append(r))
    runner.add_probe("slow", slow, timeout=0.05, cadence=0.05)
    _run_briefly(runner, duration=0.25)
    assert seen
    assert seen[0].is_success is False
    assert seen[0].wall_time is not None and seen[0].wall_time >= 0.05


def test_run_with_no_probes_raises():
    with pytest.raises(ValueError, match="no probes registered"):
        ProbeRunner().run()


def test_multiple_probes_run_independently():
    seen: list[str] = []
    runner = ProbeRunner(on_result=lambda n, _r: seen.append(n))
    runner.add_probe("a", lambda: ProbeResult(is_success=True), timeout=1.0, cadence=0.05)
    runner.add_probe("b", lambda: ProbeResult(is_success=True), timeout=1.0, cadence=0.05)
    _run_briefly(runner, duration=0.2)
    assert set(seen) == {"a", "b"}
