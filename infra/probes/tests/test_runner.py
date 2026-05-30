# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ProbeRunner basics. Output is hardcoded to the ``probes`` logger; tests
use pytest's ``caplog`` to assert on it. asyncio.wait_for bounds duration."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from marin_infra_probes import ProbeRunner


def _run_briefly(runner, duration=0.15):
    """Run the runner for ``duration`` seconds and return. asyncio.wait_for
    cancels the gather, which propagates CancelledError; we swallow it."""
    try:
        asyncio.run(asyncio.wait_for(runner._run_async(), timeout=duration))
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "probes"]


def test_success_probe_logs_ok(caplog):
    runner = ProbeRunner()
    runner.add_probe("ok", lambda: True, timeout=1.0, cadence=0.05)
    with caplog.at_level(logging.INFO, logger="probes"):
        _run_briefly(runner)
    msgs = _messages(caplog)
    assert any(m.startswith("probe ok: ok [") for m in msgs), msgs


def test_raising_probe_logs_fail(caplog):
    def boom():
        raise RuntimeError("nope")

    runner = ProbeRunner()
    runner.add_probe("boom", boom, timeout=1.0, cadence=0.05)
    with caplog.at_level(logging.INFO, logger="probes"):
        _run_briefly(runner)
    msgs = _messages(caplog)
    assert any(m.startswith("probe boom: fail [") for m in msgs), msgs


def test_timeout_probe_logs_fail(caplog):
    def slow():
        time.sleep(1.0)
        return True

    runner = ProbeRunner()
    runner.add_probe("slow", slow, timeout=0.05, cadence=0.05)
    with caplog.at_level(logging.INFO, logger="probes"):
        _run_briefly(runner, duration=0.25)
    msgs = _messages(caplog)
    assert any(m.startswith("probe slow: fail [") for m in msgs), msgs


def test_run_with_no_probes_raises():
    with pytest.raises(ValueError, match="no probes registered"):
        ProbeRunner().run()


def test_multiple_probes_run_independently(caplog):
    runner = ProbeRunner()
    runner.add_probe("a", lambda: True, timeout=1.0, cadence=0.05)
    runner.add_probe("b", lambda: True, timeout=1.0, cadence=0.05)
    with caplog.at_level(logging.INFO, logger="probes"):
        _run_briefly(runner, duration=0.2)
    msgs = _messages(caplog)
    assert any(m.startswith("probe a: ") for m in msgs), msgs
    assert any(m.startswith("probe b: ") for m in msgs), msgs
