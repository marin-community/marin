# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CollectorRunner basics. The runner logs under the ``runner`` module logger;
tests assert on it via caplog and on the samples delivered to a recording sink.
asyncio.wait_for bounds duration."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from runner import METRIC_LATENCY_MS, METRIC_UP, Collector, CollectorRunner, health_collector
from sample import Sample


def _run_briefly(runner, duration=0.15):
    """Run the runner for ``duration`` seconds and return. asyncio.wait_for
    cancels the gather, which propagates CancelledError; we swallow it."""
    try:
        asyncio.run(asyncio.wait_for(runner._run_async(), timeout=duration))
    except (TimeoutError, asyncio.CancelledError):
        pass


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "runner"]


class RecordingSink:
    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def record(self, s: Sample) -> None:
        self.samples.append(s)


def _values(sink: RecordingSink, metric: str, *, probe: str | None = None) -> list[float]:
    return [s.value for s in sink.samples if s.metric == metric and (probe is None or f'"probe": "{probe}"' in s.labels)]


def test_passing_health_check_logs_ok_and_emits_up_1(caplog):
    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(health_collector("ok", lambda: True, timeout=1.0, cadence=0.05))
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner)
    assert any(m.startswith("probe ok: ok [") for m in _messages(caplog))
    assert 1.0 in _values(sink, METRIC_UP)
    # every cycle also emits a latency sample
    assert _values(sink, METRIC_LATENCY_MS)


def test_failing_health_check_logs_fail_and_emits_up_0(caplog):
    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(health_collector("down", lambda: False, timeout=1.0, cadence=0.05))
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner)
    assert any(m.startswith("probe down: fail [") for m in _messages(caplog))
    assert _values(sink, METRIC_UP) and all(v == 0.0 for v in _values(sink, METRIC_UP))


def test_raising_collector_logs_fail_and_emits_up_0(caplog):
    def boom():
        raise RuntimeError("nope")

    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(health_collector("boom", boom, timeout=1.0, cadence=0.05))
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner)
    assert any(m.startswith("probe boom: fail [") for m in _messages(caplog))
    assert 0.0 in _values(sink, METRIC_UP)


def test_timeout_collector_logs_fail(caplog):
    def slow():
        time.sleep(1.0)
        return True

    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(health_collector("slow", slow, timeout=0.05, cadence=0.05))
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner, duration=0.25)
    assert any(m.startswith("probe slow: fail [") for m in _messages(caplog))
    # the timeout is reported as down, not a missing sample
    assert _values(sink, METRIC_UP) and all(v == 0.0 for v in _values(sink, METRIC_UP))


def test_gauge_collector_records_all_samples_stamped(caplog):
    """A non-health collector's samples are all delivered, each stamped with the
    cycle time, and the cycle logs ok (it ran)."""
    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(
        Collector("gauge", lambda: [Sample.of("m_a", 7.0, zone="z"), Sample.of("m_b", 9.0)], timeout=1.0, cadence=0.05)
    )
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner)
    assert any(m.startswith("probe gauge: ok [") for m in _messages(caplog))
    assert 7.0 in _values(sink, "m_a") and 9.0 in _values(sink, "m_b")
    assert all(s.collected_at is not None for s in sink.samples)


def test_run_with_no_collectors_raises():
    with pytest.raises(ValueError, match="no collectors registered"):
        CollectorRunner().run()


def test_multiple_collectors_run_independently():
    sink = RecordingSink()
    runner = CollectorRunner(sinks=[sink])
    runner.add(health_collector("a", lambda: True, timeout=1.0, cadence=0.05))
    runner.add(health_collector("b", lambda: True, timeout=1.0, cadence=0.05))
    _run_briefly(runner, duration=0.2)
    # both collectors produced their own up samples, labelled by probe name
    assert 1.0 in _values(sink, METRIC_UP, probe="a")
    assert 1.0 in _values(sink, METRIC_UP, probe="b")


def test_sink_failure_does_not_disrupt_collection(caplog):
    """A raising sink must neither crash the runner nor stop the other sinks from
    receiving samples — sinks are best-effort telemetry."""

    class RaisingSink:
        def record(self, s):
            raise RuntimeError("sink boom")

    good = RecordingSink()
    runner = CollectorRunner(sinks=[RaisingSink(), good])
    runner.add(health_collector("ok", lambda: True, timeout=1.0, cadence=0.05))
    with caplog.at_level(logging.INFO, logger="runner"):
        _run_briefly(runner)

    # collection continued (still logged) despite the raising sink...
    assert any(m.startswith("probe ok: ok [") for m in _messages(caplog))
    # ...and the non-failing sink still got the up sample.
    assert 1.0 in _values(good, METRIC_UP)
