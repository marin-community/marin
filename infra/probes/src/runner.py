# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The collector abstraction and the runner that drives it.

A ``Collector`` runs on a fixed cadence and returns a batch of ``Sample``s — that
single shape covers both health checks (a ``probe_up`` 1/0 sample) and gauges
(the TPU-provisioning rows). The runner schedules each collector, bounds each run
with a timeout, stamps the samples with the cycle time, logs a one-line status,
and fans the samples to the sinks.

Imports stay light (stdlib + ``sample``) so the runner is testable without the
iris/finelog clients the entrypoint wires up.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from sample import Sample

logger = logging.getLogger(__name__)

# Meta-metrics the runner stamps onto every cycle, regardless of what the
# collector returns: liveness/timing of the collector itself.
METRIC_UP = "probe_up"
METRIC_LATENCY_MS = "probe_latency_ms"


class MetricSink(Protocol):
    def record(self, sample: Sample) -> None: ...


# A collector reports a batch of samples; the runner owns scheduling, timing,
# timestamping, and delivery. A health check is just a collector that returns a
# single METRIC_UP sample.
CollectFn = Callable[[], list[Sample]]


@dataclass
class Collector:
    name: str
    collect: CollectFn
    timeout: float
    cadence: float


def health_collector(name: str, check: Callable[[], bool], *, timeout: float, cadence: float) -> Collector:
    """Adapt a boolean health check into a collector emitting one METRIC_UP sample.

    A ``False`` return reports down; a raised exception or timeout is turned into
    down by the runner (which can't trust a collector that failed mid-run).
    """
    return Collector(
        name=name,
        collect=lambda: [Sample.of(METRIC_UP, 1.0 if check() else 0.0, probe=name)],
        timeout=timeout,
        cadence=cadence,
    )


def _up_value(samples: Sequence[Sample]) -> bool:
    """The cycle's headline ok/fail for the log line: the METRIC_UP value if the
    collector emitted one (health checks), else True (a gauge collector that ran)."""
    for s in samples:
        if s.metric == METRIC_UP:
            return s.value == 1.0
    return True


class CollectorRunner:
    """Register collectors, then ``run()`` to execute each forever on its own
    cadence. Ctrl-C kills the process — samples are stateless, so there is no
    graceful-shutdown work. Each cycle logs ``probe <name>: ok|fail [<ms>ms]
    start=<utc-iso>`` and emits its samples to every sink; sink faults are logged
    and swallowed so telemetry never disrupts collection."""

    def __init__(self, sinks: Sequence[MetricSink] = ()) -> None:
        self._collectors: list[Collector] = []
        self._sinks = tuple(sinks)

    def add(self, collector: Collector) -> None:
        self._collectors.append(collector)

    def run(self) -> None:
        if not self._collectors:
            raise ValueError("no collectors registered")
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        await asyncio.gather(*(self._run_collector(c) for c in self._collectors))

    async def _run_collector(self, collector: Collector) -> None:
        while True:
            started_at = datetime.now(UTC)
            start = time.monotonic()
            try:
                samples = await asyncio.wait_for(asyncio.to_thread(collector.collect), timeout=collector.timeout)
            except TimeoutError:
                samples = [Sample.of(METRIC_UP, 0.0, probe=collector.name)]
            except Exception:
                logger.exception("collector %s raised", collector.name)
                samples = [Sample.of(METRIC_UP, 0.0, probe=collector.name)]
            wall_ms = (time.monotonic() - start) * 1000
            samples = [*samples, Sample.of(METRIC_LATENCY_MS, wall_ms, probe=collector.name)]

            up = _up_value(samples)
            logger.log(
                logging.INFO if up else logging.ERROR,
                "probe %s: %s [%dms] start=%s",
                collector.name,
                "ok" if up else "fail",
                wall_ms,
                started_at.isoformat(timespec="milliseconds"),
            )
            for s in samples:
                stamped = replace(s, collected_at=started_at)
                for sink in self._sinks:
                    try:
                        sink.record(stamped)
                    except Exception:
                        logger.exception("sink %s failed for collector %s", type(sink).__name__, collector.name)

            await asyncio.sleep(collector.cadence)
