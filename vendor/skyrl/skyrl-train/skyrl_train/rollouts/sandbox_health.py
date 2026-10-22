"""Rollout-host health measured from inside each rollout worker: a sandbox canary and event-loop lag.

Each rollout worker runs a ``SandboxHealthProbe``. Every ``canary_interval_seconds`` it creates a machine from the
worker's own task backend, runs ``true`` in it and closes it, one canary at a time; once a second it measures how
late its event loop wakes. The pool's ``SandboxHealthWatch`` collects these readings from every worker, publishes
them to telemetry, raises alarms in the driver log, and turns each step's readings into tracker metrics.

The watch raises an alarm for a worker when, since the previous collection:

* a canary was slow: longer than ``slow_seconds``, or than ``slow_baseline_ratio`` times the median of that worker's
  first ``baseline_samples`` canaries, whichever is lower;
* a canary failed, or did not finish within ``canary_timeout_seconds``;
* the worker produced no canary for ``silent_after_seconds``, or did not answer the collection at all.

The first two log a warning when they start for a worker; a missing worker logs an error. Every step that saw any of
them also logs one summary warning and sets the ``sandbox_canary_alarm`` metric.
"""

import asyncio
import contextlib
import math
import statistics
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from loguru import logger
from omegaconf import DictConfig
from shellbox.machine import Command, HostImage, Machine, MachineFactory, MachineSpec

from skyrl_train.rollout_observability import event_loop_lag
from skyrl_train.telemetry import WORKER_ROLE, telemetry

TRAINING_METRIC_PREFIX = "rollout_workers"
EVAL_METRIC_PREFIX = "eval_rollout_workers"
# The default local spec: the host's programs, a private root and no network, like a task machine.
CANARY_SPEC = MachineSpec(source=HostImage())
CANARY_ARGV = ("true",)
# A worker that has not reported within this long counts as missing for that collection.
REPORT_TIMEOUT_SECONDS = 30.0
# Readings kept per worker between collections, on each side; older ones are dropped and counted.
MAX_BUFFERED_READINGS = 4096
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("max", 1.0))

canary_seconds = telemetry.histogram("sandbox_canary_seconds", unit="s")


@dataclass(frozen=True)
class SandboxHealthConfig:
    """When rollout workers probe their host, and when a probe counts as an alarm."""

    enabled: bool
    backend: str
    canary_interval_seconds: float
    canary_timeout_seconds: float
    event_loop_lag_interval_seconds: float
    collect_interval_seconds: float
    baseline_samples: int
    slow_baseline_ratio: float
    slow_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "canary_interval_seconds",
            "canary_timeout_seconds",
            "event_loop_lag_interval_seconds",
            "collect_interval_seconds",
            "baseline_samples",
            "slow_baseline_ratio",
            "slow_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"sandbox_health.{name} must be positive; got {getattr(self, name)}")

    @classmethod
    def from_config(cls, config: DictConfig) -> "SandboxHealthConfig":
        health = config.trajectory_runner.rollout_workers.sandbox_health
        return cls(
            enabled=bool(health.enabled),
            backend=str(health.backend),
            canary_interval_seconds=float(health.canary_interval_seconds),
            canary_timeout_seconds=float(health.canary_timeout_seconds),
            event_loop_lag_interval_seconds=float(health.event_loop_lag_interval_seconds),
            collect_interval_seconds=float(health.collect_interval_seconds),
            baseline_samples=int(health.baseline_samples),
            slow_baseline_ratio=float(health.slow_baseline_ratio),
            slow_seconds=float(health.slow_seconds),
        )

    @property
    def silent_after_seconds(self) -> float:
        """How long a worker may go without finishing a canary before it counts as missing."""
        return self.canary_timeout_seconds + 3 * self.canary_interval_seconds


@runtime_checkable
class MachineFactorySource(Protocol):
    """A trajectory runner that creates task machines, so a canary can use the same backend."""

    def machine_factory(self, backend: str) -> MachineFactory | None: ...


class CanaryOutcome(StrEnum):
    OK = "ok"
    SLOW = "slow"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class CanaryReading:
    """One canary's wall time, split by machine operation when it got that far."""

    total_seconds: float
    outcome: CanaryOutcome
    create_seconds: float | None = None
    run_seconds: float | None = None
    close_seconds: float | None = None


@dataclass(frozen=True)
class WorkerHealthReport:
    """Readings a worker took since its previous report."""

    canary_enabled: bool
    host: str
    canaries: tuple[CanaryReading, ...]
    loop_lags: tuple[float, ...]
    dropped_readings: int
    seconds_since_canary: float
    """Since the last canary finished, or since the probe started when none has."""


class CanaryCommandFailed(RuntimeError):
    """The canary's ``true`` did not exit zero."""


class SandboxHealthProbe:
    """One rollout worker's canary and event-loop lag loops, buffering readings until the pool collects them."""

    def __init__(
        self,
        config: SandboxHealthConfig,
        factory: MachineFactory | None,
        *,
        host: str,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._config = config
        self._factory = factory
        self._host = host
        self._clock = clock
        self._wait = wait
        self._canaries: deque[CanaryReading] = deque()
        self._lags: deque[float] = deque()
        self._dropped = 0
        self._last_canary = clock()
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        if not self._config.enabled:
            return
        self._last_canary = self._clock()
        self._tasks.append(asyncio.create_task(self._lag_loop(), name="sandbox-health-lag"))
        if self._factory is not None:
            self._tasks.append(asyncio.create_task(self._canary_loop(), name="sandbox-health-canary"))

    async def stop(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def report(self) -> WorkerHealthReport:
        canaries, self._canaries = tuple(self._canaries), deque()
        lags, self._lags = tuple(self._lags), deque()
        dropped, self._dropped = self._dropped, 0
        return WorkerHealthReport(
            canary_enabled=self._config.enabled and self._factory is not None,
            host=self._host,
            canaries=canaries,
            loop_lags=lags,
            dropped_readings=dropped,
            seconds_since_canary=self._clock() - self._last_canary,
        )

    async def canary(self) -> CanaryReading:
        """Create a machine, run ``true`` in it and close it, recording a failure or timeout instead of raising."""
        assert self._factory is not None
        clock = self._clock
        started = clock()
        marks: list[float] = []
        machine: Machine | None = None
        outcome = CanaryOutcome.OK
        try:
            async with asyncio.timeout(self._config.canary_timeout_seconds):
                machine = await self._factory.create(CANARY_SPEC)
                marks.append(clock())
                result = await machine.run(Command(CANARY_ARGV, timeout=self._config.canary_timeout_seconds))
                marks.append(clock())
                if result.exit_code != 0:
                    raise CanaryCommandFailed(f"`true` ended with exit code {result.exit_code} ({result.reason})")
                closing, machine = machine, None
                await closing.close()
                marks.append(clock())
        except TimeoutError:
            outcome = CanaryOutcome.TIMED_OUT
            logger.warning(
                "Sandbox canary did not finish within {:g}s on {}", self._config.canary_timeout_seconds, self._host
            )
        except Exception as error:
            # A canary reports the backend's failures; it must never take the rollout worker down with it.
            outcome = CanaryOutcome.FAILED
            logger.warning("Sandbox canary failed on {}: {!r}", self._host, error)
        finally:
            if machine is not None:
                await self._close_abandoned(machine)
        finished = clock()
        self._last_canary = finished
        points = [started, *marks]
        phases = [later - earlier for earlier, later in zip(points, points[1:], strict=False)]
        reading = CanaryReading(
            total_seconds=finished - started,
            outcome=outcome,
            create_seconds=phases[0] if len(phases) > 0 else None,
            run_seconds=phases[1] if len(phases) > 1 else None,
            close_seconds=phases[2] if len(phases) > 2 else None,
        )
        self._append(self._canaries, reading)
        return reading

    async def _close_abandoned(self, machine: Machine) -> None:
        try:
            async with asyncio.timeout(self._config.canary_timeout_seconds):
                await machine.close()
        except Exception as error:
            logger.warning("Sandbox canary machine close failed on {}: {!r}", self._host, error)

    async def _canary_loop(self) -> None:
        while True:
            await self._wait(self._config.canary_interval_seconds)
            await self.canary()

    async def _lag_loop(self) -> None:
        interval = self._config.event_loop_lag_interval_seconds
        while True:
            expected = self._clock() + interval
            await self._wait(interval)
            self._append(self._lags, max(0.0, self._clock() - expected))

    def _append(self, readings: deque, value) -> None:
        if len(readings) >= MAX_BUFFERED_READINGS:
            readings.popleft()
            self._dropped += 1
        readings.append(value)


class _Alarm(StrEnum):
    SLOW = "slow"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MISSING = "missing"


_OUTCOME_ALARMS = {
    CanaryOutcome.SLOW: _Alarm.SLOW,
    CanaryOutcome.FAILED: _Alarm.FAILED,
    CanaryOutcome.TIMED_OUT: _Alarm.TIMED_OUT,
}


@dataclass
class _WorkerWindow:
    """What the watch knows about one worker: its baseline, active alarms, and readings since the last step."""

    baseline_readings: list[float] = field(default_factory=list)
    baseline_seconds: float | None = None
    active: set[_Alarm] = field(default_factory=set)
    canary_enabled: bool = False
    host: str = ""
    canaries: list[CanaryReading] = field(default_factory=list)
    loop_lags: list[float] = field(default_factory=list)
    missing_collections: int = 0
    dropped_readings: int = 0


WorkerReport = Callable[[], Awaitable[WorkerHealthReport]]


class SandboxHealthWatch:
    """Collect every rollout worker's health readings, raise alarms in the driver log, and summarize each step."""

    def __init__(self, config: SandboxHealthConfig, *, report_timeout_seconds: float = REPORT_TIMEOUT_SECONDS):
        self.config = config
        self._report_timeout = report_timeout_seconds
        self._workers: list[_WorkerWindow] = []
        self._lock = asyncio.Lock()

    def slow_threshold_seconds(self, worker: int) -> float:
        baseline = self._workers[worker].baseline_seconds
        if baseline is None:
            return self.config.slow_seconds
        return min(self.config.slow_seconds, self.config.slow_baseline_ratio * baseline)

    async def run(self, reports: Callable[[], Sequence[WorkerReport]]) -> None:
        """Collect from the current workers every ``collect_interval_seconds`` until cancelled."""
        while True:
            await asyncio.sleep(self.config.collect_interval_seconds)
            try:
                await self.collect(reports())
            except Exception:
                logger.exception("Sandbox health collection failed; the step summary will collect again")

    async def collect(self, reports: Sequence[WorkerReport]) -> None:
        async def bounded(report: WorkerReport) -> WorkerHealthReport:
            async with asyncio.timeout(self._report_timeout):
                return await report()

        async with self._lock:
            results = await asyncio.gather(*(bounded(report) for report in reports), return_exceptions=True)
            while len(self._workers) < len(results):
                self._workers.append(_WorkerWindow())
            for index, result in enumerate(results):
                if isinstance(result, BaseException):
                    self._missing(index, f"its health report failed: {result!r}")
                else:
                    self._absorb(index, result)

    def step_metrics(self, prefix: str) -> dict[str, float]:
        """Summarize readings since the previous call as tracker metrics under ``prefix``, then start a new window."""
        windows = self._workers
        canaries = [reading for window in windows for reading in window.canaries]
        lags = [lag for window in windows for lag in window.loop_lags]
        counts = {alarm: 0 for alarm in _Alarm}
        for reading in canaries:
            if (alarm := _OUTCOME_ALARMS.get(reading.outcome)) is not None:
                counts[alarm] += 1
        missing_workers = sum(1 for window in windows if window.missing_collections)
        counts[_Alarm.MISSING] = missing_workers
        alarm = any(counts.values())
        metrics = {
            f"{prefix}/sandbox_canary_workers": float(sum(window.canary_enabled for window in windows)),
            f"{prefix}/sandbox_canary_samples": float(len(canaries)),
            f"{prefix}/sandbox_canary_slow": float(counts[_Alarm.SLOW]),
            f"{prefix}/sandbox_canary_failures": float(counts[_Alarm.FAILED]),
            f"{prefix}/sandbox_canary_timeouts": float(counts[_Alarm.TIMED_OUT]),
            f"{prefix}/sandbox_canary_missing_workers": float(missing_workers),
            f"{prefix}/sandbox_canary_alarm": float(alarm),
            f"{prefix}/dropped_readings": float(sum(window.dropped_readings for window in windows)),
        }
        # A timed-out canary's elapsed time is a lower bound on its latency; it belongs in the tail.
        timed = [reading for reading in canaries if reading.outcome != CanaryOutcome.FAILED]
        metrics.update(_percentiles(f"{prefix}/sandbox_canary_seconds", (r.total_seconds for r in timed)))
        for phase in ("create", "run", "close"):
            values = (getattr(r, f"{phase}_seconds") for r in canaries)
            metrics.update(
                _percentiles(f"{prefix}/sandbox_canary_{phase}_seconds", (v for v in values if v is not None))
            )
        metrics.update(_percentiles(f"{prefix}/event_loop_lag_seconds", lags))
        thresholds = [self.slow_threshold_seconds(index) for index, w in enumerate(windows) if w.canary_enabled]
        if thresholds:
            metrics[f"{prefix}/sandbox_canary_slow_threshold_seconds"] = max(thresholds)
        for index, window in enumerate(windows):
            worker_timed = [r.total_seconds for r in window.canaries if r.outcome != CanaryOutcome.FAILED]
            if worker_timed:
                metrics[f"{prefix}/worker_{index}/sandbox_canary_max_seconds"] = max(worker_timed)
        if alarm:
            logger.bind(alarm="sandbox_canary", **{key.value: value for key, value in counts.items()}).warning(
                "Sandbox canary alarm for {}: slow={} failed={} timed_out={} missing_workers={} p95={} max={} "
                "threshold={}",
                prefix,
                counts[_Alarm.SLOW],
                counts[_Alarm.FAILED],
                counts[_Alarm.TIMED_OUT],
                missing_workers,
                *(
                    _seconds(metrics.get(f"{prefix}/{name}"))
                    for name in (
                        "sandbox_canary_seconds/p95",
                        "sandbox_canary_seconds/max",
                        "sandbox_canary_slow_threshold_seconds",
                    )
                ),
            )
        for window in windows:
            window.canaries, window.loop_lags = [], []
            window.missing_collections = window.dropped_readings = 0
        return metrics

    def _absorb(self, index: int, report: WorkerHealthReport) -> None:
        window = self._workers[index]
        window.canary_enabled = report.canary_enabled
        window.host = report.host
        window.dropped_readings += report.dropped_readings
        attributes = {"role": WORKER_ROLE, "rollout_worker": str(index), "host": report.host}
        for lag in report.loop_lags:
            event_loop_lag.record(lag, attributes=attributes)
        window.loop_lags.extend(report.loop_lags)
        seen: set[_Alarm] = set()
        for reading in report.canaries:
            reading = self._classify(index, reading)
            canary_seconds.record(reading.total_seconds, attributes={**attributes, "outcome": reading.outcome})
            if (alarm := _OUTCOME_ALARMS.get(reading.outcome)) is not None:
                seen.add(alarm)
            window.canaries.append(reading)
        for readings in (window.canaries, window.loop_lags):
            window.dropped_readings += max(0, len(readings) - MAX_BUFFERED_READINGS)
            del readings[:-MAX_BUFFERED_READINGS]
        if report.canary_enabled and report.seconds_since_canary > self.config.silent_after_seconds:
            self._missing(index, f"no canary finished in the last {report.seconds_since_canary:.0f}s")
        elif _Alarm.MISSING in window.active:
            window.active.discard(_Alarm.MISSING)
            logger.info("Rollout worker {} on {} reports sandbox canaries again", index, report.host)
        # A collection between two canaries says nothing new, so it leaves the canary alarms as they were.
        if report.canaries:
            self._transition(index, window, seen)

    def _missing(self, index: int, reason: str) -> None:
        window = self._workers[index]
        window.missing_collections += 1
        if _Alarm.MISSING not in window.active:
            logger.bind(alarm="sandbox_canary_missing", rollout_worker=index).error(
                "Rollout worker {} on {} is missing sandbox health readings: {}", index, window.host or "?", reason
            )
        window.active.add(_Alarm.MISSING)

    def _classify(self, index: int, reading: CanaryReading) -> CanaryReading:
        if reading.outcome != CanaryOutcome.OK:
            return reading
        threshold = self.slow_threshold_seconds(index)
        if reading.total_seconds > threshold:
            return replace(reading, outcome=CanaryOutcome.SLOW)
        window = self._workers[index]
        if window.baseline_seconds is None:
            window.baseline_readings.append(reading.total_seconds)
            if len(window.baseline_readings) >= self.config.baseline_samples:
                window.baseline_seconds = statistics.median(window.baseline_readings)
                logger.info(
                    "Rollout worker {} sandbox canary baseline {:.3f}s; slow above {:.3f}s",
                    index,
                    window.baseline_seconds,
                    self.slow_threshold_seconds(index),
                )
        return reading

    def _transition(self, index: int, window: _WorkerWindow, seen: set[_Alarm]) -> None:
        """Warn when a canary alarm starts for a worker, and note when it clears."""
        active = window.active - {_Alarm.MISSING}
        for alarm in sorted(seen - active):
            logger.bind(alarm=f"sandbox_canary_{alarm}", rollout_worker=index).warning(
                "Rollout worker {} on {}: sandbox canary {} (slow above {:.3f}s, timeout {:g}s)",
                index,
                window.host,
                alarm.value.replace("_", " "),
                self.slow_threshold_seconds(index),
                self.config.canary_timeout_seconds,
            )
        for alarm in sorted(active - seen):
            logger.info("Rollout worker {} on {}: sandbox canary no longer {}", index, window.host, alarm.value)
        window.active = seen | (window.active & {_Alarm.MISSING})


def _seconds(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}s"


def _percentiles(name: str, values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        f"{name}/{label}": ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]
        for label, quantile in PERCENTILES
    }


@runtime_checkable
class ReportsSandboxHealth(Protocol):
    async def sandbox_health_metrics(self, prefix: str) -> dict[str, float]: ...


async def rollout_health_metrics(training_runner: object, eval_runner: object) -> dict[str, float]:
    """The step's sandbox health metrics from the training runner and, when separate, the evaluation runner."""
    runners = {TRAINING_METRIC_PREFIX: training_runner}
    if eval_runner is not training_runner:
        runners[EVAL_METRIC_PREFIX] = eval_runner
    metrics: dict[str, float] = {}
    for prefix, runner in runners.items():
        if isinstance(runner, ReportsSandboxHealth):
            metrics.update(await runner.sandbox_health_metrics(prefix))
    return metrics


async def stop_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
