# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, bounded hardware probe lifecycle and normalized results."""

import json
import math
import os
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from datetime import timedelta

from rigging import telemetry
from rigging.nccl_ras_probe import NcclRasProbe
from rigging.nvidia_smi_probe import NvidiaSmiProbe
from rigging.probe_types import (
    DEFAULT_NCCL_RAS_TIMEOUT,
    DEFAULT_NVIDIA_SMI_TIMEOUT,
    DEFAULT_PROBE_INTERVAL,
    MAX_ATTRIBUTE_FIELDS,
    MAX_EVENT_BODY_BYTES,
    MAX_EVENT_FIELDS,
    MAX_FIELD_BYTES,
    MAX_RESULT_EVENTS,
    MAX_RESULT_METRICS,
    MAX_SHUTDOWN_TIMEOUT,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    CommandResult,
    CommandRunner,
    CommandStatus,
    MetricKind,
    Probe,
    ProbeClock,
    ProbeCollection,
    ProbeEvent,
    ProbeManagerStatus,
    ProbeMetric,
    ProbeOutcome,
    ProbeReason,
    ProbeSample,
    ProbeSink,
)
from rigging.timing import Deadline

__all__ = [
    "DEFAULT_NCCL_RAS_TIMEOUT",
    "DEFAULT_NVIDIA_SMI_TIMEOUT",
    "DEFAULT_PROBE_INTERVAL",
    "MAX_EVENT_BODY_BYTES",
    "MAX_FIELD_BYTES",
    "MAX_RESULT_EVENTS",
    "MAX_RESULT_METRICS",
    "MAX_SHUTDOWN_TIMEOUT",
    "MAX_SUBPROCESS_OUTPUT_BYTES",
    "BoundedSubprocessRunner",
    "CommandResult",
    "CommandRunner",
    "CommandStatus",
    "MetricKind",
    "Probe",
    "ProbeClock",
    "ProbeCollection",
    "ProbeEvent",
    "ProbeManager",
    "ProbeManagerStatus",
    "ProbeMetric",
    "ProbeOutcome",
    "ProbeReason",
    "ProbeSample",
    "ProbeSink",
    "SystemProbeClock",
    "TelemetryProbeSink",
    "nccl_ras",
    "nvidia_smi",
    "start",
]

_PROCESS_REAP_TIMEOUT = 0.1


class SystemProbeClock:
    """System monotonic/wall clock implementation."""

    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()

    def wait(self, event: threading.Event, timeout: float) -> bool:
        return event.wait(timeout)


class BoundedSubprocessRunner:
    """Run a process with a monotonic deadline and bounded captured stdout."""

    def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> CommandResult:
        if not argv:
            raise ValueError("argv must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")

        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError:
            return CommandResult(CommandStatus.NOT_FOUND)
        except OSError:
            return CommandResult(CommandStatus.FAILED)

        assert process.stdout is not None
        deadline = Deadline.from_seconds(timeout)
        output = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline.remaining_seconds()
                    if remaining <= 0:
                        return self._stop_result(process, CommandStatus.TIMED_OUT, output)
                    if not selector.select(remaining):
                        return self._stop_result(process, CommandStatus.TIMED_OUT, output)
                    chunk = os.read(process.stdout.fileno(), min(65_536, max_output_bytes + 1 - len(output)))
                    if not chunk:
                        selector.unregister(process.stdout)
                        break
                    output.extend(chunk)
                    if len(output) > max_output_bytes:
                        del output[max_output_bytes:]
                        return self._stop_result(process, CommandStatus.OUTPUT_LIMIT, output)
        except OSError:
            self._terminate(process)
            return CommandResult(CommandStatus.FAILED, bytes(output))
        finally:
            process.stdout.close()

        try:
            returncode = process.wait(timeout=deadline.remaining_seconds())
        except subprocess.TimeoutExpired:
            return self._stop_result(process, CommandStatus.TIMED_OUT, output)
        return CommandResult(CommandStatus.COMPLETED, bytes(output), returncode)

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> bool:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except OSError:
                return False
        try:
            process.wait(timeout=_PROCESS_REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _stop_result(
        self,
        process: subprocess.Popen[bytes],
        status: CommandStatus,
        output: bytearray,
    ) -> CommandResult:
        final_status = status if self._terminate(process) else CommandStatus.FAILED
        return CommandResult(final_status, bytes(output))


_PROBE_RUNS = telemetry.counter("probe_runs", unit="{run}")
_PROBE_DURATION = telemetry.histogram("probe_duration_seconds", unit="s")
_PROBE_UP = telemetry.gauge("probe_up")
_PROBE_LAST_SUCCESS = telemetry.gauge("probe_last_success_time_seconds", unit="s")


class TelemetryProbeSink:
    """Emit normalized probe results through the configured Rigging telemetry client."""

    def emit(self, sample: ProbeSample) -> None:
        attributes = {"probe": sample.probe, "outcome": sample.outcome.value}
        if sample.outcome is ProbeOutcome.SHUTDOWN:
            telemetry.event(
                "probe_terminal",
                {"state": ProbeReason.CLEAN_SHUTDOWN.value},
                attributes={"probe": sample.probe, "reason": ProbeReason.CLEAN_SHUTDOWN.value},
            )
            return

        _PROBE_RUNS.add(1, attributes=attributes)
        _PROBE_DURATION.record(sample.duration, attributes=attributes)
        _PROBE_UP.set(float(sample.outcome is ProbeOutcome.SUCCEEDED), attributes={"probe": sample.probe})
        if sample.outcome is ProbeOutcome.SUCCEEDED:
            _PROBE_LAST_SUCCESS.set(sample.observed_time, attributes={"probe": sample.probe})
        elif sample.reason is not None:
            body = {"state": sample.outcome.value}
            if sample.error_type:
                body["error_type"] = sample.error_type
            telemetry.event(
                "probe_terminal",
                body,
                attributes={"probe": sample.probe, "reason": sample.reason.value},
            )

        for metric in sample.metrics:
            if metric.kind is MetricKind.COUNTER:
                telemetry.counter(metric.name, unit=metric.unit).add(metric.value, attributes=metric.attributes)
            elif metric.kind is MetricKind.GAUGE:
                telemetry.gauge(metric.name, unit=metric.unit).set(metric.value, attributes=metric.attributes)
            else:
                telemetry.histogram(metric.name, unit=metric.unit).record(metric.value, attributes=metric.attributes)
        for event in sample.events:
            telemetry.event(event.name, dict(event.body), attributes=event.attributes)


class _ProbeWorker:
    def __init__(self, probe: Probe, runner: CommandRunner, sink: ProbeSink, clock: ProbeClock) -> None:
        self.probe = probe
        self.runner = runner
        self.sink = sink
        self.clock = clock
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._sample_requested = False
        self._stopping = False
        self._sink_failures = 0
        self._thread = threading.Thread(target=self._run, name=f"rigging-probe-{probe.name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def sample(self) -> None:
        with self._lock:
            self._sample_requested = True
            self._wake.set()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._wake.set()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def alive(self) -> bool:
        return self._thread.is_alive()

    def sink_failures(self) -> int:
        with self._lock:
            return self._sink_failures

    def _run(self) -> None:
        next_due = self.clock.monotonic()
        try:
            while True:
                with self._lock:
                    if self._stopping:
                        return
                    requested = self._sample_requested
                    self._sample_requested = False
                    self._wake.clear()

                now = self.clock.monotonic()
                due = now >= next_due
                if due:
                    while next_due <= now:
                        next_due += self.probe.interval
                if due or requested:
                    self._collect()
                    continue
                self.clock.wait(self._wake, max(0.0, next_due - now))
        finally:
            self._emit(
                ProbeSample(
                    self.probe.name,
                    ProbeOutcome.SHUTDOWN,
                    ProbeReason.CLEAN_SHUTDOWN,
                    0.0,
                    self.clock.time(),
                )
            )

    def _collect(self) -> None:
        started = self.clock.monotonic()
        try:
            collection = _bounded_collection(self.probe.collect(self.runner))
        except Exception as error:
            collection = ProbeCollection(
                ProbeOutcome.FAILED,
                ProbeReason.INTERNAL_ERROR,
                error_type=_exception_type(error),
            )
        duration = max(0.0, self.clock.monotonic() - started)
        self._emit(
            ProbeSample(
                probe=self.probe.name,
                outcome=collection.outcome,
                reason=collection.reason,
                duration=duration,
                observed_time=self.clock.time(),
                metrics=collection.metrics,
                events=collection.events,
                error_type=collection.error_type,
            )
        )

    def _emit(self, sample: ProbeSample) -> None:
        try:
            self.sink.emit(sample)
        except Exception:
            with self._lock:
                self._sink_failures += 1


class ProbeManager:
    """Own independently scheduled probe daemon threads."""

    def __init__(self, workers: Mapping[str, _ProbeWorker]) -> None:
        self._workers = dict(workers)
        self._shutdown_lock = threading.Lock()
        self._shutdown = False

    def sample(self, probe: str) -> None:
        """Queue an on-demand collection on the named probe's daemon thread."""
        self._workers[probe].sample()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Request stop and spend at most one shared timeout joining workers.

        A probe blocked beyond the budget remains an isolated daemon thread.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            budget = _shutdown_budget(timeout)
            self._shutdown = True
            deadline = Deadline.from_seconds(budget)
            for worker in self._workers.values():
                worker.stop()
            for worker in self._workers.values():
                worker.join(deadline.remaining_seconds())

    def status(self) -> ProbeManagerStatus:
        """Return process-local worker liveness and sink loss."""
        return ProbeManagerStatus(
            live_workers=sum(worker.alive() for worker in self._workers.values()),
            sink_failures=sum(worker.sink_failures() for worker in self._workers.values()),
        )

    def __enter__(self) -> "ProbeManager":
        return self

    def __exit__(self, *_args: object) -> None:
        self.shutdown()


def start(
    *configured_probes: Probe,
    runner: CommandRunner | None = None,
    sink: ProbeSink | None = None,
    clock: ProbeClock | None = None,
) -> ProbeManager:
    """Start each explicitly configured probe on an isolated daemon thread."""
    names = [probe.name for probe in configured_probes]
    if len(names) != len(set(names)):
        raise ValueError("probe names must be unique")
    for probe in configured_probes:
        if not math.isfinite(probe.interval) or probe.interval <= 0:
            raise ValueError("probe intervals must be positive and finite")

    command_runner = runner if runner is not None else BoundedSubprocessRunner()
    probe_sink = sink if sink is not None else TelemetryProbeSink()
    probe_clock = clock if clock is not None else SystemProbeClock()
    workers = {probe.name: _ProbeWorker(probe, command_runner, probe_sink, probe_clock) for probe in configured_probes}
    manager = ProbeManager(workers)
    for worker in workers.values():
        worker.start()
    return manager


def nvidia_smi(
    *,
    interval: timedelta = DEFAULT_PROBE_INTERVAL,
    timeout: timedelta = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> Probe:
    """Configure the portable NVIDIA inventory and slow-health probe."""
    return NvidiaSmiProbe(_duration_seconds(interval, "interval"), _duration_seconds(timeout, "timeout"))


def nccl_ras(
    *,
    interval: timedelta = DEFAULT_PROBE_INTERVAL,
    timeout: timedelta = DEFAULT_NCCL_RAS_TIMEOUT,
) -> Probe:
    """Configure the observe-only NCCL RAS summary probe."""
    return NcclRasProbe(_duration_seconds(interval, "interval"), _duration_seconds(timeout, "timeout"))


def _duration_seconds(value: timedelta, field_name: str) -> float:
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be positive and finite")
    return seconds


def _shutdown_budget(timeout: float) -> float:
    try:
        budget = float(timeout)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("shutdown timeout must be nonnegative and finite") from None
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("shutdown timeout must be nonnegative and finite")
    return min(budget, MAX_SHUTDOWN_TIMEOUT)


class _ResultLimitError(ValueError):
    pass


class _InvalidProbeResultError(ValueError):
    pass


def _bounded_collection(collection: ProbeCollection) -> ProbeCollection:
    try:
        if len(collection.metrics) > MAX_RESULT_METRICS or len(collection.events) > MAX_RESULT_EVENTS:
            raise _ResultLimitError("result count limit")
        if collection.error_type:
            _validate_field(collection.error_type)
        for metric in collection.metrics:
            _validate_field(metric.name)
            if metric.unit:
                _validate_field(metric.unit)
            try:
                numeric = float(metric.value)
            except (TypeError, ValueError, OverflowError):
                raise _InvalidProbeResultError("metric value") from None
            if not math.isfinite(numeric):
                raise _InvalidProbeResultError("metric value")
            _validate_attributes(metric.attributes)
        for event in collection.events:
            _validate_field(event.name)
            _validate_attributes(event.attributes)
            if len(event.body) > MAX_EVENT_FIELDS:
                raise _ResultLimitError("event field count")
            for key, value in event.body.items():
                _validate_field(key)
                if isinstance(value, str):
                    _validate_field(value)
                elif isinstance(value, float) and not math.isfinite(value):
                    raise _InvalidProbeResultError("event value")
                elif not isinstance(value, int | bool | float):
                    raise _InvalidProbeResultError("event value")
            if len(json.dumps(dict(event.body), allow_nan=False, separators=(",", ":")).encode()) > MAX_EVENT_BODY_BYTES:
                raise _ResultLimitError("event body bytes")
        return collection
    except _ResultLimitError:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.RESULT_LIMIT)
    except (_InvalidProbeResultError, TypeError, ValueError, OverflowError):
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.INVALID_RESULT)


def _validate_attributes(attributes: Mapping[str, str]) -> None:
    if len(attributes) > MAX_ATTRIBUTE_FIELDS:
        raise _ResultLimitError("attribute field count")
    for key, value in attributes.items():
        _validate_field(key)
        _validate_field(value)


def _validate_field(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise _InvalidProbeResultError("field must be a nonempty string")
    if len(value.encode()) > MAX_FIELD_BYTES:
        raise _ResultLimitError("field exceeds byte limit")


def _exception_type(error: Exception) -> str:
    name = type(error).__name__
    return name if len(name.encode()) <= MAX_FIELD_BYTES else "Exception"
