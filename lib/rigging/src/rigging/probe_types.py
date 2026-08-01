# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data and protocol contracts for bounded Rigging probes."""

import enum
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

DEFAULT_PROBE_INTERVAL = timedelta(minutes=10)
DEFAULT_NVIDIA_SMI_TIMEOUT = timedelta(seconds=5)
DEFAULT_NCCL_RAS_TIMEOUT = timedelta(seconds=8)
MAX_SUBPROCESS_OUTPUT_BYTES = 256 * 1024
MAX_RESULT_METRICS = 4_096
MAX_RESULT_EVENTS = 64
MAX_EVENT_BODY_BYTES = 2_048
MAX_EVENT_FIELDS = 32
MAX_ATTRIBUTE_FIELDS = 32
MAX_FIELD_BYTES = 256
MAX_SHUTDOWN_TIMEOUT = 5.0


class ProbeOutcome(enum.StrEnum):
    """Stable outcome vocabulary shared by every probe."""

    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class ProbeReason(enum.StrEnum):
    """Bounded failure and lifecycle reasons suitable for telemetry."""

    TOOL_MISSING = "tool_missing"
    NO_DEVICES = "no_devices"
    SUBPROCESS_TIMEOUT = "subprocess_timeout"
    OUTPUT_LIMIT = "output_limit"
    NONZERO_EXIT = "nonzero_exit"
    PARSE_ERROR = "parse_error"
    RESULT_LIMIT = "result_limit"
    INVALID_RESULT = "invalid_result"
    INTERNAL_ERROR = "internal_error"
    CLEAN_SHUTDOWN = "clean_shutdown"


class MetricKind(enum.StrEnum):
    """Metric instruments supported by normalized probe results."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class CommandStatus(enum.StrEnum):
    """Bounded subprocess boundary outcomes."""

    COMPLETED = "completed"
    NOT_FOUND = "not_found"
    TIMED_OUT = "timed_out"
    OUTPUT_LIMIT = "output_limit"
    FAILED = "failed"


@dataclass(frozen=True)
class CommandResult:
    """Bounded subprocess output and status."""

    status: CommandStatus
    stdout: bytes = b""
    returncode: int | None = None


class CommandRunner(Protocol):
    """Subprocess boundary used only by probe daemon threads."""

    def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> CommandResult: ...


def command_failure_collection(
    status: CommandStatus,
    *,
    timeout_metrics: tuple["ProbeMetric", ...] = (),
) -> "ProbeCollection | None":
    """Return the normalized failure for a non-completed command status."""
    if status is CommandStatus.NOT_FOUND:
        return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.TOOL_MISSING)
    if status is CommandStatus.TIMED_OUT:
        return ProbeCollection(
            ProbeOutcome.TIMED_OUT,
            ProbeReason.SUBPROCESS_TIMEOUT,
            metrics=timeout_metrics,
        )
    if status is CommandStatus.OUTPUT_LIMIT:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.OUTPUT_LIMIT)
    if status is CommandStatus.FAILED:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.INTERNAL_ERROR)
    return None


@dataclass(frozen=True)
class ProbeMetric:
    """One normalized metric emitted by a probe."""

    name: str
    kind: MetricKind
    value: float
    unit: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)


EventValue = str | int | float | bool


@dataclass(frozen=True)
class ProbeEvent:
    """One flat, bounded diagnostic event without raw command output."""

    name: str
    body: Mapping[str, EventValue]
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeCollection:
    """A probe adapter's normalized collection result."""

    outcome: ProbeOutcome
    reason: ProbeReason | None = None
    metrics: tuple[ProbeMetric, ...] = ()
    events: tuple[ProbeEvent, ...] = ()
    error_type: str = ""

    @classmethod
    def succeeded(
        cls,
        *,
        metrics: tuple[ProbeMetric, ...] = (),
        events: tuple[ProbeEvent, ...] = (),
    ) -> "ProbeCollection":
        return cls(ProbeOutcome.SUCCEEDED, metrics=metrics, events=events)


@dataclass(frozen=True)
class ProbeSample:
    """One timestamped, bounded collection observed by a probe sink."""

    probe: str
    outcome: ProbeOutcome
    reason: ProbeReason | None
    duration: float
    observed_time: float
    metrics: tuple[ProbeMetric, ...] = ()
    events: tuple[ProbeEvent, ...] = ()
    error_type: str = ""


@dataclass(frozen=True)
class ProbeManagerStatus:
    """Process-local probe worker and sink state."""

    live_workers: int
    sink_failures: int


class Probe(Protocol):
    """A configured probe collected by one isolated daemon thread."""

    @property
    def name(self) -> str: ...

    @property
    def interval(self) -> float: ...

    def collect(self, runner: CommandRunner) -> ProbeCollection: ...


class ProbeSink(Protocol):
    """Receives normalized samples on probe daemon threads."""

    def emit(self, sample: ProbeSample) -> None: ...


class ProbeClock(Protocol):
    """Clock and interruptible wait boundary for deterministic scheduling."""

    def monotonic(self) -> float: ...

    def time(self) -> float: ...

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Return True when the event wakes the wait, or False on timeout."""
        ...
