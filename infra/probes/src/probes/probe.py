# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core types for the probes service: outcomes, samples, and the Probe protocol.

A Probe is a small piece of behaviour: ``run(deadline_seconds) -> ProbeResult``.
A ProbeSpec is its registration record (name, location, cadence, deadline).
The scheduler reads ProbeSpecs; the Probe.run callable does the work. This
split lets probes be tested without importing the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class ProbeOutcome(StrEnum):
    """Terminal classification of a single probe execution."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    REMOTE_ERROR = "remote_error"
    LOCAL_ERROR = "local_error"


class ErrorClass(StrEnum):
    """Stable, low-cardinality error taxonomy. Consumers (alerts, dashboards)
    may rely on these tokens; renaming a member is a spec amendment."""

    CONNECT_ERROR = "ConnectError"
    RPC_ERROR = "RpcError"
    HTTP_ERROR = "HttpError"
    JOB_FAILED = "JobFailed"
    JOB_CANCELLED = "JobCancelled"
    SUBMIT_REJECTED = "SubmitRejected"
    READBACK_MISMATCH = "ReadbackMismatch"
    LOCAL_CONFIG_ERROR = "LocalConfigError"
    TIMEOUT = "Timeout"


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one Probe.run() call. Latency is measured by the daemon."""

    outcome: ProbeOutcome
    error_class: ErrorClass | None = None
    error_detail: str | None = None
    target_id: str | None = None
    extras: dict[str, str | int | float] | None = None

    @classmethod
    def success(
        cls,
        *,
        target_id: str | None = None,
        extras: dict[str, str | int | float] | None = None,
    ) -> ProbeResult:
        return cls(outcome=ProbeOutcome.SUCCESS, target_id=target_id, extras=extras)

    @classmethod
    def remote_error(
        cls,
        error_class: ErrorClass,
        detail: str | None = None,
        *,
        target_id: str | None = None,
        extras: dict[str, str | int | float] | None = None,
    ) -> ProbeResult:
        return cls(
            outcome=ProbeOutcome.REMOTE_ERROR,
            error_class=error_class,
            error_detail=detail,
            target_id=target_id,
            extras=extras,
        )

    @classmethod
    def local_error(
        cls,
        detail: str | None = None,
        *,
        extras: dict[str, str | int | float] | None = None,
    ) -> ProbeResult:
        return cls(
            outcome=ProbeOutcome.LOCAL_ERROR,
            error_class=ErrorClass.LOCAL_CONFIG_ERROR,
            error_detail=detail,
            extras=extras,
        )


class Probe(Protocol):
    """A single check executed by the daemon. ``run`` blocks; the daemon
    enforces the deadline externally (cancellation event + hard kill at
    deadline + grace). Probes SHOULD honor the deadline as a soft cap on
    RPC timeouts and SHOULD prefer returning a ProbeResult to raising."""

    def run(self, deadline_seconds: float) -> ProbeResult: ...


@dataclass(frozen=True)
class ProbeSpec:
    """Registration record for a Probe instance — config only, no behaviour."""

    name: str
    kind: str
    location: str | None
    cadence_seconds: int
    deadline_seconds: float
    probe: Probe

    def __post_init__(self) -> None:
        if self.cadence_seconds < 10:
            raise ValueError(f"cadence_seconds must be >= 10, got {self.cadence_seconds}")
        if self.deadline_seconds <= 0:
            raise ValueError(f"deadline_seconds must be > 0, got {self.deadline_seconds}")
        if self.deadline_seconds >= self.cadence_seconds:
            raise ValueError(
                f"deadline_seconds ({self.deadline_seconds}) must be < cadence_seconds ({self.cadence_seconds})"
            )


@dataclass(frozen=True)
class ProbeSample:
    """The persisted record for one Probe.run() invocation."""

    timestamp: datetime
    probe_name: str
    probe_kind: str
    location: str | None
    outcome: ProbeOutcome
    latency_ms: int
    error_class: ErrorClass | None
    error_detail: str | None
    target_id: str | None
    extras_json: str
    daemon_instance: str
