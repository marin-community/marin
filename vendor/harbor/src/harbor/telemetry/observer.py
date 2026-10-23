"""Thin, failure-isolated Harbor telemetry hooks."""

import contextlib
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from harbor.models.job.config import TelemetryConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent

logger = logging.getLogger(__name__)

_MAX_EVENT_TEXT_BYTES = 4096
_SHUTDOWN_TIMEOUT_SEC = 2.0
_DRAIN_POLL_INTERVAL_SEC = 0.01
_PHASE_DURATION_METRIC = "phase_duration_seconds"
_WORK_COMPLETED_METRIC = "work_completed"
_FAILURES_METRIC = "failures"
_CRITICAL_PATH_CLOCK_DOMAIN = "critical_path"
_WORK_UNIT = "{item}"
_JOB_SCOPE = "job"
_TRIAL_SCOPE = "trial"

FAILURE_KIND_LLM_REQUEST = "llm_request"
FAILURE_KIND_QUEUE_WAIT = "queue_wait"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_FAILED = "failed"
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_TIMEOUT = "timeout"


@dataclass(frozen=True)
class AttemptIdentity:
    trial_name: str
    planned_attempt: int
    infrastructure_retry: int


@dataclass(frozen=True)
class TelemetryRuntime:
    job_uid: str
    started_monotonic: float


_CURRENT_RUNTIME: ContextVar[TelemetryRuntime | None] = ContextVar(
    "harbor_telemetry_runtime", default=None
)
_CURRENT_ATTEMPT: ContextVar[AttemptIdentity | None] = ContextVar(
    "harbor_telemetry_attempt", default=None
)


def start_telemetry(
    config: TelemetryConfig | None,
    *,
    job_uid: UUID,
    job_name: str,
) -> TelemetryRuntime | None:
    """Start a job-owned exporter, or return None when export is unavailable."""
    if config is None:
        return None
    try:
        from rigging import telemetry
    except ImportError as error:
        logger.warning(
            "Harbor telemetry is configured but the telemetry extra is unavailable "
            "(%s)",
            type(error).__name__,
        )
        return None

    if telemetry.runtime_status().configured:
        logger.warning(
            "Harbor telemetry is already configured by another owner; "
            "leaving that exporter unchanged"
        )
        return None
    # Finelog indexes on the run_id column, so root_run_uid ships under that key.
    attributes = {
        "execution_uid": config.execution_uid,
        "harbor_job_name": job_name,
        "harbor_job_uid": str(job_uid),
        "role": "orchestrator",
        "run_id": config.root_run_uid,
    }
    if config.serving_job_id is not None:
        attributes["serving_job_id"] = config.serving_job_id
    telemetry.configure(
        endpoint=config.endpoint.get_secret_value(),
        service="harbor",
        attributes=attributes,
    )
    if not telemetry.runtime_status().configured:
        logger.warning("Harbor telemetry configuration did not start an exporter")
        return None

    runtime = TelemetryRuntime(
        job_uid=str(job_uid),
        started_monotonic=time.monotonic(),
    )
    telemetry.event("job_started", {"harbor_job_uid": str(job_uid)})
    return runtime


@contextlib.contextmanager
def bind_telemetry(
    runtime: TelemetryRuntime | None,
) -> Iterator[None]:
    runtime_token = _CURRENT_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _CURRENT_RUNTIME.reset(runtime_token)


@contextlib.contextmanager
def bind_attempt(attempt: AttemptIdentity) -> Iterator[None]:
    attempt_token = _CURRENT_ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _CURRENT_ATTEMPT.reset(attempt_token)


def record_job_terminal(
    runtime: TelemetryRuntime | None,
    error: BaseException | None = None,
) -> None:
    if runtime is None:
        return
    from rigging import telemetry

    if error is None:
        outcome = OUTCOME_SUCCEEDED
    elif isinstance(error, asyncio.CancelledError):
        outcome = OUTCOME_CANCELLED
    else:
        outcome = OUTCOME_FAILED
    duration = max(0.0, time.monotonic() - runtime.started_monotonic)
    attributes = {
        "clock_domain": _CRITICAL_PATH_CLOCK_DOMAIN,
        "outcome": outcome,
        "phase": _JOB_SCOPE,
    }
    telemetry.histogram(_PHASE_DURATION_METRIC, unit="s").record(
        duration, attributes=attributes
    )
    telemetry.counter(_WORK_COMPLETED_METRIC, unit=_WORK_UNIT).add(
        attributes={"outcome": outcome, "work_kind": _JOB_SCOPE}
    )
    body: dict[str, Any] = {
        "duration_seconds": duration,
        "harbor_job_uid": runtime.job_uid,
        "outcome": outcome,
    }
    if error is not None:
        body.update(_error_body(error))
    telemetry.event("job_terminal", body, attributes={"outcome": outcome})


def trial_hook(
    runtime: TelemetryRuntime | None,
) -> Callable[[TrialHookEvent], Awaitable[None]]:
    async def emit(event: TrialHookEvent) -> None:
        if runtime is None:
            return
        try:
            _record_trial_event(runtime, event)
        except Exception as error:
            logger.warning(
                "Harbor trial telemetry hook failed (%s); trial execution is unchanged",
                type(error).__name__,
            )

    return emit


def _record_trial_event(runtime: TelemetryRuntime, event: TrialHookEvent) -> None:
    from rigging import telemetry

    body = _attempt_body(runtime, event.result)
    body.setdefault("harbor_trial_name", event.trial_id)
    body["task_name"] = event.task_name
    if event.event != TrialEvent.END or event.result is None:
        telemetry.event(f"trial_{event.event.value}", body)
        return

    outcome = _result_outcome(event.result)
    body["outcome"] = outcome
    duration = _trial_duration(event.result)
    if duration is not None:
        body["duration_seconds"] = duration
        telemetry.histogram(_PHASE_DURATION_METRIC, unit="s").record(
            duration,
            attributes={
                "clock_domain": _CRITICAL_PATH_CLOCK_DOMAIN,
                "outcome": outcome,
                "phase": _TRIAL_SCOPE,
            },
        )
    telemetry.counter(_WORK_COMPLETED_METRIC, unit=_WORK_UNIT).add(
        attributes={"outcome": outcome, "work_kind": _TRIAL_SCOPE}
    )
    telemetry.event("trial_terminal", body, attributes={"outcome": outcome})


def record_failure(
    failure_kind: str,
    error: BaseException,
) -> None:
    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        return
    from rigging import telemetry

    status_code = getattr(error, "status_code", None)
    if failure_kind == FAILURE_KIND_QUEUE_WAIT:
        status = OUTCOME_CANCELLED
    elif isinstance(status_code, int):
        status = f"{status_code // 100}xx"
    else:
        status = "transport_error"
    attributes = {"failure_kind": failure_kind, "status": status}
    telemetry.counter(_FAILURES_METRIC).add(attributes=attributes)
    telemetry.event(
        "failure",
        {
            **_attempt_body(runtime),
            **_error_body(error),
            "failure_kind": failure_kind,
            "status_code": status_code,
        },
        attributes=attributes,
    )


def record_queue_wait(duration: float, *, outcome: str) -> None:
    """Record time spent waiting for a trial-attempt admission permit."""
    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        return
    from rigging import telemetry

    attributes = {
        "clock_domain": _CRITICAL_PATH_CLOCK_DOMAIN,
        "outcome": outcome,
        "phase": FAILURE_KIND_QUEUE_WAIT,
    }
    telemetry.histogram(_PHASE_DURATION_METRIC, unit="s").record(
        max(0.0, duration), attributes=attributes
    )
    telemetry.event(
        "trial_queue_wait",
        {
            **_attempt_body(runtime),
            "duration_seconds": max(0.0, duration),
            "outcome": outcome,
        },
        attributes={"outcome": outcome},
    )


def shutdown_telemetry(runtime: TelemetryRuntime | None) -> None:
    if runtime is None:
        return
    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SEC
    _wait_for_export_queue(deadline)
    from rigging import telemetry

    telemetry.shutdown(max(0.0, deadline - time.monotonic()))


def _wait_for_export_queue(deadline: float) -> None:
    from rigging import telemetry

    status = telemetry.runtime_status()
    while status.configured and status.queued_records:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_DRAIN_POLL_INTERVAL_SEC, remaining))
        status = telemetry.runtime_status()


def _attempt_body(
    runtime: TelemetryRuntime,
    result: TrialResult | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"harbor_job_uid": runtime.job_uid}
    attempt = _CURRENT_ATTEMPT.get()
    trial_name = None
    if attempt is not None:
        body.update(
            {
                "infrastructure_retry": attempt.infrastructure_retry,
                "planned_attempt": attempt.planned_attempt,
            }
        )
        trial_name = attempt.trial_name
    if result is not None:
        body.update(
            {
                "result_uid": str(result.id),
                "result_uri": str(result.trial_uri),
            }
        )
        trial_name = result.trial_name
        if result.exception_info is not None:
            body.update(
                {
                    "error_type": result.exception_info.exception_type,
                    "message": _bound_text(result.exception_info.exception_message),
                }
            )
    if trial_name is not None:
        body["harbor_trial_name"] = trial_name
    return body


def _trial_duration(result: TrialResult) -> float | None:
    if result.started_at is None or result.finished_at is None:
        return None
    return max(0.0, (result.finished_at - result.started_at).total_seconds())


def _result_outcome(result: TrialResult) -> str:
    info = result.exception_info
    if info is None:
        return OUTCOME_SUCCEEDED
    if info.exception_type == "CancelledError":
        return OUTCOME_CANCELLED
    if "Timeout" in info.exception_type:
        return OUTCOME_TIMEOUT
    return OUTCOME_FAILED


def _error_body(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "message": _bound_text(str(error)),
    }


def _bound_text(value: str) -> str:
    # Exception messages can embed agent output containing unpaired Unicode
    # surrogates; escape rather than raise so event emission never fails.
    encoded = value.encode("utf-8", errors="backslashreplace")
    return encoded[:_MAX_EVENT_TEXT_BYTES].decode("utf-8", "ignore")
