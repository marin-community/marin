"""Typed task outcomes shared by custom and lm-eval benchmark routes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from numbers import Real
from typing import Any, Literal

from .failures import FailureCategory as FailureCategory
from .failures import FailurePhase as FailurePhase
from .failures import classify_task_exception as classify_task_exception

TASK_OUTCOME_SCHEMA_VERSION = 1


class TaskStatus(StrEnum):
    """Terminal state of one requested evaluation task."""

    SUCCEEDED = "succeeded"
    EXPORTED = "exported"
    FAILED = "failed"


class TaskRoute(StrEnum):
    """Adapter that executed a requested task."""

    CUSTOM = "Evalchemy chat benchmark"
    LM_EVAL = "lm-eval"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskFailure:
    """Machine-readable reason a task did not produce a valid outcome."""

    category: FailureCategory
    message: str
    exception_type: str | None = None


@dataclass(frozen=True)
class TaskOutcome:
    """Task metrics with distinct-problem counts and independently sampled-answer counts."""

    task_name: str
    route: TaskRoute
    status: TaskStatus
    metrics: Mapping[str, Any]
    expected_count: int | None
    generated_count: int | None
    scored_count: int | None
    failure: TaskFailure | None = None
    failure_counts: Mapping[FailureCategory, int] = field(default_factory=dict)
    completion_response_summary: Mapping[str, int] = field(default_factory=dict)
    expected_sample_count: int | None = None
    generated_sample_count: int | None = None
    scored_sample_count: int | None = None
    schema_version: Literal[1] = TASK_OUTCOME_SCHEMA_VERSION

    @classmethod
    def failed(
        cls,
        task_name: str,
        route: TaskRoute,
        category: FailureCategory,
        message: str,
        *,
        exception: BaseException | None = None,
    ) -> TaskOutcome:
        failure_message = message.strip()
        if not failure_message:
            failure_message = type(exception).__name__ if exception is not None else category.value
        return cls(
            task_name=task_name,
            route=route,
            status=TaskStatus.FAILED,
            metrics={},
            expected_count=None,
            generated_count=None,
            scored_count=None,
            failure=TaskFailure(
                category=category,
                message=failure_message,
                exception_type=type(exception).__name__ if exception is not None else None,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible task-outcome envelope."""
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "route": self.route,
            "status": self.status,
            "metrics": dict(self.metrics),
            "expected_count": self.expected_count,
            "generated_count": self.generated_count,
            "scored_count": self.scored_count,
            "failure": asdict(self.failure) if self.failure is not None else None,
            "failure_counts": dict(self.failure_counts),
            "completion_response_summary": dict(self.completion_response_summary),
            "expected_sample_count": self.expected_sample_count,
            "generated_sample_count": self.generated_sample_count,
            "scored_sample_count": self.scored_sample_count,
        }


class EvaluationRunError(RuntimeError):
    """Raised when a result package has missing or invalid outcome evidence."""

    def __init__(self, outcomes: Sequence[TaskOutcome]):
        self.outcomes = tuple(outcomes)
        failures = [
            f"{outcome.task_name}: {outcome.failure.message if outcome.failure else outcome.status}"
            for outcome in self.outcomes
            if outcome.status is TaskStatus.FAILED
        ]
        super().__init__("Evaluation did not complete: " + "; ".join(failures))


def validate_requested_outcomes(requested_tasks: Sequence[str], outcomes: Sequence[TaskOutcome]) -> None:
    """Require one outcome for each task while allowing reported task failures."""
    by_task: dict[str, list[TaskOutcome]] = {}
    for outcome in outcomes:
        by_task.setdefault(outcome.task_name, []).append(outcome)

    completed = list(outcomes)
    for task_name in dict.fromkeys(requested_tasks):
        task_outcomes = by_task.get(task_name, [])
        if not task_outcomes:
            completed.append(
                TaskOutcome.failed(
                    task_name,
                    TaskRoute.UNKNOWN,
                    FailureCategory.INCOMPLETE_EVALUATION,
                    "requested task produced no outcome",
                )
            )
        elif len(task_outcomes) > 1:
            completed.append(
                TaskOutcome.failed(
                    task_name,
                    task_outcomes[0].route,
                    FailureCategory.INVALID_RESULT,
                    "requested task produced multiple outcomes",
                )
            )

    if len(completed) != len(outcomes):
        raise EvaluationRunError(completed)


def validate_result_document(result: Mapping[str, Any]) -> None:
    """Reject aggregate persistence that bypasses the typed outcome boundary."""
    serialized = result.get("task_outcomes")
    if not isinstance(serialized, Mapping) or not serialized:
        raise EvaluationRunError(
            [
                TaskOutcome.failed(
                    "<run>",
                    TaskRoute.UNKNOWN,
                    FailureCategory.INCOMPLETE_EVALUATION,
                    "result document has no task outcomes",
                )
            ]
        )
    from .preflight import validate_task_preparations  # noqa: PLC0415 - break preflight/outcome cycle

    try:
        validate_task_preparations(result.get("task_preparations", {}), [str(name) for name in serialized])
    except (TypeError, ValueError) as exc:
        raise EvaluationRunError(
            [
                TaskOutcome.failed(
                    "<run>",
                    TaskRoute.UNKNOWN,
                    FailureCategory.INVALID_RESULT,
                    str(exc),
                )
            ]
        ) from exc
    failed = []
    for task_name, value in serialized.items():
        try:
            _task_outcome_from_mapping(str(task_name), value)
        except (TypeError, ValueError) as exc:
            failed.append(
                TaskOutcome.failed(
                    str(task_name),
                    _serialized_route(value),
                    FailureCategory.INVALID_RESULT,
                    str(exc),
                )
            )
    if failed:
        raise EvaluationRunError(failed)


def _task_outcome_from_mapping(task_name: str, value: Any) -> TaskOutcome:
    if not isinstance(value, Mapping):
        raise TypeError("task outcome must be a mapping")
    if value.get("schema_version") != TASK_OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"task outcome schema_version must be {TASK_OUTCOME_SCHEMA_VERSION}")
    if value.get("task_name") != task_name:
        raise ValueError("task outcome key does not match task_name")
    try:
        status = TaskStatus(value.get("status"))
        route = TaskRoute(value.get("route"))
    except (TypeError, ValueError):
        raise ValueError("task outcome has an unknown status or route") from None
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("task outcome metrics must be a mapping")
    failure_value = value.get("failure")
    if status is TaskStatus.FAILED:
        if not isinstance(failure_value, Mapping):
            raise ValueError("failed task outcome has no failure detail")
        try:
            failure = TaskFailure(
                category=FailureCategory(failure_value["category"]),
                message=failure_value["message"],
                exception_type=failure_value.get("exception_type"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("failed task outcome has invalid failure detail") from exc
        if not isinstance(failure.message, str) or not failure.message:
            raise ValueError("failed task outcome has no failure message")
        if failure.exception_type is not None and not isinstance(failure.exception_type, str):
            raise ValueError("failed task outcome has invalid exception type")
    elif failure_value is not None:
        raise ValueError("completed task outcome contains a failure")
    else:
        failure = None
    raw_failure_counts = value.get("failure_counts", {})
    if not isinstance(raw_failure_counts, Mapping):
        raise TypeError("task outcome failure_counts must be a mapping")
    try:
        failure_counts = {FailureCategory(category): count for category, count in raw_failure_counts.items()}
    except ValueError as exc:
        raise ValueError("task outcome has an unknown failure category") from exc
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in failure_counts.values()):
        raise ValueError("task outcome failure counts must be positive integers")
    response_summary = value.get("completion_response_summary", {})
    if not isinstance(response_summary, Mapping) or any(
        not isinstance(name, str) or not isinstance(count, int) or isinstance(count, bool) or count <= 0
        for name, count in response_summary.items()
    ):
        raise ValueError("task outcome completion_response_summary must contain positive counts")
    outcome = TaskOutcome(
        task_name=task_name,
        route=route,
        status=status,
        metrics=dict(metrics),
        expected_count=_serialized_count(value, "expected_count"),
        generated_count=_serialized_count(value, "generated_count"),
        scored_count=_serialized_count(value, "scored_count"),
        failure=failure,
        failure_counts=failure_counts,
        completion_response_summary=dict(response_summary),
        expected_sample_count=_serialized_count(value, "expected_sample_count"),
        generated_sample_count=_serialized_count(value, "generated_sample_count"),
        scored_sample_count=_serialized_count(value, "scored_sample_count"),
    )
    if violation := _outcome_contract_violation(outcome):
        raise ValueError(violation.message)
    return outcome


def _serialized_route(value: Any) -> TaskRoute:
    if not isinstance(value, Mapping):
        return TaskRoute.UNKNOWN
    try:
        return TaskRoute(value.get("route", TaskRoute.UNKNOWN))
    except ValueError:
        return TaskRoute.UNKNOWN


def _outcome_contract_violation(outcome: TaskOutcome) -> TaskFailure | None:
    if outcome.expected_sample_count is not None:
        generated = outcome.generated_sample_count
        scored = outcome.scored_sample_count
        if generated is None or generated > outcome.expected_sample_count:
            return TaskFailure(FailureCategory.INVALID_RESULT, "generated sample count exceeds requested samples")
        if scored is not None and scored > generated:
            return TaskFailure(FailureCategory.INVALID_RESULT, "scored sample count exceeds generated samples")
    if outcome.status is TaskStatus.FAILED:
        if outcome.failure is None:
            return TaskFailure(FailureCategory.INVALID_RESULT, "failed outcome has no failure detail")
        if outcome.metrics:
            return TaskFailure(FailureCategory.INVALID_RESULT, "failed outcome contains metrics")
        return None
    if outcome.failure is not None:
        return TaskFailure(FailureCategory.INVALID_RESULT, "completed task outcome contains a failure")
    if outcome.status is TaskStatus.EXPORTED:
        if outcome.generated_count == 0:
            return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task exported zero generations")
        return None
    if not _contains_finite_number(outcome.metrics):
        return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task returned no finite numeric metrics")
    if 0 in {
        count for count in (outcome.expected_count, outcome.generated_count, outcome.scored_count) if count is not None
    }:
        return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task reported zero sample coverage")
    return None


def _serialized_count(value: Mapping[str, Any], key: str) -> int | None:
    count = value.get(key)
    if count is None:
        return None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"task outcome {key} must be a non-negative integer or null")
    return count


def _contains_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_finite_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_finite_number(item) for item in value)
    return False
