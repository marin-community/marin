"""Typed, model-independent preparation for every evaluation task route."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.resources import files
from typing import Any, Protocol, runtime_checkable

from .eval_spec import RequestKind
from .failures import FailureCategory, FailurePhase, ModelRequestValidationError, classify_task_exception
from .task_outcome import TaskFailure, TaskRoute

TASK_PREPARATION_SCHEMA_VERSION = 1


class ResourceKind(StrEnum):
    """Capabilities a benchmark may require before model execution."""

    PACKAGE_FILE = "package_file"
    PYTHON_DEPENDENCY = "python_dependency"
    NETWORK_DATASET = "network_dataset"


@runtime_checkable
class ResourceRequirement(Protocol):
    """A serializable capability checked during static preparation."""

    kind: ResourceKind

    def validate(self) -> None: ...

    def to_dict(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class PackageFileRequirement:
    package: str
    relative_path: str
    kind: ResourceKind = ResourceKind.PACKAGE_FILE

    def validate(self) -> None:
        if not self.relative_path:
            raise ValueError("package file relative_path must not be empty")
        resource = files(self.package).joinpath(self.relative_path)
        if not resource.is_file():
            raise FileNotFoundError(
                f"required package file does not exist: {self.package}:{self.relative_path}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "package": self.package, "relative_path": self.relative_path}


@dataclass(frozen=True)
class PythonDependencyRequirement:
    module_name: str
    kind: ResourceKind = ResourceKind.PYTHON_DEPENDENCY

    def validate(self) -> None:
        if importlib.util.find_spec(self.module_name) is None:
            raise ModuleNotFoundError(f"required Python dependency is unavailable: {self.module_name}")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "module_name": self.module_name}


@dataclass(frozen=True)
class NetworkDatasetRequirement:
    dataset_name: str
    kind: ResourceKind = ResourceKind.NETWORK_DATASET

    def validate(self) -> None:
        if not self.dataset_name:
            raise ValueError("network dataset locator must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "dataset_name": self.dataset_name}


class TaskPreparationStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskPreparation:
    """Terminal result of preparing one requested task before model startup."""

    task_name: str
    route: TaskRoute
    status: TaskPreparationStatus
    resources: tuple[ResourceRequirement, ...]
    failure: TaskFailure | None = None
    schema_version: int = TASK_PREPARATION_SCHEMA_VERSION

    @classmethod
    def failed(
        cls,
        task_name: str,
        route: TaskRoute,
        resources: Sequence[ResourceRequirement],
        exception: BaseException,
    ) -> "TaskPreparation":
        return cls(
            task_name=task_name,
            route=route,
            status=TaskPreparationStatus.FAILED,
            resources=tuple(resources),
            failure=TaskFailure(
                category=classify_task_exception(FailurePhase.PREPARATION, exception),
                message=str(exception),
                exception_type=type(exception).__name__,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "route": self.route,
            "status": self.status,
            "resources": [resource.to_dict() for resource in self.resources],
            "failure": asdict(self.failure) if self.failure is not None else None,
        }


class EvaluationPreflightError(RuntimeError):
    """Raised when one or more requested tasks cannot be prepared."""

    def __init__(self, preparations: Sequence[TaskPreparation]):
        self.preparations = tuple(preparations)
        failures = [
            f"{item.task_name}: {item.failure.message}"
            for item in preparations
            if item.status is TaskPreparationStatus.FAILED and item.failure is not None
        ]
        super().__init__("Evaluation preflight failed: " + "; ".join(failures))


def validate_model_request(request) -> None:
    """Validate the common request shape after model-specific prompt preparation."""
    if request.kind is not RequestKind.GENERATE:
        return
    prompt, parameters = request.prompt, request.generation
    string_prompt = isinstance(prompt, str) and bool(prompt.strip())
    chat_prompt = _is_non_empty_chat_prompt(prompt)
    if not string_prompt and not chat_prompt:
        raise ModelRequestValidationError(
            "generation request prompt must be non-empty text or a non-empty chat message sequence"
        )
    if not isinstance(parameters, Mapping):
        raise ModelRequestValidationError("generation request parameters must be a mapping")


def _is_non_empty_chat_prompt(prompt: Any) -> bool:
    """Recognize raw messages and lm-eval's serialized chat request wrapper."""
    serialized_prompt = getattr(prompt, "prompt", None)
    if isinstance(prompt, tuple) and len(prompt) == 1 and isinstance(serialized_prompt, str):
        try:
            prompt = json.loads(serialized_prompt)
        except json.JSONDecodeError as exc:
            raise ModelRequestValidationError("serialized chat request must contain valid JSON") from exc

    return (
        isinstance(prompt, Sequence)
        and not isinstance(prompt, (str, bytes))
        and bool(prompt)
        and all(isinstance(message, Mapping) for message in prompt)
        and any(
            isinstance(message.get("content"), str) and bool(message["content"].strip())
            for message in prompt
        )
    )


def prepare_task(
    task_name: str,
    route: TaskRoute,
    resources: Sequence[ResourceRequirement],
    validate: Callable[[], None],
) -> TaskPreparation:
    """Validate one task without constructing or contacting a model."""
    try:
        for resource in resources:
            resource.validate()
        validate()
    except Exception as exc:
        return TaskPreparation.failed(task_name, route, resources, exc)
    return TaskPreparation(
        task_name=task_name,
        route=route,
        status=TaskPreparationStatus.READY,
        resources=tuple(resources),
    )


def prepare_requested_tasks(
    task_names: Sequence[str],
    task_routes: Mapping[str, TaskRoute],
    custom_task_manager: "CustomTaskManager",
    lm_eval_task_manager: "LMEvalTaskManager",
) -> list[TaskPreparation]:
    """Prepare all requested routes and fail once with every typed fault."""
    preparations: list[TaskPreparation] = []
    load_failures = getattr(custom_task_manager, "load_failures", {})
    for task_name in task_names:
        route = TaskRoute(task_routes[task_name])
        if route is TaskRoute.CUSTOM:
            load_error = load_failures.get(task_name)
            if load_error is not None:
                preparations.append(TaskPreparation.failed(task_name, route, (), load_error))
                continue
            benchmark = custom_task_manager.get_benchmark(task_name)
            if benchmark is None:
                preparations.append(
                    TaskPreparation.failed(
                        task_name,
                        route,
                        (),
                        LookupError("custom benchmark was not constructed"),
                    )
                )
                continue
            preparations.append(benchmark.prepare(task_name))
            continue

        resources = (NetworkDatasetRequirement(task_name),)
        preparations.append(
            prepare_task(
                task_name,
                route,
                resources,
                lambda task_name=task_name: _load_lm_eval_task(lm_eval_task_manager, task_name),
            )
        )

    if any(item.status is TaskPreparationStatus.FAILED for item in preparations):
        raise EvaluationPreflightError(preparations)
    return preparations


class PreparedBenchmark(Protocol):
    def prepare(self, task_name: str | None = None) -> TaskPreparation: ...


class CustomTaskManager(Protocol):
    load_failures: Mapping[str, BaseException]

    def get_benchmark(self, task_name: str) -> PreparedBenchmark | None: ...


class LMEvalTaskManager(Protocol):
    def load_task_or_group(self, task_list: list[str]) -> Mapping[str, object]: ...


def validate_task_preparations(serialized: Any, expected_tasks: Sequence[str]) -> None:
    """Validate persisted preparation envelopes without re-running their checks."""
    if not isinstance(serialized, Mapping):
        raise TypeError("task_preparations must be a mapping")
    if not serialized:
        return  # Compatibility with result documents created before schema version 1.
    if set(serialized) != set(expected_tasks):
        raise ValueError("task preparation keys do not match task outcome keys")
    for task_name, value in serialized.items():
        _task_preparation_from_mapping(str(task_name), value)


def _task_preparation_from_mapping(task_name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("task preparation must be a mapping")
    if value.get("schema_version") != TASK_PREPARATION_SCHEMA_VERSION:
        raise ValueError(f"task preparation schema_version must be {TASK_PREPARATION_SCHEMA_VERSION}")
    if value.get("task_name") != task_name:
        raise ValueError("task preparation key does not match task_name")
    try:
        status = TaskPreparationStatus(value.get("status"))
        TaskRoute(value.get("route"))
    except (TypeError, ValueError):
        raise ValueError("task preparation has an unknown status or route") from None
    resources = value.get("resources")
    if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes)):
        raise TypeError("task preparation resources must be a sequence")
    for resource in resources:
        _validate_serialized_resource(resource)
    failure = value.get("failure")
    if status is TaskPreparationStatus.READY and failure is not None:
        raise ValueError("ready task preparation contains a failure")
    if status is TaskPreparationStatus.FAILED:
        valid_categories = {FailureCategory.PREPARATION, FailureCategory.RESOURCE}
        if not isinstance(failure, Mapping) or failure.get("category") not in valid_categories:
            raise ValueError("failed task preparation lacks a preparation or resource failure")


def _validate_serialized_resource(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("task preparation resource must be a mapping")
    try:
        kind = ResourceKind(value.get("kind"))
    except (TypeError, ValueError):
        raise ValueError("task preparation resource has an unknown kind") from None
    fields = {
        ResourceKind.PACKAGE_FILE: ("package", "relative_path"),
        ResourceKind.PYTHON_DEPENDENCY: ("module_name",),
        ResourceKind.NETWORK_DATASET: ("dataset_name",),
    }[kind]
    if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
        raise ValueError(f"task preparation {kind} resource has an invalid locator")


def _load_lm_eval_task(task_manager: LMEvalTaskManager, task_name: str) -> None:
    loaded = task_manager.load_task_or_group([task_name])
    if not isinstance(loaded, Mapping) or not loaded:
        raise LookupError(f"lm-eval did not construct requested task {task_name}")
