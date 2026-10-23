"""Registry-wide conformance to Evalchemy's shared task lifecycle."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from lm_eval.tasks import TaskManager as LMEvalTaskManager
from eval.sources import SOURCES

from .preflight import ResourceRequirement
from .task_outcome import TaskRoute


@dataclass(frozen=True)
class TaskContract:
    """One task name bound to a framework-owned lifecycle route."""

    task_name: str
    route: TaskRoute

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task contract name must not be empty")
        object.__setattr__(self, "route", TaskRoute(self.route))


def validate_task_contracts(contracts: Iterable[TaskContract]) -> None:
    """Reject duplicate or ambiguous task-to-route bindings."""
    names: dict[str, TaskRoute] = {}
    for contract in contracts:
        if contract.task_name in names:
            prior_route = names[contract.task_name]
            if prior_route is not contract.route:
                raise ValueError(
                    f"task {contract.task_name!r} belongs to both {prior_route.value} and {contract.route.value}"
                )
            raise ValueError(f"duplicate task contract: {contract.task_name}")
        names[contract.task_name] = contract.route


def build_task_contract_registry(
    custom_tasks: Iterable[str],
    lm_eval_tasks: Iterable[str],
) -> tuple[TaskContract, ...]:
    """Enumerate every task from both registries under one lifecycle schema."""
    contracts = tuple(
        [TaskContract(name, TaskRoute.CUSTOM) for name in sorted(custom_tasks)]
        + [TaskContract(name, TaskRoute.LM_EVAL) for name in sorted(lm_eval_tasks)]
    )
    validate_task_contracts(contracts)
    return contracts


def discover_task_contracts(lm_eval_include_dir: Path) -> tuple[TaskContract, ...]:
    """Discover registered task names, including names sharing a source module."""
    lm_eval_names = set(LMEvalTaskManager(include_path=[str(lm_eval_include_dir)]).all_tasks)
    return build_task_contract_registry(SOURCES, lm_eval_names)


def find_custom_benchmark_classes(
    module: ModuleType,
    base_class: type,
) -> list[type]:
    """Return benchmark subclasses defined by the supplied module, including abstract ones."""
    return [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if issubclass(candidate, base_class)
        and candidate is not base_class
        and candidate.__module__ == module.__name__
    ]


def validate_custom_benchmark_class(benchmark_class: type, base_class: type) -> None:
    """Audit a loaded custom benchmark's framework-facing declarations."""
    if (
        not issubclass(benchmark_class, base_class)
        or benchmark_class is base_class
        or inspect.isabstract(benchmark_class)
    ):
        raise TypeError("custom benchmark must be a concrete BaseBenchmark subclass")
    requirements = benchmark_class.RESOURCE_REQUIREMENTS
    if not isinstance(requirements, tuple):
        raise TypeError("custom benchmark RESOURCE_REQUIREMENTS must be a tuple")
    for requirement in requirements:
        if not isinstance(requirement, ResourceRequirement):
            raise TypeError("custom benchmark resource requirements must implement the shared contract")
    for method_name in ("prepare", "describe", "iter_specs"):
        if not callable(getattr(benchmark_class, method_name, None)):
            raise TypeError(f"custom benchmark is missing lifecycle method: {method_name}")
