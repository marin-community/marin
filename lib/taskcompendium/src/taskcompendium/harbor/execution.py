# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read source-declared Harbor limits without filling harness defaults."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from taskcompendium.models import ExecutionRequirements, TaskSpec


@dataclass(frozen=True)
class NativeExecutionLimits:
    """Source allocation fields, including incomplete declarations."""

    cpus: int | float | None
    memory_mb: int | float | None
    storage_mb: int | float | None
    agent_timeout_sec: int | float | None
    verifier_timeout_sec: int | float | None

    @property
    def declared(self) -> dict[str, int | float]:
        return {name: value for name, value in asdict(self).items() if value is not None}

    @property
    def complete(self) -> bool:
        return len(self.declared) == len(ExecutionRequirements.model_fields)

    @property
    def requirements(self) -> ExecutionRequirements | None:
        return ExecutionRequirements.model_validate(self.declared) if self.complete else None


def native_execution_limits(config: Mapping[str, Any]) -> NativeExecutionLimits:
    """Read only allocation keys explicitly present in the source task configuration."""
    environment = config.get("environment") or {}
    return NativeExecutionLimits(
        cpus=environment.get("cpus"),
        memory_mb=environment.get("memory_mb"),
        storage_mb=environment.get("storage_mb"),
        agent_timeout_sec=(config.get("agent") or {}).get("timeout_sec"),
        verifier_timeout_sec=(config.get("verifier") or {}).get("timeout_sec"),
    )


def task_with_declared_execution_requirements(task: TaskSpec, limits: NativeExecutionLimits) -> TaskSpec:
    """Lift complete source declarations while rejecting conflicting existing requirements."""
    if task.execution_requirements is not None:
        for name, value in limits.declared.items():
            if getattr(task.execution_requirements, name) != value:
                raise ValueError(f"{task.id}: {name} conflicts with its source-declared execution limit")
        return task
    requirements = limits.requirements
    return task if requirements is None else task.model_copy(update={"execution_requirements": requirements})
