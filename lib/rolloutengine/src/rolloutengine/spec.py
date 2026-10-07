# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serializable runtime configuration for one task attempt."""

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import NetworkPolicy
from taskcompendium.models import TaskSpec


class MachineRuntimeSpec(BaseModel):
    """Deployment settings for one Shellbox machine."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    backend: str = Field(min_length=1)
    network: NetworkPolicy
    cpus: int | None = Field(gt=0)
    memory_mb: int | None = Field(gt=0)
    storage_mb: int | None = Field(gt=0)
    gpus: int = Field(ge=0)
    user: str | None
    startup_timeout: float | None = Field(gt=0)
    cleanup_timeout: float | None = Field(gt=0)


class TaskRuntimeSpec(BaseModel):
    """Machine selections for task execution and private verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_machine: MachineRuntimeSpec | None
    verifier_machine: MachineRuntimeSpec | None


class TaskSessionSpec(BaseModel):
    """Session selection and deadlines for one attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    task_session: str = Field(min_length=1)
    max_turns: int = Field(gt=0)
    model_turn_timeout: float | None = Field(gt=0)
    tool_turn_timeout: float | None = Field(gt=0)
    total_turn_timeout: float | None = Field(gt=0)
    attempt_timeout: float | None = Field(gt=0)
    verifier_timeout: float | None = Field(gt=0)
    cleanup_timeout: float = Field(gt=0)


class LoweredTaskSpec(BaseModel):
    """An unchanged task definition and its selected runtime configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskSpec
    runtime: TaskRuntimeSpec
    session: TaskSessionSpec
