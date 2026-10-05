# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution settings supplied separately from a task definition."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.environment import EnvironmentCommand, EnvironmentFile, HealthcheckSpec, validate_environment_files


class StageExecution(BaseModel):
    """Prepare a stage and select its agent user and deadline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workdir_files: tuple[EnvironmentFile, ...] = ()
    setup: tuple[EnvironmentCommand, ...] = ()
    healthcheck: HealthcheckSpec | None = None
    agent_timeout: float | None = Field(default=None, gt=0)
    agent_user: str | None = None

    @model_validator(mode="after")
    def validate_files(self) -> "StageExecution":
        validate_environment_files(self.workdir_files)
        return self


class TaskExecution(BaseModel):
    """Deadlines, users, and stage preparation for one execution of a task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_timeout: float | None = Field(default=None, gt=0)
    agent_timeout: float | None = Field(default=None, gt=0)
    agent_user: str | None = None
    stages: dict[str, StageExecution] = Field(default_factory=dict)
