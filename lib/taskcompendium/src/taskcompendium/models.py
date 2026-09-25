# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = "0.3"


class AnswerType(StrEnum):
    """The kind of result the task asks the model to produce."""

    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    WORKSPACE_STATE = "workspace_state"
    NATIVE_ACTION = "native_action"


class Source(BaseModel):
    """Pinned provenance for the source row and the importer that converted it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    revision: str
    row: str
    importer_revision: str

    @model_validator(mode="after")
    def validate_source(self) -> "Source":
        if not all((self.dataset, self.revision, self.row, self.importer_revision)):
            raise ValueError("Complete source provenance is required")
        return self


class ExactAnswer(BaseModel):
    """An exact reference answer and its text-normalization rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: str
    ignore_case: bool = True
    ignore_whitespace: bool = True

    @model_validator(mode="after")
    def validate_expected(self) -> "ExactAnswer":
        if not self.expected.strip():
            raise ValueError("An exact answer is required")
        return self


class TaskRequirements(BaseModel):
    """Environment functionality required to run the task.

    ``capabilities`` contains generic operations such as ``filesystem`` or
    ``shell``. ``action_interfaces`` contains named stateful tool surfaces such
    as ``workplace:v1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[str, ...] = ()
    action_interfaces: tuple[str, ...] = ()


class TaskSpec(BaseModel):
    """The private definition of one deterministic answer task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    instructions: str
    verifier: ExactAnswer
    source: Source
    requirements: TaskRequirements
    answer_type: AnswerType
    permitted_submission_conventions: tuple[str, ...] | None = None
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_specification(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
        if self.permitted_submission_conventions is not None:
            if not self.permitted_submission_conventions:
                raise ValueError("At least one submission convention must be permitted")
            if len(set(self.permitted_submission_conventions)) != len(self.permitted_submission_conventions):
                raise ValueError("Permitted submission conventions must be unique")
        return self
