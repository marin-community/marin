# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = "0.2"


class AnswerFormat(StrEnum):
    """A model-visible envelope for the semantic answer."""

    PLAIN = "plain"
    JSON = "json"


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
    permitted_answer_formats: tuple[AnswerFormat, ...] = (AnswerFormat.PLAIN,)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_specification(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
        if not self.permitted_answer_formats:
            raise ValueError("At least one answer format must be permitted")
        if len(set(self.permitted_answer_formats)) != len(self.permitted_answer_formats):
            raise ValueError("Permitted answer formats must be unique")
        return self
