# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.resources import SHA256_PATTERN, TaskResource, validate_resource_paths

SCHEMA_VERSION = "0.5"


class AnswerType(StrEnum):
    """The kind of result the task asks the model to produce."""

    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    WORKSPACE_STATE = "workspace_state"
    STATE = "state"
    NATIVE_ACTION = "native_action"


class VerifierKind(StrEnum):
    """The registered grader used to check a submission."""

    EXACT_ANSWER = "exact_answer"
    STATE_MATCH = "state_match"


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


class VerifierSpec(BaseModel):
    """A private verifier selection and its pinned configuration.

    ``kind`` selects a verifier class. ``parameters_json`` is its private
    JSON-encoded configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: VerifierKind
    parameters_json: str = Field(repr=False)


class TaskRequirements(BaseModel):
    """Environment functionality required to run the task.

    ``capabilities`` contains generic operations such as ``filesystem`` or
    ``shell``. ``action_interfaces`` contains named stateful tool surfaces such
    as ``workplace:v1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[str, ...] = ()
    action_interfaces: tuple[str, ...] = ()
    seed_sha256: str | None = None

    @model_validator(mode="after")
    def validate_seed(self) -> "TaskRequirements":
        if self.seed_sha256 is not None and not SHA256_PATTERN.fullmatch(self.seed_sha256):
            raise ValueError("An immutable seed requires a lowercase SHA256 digest")
        return self


class TaskSpec(BaseModel):
    """The private definition of one deterministic answer task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    instructions: str
    verifier: VerifierSpec
    source: Source
    requirements: TaskRequirements
    answer_type: AnswerType
    resources: tuple[TaskResource, ...] = ()
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_specification(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
        validate_resource_paths(self.resources)
        return self
