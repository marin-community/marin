# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

SCHEMA_VERSION = "0.4"


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


class VerifierSpec(BaseModel):
    """A private verifier kind and immutable JSON parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    parameters_json: str = Field(alias="parameters", repr=False)

    @field_validator("parameters_json", mode="before")
    @classmethod
    def freeze_parameters(cls, value: Mapping[str, Any]) -> str:
        if not isinstance(value, Mapping):
            raise ValueError("Verifier parameters must be a mapping")
        return json.dumps(value, sort_keys=True, allow_nan=False)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {"kind": self.kind, "parameters": self.parameters}

    @property
    def parameters(self) -> dict[str, Any]:
        """Return a copy of the private parameter payload."""
        return json.loads(self.parameters_json)


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCallComparatorConfig:
    numeric_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.numeric_tolerance is not None and (
            isinstance(self.numeric_tolerance, bool)
            or not isfinite(self.numeric_tolerance)
            or self.numeric_tolerance < 0
        ):
            raise ValueError("Numeric tolerance must be finite and nonnegative")


class NativeFunction(BaseModel):
    """Advertised output function, without an execution binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameters: dict[str, Any]
    description: str | None = None
    strict: bool | None = None


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
    verifier: VerifierSpec
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
