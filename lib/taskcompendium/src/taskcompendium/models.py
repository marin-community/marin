# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

SCHEMA_VERSION = "0.3"


class AnswerFormat(StrEnum):
    """A model-visible envelope for the semantic answer."""

    PLAIN = "plain"
    JSON = "json"
    FINAL_ACTION = "final_action"


@dataclass(frozen=True)
class Source:
    """Pinned provenance for the source row and the importer that converted it."""

    dataset: str
    revision: str
    row: str
    importer_revision: str

    def __post_init__(self) -> None:
        if not all((self.dataset, self.revision, self.row, self.importer_revision)):
            raise ValueError("Complete source provenance is required")


@dataclass(frozen=True)
class VerifierSpec:
    """A private verifier kind and the parameters passed to its handler."""

    kind: str
    parameters: dict[str, Any]


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


@dataclass(frozen=True)
class NativeFunction:
    """Advertised output function, without an execution binding."""

    name: str
    parameters: dict[str, Any]
    description: str | None = None
    strict: bool | None = None


@dataclass(frozen=True)
class TaskRequirements:
    """Environment functionality required to run the task.

    ``capabilities`` contains generic operations such as ``filesystem`` or
    ``shell``. ``action_interfaces`` contains named stateful tool surfaces such
    as ``workplace:v1``.
    """

    capabilities: tuple[str, ...] = ()
    action_interfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """The private definition of one deterministic answer task."""

    id: str
    instructions: str
    verifier: VerifierSpec
    source: Source
    requirements: TaskRequirements
    permitted_answer_formats: tuple[AnswerFormat, ...] = (AnswerFormat.PLAIN,)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
        if not self.permitted_answer_formats:
            raise ValueError("At least one answer format must be permitted")
        if len(set(self.permitted_answer_formats)) != len(self.permitted_answer_formats):
            raise ValueError("Permitted answer formats must be unique")
