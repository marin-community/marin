# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

SCHEMA_VERSION = "0.2"


@dataclass(frozen=True)
class Source:
    dataset: str
    revision: str
    row: str
    importer_revision: str

    def __post_init__(self) -> None:
        if not all((self.dataset, self.revision, self.row, self.importer_revision)):
            raise ValueError("Complete source provenance is required")


@dataclass(frozen=True)
class VerifierSpec:
    """Private verifier identity and parameters; handlers register by kind."""

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
    """Operations and action interfaces the task needs from a target."""

    capabilities: tuple[str, ...] = ()
    action_interfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """One pinned semantic task, independent of output format or Harbor launch."""

    id: str
    instructions: str
    verifier: VerifierSpec
    source: Source
    requirements: TaskRequirements
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
