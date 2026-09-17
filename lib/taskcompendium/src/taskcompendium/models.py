# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic, single-turn answer task."""

from dataclasses import dataclass

SCHEMA_VERSION = "0.1"


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
class ExactAnswer:
    """Private reference and comparison rules, never part of rendered instructions."""

    expected: str
    ignore_case: bool = True
    ignore_whitespace: bool = True

    def __post_init__(self) -> None:
        if not self.expected.strip():
            raise ValueError("An exact answer is required")


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
    verifier: ExactAnswer
    source: Source
    requirements: TaskRequirements
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("A task id and instructions are required")
