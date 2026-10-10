# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and decode rows from a staged source and its staged auxiliary inputs."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import EnvironmentRequirements

type StagedInputs = Mapping[str, StoragePath]
"""Staged auxiliary input roots by name."""


@dataclass(frozen=True)
class SourceFileOverride:
    """Actual local bytes substituted for one declared logical source filename."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ConversionContext:
    """What a source's callables receive with each file or row.

    ``inputs`` holds the staged auxiliary inputs by name. ``grader_environment`` runs the source's
    sandboxed graders in its grader image; it is ``None`` when the source names no grader image.
    """

    inputs: StagedInputs
    grader_environment: EnvironmentRequirements | None


def required_grader_environment(context: ConversionContext) -> EnvironmentRequirements:
    """The grader image environment, for a converter that ships a sandboxed grader."""
    if context.grader_environment is None:
        raise ValueError("A sandboxed grader requires the source to name a grader image")
    return context.grader_environment


class FileParts(Protocol):
    """Produce each staged file's rows in ``count`` independent parts, so that several workers share one file.

    A file holds ``size`` rows, indexed from 0 without producing them; the indices form the rows'
    locators. Each call yields one part's rows with their indices, only those in ``indices`` when
    given; together the parts yield every requested index once. A sample therefore draws its row
    indices first and produces only those rows.
    """

    @property
    def count(self) -> int: ...

    def size(self, file: StoragePath, context: ConversionContext) -> int: ...

    def __call__(
        self, file: StoragePath, context: ConversionContext, part: int, indices: frozenset[int] | None
    ) -> Iterator[tuple[int, dict[str, Any]]]: ...


class SourceFormat(StrEnum):
    PARQUET = "parquet"
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    GENERATED = "generated"


@dataclass(frozen=True)
class SourceFiles:
    """Rows of one staged source and the provenance recorded on each task.

    ``select`` and ``decode`` run on raw records before sampling, so a panel is drawn from the
    intended rows. ``read`` replaces the format reader for files that are not one record per row;
    ``parts`` replaces it for a file whose rows are expensive to produce but whose row count is known,
    such as a generator.
    Each callable also receives the source's conversion context.
    """

    dataset: str
    revision: str
    patterns: tuple[str, ...]
    format: SourceFormat
    select: Callable[[dict[str, Any], ConversionContext], bool] | None = None
    decode: Callable[[dict[str, Any], ConversionContext], dict[str, Any]] | None = None
    read: Callable[[StoragePath, ConversionContext], Iterator[dict[str, Any]]] | None = None
    parts: FileParts | None = None

    def __post_init__(self) -> None:
        if self.read is not None and self.parts is not None:
            raise ValueError("A source reads its files either whole or in parts")
        if self.select is not None and self.parts is not None:
            raise ValueError("A parted source samples rows by index before reading them, so it cannot select rows")
