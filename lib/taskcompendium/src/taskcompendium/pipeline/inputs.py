# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and decode rows from a staged source and its staged auxiliary inputs."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import EnvironmentRequirements

type StagedInputs = Mapping[str, StoragePath]
"""Staged auxiliary input roots by name."""


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
    intended rows. ``read`` replaces the format reader for files that are not one record per row.
    Each callable also receives the source's conversion context.
    """

    dataset: str
    revision: str
    patterns: tuple[str, ...]
    format: SourceFormat
    select: Callable[[dict[str, Any], ConversionContext], bool] | None = None
    decode: Callable[[dict[str, Any], ConversionContext], dict[str, Any]] | None = None
    read: Callable[[StoragePath, ConversionContext], Iterator[dict[str, Any]]] | None = None
