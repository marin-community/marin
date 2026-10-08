# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and decode rows from a staged source and its staged auxiliary inputs."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rigging.filesystem.storage_path import StoragePath

type StagedInputs = Mapping[str, StoragePath]
"""Staged auxiliary input roots by name."""


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
    Each callable also receives the staged auxiliary inputs by name.
    """

    dataset: str
    revision: str
    patterns: tuple[str, ...]
    format: SourceFormat
    select: Callable[[dict[str, Any], StagedInputs], bool] | None = None
    decode: Callable[[dict[str, Any], StagedInputs], dict[str, Any]] | None = None
    read: Callable[[StoragePath, StagedInputs], Iterator[dict[str, Any]]] | None = None
