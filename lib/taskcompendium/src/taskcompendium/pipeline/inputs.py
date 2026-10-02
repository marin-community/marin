# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned downloads and staged record selection owned by a recipe."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rigging.filesystem.storage_path import StoragePath


class SourceFormat(StrEnum):
    PARQUET = "parquet"
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    GENERATED = "generated"


@dataclass(frozen=True)
class SourceFiles:
    patterns: tuple[str, ...]
    format: SourceFormat
    selector: Callable[[dict[str, Any], StoragePath], bool] | None = None
    decoder: Callable[[dict[str, Any], StoragePath], dict[str, Any]] | None = None


@dataclass(frozen=True)
class HubDownload:
    dataset: str
    revision: str
    patterns: tuple[str, ...]
    subdirectory: str = ""


@dataclass(frozen=True)
class UrlDownload:
    url: str
    filename: str


@dataclass(frozen=True)
class RecipeInputs:
    files: SourceFiles
    downloads: tuple[HubDownload | UrlDownload, ...]


def hub_inputs(dataset: str, revision: str, files: SourceFiles) -> RecipeInputs:
    return RecipeInputs(files, (HubDownload(dataset, revision, files.patterns),))
