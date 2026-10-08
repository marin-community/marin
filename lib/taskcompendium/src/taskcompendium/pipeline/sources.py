# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode records from pinned source files staged by an acquisition artifact."""

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_jsonl, load_parquet

from taskcompendium.pipeline.fingerprints import callable_identity
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat, StagedInputs
from taskcompendium.pipeline.models import SourceRecipe


def source_files_identity(spec: SourceFiles) -> dict[str, Any]:
    """Stable artifact identity for a staged reader and its selection rules."""
    return {
        "revision": "2",
        "dataset": spec.dataset,
        "source_revision": spec.revision,
        "patterns": spec.patterns,
        "format": spec.format.value,
        "select": callable_identity(spec.select) if spec.select is not None else None,
        "decode": callable_identity(spec.decode) if spec.decode is not None else None,
        "read": callable_identity(spec.read) if spec.read is not None else None,
        # Present only for parted sources, so whole-file sources keep their identities.
        **({"parts": callable_identity(spec.parts)} if spec.parts is not None else {}),
    }


def staged_inputs(paths: Mapping[str, str]) -> StagedInputs:
    """Resolve staged auxiliary input paths for the source callables."""
    return {name: StoragePath(path) for name, path in paths.items()}


def conversion_context(recipe: SourceRecipe) -> ConversionContext:
    """The staged inputs and grader image environment the recipe's source callables receive."""
    return ConversionContext(staged_inputs(recipe.inputs), recipe.grader_environment)


def staged_files(path: str, spec: SourceFiles) -> tuple[str, ...]:
    """List selected files by pinned relative path, rejecting missing declarations."""
    root = StoragePath(path)
    _, root_path = url_to_fs(path)
    filesystem_root = StoragePath(root_path)
    files = set()
    for pattern in spec.patterns:
        for file in (root / pattern).glob():
            _, file_path = url_to_fs(str(file))
            relative = StoragePath(file_path).relative_to(filesystem_root)
            if not any(part.startswith(".") for part in relative.split("/")):
                files.add(relative)
    if not files:
        raise FileNotFoundError(f"No staged files match {spec.patterns} under {path}")
    return tuple(sorted(files))


def row_locator(file: str, index: int) -> str:
    """The stable locator of a staged file's row."""
    return f"{file}:{index}"


@dataclass(frozen=True)
class SourceShard:
    """One staged file, or one part of a file that its source reads in parts.

    ``indices`` restricts a part to those rows; ``None`` reads all of them.
    """

    file: str
    part: int
    parts: int
    indices: frozenset[int] | None

    @property
    def name(self) -> str:
        """A name unique among the source's shards: the file, with its part when it has several."""
        return self.file if self.parts == 1 else f"{self.file}#part-{self.part}"


def source_shards(path: str, spec: SourceFiles) -> tuple[SourceShard, ...]:
    """The independently readable shards of the selected files, in file and part order."""
    parts = spec.parts.count if spec.parts is not None else 1
    if parts < 1:
        raise ValueError("A parted source requires at least one part")
    return tuple(SourceShard(file, part, parts, None) for file in staged_files(path, spec) for part in range(parts))


def _decoded_rows(path: StoragePath, source_format: SourceFormat) -> Iterator[dict[str, Any]]:
    if source_format == SourceFormat.PARQUET:
        yield from load_parquet(str(path))
    elif source_format == SourceFormat.JSONL:
        yield from load_jsonl(str(path))
    elif source_format == SourceFormat.JSON:
        with path.open("rt") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON record array in {path}")
        yield from payload
    elif source_format == SourceFormat.CSV:
        with path.open("rt", encoding="utf-8-sig") as stream:
            yield from csv.DictReader(stream)
    else:
        raise ValueError(f"Unsupported staged source format: {source_format}")


def _indexed_rows(
    file: StoragePath, shard: SourceShard, spec: SourceFiles, context: ConversionContext
) -> Iterator[tuple[int, dict[str, Any]]]:
    if spec.parts is not None:
        return spec.parts(file, context, shard.part, shard.indices)
    assert shard.indices is None, "Only a parted source reads selected rows"
    return enumerate(spec.read(file, context) if spec.read is not None else _decoded_rows(file, spec.format))


def staged_raw_file_rows(
    path: str, shard: SourceShard, spec: SourceFiles, context: ConversionContext
) -> Iterator[dict[str, Any]]:
    """Yield one shard's selected source rows before decoding with their original stable locators."""
    relative_file = shard.file
    if relative_file.startswith("/") or ".." in relative_file.split("/"):
        raise ValueError(f"Source file must be relative to its staged root: {relative_file}")
    for index, row in _indexed_rows(StoragePath(path) / relative_file, shard, spec, context):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {relative_file}:{index}")
        if spec.select is not None and not spec.select(row, context):
            continue
        yield {"index": index, "locator": row_locator(relative_file, index), "data": row}


def decode_staged_row(record: dict[str, Any], spec: SourceFiles, context: ConversionContext) -> dict[str, Any]:
    """Apply a source decoder to an already selected row without changing its locator."""
    data = spec.decode(record["data"], context) if spec.decode is not None else record["data"]
    return {**record, "data": data}


def staged_file_rows(
    path: str, shard: SourceShard, spec: SourceFiles, context: ConversionContext
) -> Iterator[dict[str, Any]]:
    """Yield one shard's selected, decoded records with their original file and row locators."""
    for record in staged_raw_file_rows(path, shard, spec, context):
        yield decode_staged_row(record, spec, context)
