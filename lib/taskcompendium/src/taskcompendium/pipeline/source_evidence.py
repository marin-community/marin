# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover selected original rows from the curation invocation's pinned inputs."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import open_file

from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.sources import (
    SourceShard,
    canonical_source_json,
    staged_raw_file_rows,
)

READ_BLOCK_BYTES = 1_048_576


@dataclass(frozen=True)
class OriginalRow:
    task_id: str
    locator: str
    raw_input_sha256: str | None


@dataclass(frozen=True)
class SourceEvidenceFile:
    file: str
    path: str
    rows: dict[int, OriginalRow]
    prepared_path: str | None = None


def _native_rows(path: str, selected: frozenset[int]) -> Iterator[tuple[int, dict[str, Any]]]:
    with open_file(path, block_size=READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            indices = sorted(index - offset for index in selected if offset <= index < offset + size)
            if indices:
                table = parquet.read_row_group(group).take(pa.array(indices, type=pa.int64()))
                yield from zip((offset + index for index in indices), table.to_pylist(), strict=True)
            offset += size


def _prepared_rows(path: str, selected: frozenset[int]) -> Iterator[tuple[int, dict[str, Any]]]:
    values = pa.array(sorted(selected), type=pa.int64())
    with open_file(path, block_size=READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        for group in range(parquet.num_row_groups):
            if parquet.metadata.row_group(group).num_rows == 0:
                continue
            identities = parquet.read_row_group(group, columns=["source_index"])
            matches = pc.call_function(
                "is_in", [identities["source_index"]], options=pc.SetLookupOptions(value_set=values)
            )
            indices = pc.call_function("indices_nonzero", [matches]).to_pylist()
            if indices:
                table = parquet.read_row_group(group, columns=["source_index", "raw_json"])
                for row in table.take(pa.array(indices, type=pa.int64())).to_pylist():
                    yield row["source_index"], json.loads(row["raw_json"])


def _source_rows(
    file: SourceEvidenceFile, spec: SourceFiles, context: ConversionContext
) -> Iterator[tuple[int, dict[str, Any]]]:
    selected = frozenset(file.rows)
    if file.prepared_path is not None:
        yield from _prepared_rows(file.prepared_path, selected)
        return
    if spec.format == SourceFormat.PARQUET and spec.read is None and spec.parts is None:
        yield from _native_rows(file.path, selected)
        return
    parts = spec.parts.count if spec.parts is not None else 1
    for part in range(parts):
        shard = SourceShard(file.file, part, parts, selected if spec.parts is not None else None, read_path=file.path)
        for row in staged_raw_file_rows(str(StoragePath(file.path).parent), shard, spec, context):
            if row["index"] in selected:
                yield row["index"], row["data"]


def original_rows(
    file: SourceEvidenceFile, *, spec: SourceFiles, context: ConversionContext, preview_characters: int
) -> Iterator[dict[str, Any]]:
    """Yield bounded original-row evidence, rejecting changed or missing source data."""
    found = set()
    for index, data in _source_rows(file, spec, context):
        if index in found:
            raise ValueError(f"Duplicate original source row: {file.file}:{index}")
        found.add(index)
        selected = file.rows[index]
        document = canonical_source_json(data)
        digest = hashlib.sha256(document.encode()).hexdigest()
        if selected.raw_input_sha256 is not None and selected.raw_input_sha256 != digest:
            if spec.format != SourceFormat.GENERATED or file.prepared_path is not None:
                raise ValueError(f"Original source row changed: {selected.locator}")
            yield {
                "task_id": selected.task_id,
                "evidence": {
                    "source": {
                        "locator": selected.locator,
                        "raw_input_sha256": selected.raw_input_sha256,
                        "regenerated_sha256": digest,
                        "available": False,
                    }
                },
                "omissions": ["source: original generated row could not be reproduced"],
            }
            continue
        truncated = len(document) > preview_characters
        yield {
            "task_id": selected.task_id,
            "evidence": {
                "source": {
                    "locator": selected.locator,
                    "available": True,
                    "raw_input_sha256": digest,
                    "characters": len(document),
                    "preview": document[:preview_characters],
                    "truncated": truncated,
                }
            },
            "omissions": ["source: original row preview truncated"] if truncated else [],
        }
    missing = set(file.rows) - found
    if missing:
        raise ValueError(f"Original source rows unavailable in {file.path}: {sorted(missing)}")
