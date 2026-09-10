# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading TaskTrove source parquets one shard at a time.

A shard is a run of row groups in one ``<source>/tasks.parquet`` holding about
``ROWS_PER_SHARD`` tasks. Sources differ by three orders of magnitude in rows per row group, so
sharding on row counts rather than files keeps every stage's unit of work about the same size.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

TASKS_GLOB = "*/tasks.parquet"
ROWS_PER_SHARD = 2000
"""Row groups of one source parquet are grouped into shards of about this many tasks, so a
600k-task source spreads over hundreds of workers instead of one."""


@dataclass(frozen=True)
class TaskRow:
    """One TaskTrove row and where it sits in its source parquet."""

    source: str
    path: str
    row_group: int
    row_in_group: int
    task_binary: bytes


@dataclass(frozen=True)
class TaskShard:
    """A contiguous run of row groups in one source parquet: the unit of work for every stage."""

    parquet_path: str
    first_row_group: int
    end_row_group: int


def source_name(parquet_path: str) -> str:
    return StoragePath(parquet_path).parent.name


def task_shards(input_path: str, rows_per_shard: int = ROWS_PER_SHARD) -> list[TaskShard]:
    """Split every source parquet under ``input_path`` into shards of about ``rows_per_shard`` tasks."""
    shards: list[TaskShard] = []
    for parquet in sorted((StoragePath(input_path) / TASKS_GLOB).glob(), key=str):
        with parquet.open("rb") as handle:
            metadata = pq.ParquetFile(handle).metadata
        start, rows = 0, 0
        for rg in range(metadata.num_row_groups):
            rows += metadata.row_group(rg).num_rows
            if rows >= rows_per_shard:
                shards.append(TaskShard(str(parquet), start, rg + 1))
                start, rows = rg + 1, 0
        if start < metadata.num_row_groups:
            shards.append(TaskShard(str(parquet), start, metadata.num_row_groups))
    return shards


def iter_task_rows(parquet_path: str, first_row_group: int = 0, end_row_group: int | None = None) -> Iterator[TaskRow]:
    """Yield the tasks in one source parquet's row groups, one row group in memory at a time."""
    source = source_name(parquet_path)
    with StoragePath(parquet_path).open("rb") as handle:
        pf = pq.ParquetFile(handle)
        stop = pf.num_row_groups if end_row_group is None else end_row_group
        for rg in range(first_row_group, stop):
            table = pf.read_row_group(rg, columns=["path", "task_binary"])
            paths = table.column("path").to_pylist()
            blobs = table.column("task_binary").to_pylist()
            for i, (path, blob) in enumerate(zip(paths, blobs, strict=True)):
                yield TaskRow(source, path, rg, i, blob)


def iter_shard_rows(shard: TaskShard) -> Iterator[TaskRow]:
    return iter_task_rows(shard.parquet_path, shard.first_row_group, shard.end_row_group)
