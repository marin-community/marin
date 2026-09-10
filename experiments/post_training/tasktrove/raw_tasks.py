# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The raw TaskTrove parquets as a Zephyr dataset, plus the worker size every stage uses."""

from collections.abc import Iterator

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.dataset import Dataset
from zephyr.input_file import DEFAULT_FILE_PATH_COLUMN

TASKS_GLOB = "*/tasks.parquet"
APPROX_SHARD_BYTES = 32 << 20
"""Parquet files split into shards of about this many uncompressed bytes, cut at row-group
boundaries; a row group larger than this is one shard on its own."""
WORKING_SHARDS = 64
"""Loaded rows are shuffled by path into this many even shards before per-task work. Sixty-four
workers are enough for this few-gigabyte corpus while still splitting sources whose input is one
large row group. (``reshard`` alone moves whole intermediate chunks rather than individual rows.)"""
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="4g")
"""One Zephyr worker per shard. Twenty sources store every task in a single row group, the largest
about 400 MB uncompressed, and a worker holds one decoded row group plus its Python rows."""


def source_name(parquet_path: str) -> str:
    return StoragePath(parquet_path).parent.name


def _with_source(row: dict) -> dict:
    row["source"] = source_name(row.pop(DEFAULT_FILE_PATH_COLUMN))
    return row


def _row_key(row: dict) -> tuple[str, str]:
    return (row["source"], row["path"])


def _one_row(_key: tuple[str, str], rows: Iterator[dict]) -> Iterator[dict]:
    return rows


def raw_rows(input_path: str) -> Dataset[dict]:
    """Every row under ``input_path`` as ``{source, path, task_binary}``, one shard per parquet split."""
    files = Dataset.from_files(str(StoragePath(input_path) / TASKS_GLOB))
    rows = files.load_parquet(
        columns=["path", "task_binary", DEFAULT_FILE_PATH_COLUMN],
        approx_shard_bytes=APPROX_SHARD_BYTES,
        include_file_paths=True,
    )
    return rows.map(_with_source)


def raw_tasks(input_path: str) -> Dataset[dict]:
    """``raw_rows`` shuffled by path into even working shards for per-task stages."""
    return raw_rows(input_path).group_by(key=_row_key, reducer=_one_row, num_output_shards=WORKING_SHARDS)
