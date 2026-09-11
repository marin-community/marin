# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove source policy and raw Zephyr dataset."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.dataset import Dataset
from zephyr.input_file import DEFAULT_FILE_PATH_COLUMN

TASKTROVE_HF_ID = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "0292300"
TASKS_GLOB = "*/tasks.parquet"
APPROX_SHARD_BYTES = 32 << 20
WORKING_SHARDS = 64
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="4g")
_VERDICTS_PATH = Path(__file__).with_name("source_verdicts.json")
_REVIEWED_DEFECTS_PATH = Path(__file__).with_name("reviewed_defects.json")


class SourceVerdict(StrEnum):
    KEEP = "keep"
    DROP = "drop"


@dataclass(frozen=True)
class SourceInfo:
    source: str
    verdict: SourceVerdict
    family: str
    reason: str


def load_source_verdicts() -> dict[str, SourceInfo]:
    raw = json.loads(_VERDICTS_PATH.read_text())
    return {
        source: SourceInfo(source, SourceVerdict(row["verdict"]), row["family"], row["reason"])
        for source, row in raw.items()
    }


@cache
def load_reviewed_defects() -> dict[tuple[str, str], str]:
    """Return task rows rejected after manual review, keyed by source and path."""
    rows = json.loads(_REVIEWED_DEFECTS_PATH.read_text())
    defects = {(row["source"], row["path"]): row["reason"] for row in rows}
    if len(defects) != len(rows):
        raise ValueError(f"duplicate source/path in {_REVIEWED_DEFECTS_PATH}")
    return defects


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
    """Load every source row and annotate it with the source directory name."""
    files = Dataset.from_files(str(StoragePath(input_path) / TASKS_GLOB))
    rows = files.load_parquet(
        columns=["path", "task_binary", DEFAULT_FILE_PATH_COLUMN],
        approx_shard_bytes=APPROX_SHARD_BYTES,
        include_file_paths=True,
    )
    return rows.map(_with_source)


def raw_tasks(input_path: str) -> Dataset[dict]:
    """Shuffle raw rows by task path into 64 balanced working shards."""
    return raw_rows(input_path).group_by(key=_row_key, reducer=_one_row, num_output_shards=WORKING_SHARDS)
