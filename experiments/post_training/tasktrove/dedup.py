# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop repeated instructions within a source and apply the optional per-source cap.

Only converted rows are candidates; unconverted rows pass through untouched so the ledger keeps
their reasons. Two tasks are duplicates when their ``instruction_key`` columns match (the
normalized instruction hash the converted step wrote). The lowest path wins; with a cap,
survivors are the first ``max_tasks_per_source`` by a seeded hash of the path so reruns pick the
same rows. Both passes are Zephyr ``group_by`` stages over the converted rows.
"""

import hashlib
import logging
from collections.abc import Iterator

from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.convert import CONVERTED_GLOB, CONVERTED_SCHEMA
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.raw_tasks import APPROX_SHARD_BYTES, RAW_SHARDS, WORKER_RESOURCES

logger = logging.getLogger(__name__)

DEDUPED_GLOB = "deduped/*.parquet"
_CAP_SEED = b"tasktrove-clean-cap-v1"


class DedupStatus:
    DUPLICATE = "duplicate"
    CAPPED = "capped"


def cap_rank(path: str) -> str:
    return hashlib.sha256(_CAP_SEED + path.encode()).hexdigest()


def dedup_key(row: dict) -> tuple[str, str, str]:
    """Converted rows group by instruction within their source; every other row is its own group."""
    if row["status"] == ConvertStatus.CONVERTED:
        return ("instruction", row["source"], row["instruction_key"])
    return ("path", row["source"], row["path"])


def _dropped(row: dict, status: str) -> dict:
    return {**row, "status": status, "task_binary": None, "solution_binary": None}


def keep_first(_key: tuple, rows: Iterator[dict]) -> Iterator[dict]:
    yield next(rows)
    for row in rows:
        yield _dropped(row, DedupStatus.DUPLICATE)


def cap_source(rows: Iterator[dict], max_tasks: int) -> Iterator[dict]:
    kept = 0
    for row in rows:
        if row["status"] != ConvertStatus.CONVERTED:
            yield row
            continue
        kept += 1
        yield row if kept <= max_tasks else _dropped(row, DedupStatus.CAPPED)


def dedup_tasks(converted_path: str, output_path: str, max_tasks_per_source: int | None) -> None:
    """Zephyr stage: rewrite the converted rows with duplicate and capped rows marked and their binaries dropped."""
    files = Dataset.from_files(str(StoragePath(converted_path) / CONVERTED_GLOB))
    ds = files.load_parquet(approx_shard_bytes=APPROX_SHARD_BYTES)
    ds = ds.group_by(key=dedup_key, reducer=keep_first, sort_by=lambda row: row["path"], num_output_shards=RAW_SHARDS)
    if max_tasks_per_source is not None:
        ds = ds.group_by(
            key=lambda row: row["source"],
            reducer=lambda _source, rows: cap_source(rows, max_tasks_per_source),
            sort_by=lambda row: cap_rank(row["path"]),
            num_output_shards=RAW_SHARDS,
        )
    ds = ds.write_parquet(str(StoragePath(output_path) / "deduped/part-{shard:05d}.parquet"), schema=CONVERTED_SCHEMA)
    ZephyrContext(name="tasktrove-dedup", resources=WORKER_RESOURCES).execute(ds)
