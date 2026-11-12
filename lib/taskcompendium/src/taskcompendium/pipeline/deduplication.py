# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mandatory exact within-source exclusion before resource admission or quality review."""

import json
import pickle
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import load_parquet

from taskcompendium.identity import task_identity
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.conversion import ConvertedRow
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.models import NormalizedTask
from taskcompendium.pipeline.shard_outputs import ShardOutput, write_shard_outputs
from taskcompendium.pipeline.sources import raw_input_sha256

DUPLICATE_WITHIN_SOURCE = "DUPLICATE_WITHIN_SOURCE"
DEDUPLICATION_REVISION = "1"
RETAINED_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("source_locator", pa.string()),
        ("raw_input_sha256", pa.string()),
        ("converted", pa.binary()),
    ]
)
DROP_SCHEMA = pa.schema([("task_id", pa.string()), ("kept_id", pa.string()), ("reason", pa.string())])


def unique_task_indices(tasks: Sequence[TaskSpec]) -> tuple[list[int], list[dict[str, str]]]:
    """Select exact contracts by lowest task ID and return every dropped-to-kept mapping."""
    seen: dict[str, str] = {}
    identifiers: set[str] = set()
    kept, dropped = [], []
    for index in sorted(range(len(tasks)), key=lambda index: tasks[index].id):
        task = tasks[index]
        if task.id in identifiers:
            raise ValueError(f"Duplicate task ID prevents an unambiguous drop manifest: {task.id}")
        identifiers.add(task.id)
        key = task_identity(task)
        if key in seen:
            dropped.append({"task_id": task.id, "kept_id": seen[key], "reason": DUPLICATE_WITHIN_SOURCE})
        else:
            seen[key] = task.id
            kept.append(index)
    return kept, dropped


@dataclass(frozen=True)
class DeduplicatedPopulation:
    path: str
    input_rows: int
    raw_count: int
    unique_count: int
    within_source_duplicates: int

    def rows(self) -> Dataset[ConvertedRow]:
        return Dataset.from_list(
            sorted(str(path) for path in (StoragePath(self.path) / "retained/*.parquet").glob())
        ).flat_map(read_converted)

    def evidence(self) -> dict[str, Any]:
        return {
            "revision": DEDUPLICATION_REVISION,
            "raw_count": self.raw_count,
            "unique_count": self.unique_count,
            "within_source_duplicates": self.within_source_duplicates,
            "drop_manifest_uri": str(StoragePath(self.path) / "drops/*.parquet"),
        }


def read_converted(path: str) -> Iterator[ConvertedRow]:
    for row in load_parquet(path):
        yield pickle.loads(row["converted"])


def converted_key(row: ConvertedRow) -> str:
    return task_identity(row.result.task) if isinstance(row.result, NormalizedTask) else row.raw.id


def converted_order(row: ConvertedRow) -> str:
    return row.raw.id


def duplicate_group(_key: str, rows: Iterator[ConvertedRow]) -> Iterator[dict[str, Any]]:
    kept = None
    for row in rows:
        duplicate = kept if isinstance(row.result, NormalizedTask) else None
        if isinstance(row.result, NormalizedTask) and kept is None:
            kept = row.raw.id
        yield {"row": row, "duplicate_of": duplicate}


def retained_row(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["duplicate_of"] is not None:
        return None
    row = record["row"]
    return {
        "task_id": row.raw.id,
        "source_locator": row.raw.source.row,
        "raw_input_sha256": raw_input_sha256(row.original_data),
        "converted": pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL),
    }


def dropped_row(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["duplicate_of"] is None:
        return None
    return {"task_id": record["row"].raw.id, "kept_id": record["duplicate_of"], "reason": DUPLICATE_WITHIN_SOURCE}


def write_deduplicated(rows: Iterator[dict[str, Any]], shard: ShardInfo, *, path: str) -> Iterator[Counter]:
    counts = Counter()

    def counted() -> Iterator[dict[str, Any]]:
        for record in rows:
            counts["input_rows"] += 1
            counts["raw_count"] += isinstance(record["row"].result, NormalizedTask)
            counts["within_source_duplicates"] += record["duplicate_of"] is not None
            yield record

    write_shard_outputs(
        counted(),
        shard,
        [
            ShardOutput(str(StoragePath(path) / "retained/part-{shard:05d}.parquet"), RETAINED_SCHEMA, retained_row),
            ShardOutput(str(StoragePath(path) / "drops/part-{shard:05d}.parquet"), DROP_SCHEMA, dropped_row),
        ],
    )
    yield counts


def deduplicate_population(
    rows: Dataset[ConvertedRow], context: ZephyrContext, path: str, *, telemetry: PhaseTelemetry | None = None
) -> DeduplicatedPopulation:
    """Persist unique converted contracts and every duplicate mapping, in lowest-task-ID order."""
    manifest = StoragePath(path) / "manifest.json"
    if manifest.exists():
        return DeduplicatedPopulation(**json.loads(manifest.read_text()))
    if StoragePath(path).exists():
        raise FileExistsError(f"Incomplete deduplication output requires a new artifact version: {path}")
    counts = Counter()
    result = execute_phase(
        context,
        rows.group_by(
            converted_key,
            reducer=duplicate_group,
            sort_by=converted_order,
            num_output_shards=min(64, context.max_workers or 1),
        ).map_shard(partial(write_deduplicated, path=path)),
        telemetry=telemetry,
        operation="deduplicate",
    )
    for part in result.results:
        counts.update(part)
    population = DeduplicatedPopulation(
        path,
        counts["input_rows"],
        counts["raw_count"],
        counts["raw_count"] - counts["within_source_duplicates"],
        counts["within_source_duplicates"],
    )
    manifest.write_text(json.dumps(asdict(population), indent=2) + "\n")
    return population


def unique_documents(
    documents: Iterable[tuple[dict, dict]], dropped: Callable[[dict[str, str]], None]
) -> Iterator[tuple[dict, dict]]:
    """Deduplicate a complete population on disk before yielding its first retained task."""
    with TemporaryDirectory(prefix="task-dedup-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "tasks.sqlite"))) as database:
            database.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, contract TEXT, document TEXT, runtime TEXT)")
            for document, runtime in documents:
                task = TaskSpec.model_validate(document)
                database.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?)",
                    (task.id, task_identity(task), task.model_dump_json(), json.dumps(runtime)),
                )
            database.execute("CREATE INDEX contracts ON tasks(contract)")
            database.execute("CREATE TABLE kept AS SELECT contract, MIN(id) AS id FROM tasks GROUP BY contract")
            database.execute("CREATE UNIQUE INDEX kept_contracts ON kept(contract)")
            for identifier, canonical in database.execute(
                "SELECT t.id, k.id FROM tasks t JOIN kept k ON t.contract = k.contract WHERE t.id != k.id ORDER BY t.id"
            ):
                dropped({"task_id": identifier, "kept_id": canonical, "reason": DUPLICATE_WITHIN_SOURCE})
            for document, runtime in database.execute(
                "SELECT t.document, t.runtime FROM tasks t JOIN kept k ON t.id = k.id ORDER BY t.id"
            ):
                yield json.loads(document), json.loads(runtime)
