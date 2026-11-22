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
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import load_parquet

from taskcompendium.identity import task_identity, task_spec_json
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
ROWS_PER_DEDUP_SHARD = 1000
MAX_DEDUP_SHARDS = 64
KEY_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("task_id", pa.string()),
        ("normalized", pa.bool_()),
        ("payload", pa.string()),
        ("row_index", pa.int64()),
    ]
)
DROP_SCHEMA = pa.schema([("task_id", pa.string()), ("kept_id", pa.string()), ("reason", pa.string())])


class DeduplicationEvidence(TypedDict):
    revision: str
    raw_count: int
    unique_count: int
    within_source_duplicates: int
    drop_manifest_uri: str


class DeduplicationRecord(TypedDict):
    row: ConvertedRow
    duplicate_of: str | None


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
    drops_path: str | None = None
    """The population directory holding this population's drop manifest, when ``path`` does not.

    A derived population, such as the requirement backfill's, reuses the deduplication's drops
    instead of writing its own, because it changes declared fields and labels but drops nothing.
    """

    def rows(self) -> Dataset[ConvertedRow]:
        return Dataset.from_list(
            sorted(str(path) for path in (StoragePath(self.path) / "retained/*.parquet").glob())
        ).flat_map(read_converted)

    def evidence(self) -> DeduplicationEvidence:
        return {
            "revision": DEDUPLICATION_REVISION,
            "raw_count": self.raw_count,
            "unique_count": self.unique_count,
            "within_source_duplicates": self.within_source_duplicates,
            "drop_manifest_uri": str(StoragePath(self.drops_path or self.path) / "drops/*.parquet"),
        }


def read_converted(path: str) -> Iterator[ConvertedRow]:
    for row in load_parquet(path):
        yield pickle.loads(row["converted"])


def converted_key(row: ConvertedRow) -> str:
    return task_identity(row.result.task) if isinstance(row.result, NormalizedTask) else row.raw.id


class DeduplicationKey(TypedDict):
    key: str
    task_id: str
    normalized: bool
    payload: str
    row_index: int


class DeduplicationDecision(DeduplicationKey):
    duplicate_of: str | None


@dataclass(frozen=True)
class PreparedShard:
    keys: str
    input_rows: int


def read_deduplication_keys(path: str) -> Iterator[DeduplicationKey]:
    for row in load_parquet(path):
        yield DeduplicationKey(**row)


def prepared_payload(item: tuple[int, ConvertedRow]) -> dict[str, Any]:
    record = retained_row({"row": item[1], "duplicate_of": None})
    assert record is not None
    return record


def prepared_key(item: tuple[int, ConvertedRow], *, payload: str) -> dict[str, Any]:
    index, row = item
    return {
        "key": converted_key(row),
        "task_id": row.raw.id,
        "normalized": isinstance(row.result, NormalizedTask),
        "payload": payload,
        "row_index": index,
    }


def prepare_deduplication(rows: Iterator[ConvertedRow], shard: ShardInfo, *, path: str) -> Iterator[PreparedShard]:
    payload = str(StoragePath(path) / f"payload/part-{shard.shard_idx:05d}.parquet")
    count = 0

    def indexed() -> Iterator[tuple[int, ConvertedRow]]:
        nonlocal count
        for index, row in enumerate(rows):
            count += 1
            yield index, row

    paths = write_shard_outputs(
        indexed(),
        shard,
        [
            ShardOutput(str(StoragePath(path) / "payload/part-{shard:05d}.parquet"), RETAINED_SCHEMA, prepared_payload),
            ShardOutput(
                str(StoragePath(path) / "keys/part-{shard:05d}.parquet"),
                KEY_SCHEMA,
                partial(prepared_key, payload=payload),
            ),
        ],
    )
    yield PreparedShard(paths[1], count)


def key_identity(row: DeduplicationKey) -> str:
    return row["key"]


def key_order(row: DeduplicationKey) -> str:
    return row["task_id"]


def duplicate_group(_key: str, rows: Iterator[DeduplicationKey]) -> Iterator[DeduplicationDecision]:
    kept = None
    for row in rows:
        duplicate = kept if row["normalized"] else None
        if row["normalized"] and kept is None:
            kept = row["task_id"]
        yield {**row, "duplicate_of": duplicate}


def payload_identity(row: DeduplicationDecision) -> str:
    return row["payload"]


def payload_order(row: DeduplicationDecision) -> int:
    return row["row_index"]


def rejoin_payload(payload: str, decisions: Iterator[DeduplicationDecision]) -> Iterator[DeduplicationRecord]:
    # Only compact references pass through either shuffle. Original payloads stream in source order.
    for index, (row, decision) in enumerate(zip_longest(load_parquet(payload), decisions)):
        if row is None or decision is None or decision["row_index"] != index or row["task_id"] != decision["task_id"]:
            raise ValueError(f"Deduplication decisions do not cover their original payload exactly: {payload}")
        yield {"row": pickle.loads(row["converted"]), "duplicate_of": decision["duplicate_of"]}


def retained_row(record: DeduplicationRecord) -> dict[str, Any] | None:
    if record["duplicate_of"] is not None:
        return None
    row = record["row"]
    return {
        "task_id": row.raw.id,
        "source_locator": row.raw.source.row,
        "raw_input_sha256": raw_input_sha256(row.original_data),
        "converted": pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL),
    }


def dropped_row(record: DeduplicationRecord) -> dict[str, str] | None:
    if record["duplicate_of"] is None:
        return None
    return {"task_id": record["row"].raw.id, "kept_id": record["duplicate_of"], "reason": DUPLICATE_WITHIN_SOURCE}


def write_deduplicated(rows: Iterator[DeduplicationRecord], shard: ShardInfo, *, path: str) -> Iterator[Counter]:
    counts = Counter()

    def counted() -> Iterator[DeduplicationRecord]:
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
    """Persist unique converted contracts and drops, retaining the lowest task ID per contract."""
    manifest = StoragePath(path) / "manifest.json"
    if manifest.exists():
        return DeduplicatedPopulation(**json.loads(manifest.read_text()))
    if StoragePath(path).exists():
        raise FileExistsError(f"Incomplete deduplication output requires a new artifact version: {path}")
    prepared = execute_phase(
        context,
        rows.map_shard(partial(prepare_deduplication, path=path)),
        telemetry=telemetry,
        operation="deduplicate_prepare",
    ).results
    input_rows = sum(part.input_rows for part in prepared)
    partitions = min(MAX_DEDUP_SHARDS, max(1, (input_rows + ROWS_PER_DEDUP_SHARD - 1) // ROWS_PER_DEDUP_SHARD))
    counts = Counter()
    result = execute_phase(
        context,
        Dataset.from_list([part.keys for part in prepared])
        .flat_map(read_deduplication_keys)
        .group_by(
            key_identity,
            reducer=duplicate_group,
            sort_by=key_order,
            num_output_shards=partitions,
        )
        .group_by(
            payload_identity,
            reducer=rejoin_payload,
            sort_by=payload_order,
            num_output_shards=partitions,
        )
        .map_shard(partial(write_deduplicated, path=path)),
        telemetry=telemetry,
        operation="deduplicate",
    )
    for part in result.results:
        counts.update(part)
    if counts["input_rows"] != input_rows:
        raise ValueError("Deduplication output does not cover the prepared input population")
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
                task = TaskSpec.model_validate_json(json.dumps(document))
                database.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?)",
                    (task.id, task_identity(task), task_spec_json(task), json.dumps(runtime)),
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
