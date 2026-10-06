# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream versioned task specifications through Parquet."""

from collections.abc import Iterable, Iterator
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import TaskSpec

TASK_SCHEMA = pa.schema([pa.field("task_spec", pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def _read_task_records(path: str) -> Iterator[str]:
    with StoragePath(path).open("rb") as source:
        parquet = pq.ParquetFile(source)
        if parquet.schema_arrow.names != TASK_SCHEMA.names:
            raise ValueError("Task Parquet requires exactly one task_spec column")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            yield from batch.column("task_spec").to_pylist()


def read_tasks(path: str) -> Iterator[TaskSpec]:
    """Read tasks in bounded batches."""
    for value in _read_task_records(path):
        yield TaskSpec.model_validate_json(value)


def _write_task_records(path: str, records: Iterable[str]) -> None:
    task_iterator = iter(records)
    with StoragePath(path).open("wb") as destination, pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        while batch := list(islice(task_iterator, PARQUET_BATCH_SIZE)):
            writer.write_table(pa.Table.from_pydict({"task_spec": batch}, TASK_SCHEMA))


def write_tasks(path: str, tasks: Iterable[TaskSpec]) -> None:
    """Write tasks without a dataset-sized in-memory table."""
    _write_task_records(path, (task.model_dump_json() for task in tasks))


def read_task_records(path: str) -> Iterator[str]:
    """Read validated TaskSpec JSON without changing its persisted fields."""
    for value in _read_task_records(path):
        TaskSpec.model_validate_json(value)
        yield value


def _validated_task_records(records: Iterable[str]) -> Iterator[str]:
    for value in records:
        TaskSpec.model_validate_json(value)
        yield value


def write_task_records(path: str, records: Iterable[str]) -> None:
    """Write validated original TaskSpec JSON in bounded batches."""
    _write_task_records(path, _validated_task_records(records))
