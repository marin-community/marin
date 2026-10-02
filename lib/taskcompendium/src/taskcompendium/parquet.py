# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream versioned task specifications through Parquet."""

from collections.abc import Iterator
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import TaskSpec

TASK_SPEC_COLUMN = "task_spec"
TASK_SCHEMA = pa.schema([pa.field(TASK_SPEC_COLUMN, pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def read_tasks(path: str) -> Iterator[TaskSpec]:
    """Read schema-valid tasks in bounded batches without resolving runtimes."""
    with StoragePath(path).open("rb") as source:
        parquet = pq.ParquetFile(source)
        if parquet.schema_arrow.names != TASK_SCHEMA.names:
            raise ValueError("Task Parquet requires exactly one task_spec column")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            for value in batch.column(TASK_SPEC_COLUMN).to_pylist():
                yield TaskSpec.model_validate_json(value)


def write_tasks(path: str, tasks: Iterator[TaskSpec]) -> None:
    """Write tasks without a dataset-sized in-memory table."""
    with StoragePath(path).open("wb") as destination, pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        while batch := list(islice(tasks, PARQUET_BATCH_SIZE)):
            writer.write_table(
                pa.Table.from_pydict({TASK_SPEC_COLUMN: [task.model_dump_json() for task in batch]}, TASK_SCHEMA)
            )
