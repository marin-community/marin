# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard write primitive: stream Arrow batches to one immutable, atomically-published Parquet object.

:class:`ShardWriter` is the single write path shared by a store flush (buffered rows) and a
compaction merge (a k-way-merged stream). Callers group fixed-schema rows toward
:data:`ROW_GROUP_ROWS` and :data:`ROW_GROUP_TARGET_BYTES`, isolating a row that exceeds either
target. Chunked objects bound each binary value separately.
The shard is written to a temp sibling key and renamed into place on close, so its manifest commit
can never expose a half-written object.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath

# Pin level 0 explicitly; PyArrow otherwise chooses its codec default (currently zstd level 1).
_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 0

# Cap a shard's row groups at this many rows (finelog uses the same figure). A row group is the unit a
# reader prunes on: with a filter it reads only the groups whose footer statistics can match, so a
# large flush that wrote one giant group would defeat the pruning. Bounding the group keeps a
# ``name ==`` blob lookup or a ``task ==`` sample scan reading a fraction of a big shard.
ROW_GROUP_ROWS = 16_384
ROW_GROUP_TARGET_BYTES = 100 * 1024 * 1024


def estimate_python_bytes(value: object) -> int:
    """Estimate a Python value's payload size before Arrow conversion."""
    if isinstance(value, memoryview):
        return value.nbytes
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(len(str(key)) + estimate_python_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(estimate_python_bytes(item) for item in value)
    return 8


def _table_groups(row_tables: Iterable[pa.Table]) -> Iterator[pa.Table]:
    group: list[pa.Table] = []
    group_rows = 0
    group_bytes = 0
    for table in row_tables:
        if group and (
            group_rows + table.num_rows > ROW_GROUP_ROWS or group_bytes + table.nbytes > ROW_GROUP_TARGET_BYTES
        ):
            yield pa.concat_tables(group)
            group = []
            group_rows = 0
            group_bytes = 0
        group.append(table)
        group_rows += table.num_rows
        group_bytes += table.nbytes
    if group:
        yield pa.concat_tables(group)


def row_groups(rows: Iterable[dict], schema: pa.Schema) -> Iterator[pa.Table]:
    """Convert rows once and group their Arrow buffers by row count and actual byte size."""
    yield from _table_groups(pa.Table.from_pylist([row], schema=schema) for row in rows)


def table_row_groups(table: pa.Table) -> Iterator[pa.Table]:
    """Split an existing Arrow table into bounded row groups without converting its values."""
    yield from _table_groups(table.slice(offset, 1) for offset in range(table.num_rows))


@dataclass(frozen=True)
class ShardWriteResult:
    """Integrity metadata for one atomically published shard."""

    size_bytes: int
    content_sha256: str


class _HashingSink:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._hash = hashlib.sha256()
        self.size_bytes = 0

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        written = self._handle.write(data)
        if written is None:
            written = len(data)
        self._hash.update(memoryview(data)[:written])
        self.size_bytes += written
        return written

    def tell(self) -> int:
        return self._handle.tell()

    def flush(self) -> None:
        self._handle.flush()

    def result(self) -> ShardWriteResult:
        return ShardWriteResult(size_bytes=self.size_bytes, content_sha256=self._hash.hexdigest())


class ShardWriter:
    """Streams record batches to one immutable Parquet shard, one row group per ``write_table`` call.

    Open it on the final object path; it writes to a temp sibling and renames into place on
    :meth:`close` (or context-manager exit). Every appended table must share the writer's schema.
    """

    def __init__(self, path: str, schema: pa.Schema) -> None:
        self._schema = schema
        StoragePath(path).parent.mkdirs()
        self._stack = contextlib.ExitStack()
        temp_path = self._stack.enter_context(atomic_rename(path))
        handle = self._stack.enter_context(StoragePath(temp_path).open("wb"))
        self._sink = _HashingSink(handle)
        self._writer = self._stack.enter_context(
            pq.ParquetWriter(
                self._sink,
                schema,
                compression=_COMPRESSION,
                compression_level=_COMPRESSION_LEVEL,
            )
        )
        self._result: ShardWriteResult | None = None

    def write_table(self, table: pa.Table) -> None:
        """Append ``table``, split into row groups of at most :data:`ROW_GROUP_ROWS` rows; its schema
        must match the writer's. A caller streaming a merge passes one bounded batch per call, so this
        usually writes a single group."""
        self._writer.write_table(table, row_group_size=ROW_GROUP_ROWS)

    def close(self) -> ShardWriteResult:
        """Finalize the Parquet footer, flush the handle, and rename the shard into place."""
        if self._result is None:
            self._stack.close()
            self._result = self._sink.result()
        return self._result

    @property
    def result(self) -> ShardWriteResult:
        if self._result is None:
            raise RuntimeError("shard writer has not closed")
        return self._result

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.close()
            return
        self._stack.__exit__(exc_type, exc_val, exc_tb)


def write_table(path: str, table: pa.Table) -> ShardWriteResult:
    """Write ``table`` atomically in row groups bounded by actual Arrow buffer size."""
    with ShardWriter(path, table.schema) as writer:
        for group in table_row_groups(table):
            writer.write_table(group)
    return writer.result
