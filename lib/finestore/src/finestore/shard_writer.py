# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard write primitive: stream Arrow batches to one immutable, atomically-published Parquet object.

:class:`ShardWriter` is the single write path shared by a store flush (buffered rows) and a
compaction merge (a k-way-merged stream). Each :meth:`ShardWriter.write_table` call appends its rows
as row groups capped at :data:`ROW_GROUP_ROWS`, so a large flush still splits into prunable groups
and a caller streaming a merge holds only one bounded batch at a time rather than the whole table.
The shard is written to a temp sibling key and renamed into place on close, so its manifest commit
can never expose a half-written object.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath

# zstd matches finelog's segment encoding: strong ratio on the repetitive token-id / logprob / JSON
# columns eval data is dominated by, for negligible CPU at write time.
_COMPRESSION = "zstd"

# Cap a shard's row groups at this many rows (finelog uses the same figure). A row group is the unit a
# reader prunes on: with a filter it reads only the groups whose footer statistics can match, so a
# large flush that wrote one giant group would defeat the pruning. Bounding the group keeps a
# ``name ==`` blob lookup or a ``task ==`` sample scan reading a fraction of a big shard.
ROW_GROUP_ROWS = 16_384


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

    def __init__(self, path: str, schema: pa.Schema, *, compression: str = _COMPRESSION) -> None:
        self._schema = schema
        StoragePath(path).parent.mkdirs()
        self._stack = contextlib.ExitStack()
        temp_path = self._stack.enter_context(atomic_rename(path))
        handle = self._stack.enter_context(StoragePath(temp_path).open("wb"))
        self._sink = _HashingSink(handle)
        self._writer = self._stack.enter_context(pq.ParquetWriter(self._sink, schema, compression=compression))
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
    """Write ``table`` to the shard object ``path`` atomically as one row group, compressed with zstd."""
    with ShardWriter(path, table.schema) as writer:
        writer.write_table(table)
    return writer.result
