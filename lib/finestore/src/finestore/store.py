# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The writer: a :class:`DataStore` of append-only :class:`DataTable` shard streams.

A store is one archive rooted at a URL prefix, written by one ``writer_id``. Appends buffer in
memory; a background thread flushes each table's buffer to an immutable Parquet shard on a time
ceiling (or when the buffer's payload grows past a byte cap). :meth:`DataStore.flush` and
:meth:`DataStore.close` block until every buffered row is durable, which is the "writes block until
persisted" guarantee. Concurrent writers of the same run each use a distinct ``writer_id`` and write
under their own key prefix, so they never coordinate; the reader composes and deduplicates them.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import pyarrow as pa
from rigging.filesystem import StoragePath

from finestore.compaction import compact
from finestore.layout import (
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    SEQ_COLUMN,
    WRITER_COLUMN,
    ArchiveMetadata,
    FineStoreLayout,
    SealMarker,
    TableMetadata,
    build_uri,
)
from finestore.reader import CompositeReader
from finestore.shard_writer import write_table

logger = logging.getLogger(__name__)

# Flush on a time ceiling by default: eval writes are latency-sensitive and modestly sized, so a
# small interval keeps shards fresh without a manifest commit per flush.
DEFAULT_FLUSH_INTERVAL = 5.0
# Flush early when a table's buffered payload crosses this, so memory (and a single shard's size)
# stays bounded regardless of row count. Bytes, not rows: one row can be a multi-MB trajectory blob
# and another a small sample, so a row count is a poor proxy for the RAM a buffer holds. This is the
# only backstop against an unbounded buffer -- finelog additionally floors the flush rate at one
# per second, which eval's write volume does not need.
DEFAULT_MAX_BUFFER_BYTES = 100 * 1024 * 1024


def _default_writer_id() -> str:
    """A writer identity unique enough that two writers never share a key prefix."""
    return uuid.uuid4().hex


def _drop_empty_struct_keys(rows: list[dict]) -> list[dict]:
    """Drop keys whose values are only ``None`` or empty dicts across the batch.

    Parquet cannot store a zero-field struct, so a dict column that no row populates (e.g. a batch of
    ungraded samples whose ``metrics`` are all ``{}``) would fail to write. Dropping the column here
    is transparent: read-time schema unification restores it as absent, and the row model's default
    fills it back in. Non-empty dicts are untouched.
    """
    if not rows:
        return rows
    keys = set().union(*(row.keys() for row in rows))
    drop = set()
    for key in keys:
        saw_dict = False
        nonempty = False
        for row in rows:
            value = row.get(key)
            if isinstance(value, dict):
                saw_dict = True
                if value:
                    nonempty = True
                    break
        if saw_dict and not nonempty:
            drop.add(key)
    if not drop:
        return rows
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


def _estimate_bytes(value: object) -> int:
    """A cheap estimate of a value's in-RAM payload size, summed over a row to bound the write buffer.

    Counts the sizes that vary by orders of magnitude -- byte and text payloads by length, containers
    by recursion -- and charges a flat width for scalars, ignoring per-object overhead. It only has to
    be good enough to keep a buffer's memory near a target, which a row count cannot do once one row is
    a multi-MB blob and the next a small sample.
    """
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, dict):
        return sum(len(str(key)) + _estimate_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_bytes(item) for item in value)
    return 8


def _with_stamp_columns(schema: pa.Schema) -> pa.Schema:
    """Append finestore's ``_seq``/``_writer`` stamp columns to a caller's schema when absent.

    A caller pins only its own columns; the write path stamps ``_seq``/``_writer`` on every row, so the
    schema a flush validates against must carry them too. Idempotent: a schema that already declares
    them (e.g. one round-tripped from a shard) is returned unchanged.
    """
    names = set(schema.names)
    fields = list(schema)
    if SEQ_COLUMN not in names:
        fields.append(pa.field(SEQ_COLUMN, pa.int64()))
    if WRITER_COLUMN not in names:
        fields.append(pa.field(WRITER_COLUMN, pa.string()))
    return pa.schema(fields)


class FlushScheduler(Protocol):
    """The store-side controls a :class:`DataTable` needs, kept narrow so the table does not reach
    into the store's internals: ask for a background flush when a buffer crosses its cap, and re-raise
    a failure the background flusher hit."""

    def request_flush(self) -> None: ...

    def raise_if_failed(self) -> None: ...


class DataStore:
    """A writable columnar archive under ``root`` for one ``writer_id``."""

    def __init__(
        self,
        root: str,
        *,
        writer_id: str | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        self.root = root.rstrip("/")
        self._layout = FineStoreLayout(self.root)
        self.writer_id = writer_id or _default_writer_id()
        self._flush_interval = flush_interval
        self._max_buffer_bytes = max_buffer_bytes
        self._tables: dict[str, DataTable] = {}
        self._register_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        # Stamp the archive-wide metadata at open. Concurrent writers write identical content, so a
        # last-writer-wins overwrite is harmless.
        StoragePath(self._layout.archive_path).write_text(ArchiveMetadata().model_dump_json(indent=2))
        self._thread = threading.Thread(target=self._run, name="finestore-flush", daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, root: str, **kwargs) -> DataStore:
        """Open an archive for writing under ``root``, produced by one ``writer_id``."""
        return cls(root, **kwargs)

    # -- table registration ------------------------------------------------------------------------

    def table(
        self,
        name: str,
        *,
        schema: pa.Schema | None = None,
        merge_key: Sequence[str] | None = None,
        schema_version: int = 1,
    ) -> DataTable:
        """Register (or fetch) a table and return a handle for appending to it.

        ``merge_key`` names the columns a reader collapses duplicates on (keeping the latest); it is
        persisted to the table's ``_schema.json`` so readers that did not open the writer can recover
        it. Calling ``table`` again with the same ``name`` returns the same handle, so every appender
        shares one buffer and one sequence counter.

        On registration the table's sequence counter resumes above the highest ``_seq`` any shard has
        already persisted, so a row this writer appends now always outranks a row a prior session left
        behind — a re-run of an existing key wins, and nothing can shadow it.
        """
        with self._register_lock:
            existing = self._tables.get(name)
            if existing is not None:
                return existing
            key = tuple(merge_key) if merge_key is not None else None
            start_seq = CompositeReader(self.root).max_seq(name) + 1
            table = DataTable(
                name,
                writer_id=self.writer_id,
                layout=self._layout,
                max_buffer_bytes=self._max_buffer_bytes,
                scheduler=self,
                merge_key=key,
                schema=schema,
                schema_version=schema_version,
                start_seq=start_seq,
            )
            self._tables[name] = table
            self._write_schema_meta(table)
            return table

    def _write_schema_meta(self, table: DataTable) -> None:
        """Persist the table's merge key and logical schema version to its ``_schema.json``."""
        meta = TableMetadata(merge_key=table.merge_key, schema_version=table.schema_version)
        StoragePath(self._layout.schema_path(table.name)).write_text(meta.model_dump_json(indent=2))

    def write(self, name: str, metadata: Mapping[str, object] | None, data: bytes) -> str:
        """Append one opaque blob to the reserved ``blobs`` table; return its ``finestore://`` URI.

        The blob is buffered and flushed with every other table, so many small writes never block on a
        per-object round trip -- the point of routing them through the archive rather than one object
        each. The payload rides inline as a Parquet binary column and ``metadata`` as a JSON string the
        reader can project without touching the payload. The table's merge key is the blob name, so a
        re-write supersedes the prior one and a compacted shard is sorted by name for a pruned lookup.
        """
        blobs = self._tables.get(BLOBS_TABLE) or self.table(BLOBS_TABLE, merge_key=(BLOB_NAME_COLUMN,))
        blobs.append(
            {
                BLOB_NAME_COLUMN: name,
                "size": len(data),
                "metadata_json": json.dumps(dict(metadata or {})),
                "data": data,
            }
        )
        return build_uri(BLOBS_TABLE, name)

    # -- flush / seal / close ----------------------------------------------------------------------

    def flush(self) -> None:
        """Write every table's buffered rows to shards now, blocking until they are durable."""
        self.raise_if_failed()
        for table in list(self._tables.values()):
            table.flush()

    def seal(self) -> None:
        """Flush, compact every table to one deduplicated generation, then mark the archive sealed.

        Sealing is the materialize contract: afterward each table is a single deduplicated Parquet
        shard and every blob is its own object, so a plain Parquet reader over the archive sees each
        row once without having to apply finestore's generation/``_seq`` dedup rule. Compaction is
        still only an optimization for the finestore reader, which deduplicates either way.
        """
        self.flush()
        for name in list(self._tables):
            compact(self.root, name)
        StoragePath(self._layout.sealed_path).write_text(SealMarker(writer=self.writer_id).model_dump_json())

    def close(self) -> None:
        """Stop the background thread and flush any remaining rows."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        self.flush()

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- background flusher ------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()
            try:
                self.flush()
            except BaseException as exc:
                self._error = exc
                logger.exception("finestore background flush failed")
                return

    # -- FlushScheduler (the seam DataTable talks to) ----------------------------------------------

    def request_flush(self) -> None:
        """Wake the background flusher early because a table's buffer crossed its byte cap."""
        self._wake.set()

    def raise_if_failed(self) -> None:
        """Re-raise the exception the background flusher died with, if any."""
        if self._error is not None:
            raise self._error


class DataTable:
    """An append-only table: it buffers appended rows and flushes them to immutable shards.

    The table owns its pending rows, its per-writer sequence counter, and how they persist. Reads go
    through :class:`CompositeReader`.
    """

    def __init__(
        self,
        name: str,
        *,
        writer_id: str,
        layout: FineStoreLayout,
        max_buffer_bytes: int,
        scheduler: FlushScheduler,
        merge_key: tuple[str, ...] | None,
        schema: pa.Schema | None,
        schema_version: int,
        start_seq: int = 0,
    ) -> None:
        self.name = name
        self.merge_key = merge_key
        self.schema_version = schema_version
        self._writer_id = writer_id
        self._layout = layout
        self._max_buffer_bytes = max_buffer_bytes
        self._scheduler = scheduler
        self._schema = _with_stamp_columns(schema) if schema is not None else None
        self._pending: list[dict] = []
        self._pending_bytes = 0
        self._next_seq = start_seq
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        """Buffer one row. Non-blocking: the row is durable after the next flush tick or an explicit
        ``flush``. Appends from one writer are ordered by a monotonic ``_seq``; the reader keeps the
        highest ``_seq`` per primary key."""
        self.extend([row])

    def extend(self, rows: Iterable[dict]) -> None:
        """Buffer many rows for the next flush (see :meth:`append`)."""
        self._scheduler.raise_if_failed()
        with self._lock:
            for row in rows:
                stamped = dict(row)
                stamped[SEQ_COLUMN] = self._next_seq
                stamped[WRITER_COLUMN] = self._writer_id
                self._next_seq += 1
                self._pending.append(stamped)
                self._pending_bytes += _estimate_bytes(row)
            over_cap = self._pending_bytes >= self._max_buffer_bytes
        if over_cap:
            self._scheduler.request_flush()

    def flush(self) -> None:
        """Write this table's buffered rows to one immutable shard now, blocking until it is durable.

        Swaps the pending rows out under the table lock, then writes outside it, so a concurrent
        flush of the same table (the background thread and a caller) each claim a disjoint batch and
        never double-write a row.
        """
        self._scheduler.raise_if_failed()
        with self._lock:
            if not self._pending:
                return
            rows = self._pending
            self._pending = []
            self._pending_bytes = 0
        table = pa.Table.from_pylist(_drop_empty_struct_keys(rows), schema=self._schema)
        min_seq = min(row[SEQ_COLUMN] for row in rows)
        path = self._layout.shard_path(self.name, self._writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        write_table(path, table)
        logger.debug("finestore flushed %d rows to %s", len(rows), path)
