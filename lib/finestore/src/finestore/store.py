# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The writer: a :class:`DataStore` of append-only :class:`DataTable` shard streams.

A store is one archive rooted at a URL prefix, written by one ``writer_id``. Appends buffer in
memory; a background thread flushes each table's buffer to an immutable Parquet shard on a time
ceiling (or when the buffer grows past a row cap). :meth:`DataStore.flush` and
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

import pyarrow as pa
from rigging.filesystem import StoragePath

from finestore.layout import (
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    SEQ_COLUMN,
    WRITER_COLUMN,
    FineStoreLayout,
    TableMetadata,
    build_uri,
)
from finestore.shard_writer import write_table

logger = logging.getLogger(__name__)

# Flush on a time ceiling by default: eval writes are latency-sensitive and modestly sized, so a
# small interval keeps shards fresh without a manifest commit per flush.
DEFAULT_FLUSH_INTERVAL = 5.0
# A safety cap so a burst of appends between ticks cannot grow the buffer without bound; a table
# that crosses it is flushed immediately rather than waiting for the next tick.
DEFAULT_MAX_BUFFER_ROWS = 20_000


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


class DataStore:
    """A writable columnar archive under ``root`` for one ``writer_id``."""

    def __init__(
        self,
        root: str,
        *,
        writer_id: str | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_rows: int = DEFAULT_MAX_BUFFER_ROWS,
    ) -> None:
        self.root = root.rstrip("/")
        self._layout = FineStoreLayout(self.root)
        self.writer_id = writer_id or _default_writer_id()
        self._flush_interval = flush_interval
        self._max_buffer_rows = max_buffer_rows
        self._tables: dict[str, DataTable] = {}
        self._register_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
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
        """
        with self._register_lock:
            existing = self._tables.get(name)
            if existing is not None:
                return existing
            key = tuple(merge_key) if merge_key is not None else None
            table = DataTable(self, name, merge_key=key, schema=schema, schema_version=schema_version)
            self._tables[name] = table
            self._write_schema_meta(table)
            return table

    def _write_schema_meta(self, table: DataTable) -> None:
        """Persist the table's merge key and versions (the archive's only metadata object)."""
        meta = TableMetadata(merge_key=table.merge_key, schema_version=table.schema_version)
        StoragePath(self._layout.schema_path(table.name)).write_text(meta.model_dump_json(indent=2))

    def write(self, name: str, metadata: Mapping[str, object] | None, data: bytes) -> str:
        """Append one opaque blob to the reserved ``blobs`` table; return its ``finestore://`` URI.

        The payload is stored inline as a Parquet binary column. ``metadata`` is kept verbatim as a
        JSON string for the reader to surface.
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
        self._raise_if_failed()
        for table in list(self._tables.values()):
            table.flush()

    def seal(self) -> None:
        """Flush, then mark the archive sealed so readers know the run is complete."""
        self.flush()
        StoragePath(self._layout.sealed_path).write_text(json.dumps({"writer": self.writer_id}))

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

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


class DataTable:
    """An append-only table: it buffers appended rows and flushes them to immutable shards.

    The table owns its pending rows, its per-writer sequence counter, and how they persist. Reads go
    through :class:`CompositeReader`.
    """

    def __init__(
        self,
        store: DataStore,
        name: str,
        *,
        merge_key: tuple[str, ...] | None,
        schema: pa.Schema | None,
        schema_version: int,
    ) -> None:
        self._store = store
        self.name = name
        self.merge_key = merge_key
        self.schema_version = schema_version
        self._schema = schema
        self._pending: list[dict] = []
        self._next_seq = 0
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        """Buffer one row. Non-blocking: the row is durable after the next flush tick or an explicit
        ``flush``. Appends from one writer are ordered by a monotonic ``_seq``; the reader keeps the
        highest ``_seq`` per primary key."""
        self.extend([row])

    def extend(self, rows: Iterable[dict]) -> None:
        """Buffer many rows for the next flush (see :meth:`append`)."""
        self._store._raise_if_failed()
        with self._lock:
            for row in rows:
                stamped = dict(row)
                stamped[SEQ_COLUMN] = self._next_seq
                stamped[WRITER_COLUMN] = self._store.writer_id
                self._next_seq += 1
                self._pending.append(stamped)
            over_cap = len(self._pending) >= self._store._max_buffer_rows
        if over_cap:
            self._store._wake.set()

    def flush(self) -> None:
        """Write this table's buffered rows to one immutable shard now, blocking until it is durable.

        Swaps the pending rows out under the table lock, then writes outside it, so a concurrent
        flush of the same table (the background thread and a caller) each claim a disjoint batch and
        never double-write a row.
        """
        self._store._raise_if_failed()
        with self._lock:
            if not self._pending:
                return
            rows = self._pending
            self._pending = []
        table = pa.Table.from_pylist(_drop_empty_struct_keys(rows), schema=self._schema)
        min_seq = min(row[SEQ_COLUMN] for row in rows)
        path = self._store._layout.shard_path(self.name, self._store.writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        write_table(path, table)
        logger.debug("finestore flushed %d rows to %s", len(rows), path)
