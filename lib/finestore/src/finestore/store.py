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
import os
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pyarrow as pa
from rigging.filesystem import StoragePath

from finestore import _io
from finestore.layout import (
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    SEQ_COLUMN,
    WRITER_COLUMN,
    build_uri,
    schema_path,
    sealed_path,
    shard_path,
)
from finestore.reader import CompositeReader

logger = logging.getLogger(__name__)

# Flush on a time ceiling by default: eval writes are latency-sensitive and modestly sized, so a
# small interval keeps shards fresh without a manifest commit per flush.
DEFAULT_FLUSH_INTERVAL = 5.0
# A safety cap so a burst of appends between ticks cannot grow the buffer without bound; a table
# that crosses it is flushed immediately rather than waiting for the next tick.
DEFAULT_MAX_BUFFER_ROWS = 20_000


def _default_writer_id() -> str:
    """A per-process writer identity, unique enough that two writers never share a key prefix."""
    return f"{os.getpid()}-{uuid.uuid4().hex[:8]}"


@dataclass
class _Buffer:
    """One table's pending rows, its dedup primary key, and its per-writer sequence counter."""

    name: str
    primary_key: tuple[str, ...] | None
    schema: pa.Schema | None
    schema_version: int
    pending: list[dict] = field(default_factory=list)
    next_seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


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
        mode: str = "append",
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_rows: int = DEFAULT_MAX_BUFFER_ROWS,
    ) -> None:
        if mode not in ("append", "read"):
            raise ValueError(f"mode must be 'append' or 'read', got {mode!r}")
        self.root = root.rstrip("/")
        self.writer_id = writer_id or _default_writer_id()
        self.mode = mode
        self._flush_interval = flush_interval
        self._max_buffer_rows = max_buffer_rows
        self._tables: dict[str, _Buffer] = {}
        self._register_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        if mode == "append":
            self._thread = threading.Thread(target=self._run, name="finestore-flush", daemon=True)
            self._thread.start()

    @classmethod
    def open(cls, root: str, **kwargs) -> DataStore:
        """Open an archive for writing (``mode='append'``) or reading (``mode='read'``)."""
        return cls(root, **kwargs)

    # -- table registration ------------------------------------------------------------------------

    def table(
        self,
        name: str,
        *,
        schema: pa.Schema | None = None,
        primary_key: Sequence[str] | None = None,
        schema_version: int = 1,
    ) -> DataTable:
        """Register (or fetch) a table and return a handle for appending and reading it.

        ``primary_key`` names the columns a reader deduplicates on; it is persisted to the table's
        ``_schema.json`` so readers that did not open the writer can recover it.
        """
        with self._register_lock:
            buffer = self._tables.get(name)
            if buffer is None:
                pk = tuple(primary_key) if primary_key is not None else None
                buffer = _Buffer(name=name, primary_key=pk, schema=schema, schema_version=schema_version)
                self._tables[name] = buffer
                if self.mode == "append":
                    self._write_schema_meta(buffer)
        return DataTable(self, name)

    def _write_schema_meta(self, buffer: _Buffer) -> None:
        """Persist the table's primary key and schema version (the archive's only metadata object)."""
        meta = {
            "primary_key": list(buffer.primary_key) if buffer.primary_key else None,
            "schema_version": buffer.schema_version,
        }
        StoragePath(schema_path(self.root, buffer.name)).write_text(json.dumps(meta, indent=2))

    # -- append ------------------------------------------------------------------------------------

    def _append(self, table: str, rows: Iterable[dict]) -> None:
        self._raise_if_failed()
        if self.mode != "append":
            raise RuntimeError("cannot append to a store opened in read mode")
        buffer = self._tables.get(table)
        if buffer is None:
            raise KeyError(f"table {table!r} is not registered; call store.table({table!r}, ...) first")
        over_cap = False
        with buffer.lock:
            for row in rows:
                stamped = dict(row)
                stamped[SEQ_COLUMN] = buffer.next_seq
                stamped[WRITER_COLUMN] = self.writer_id
                buffer.next_seq += 1
                buffer.pending.append(stamped)
            over_cap = len(buffer.pending) >= self._max_buffer_rows
        if over_cap:
            self._wake.set()

    def write(self, name: str, metadata: Mapping[str, object] | None, data: bytes) -> str:
        """Append one opaque blob to the reserved ``blobs`` table; return its ``finestore://`` URI.

        The payload is stored inline as a Parquet binary column. ``metadata`` is kept verbatim as a
        JSON string for the reader to surface.
        """
        if BLOBS_TABLE not in self._tables:
            self.table(BLOBS_TABLE, primary_key=(BLOB_NAME_COLUMN,))
        self._append(
            BLOBS_TABLE,
            [
                {
                    BLOB_NAME_COLUMN: name,
                    "size": len(data),
                    "metadata_json": json.dumps(dict(metadata or {})),
                    "data": data,
                }
            ],
        )
        return build_uri(BLOBS_TABLE, name)

    # -- flush / seal / close ----------------------------------------------------------------------

    def flush(self) -> None:
        """Write every table's buffered rows to shards now, blocking until they are durable."""
        self._raise_if_failed()
        with self._flush_lock:
            for buffer in list(self._tables.values()):
                self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: _Buffer) -> None:
        with buffer.lock:
            if not buffer.pending:
                return
            rows = buffer.pending
            buffer.pending = []
        table = pa.Table.from_pylist(_drop_empty_struct_keys(rows), schema=buffer.schema)
        min_seq = min(row[SEQ_COLUMN] for row in rows)
        path = shard_path(self.root, buffer.name, self.writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        _io.write_table(path, table)
        logger.debug("finestore flushed %d rows to %s", len(rows), path)

    def seal(self) -> None:
        """Flush, then mark the archive sealed so readers know the run is complete."""
        self.flush()
        StoragePath(sealed_path(self.root)).write_text(json.dumps({"writer": self.writer_id}))

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

    # -- read convenience --------------------------------------------------------------------------

    def reader(self) -> CompositeReader:
        """A reader over this archive (composes every writer's shards)."""
        return CompositeReader(self.root)

    def resolve(self, uri: str):
        """Resolve a ``finestore://`` reference against this archive (see :meth:`CompositeReader.resolve`)."""
        return self.reader().resolve(uri)


class DataTable:
    """A handle to one table: append on the write side, scan/point on the read side."""

    def __init__(self, store: DataStore, name: str) -> None:
        self._store = store
        self.name = name

    def append(self, row: dict) -> None:
        """Buffer one row for the next flush."""
        self._store._append(self.name, [row])

    def extend(self, rows: Iterable[dict]) -> None:
        """Buffer many rows for the next flush."""
        self._store._append(self.name, rows)

    def scan(self, *, columns: Sequence[str] | None = None, where=None):
        """Read this table (deduplicated) — see :meth:`CompositeReader.scan`."""
        return self._store.reader().scan(self.name, columns=columns, where=where)

    def point(self, **keys):
        """Look up one row by its primary-key columns — see :meth:`CompositeReader.point`."""
        return self._store.reader().point(self.name, **keys)
