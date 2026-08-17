# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transactional FineStore writer with bounded in-memory level-zero batches."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

import pyarrow as pa

from finestore.commit import ClearSeal, CommitCoordinator, CommitDelta, TableAddition, initialize_archive, write_schema
from finestore.compaction import CompactionResult
from finestore.compaction import compact as compact_table
from finestore.layout import (
    BLOB_DATA_COLUMN,
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    RESERVED_COLUMNS,
    SEQ_COLUMN,
    WRITER_COLUMN,
    CommitToken,
    FineStoreLayout,
    OnConflict,
    SealMarker,
    Shard,
    TableMetadata,
    build_uri,
)
from finestore.migrations import migrate_for_write
from finestore.reader import ReadView
from finestore.shard_writer import write_table

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_MAX_BUFFER_BYTES = 100 * 1024 * 1024
DEFAULT_COMPACTION_SHARDS = 8


def _default_writer_id() -> str:
    return uuid.uuid4().hex


class PrimaryKeyConflict(ValueError):
    """Two rows in one writer session share a key but have different payloads."""


class TransactionTooLarge(ValueError):
    """A transaction exceeded its configured in-memory byte limit."""


def _row_digest(row: Mapping[str, object]) -> bytes:
    payload = {name: value for name, value in row.items() if name not in (SEQ_COLUMN, WRITER_COLUMN)}
    return hashlib.blake2b(json.dumps(payload, sort_keys=True, default=repr).encode(), digest_size=16).digest()


def _drop_empty_struct_keys(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    keys = set().union(*(row.keys() for row in rows))
    drop = set()
    for key in keys:
        dicts = [row.get(key) for row in rows if isinstance(row.get(key), dict)]
        if dicts and not any(dicts):
            drop.add(key)
    return rows if not drop else [{key: value for key, value in row.items() if key not in drop} for row in rows]


def _estimate_bytes(value: object) -> int:
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, dict):
        return sum(len(str(key)) + _estimate_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_bytes(item) for item in value)
    return 8


def _with_stamp_columns(schema: pa.Schema) -> pa.Schema:
    reserved = set(schema.names) & RESERVED_COLUMNS
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"table schema uses reserved FineStore columns: {names}")
    fields = list(schema)
    fields.append(pa.field(SEQ_COLUMN, pa.int64()))
    fields.append(pa.field(WRITER_COLUMN, pa.string()))
    return pa.schema(fields)


def _reject_reserved_columns(columns: Iterable[str]) -> None:
    reserved = set(columns) & RESERVED_COLUMNS
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"row uses reserved FineStore column: {names}")


@dataclass(frozen=True)
class PendingRows:
    """Rows exclusively claimed from a table buffer for one commit attempt."""

    rows: tuple[dict, ...]
    estimated_bytes: int


class FlushScheduler(Protocol):
    """Store operations needed by a table handle."""

    def request_flush(self) -> None: ...

    def raise_if_failed(self) -> None: ...

    def flush_table(self, table: DataTable) -> CommitToken | None: ...


class DataStore:
    """A writable FineStore archive for one writer identity."""

    def __init__(
        self,
        root: str,
        *,
        writer_id: str | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        compaction_shards: int = DEFAULT_COMPACTION_SHARDS,
    ) -> None:
        self.root = root
        self._layout = FineStoreLayout(self.root)
        initialize_archive(self._layout)
        self._commits = CommitCoordinator(self._layout)
        self.writer_id = writer_id or _default_writer_id()
        self._flush_interval = flush_interval
        self._max_buffer_bytes = max_buffer_bytes
        self._compaction_shards = compaction_shards
        self._tables: dict[str, DataTable] = {}
        self._register_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        snapshot = self._commits.snapshot()
        if snapshot.manifest.sealed is not None:
            self._commits.commit(CommitDelta(seal_update=ClearSeal()), base=snapshot)
            logger.info("FineStore reopened sealed archive %s for writing", self.root)
        self._thread = threading.Thread(target=self._run, name="finestore-maintenance", daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, root: str, **kwargs) -> DataStore:
        migrate_for_write(root)
        return cls(root, **kwargs)

    def table(
        self,
        name: str,
        *,
        primary_key: Sequence[str],
        schema: pa.Schema | None = None,
        schema_version: int = 1,
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> DataTable:
        """Register or return a table handle.

        Registration writes content-addressed metadata. A transaction can then refer to the
        table by name without repeating its schema contract.
        """
        with self._register_lock:
            existing = self._tables.get(name)
            if existing is not None:
                return existing
            metadata = TableMetadata(
                primary_key=tuple(primary_key), schema_version=schema_version, on_conflict=on_conflict
            )
            _reject_reserved_columns(primary_key)
            metadata_path = write_schema(self._layout, metadata)
            view = self.read_view()
            existing_metadata_path = view.table_metadata_path(name)
            if existing_metadata_path is not None and existing_metadata_path != metadata_path:
                raise ValueError(f"table {name!r} already exists with different metadata")
            table = DataTable(
                name,
                writer_id=self.writer_id,
                layout=self._layout,
                metadata_path=metadata_path,
                max_buffer_bytes=self._max_buffer_bytes,
                scheduler=self,
                primary_key=metadata.primary_key,
                schema=schema,
                schema_version=schema_version,
                on_conflict=on_conflict,
                start_seq=view.max_seq(name) + 1,
            )
            self._tables[name] = table
            return table

    def write_object(self, name: str, data: bytes, metadata: Mapping[str, object] | None = None) -> str:
        """Buffer one named byte object and return its ``finestore://`` URI."""
        self._blob_table().append(self._blob_row(name, data, metadata))
        return build_uri(BLOBS_TABLE, name)

    def _blob_table(self) -> DataTable:
        return self._tables.get(BLOBS_TABLE) or self.table(
            BLOBS_TABLE, primary_key=(BLOB_NAME_COLUMN,), on_conflict=OnConflict.SUPERSEDE
        )

    @staticmethod
    def _blob_row(name: str, data: bytes, metadata: Mapping[str, object] | None) -> dict:
        return {
            BLOB_NAME_COLUMN: name,
            "size": len(data),
            "metadata_json": json.dumps(dict(metadata or {})),
            BLOB_DATA_COLUMN: data,
        }

    @staticmethod
    def estimate_object_bytes(name: str, data: bytes, metadata: Mapping[str, object] | None = None) -> int:
        """Return the transaction byte estimate for one named object."""
        return _estimate_bytes(DataStore._blob_row(name, data, metadata))

    def read_view(self) -> ReadView:
        """Pin a snapshot using one HEAD read."""
        return ReadView(self.root)

    def transaction(self, *, max_bytes: int | None = None) -> Transaction:
        """Create a transaction bounded by ``max_bytes`` of estimated payload data."""
        self.raise_if_failed()
        transaction_limit = self._max_buffer_bytes if max_bytes is None else max_bytes
        if transaction_limit <= 0:
            raise ValueError("max_bytes must be positive")
        return Transaction(self, max_bytes=transaction_limit)

    def flush(self) -> CommitToken | None:
        """Commit all buffers with one HEAD update, or return ``None`` when empty."""
        self.raise_if_failed()
        return self._flush_tables(list(self._tables.values()))

    def flush_table(self, table: DataTable) -> CommitToken | None:
        """Commit only ``table``'s current buffer."""
        self.raise_if_failed()
        return self._flush_tables([table])

    def _flush_tables(self, tables: Sequence[DataTable]) -> CommitToken | None:
        with self._commit_lock:
            claimed = {table: table._claim_pending() for table in tables}
            claimed = {table: pending for table, pending in claimed.items() if pending is not None}
            if not claimed:
                return None
            try:
                additions = {
                    table.name: TableAddition(metadata_path=table.metadata_path, shards=(table._write(pending),))
                    for table, pending in claimed.items()
                }
                return self._commits.commit(CommitDelta(additions=additions, seal_update=ClearSeal()))
            except BaseException:
                for table, pending in claimed.items():
                    table._restore(pending)
                raise

    def _commit_transaction(self, rows: dict[str, list[dict]]) -> CommitToken:
        with self._commit_lock:
            additions = {}
            for name, pending_rows in rows.items():
                table = self._tables.get(name)
                if table is None:
                    raise KeyError(f"table {name!r} must be registered before the transaction")
                stamped = table._stamp(pending_rows)
                pending = PendingRows(tuple(stamped), sum(_estimate_bytes(row) for row in pending_rows))
                additions[name] = TableAddition(metadata_path=table.metadata_path, shards=(table._write(pending),))
            if not additions:
                snapshot = self._commits.snapshot()
                if snapshot.token is None:
                    return self._commits.commit(CommitDelta())
                return snapshot.token
            return self._commits.commit(CommitDelta(additions=additions, seal_update=ClearSeal()))

    def compact(self, table: str) -> CompactionResult:
        """Logically compact one table against the current manifest."""
        with self._commit_lock:
            return compact_table(self.root, table, coordinator=self._commits)

    def maintain(self) -> CommitToken | None:
        """Flush pending rows and compact tables that crossed the shard threshold."""
        token = self.flush()
        view = self.read_view()
        for name in view.table_names():
            if len(view.list_shards(name)) >= self._compaction_shards:
                self.compact(name)
                view = self.read_view()
                token = view.token
        return token

    def seal(self) -> CommitToken:
        """Flush, compact active tables, and publish sealed state in the manifest."""
        self.flush()
        superseded = {}
        for name in self.read_view().table_names():
            result = self.compact(name)
            if result.superseded:
                superseded[name] = result.superseded
        return self._commits.commit(CommitDelta(seal_update=SealMarker(writer=self.writer_id, superseded=superseded)))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        self._thread.join()
        self.flush()

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                self.maintain()
            except BaseException as exc:
                self._error = exc
                logger.exception("FineStore background maintenance failed")
                return

    def request_flush(self) -> None:
        self._wake.set()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


class TransactionTable:
    """One table's rows inside a transaction."""

    def __init__(self, transaction: Transaction, name: str) -> None:
        self._transaction = transaction
        self._name = name

    def add(self, row: dict) -> None:
        self._transaction._add(self._name, row)

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)


class Transaction:
    """A bounded set of rows and objects published through one commit token."""

    def __init__(self, store: DataStore, *, max_bytes: int) -> None:
        self._store = store
        self._base = store.read_view()
        self._rows: dict[str, list[dict]] = {}
        self._objects: dict[str, bytes] = {}
        self._max_bytes = max_bytes
        self._estimated_bytes = 0
        self._closed = False
        self.token: CommitToken | None = None

    def table(self, name: str) -> TransactionTable:
        if name not in self._store._tables:
            raise KeyError(f"table {name!r} must be registered before the transaction")
        return TransactionTable(self, name)

    def _add(self, name: str, row: dict) -> None:
        if self._closed:
            raise RuntimeError("transaction is closed")
        copied = dict(row)
        estimated_bytes = _estimate_bytes(copied)
        if self._estimated_bytes + estimated_bytes > self._max_bytes:
            raise TransactionTooLarge(f"transaction payload exceeds {self._max_bytes} bytes; commit a smaller batch")
        self._estimated_bytes += estimated_bytes
        self._rows.setdefault(name, []).append(copied)

    def write_object(self, name: str, data: bytes, metadata: Mapping[str, object] | None = None) -> str:
        """Buffer one named byte object and return its ``finestore://`` URI."""
        if self._closed:
            raise RuntimeError("transaction is closed")
        self._store._blob_table()
        self._add(BLOBS_TABLE, self._store._blob_row(name, data, metadata))
        self._objects[name] = data
        return build_uri(BLOBS_TABLE, name)

    def lookup(self, name: str) -> bytes | None:
        """Read this transaction's latest object value, then its pinned committed base."""
        if name in self._objects:
            return self._objects[name]
        return self._base.read_blob(name)

    def commit(self) -> CommitToken:
        if self._closed:
            if self.token is None:
                raise RuntimeError("transaction closed without committing")
            return self.token
        self.token = self._store._commit_transaction(self._rows)
        self._closed = True
        return self.token

    def __enter__(self) -> Transaction:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.commit()
        else:
            self._closed = True


class DataTable:
    """An append buffer whose claimed batches become immutable level-zero shards."""

    def __init__(
        self,
        name: str,
        *,
        writer_id: str,
        layout: FineStoreLayout,
        metadata_path: str,
        max_buffer_bytes: int,
        scheduler: FlushScheduler,
        primary_key: tuple[str, ...],
        schema: pa.Schema | None,
        schema_version: int,
        on_conflict: OnConflict = OnConflict.ERROR,
        start_seq: int = 0,
    ) -> None:
        self.name = name
        self.primary_key = primary_key
        self.schema_version = schema_version
        self.on_conflict = on_conflict
        self.metadata_path = metadata_path
        self._writer_id = writer_id
        self._layout = layout
        self._max_buffer_bytes = max_buffer_bytes
        self._scheduler = scheduler
        self._schema = _with_stamp_columns(schema) if schema is not None else None
        self._pending: list[dict] = []
        self._pending_bytes = 0
        self._next_seq = start_seq
        self._lock = threading.Lock()
        self._digests: dict[tuple, bytes] | None = {} if on_conflict is OnConflict.ERROR else None

    def append(self, row: dict) -> None:
        self.extend([row])

    def _check_conflict(self, row: dict) -> None:
        if self._digests is None:
            return
        key = tuple(row.get(name) for name in self.primary_key)
        digest = _row_digest(row)
        previous = self._digests.get(key)
        if previous is None:
            self._digests[key] = digest
            return
        if previous != digest:
            pairs = ", ".join(f"{name}={row.get(name)!r}" for name in self.primary_key)
            raise PrimaryKeyConflict(
                f"table {self.name!r} already holds a different row for primary key ({pairs}); "
                "declare on_conflict=SUPERSEDE when replacement is intended"
            )

    def _stamp(self, rows: Iterable[dict]) -> list[dict]:
        stamped_rows = []
        with self._lock:
            for row in rows:
                _reject_reserved_columns(row)
                self._check_conflict(row)
                stamped = dict(row)
                stamped[SEQ_COLUMN] = self._next_seq
                stamped[WRITER_COLUMN] = self._writer_id
                self._next_seq += 1
                stamped_rows.append(stamped)
        return stamped_rows

    def extend(self, rows: Iterable[dict]) -> None:
        self._scheduler.raise_if_failed()
        rows = list(rows)
        stamped = self._stamp(rows)
        with self._lock:
            self._pending.extend(stamped)
            self._pending_bytes += sum(_estimate_bytes(row) for row in rows)
            over_cap = self._pending_bytes >= self._max_buffer_bytes
        if over_cap:
            self._scheduler.request_flush()

    def _claim_pending(self) -> PendingRows | None:
        with self._lock:
            if not self._pending:
                return None
            pending = PendingRows(tuple(self._pending), self._pending_bytes)
            self._pending = []
            self._pending_bytes = 0
            return pending

    def _restore(self, pending: PendingRows) -> None:
        with self._lock:
            self._pending = list(pending.rows) + self._pending
            self._pending_bytes += pending.estimated_bytes

    def _write(self, pending: PendingRows) -> Shard:
        rows = list(pending.rows)
        table = pa.Table.from_pylist(_drop_empty_struct_keys(rows), schema=self._schema)
        sort_columns = [(name, "ascending") for name in self.primary_key if name in table.column_names]
        primary_key_sorted = len(sort_columns) == len(self.primary_key)
        if primary_key_sorted:
            table = table.sort_by(sort_columns)
        min_seq = min(row[SEQ_COLUMN] for row in rows)
        max_seq = max(row[SEQ_COLUMN] for row in rows)
        path = self._layout.shard_path(self.name, self._writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        written = write_table(path, table)
        return Shard(
            path=path,
            writer=self._writer_id,
            generation=0,
            rows=len(rows),
            min_seq=min_seq,
            max_seq=max_seq,
            size_bytes=written.size_bytes,
            content_sha256=written.content_sha256,
            primary_key_sorted=primary_key_sorted,
        )

    def flush(self) -> CommitToken | None:
        return self._scheduler.flush_table(self)
