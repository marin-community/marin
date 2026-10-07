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
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import ExponentialBackoff

from finestore.commit import ClearSeal, CommitCoordinator, CommitDelta, TableAddition, initialize_archive, write_schema
from finestore.compaction import CompactionResult, compact_table
from finestore.layout import (
    CHUNKED_BLOBS_FEATURE,
    RESERVED_COLUMNS,
    BlobColumns,
    BlobTables,
    CommitToken,
    FineStoreLayout,
    OnConflict,
    SealMarker,
    Shard,
    SystemColumns,
    TableMetadata,
    build_uri,
)
from finestore.migrations import migrate_for_write
from finestore.reader import ReadView
from finestore.shard_writer import ROW_GROUP_TARGET_BYTES, ShardWriter, table_row_groups

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_MAX_BUFFER_BYTES = ROW_GROUP_TARGET_BYTES
DEFAULT_COMPACTION_LEVELS = (16, 4, 4)
OBJECT_PART_BYTES = 8 * 1024 * 1024
_MAINTENANCE_BACKOFF_INITIAL = 1.0
_MAINTENANCE_BACKOFF_MAXIMUM = 10 * 60.0

_BLOB_SCHEMA = pa.schema(
    [
        pa.field(BlobColumns.NAME, pa.string()),
        pa.field(BlobColumns.SIZE, pa.int64()),
        pa.field(BlobColumns.METADATA, pa.string()),
        pa.field(BlobColumns.DATA, pa.binary()),
        pa.field(BlobColumns.PART_COUNT, pa.int64()),
    ]
)
_BLOB_PART_SCHEMA = pa.schema(
    [
        pa.field(BlobColumns.NAME, pa.string()),
        pa.field(BlobColumns.PART, pa.int64()),
        pa.field(BlobColumns.DATA, pa.binary()),
    ]
)


def _default_writer_id() -> str:
    return uuid.uuid4().hex


class PrimaryKeyConflict(ValueError):
    """Two rows in one writer session share a key but have different payloads."""


class TransactionTooLarge(ValueError):
    """A transaction exceeded its configured in-memory byte limit."""


def _row_digest(row: Mapping[str, object]) -> bytes:
    payload = {name: value for name, value in row.items() if name not in (SystemColumns.SEQUENCE, SystemColumns.WRITER)}
    return hashlib.blake2b(json.dumps(payload, sort_keys=True, default=repr).encode(), digest_size=16).digest()


def _drop_empty_struct_fields(schema: pa.Schema) -> pa.Schema:
    fields = [field for field in schema if not (pa.types.is_struct(field.type) and field.type.num_fields == 0)]
    return pa.schema(fields, metadata=schema.metadata)


def _align_arrow_row(table: pa.Table, schema: pa.Schema) -> pa.Table:
    if table.schema == schema:
        return table
    aligned = pa.concat_tables([schema.empty_table(), table], promote_options="permissive")
    return aligned.select(schema.names)


def _with_stamp_columns(schema: pa.Schema) -> pa.Schema:
    reserved = set(schema.names) & RESERVED_COLUMNS
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"table schema uses reserved FineStore columns: {names}")
    fields = list(schema)
    fields.append(pa.field(SystemColumns.SEQUENCE, pa.int64()))
    fields.append(pa.field(SystemColumns.WRITER, pa.string()))
    return pa.schema(fields)


def _reject_reserved_columns(columns: Iterable[str]) -> None:
    reserved = set(columns) & RESERVED_COLUMNS
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"row uses reserved FineStore column: {names}")


def _addition_delta(additions: dict[str, TableAddition]) -> CommitDelta:
    required_features = frozenset({CHUNKED_BLOBS_FEATURE}) if BlobTables.PARTS in additions else frozenset()
    return CommitDelta(additions=additions, required_features=required_features, seal_update=ClearSeal())


@dataclass(frozen=True)
class ArrowRow:
    """One logical row and its retained Arrow buffers."""

    table: pa.Table
    key: tuple
    sort_key: tuple | None
    digest: bytes | None
    sequence: int | None = None


@dataclass(frozen=True)
class PendingRows:
    """Rows exclusively claimed from a table buffer for one commit attempt."""

    rows: tuple[ArrowRow, ...]
    nbytes: int


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
        compaction_levels: Sequence[int] = DEFAULT_COMPACTION_LEVELS,
    ) -> None:
        self.root = root
        self._layout = FineStoreLayout(self.root)
        initialize_archive(self._layout)
        self._commits = CommitCoordinator(self._layout)
        self.writer_id = writer_id or _default_writer_id()
        self._flush_interval = flush_interval
        self._max_buffer_bytes = max_buffer_bytes
        self._compaction_levels = tuple(compaction_levels)
        if not self._compaction_levels or any(limit < 1 for limit in self._compaction_levels):
            raise ValueError("compaction_levels must contain positive shard limits")
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
        descriptor, parts = self._blob_rows(name, data, metadata)
        blob_table = self._blob_table()
        part_table = self._blob_part_table() if parts else None
        with self._commit_lock:
            if part_table is not None:
                part_table.extend(parts)
            blob_table.append(descriptor)
        return build_uri(BlobTables.DESCRIPTORS, name)

    def _blob_table(self) -> DataTable:
        return self._tables.get(BlobTables.DESCRIPTORS) or self.table(
            BlobTables.DESCRIPTORS,
            primary_key=(BlobColumns.NAME,),
            schema=_BLOB_SCHEMA,
            on_conflict=OnConflict.SUPERSEDE,
        )

    def _blob_part_table(self) -> DataTable:
        return self._tables.get(BlobTables.PARTS) or self.table(
            BlobTables.PARTS,
            primary_key=(BlobColumns.NAME, BlobColumns.PART),
            schema=_BLOB_PART_SCHEMA,
            on_conflict=OnConflict.SUPERSEDE,
        )

    @staticmethod
    def _blob_rows(name: str, data: bytes, metadata: Mapping[str, object] | None) -> tuple[dict, list[dict]]:
        descriptor: dict[str, object] = {
            BlobColumns.NAME: name,
            BlobColumns.SIZE: len(data),
            BlobColumns.METADATA: json.dumps(dict(metadata or {})),
        }
        if len(data) <= OBJECT_PART_BYTES:
            descriptor[BlobColumns.DATA] = data
            return descriptor, []
        parts = [
            {
                BlobColumns.NAME: name,
                BlobColumns.PART: index,
                BlobColumns.DATA: memoryview(data)[offset : offset + OBJECT_PART_BYTES],
            }
            for index, offset in enumerate(range(0, len(data), OBJECT_PART_BYTES))
        ]
        descriptor[BlobColumns.PART_COUNT] = len(parts)
        return descriptor, parts

    def read_view(self) -> ReadView:
        """Pin a snapshot using one HEAD read."""
        return ReadView(self.root)

    def transaction(self, *, max_bytes: int | None = None) -> Transaction:
        """Create a transaction bounded by actual Arrow buffer bytes."""
        self.raise_if_failed()
        transaction_limit = self._max_buffer_bytes if max_bytes is None else max_bytes
        if transaction_limit <= 0:
            raise ValueError("max_bytes must be positive")
        return Transaction(self, max_bytes=transaction_limit)

    def unbounded_transaction(self) -> Transaction:
        """Create a transaction without an Arrow buffer byte limit."""
        self.raise_if_failed()
        return Transaction(self, max_bytes=None)

    def flush(self) -> CommitToken | None:
        """Commit all buffers with one HEAD update, or return ``None`` when empty."""
        self.raise_if_failed()
        return self._flush_tables(list(self._tables.values()))

    def flush_table(self, table: DataTable) -> CommitToken | None:
        """Commit ``table``'s buffer, pairing blob descriptor and part buffers when needed."""
        self.raise_if_failed()
        if table.name in (BlobTables.DESCRIPTORS, BlobTables.PARTS):
            tables = [
                related
                for name in (BlobTables.PARTS, BlobTables.DESCRIPTORS)
                if (related := self._tables.get(name)) is not None
            ]
            return self._flush_tables(tables)
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
                return self._commits.commit(_addition_delta(additions))
            except BaseException:
                for table, pending in claimed.items():
                    table._restore(pending)
                raise

    def _commit_transaction(self, rows: dict[str, list[ArrowRow]]) -> CommitToken:
        with self._commit_lock:
            additions = {}
            for name, pending_rows in rows.items():
                table = self._tables.get(name)
                if table is None:
                    raise KeyError(f"table {name!r} must be registered before the transaction")
                stamped = table._stamp_arrow_rows(pending_rows)
                pending = PendingRows(tuple(stamped), sum(row.table.nbytes for row in stamped))
                additions[name] = TableAddition(metadata_path=table.metadata_path, shards=(table._write(pending),))
            if not additions:
                snapshot = self._commits.snapshot()
                if snapshot.token is None:
                    return self._commits.commit(CommitDelta())
                return snapshot.token
            return self._commits.commit(_addition_delta(additions))

    def compact(self, table: str, *, generation: int | None = None) -> CompactionResult:
        """Logically compact one table against the current manifest."""
        with self._commit_lock:
            return compact_table(self.root, table, generation=generation, coordinator=self._commits)

    def maintain(self) -> CommitToken | None:
        """Flush pending rows and compact each overfull level once."""
        token = self.flush()
        view = self.read_view()
        for name in view.table_names():
            for generation, shard_limit in enumerate(self._compaction_levels):
                level_shards = [shard for shard in view.list_shards(name) if shard.generation == generation]
                if len(level_shards) <= shard_limit:
                    continue
                self.compact(name, generation=generation)
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
            backoff = ExponentialBackoff(
                initial=_MAINTENANCE_BACKOFF_INITIAL,
                maximum=_MAINTENANCE_BACKOFF_MAXIMUM,
                factor=2.0,
                jitter=0.25,
            )
            failures = 0
            while not self._stop.is_set():
                try:
                    self.maintain()
                    break
                except Exception as exc:
                    if not is_transient_s3_error(exc):
                        self._error = exc
                        logger.exception("FineStore background maintenance failed")
                        return
                    failures += 1
                    delay = backoff.next_interval()
                    logger.warning(
                        "FineStore background maintenance failed for %s (consecutive_failures=%d); "
                        "retrying in %.2fs: %s",
                        self.root,
                        failures,
                        delay,
                        exc,
                    )
                    if self._stop.wait(delay):
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
        self._transaction._add_rows({self._name: [row]})

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)


class Transaction:
    """A set of retained Arrow rows and objects published through one commit token."""

    def __init__(self, store: DataStore, *, max_bytes: int | None) -> None:
        self._store = store
        self._base = store.read_view()
        self._rows: dict[str, list[ArrowRow]] = {}
        self._objects: dict[str, bytes] = {}
        self._max_bytes = max_bytes
        self._buffer_bytes = 0
        self._closed = False
        self.token: CommitToken | None = None

    def table(self, name: str) -> TransactionTable:
        if name not in self._store._tables:
            raise KeyError(f"table {name!r} must be registered before the transaction")
        return TransactionTable(self, name)

    def _add_rows(self, rows: dict[str, list[dict]]) -> None:
        if self._closed:
            raise RuntimeError("transaction is closed")
        buffered = {}
        for name, table_rows in rows.items():
            table = self._store._tables.get(name)
            if table is None:
                raise KeyError(f"table {name!r} must be registered before the transaction")
            buffered[name] = table._arrow_rows(table_rows)
        added_bytes = sum(row.table.nbytes for table_rows in buffered.values() for row in table_rows)
        if self._max_bytes is not None and self._buffer_bytes + added_bytes > self._max_bytes:
            raise TransactionTooLarge(f"transaction payload exceeds {self._max_bytes} bytes; commit a smaller batch")
        self._buffer_bytes += added_bytes
        for name, table_rows in buffered.items():
            self._rows.setdefault(name, []).extend(table_rows)

    def write_object(self, name: str, data: bytes, metadata: Mapping[str, object] | None = None) -> str:
        """Buffer one named byte object and return its ``finestore://`` URI."""
        if self._closed:
            raise RuntimeError("transaction is closed")
        blob_table = self._store._blob_table()
        descriptor, parts = self._store._blob_rows(name, data, metadata)
        rows = {blob_table.name: [descriptor]}
        if parts:
            part_table = self._store._blob_part_table()
            rows[part_table.name] = parts
        self._add_rows(rows)
        self._objects[name] = data
        return build_uri(BlobTables.DESCRIPTORS, name)

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
        self._pending: list[ArrowRow] = []
        self._pending_bytes = 0
        self._next_seq = start_seq
        self._lock = threading.Lock()
        self._digests: dict[tuple, bytes] | None = {} if on_conflict is OnConflict.ERROR else None

    def append(self, row: dict) -> None:
        self.extend([row])

    def _check_conflict(self, row: ArrowRow) -> None:
        if self._digests is None:
            return
        assert row.digest is not None
        previous = self._digests.get(row.key)
        if previous is None:
            self._digests[row.key] = row.digest
            return
        if previous != row.digest:
            pairs = ", ".join(f"{name}={value!r}" for name, value in zip(self.primary_key, row.key, strict=True))
            raise PrimaryKeyConflict(
                f"table {self.name!r} already holds a different row for primary key ({pairs}); "
                "declare on_conflict=SUPERSEDE when replacement is intended"
            )

    def _arrow_rows(self, rows: Iterable[dict]) -> list[ArrowRow]:
        buffered = []
        for row in rows:
            _reject_reserved_columns(row)
            key = tuple(row.get(name) for name in self.primary_key)
            sort_key = (
                tuple((value is None, value) for value in key) if all(name in row for name in self.primary_key) else None
            )
            digest = _row_digest(row) if self._digests is not None else None
            placeholder = dict(row)
            placeholder[SystemColumns.SEQUENCE] = 0
            placeholder[SystemColumns.WRITER] = self._writer_id
            table = pa.Table.from_pylist([placeholder], schema=self._schema)
            buffered.append(ArrowRow(table, key, sort_key, digest))
        return buffered

    def _stamp_arrow_rows(self, rows: Iterable[ArrowRow]) -> list[ArrowRow]:
        stamped = []
        with self._lock:
            for row in rows:
                self._check_conflict(row)
                sequence = self._next_seq
                self._next_seq += 1
                index = row.table.schema.get_field_index(SystemColumns.SEQUENCE)
                table = row.table.set_column(
                    index,
                    pa.field(SystemColumns.SEQUENCE, pa.int64()),
                    pa.array([sequence], type=pa.int64()),
                )
                stamped.append(ArrowRow(table, row.key, row.sort_key, row.digest, sequence))
        return stamped

    def extend(self, rows: Iterable[dict]) -> None:
        self._scheduler.raise_if_failed()
        buffered = self._stamp_arrow_rows(self._arrow_rows(rows))
        with self._lock:
            self._pending.extend(buffered)
            self._pending_bytes += sum(row.table.nbytes for row in buffered)
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
            self._pending_bytes += pending.nbytes

    def _write(self, pending: PendingRows) -> Shard:
        rows = list(pending.rows)
        primary_key_sorted = all(row.sort_key is not None for row in rows)
        if primary_key_sorted:
            rows.sort(key=lambda row: row.sort_key or ())
        assert all(row.sequence is not None for row in rows)
        min_seq = min(row.sequence for row in rows if row.sequence is not None)
        max_seq = max(row.sequence for row in rows if row.sequence is not None)
        path = self._layout.shard_path(self.name, self._writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        schema = self._schema or pa.unify_schemas([row.table.schema for row in rows], promote_options="permissive")
        schema = _drop_empty_struct_fields(schema)
        arrow_rows = (_align_arrow_row(row.table, schema) for row in rows)
        with ShardWriter(path, schema) as writer:
            for group in table_row_groups(self.name, arrow_rows):
                writer.write_table(group)
        written = writer.result
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
