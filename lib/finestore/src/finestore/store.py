# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transactional FineStore writer with bounded in-memory level-zero batches."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Protocol

import pyarrow as pa
from rigging import telemetry
from rigging.filesystem.distributed_lock import HEARTBEAT_INTERVAL, LeaseLostError, create_lock, lease_refresh
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import launch_user
from rigging.timing import ExponentialBackoff, retry_with_backoff

from finestore.commit import (
    ArchiveSnapshot,
    Attribution,
    ClearSeal,
    CommitConflict,
    CommitCoordinator,
    CommitDelta,
    Precondition,
    TableAddition,
    check_preconditions,
    initialize_archive,
    write_schema,
)
from finestore.compaction import COMPACTION_TARGET_BYTES, CompactionResult, compact_level_zero_tables, compact_table
from finestore.layout import (
    CHUNKED_BLOBS_FEATURE,
    INITIAL_TABLE_INCARNATION,
    RESERVED_COLUMNS,
    STAGED_TABLE_DIR,
    BlobColumns,
    BlobTables,
    CommitToken,
    FineStoreLayout,
    KeyRange,
    Manifest,
    ManifestTable,
    OnConflict,
    SealMarker,
    Shard,
    SystemColumns,
    TableMetadata,
    TableNameConflict,
    blob_table_metadata,
    build_uri,
)
from finestore.migrations import migrate_for_write
from finestore.reader import ReadView
from finestore.shard_writer import ROW_GROUP_TARGET_BYTES, ShardWriter, table_row_groups

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_MAX_BUFFER_BYTES = ROW_GROUP_TARGET_BYTES
DEFAULT_COMPACTION_LEVELS = (16, 4, 4)
_CLOSE_COMPACTION_MAX_INPUT_BYTES = 64 * 1024 * 1024
_CLOSE_COMPACTION_MAX_INPUT_ROWS = 16_384
OBJECT_PART_BYTES = 8 * 1024 * 1024
INLINE_BLOB_BYTES = 128 * 1024
_MAINTENANCE_BACKOFF_INITIAL = 1.0
_MAINTENANCE_BACKOFF_MAXIMUM = 10 * 60.0
DEFAULT_UPDATE_ATTEMPTS = 20
# retry_with_backoff copies the schedule, so one instance serves every update.
_UPDATE_BACKOFF = ExponentialBackoff(initial=0.05, maximum=2.0, factor=2.0, jitter=0.25)
_SHOWN_KEYS = 5

_LEASE_ACQUIRED = telemetry.counter("finestore_maintenance_lease_acquired", unit="{acquisition}")
_LEASE_LOST = telemetry.counter("finestore_maintenance_lease_lost", unit="{loss}")

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


class RowConflict(CommitConflict):
    """Another writer committed keys this transaction read after its snapshot was pinned."""

    def __init__(self, table: str, keys: Sequence[tuple]) -> None:
        self.table = table
        self.keys = tuple(keys)
        shown = ", ".join(repr(key) for key in self.keys[:_SHOWN_KEYS])
        more = f" and {len(self.keys) - _SHOWN_KEYS} more" if len(self.keys) > _SHOWN_KEYS else ""
        super().__init__(f"table {table!r} rows changed after this transaction read them: {shown}{more}")


def _rows_unchanged_since(base: ReadView, guards: Mapping[str, frozenset[tuple]]) -> Precondition:
    sequence = base.token.sequence if base.token is not None else 0
    incarnations = {table: base.table_version(table).incarnation for table in guards}

    def check(manifest: Manifest) -> None:
        if manifest.sequence <= sequence:
            return
        view = ReadView(base.root, ArchiveSnapshot(manifest, token=None))
        for table, keys in guards.items():
            if manifest.table_version(table).incarnation != incarnations[table]:
                raise CommitConflict(f"table {table!r} was dropped or replaced after this transaction read it")
            changed = view.keys_committed_after(table, keys, sequence)
            if changed:
                raise RowConflict(table, changed)

    return check


def _attribution(author: str | None, message: str | None) -> Attribution | None:
    if message is None:
        if author is not None:
            raise ValueError("a commit author needs a message")
        return None
    author = author or launch_user()
    if not author:
        raise ValueError("no commit author given and none resolves from the launch; pass author=")
    return Attribution(author=author, message=message)


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


def _addition_delta(additions: dict[str, TableAddition], *, delta: CommitDelta | None = None) -> CommitDelta:
    base = delta or CommitDelta()
    required_features = frozenset({CHUNKED_BLOBS_FEATURE}) if BlobTables.PARTS in additions else frozenset()
    return replace(
        base, additions=additions, required_features=base.required_features | required_features, seal_update=ClearSeal()
    )


def _require_user_table(name: str) -> None:
    if name in (BlobTables.DESCRIPTORS, BlobTables.PARTS):
        raise ValueError(f"table {name!r} belongs to the named-object store")


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


@dataclass(frozen=True)
class _TransactionRows:
    writer: DataTable
    rows: list[ArrowRow]


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
        compaction_target_bytes: int = COMPACTION_TARGET_BYTES,
    ) -> None:
        self.root = root
        self._layout = FineStoreLayout(self.root)
        initialize_archive(self._layout)
        self._commits = CommitCoordinator(self._layout)
        self.writer_id = writer_id or _default_writer_id()
        self._flush_interval = flush_interval
        self._max_buffer_bytes = max_buffer_bytes
        self._compaction_levels = tuple(compaction_levels)
        self._compaction_target_bytes = compaction_target_bytes
        if not self._compaction_levels or any(limit < 1 for limit in self._compaction_levels):
            raise ValueError("compaction_levels must contain positive shard limits")
        self._tables: dict[str, DataTable] = {}
        self._register_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._lease = create_lock(self._layout.maintenance_lease_path, worker_id=self.writer_id)
        self._lease_epoch: int | None = None
        self._lease_refreshed = 0.0
        # Serializes maintenance passes from the background thread and from callers, so the
        # lease is acquired, refreshed, and released by one pass at a time.
        self._maintenance_lock = threading.RLock()
        snapshot = self._commits.last_snapshot()
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
            view = self.read_view()
            version = view.table_version(name)
            table = self._table_writer(
                name,
                metadata=metadata,
                schema=schema,
                incarnation=version.incarnation,
                start_seq=view.max_seq(name) + 1,
            )
            if version.metadata_path is not None and version.metadata_path != table.metadata_path:
                raise ValueError(f"table {name!r} already exists with different metadata")
            self._tables[name] = table
            return table

    def _table_writer(
        self,
        name: str,
        *,
        metadata: TableMetadata,
        schema: pa.Schema | None,
        incarnation: int,
        start_seq: int,
        storage_table: str | None = None,
    ) -> DataTable:
        _reject_reserved_columns(metadata.primary_key)
        return DataTable(
            name,
            writer_id=self.writer_id,
            layout=self._layout,
            metadata_path=write_schema(self._layout, metadata),
            incarnation=incarnation,
            max_buffer_bytes=self._max_buffer_bytes,
            scheduler=self,
            primary_key=metadata.primary_key,
            schema=schema,
            schema_version=metadata.schema_version,
            on_conflict=metadata.on_conflict,
            start_seq=start_seq,
            storage_table=storage_table,
        )

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
        metadata = blob_table_metadata(BlobTables.DESCRIPTORS)
        return self._tables.get(BlobTables.DESCRIPTORS) or self.table(
            BlobTables.DESCRIPTORS,
            primary_key=metadata.primary_key,
            schema=_BLOB_SCHEMA,
            schema_version=metadata.schema_version,
            on_conflict=metadata.on_conflict,
        )

    def _blob_part_table(self) -> DataTable:
        metadata = blob_table_metadata(BlobTables.PARTS)
        return self._tables.get(BlobTables.PARTS) or self.table(
            BlobTables.PARTS,
            primary_key=metadata.primary_key,
            schema=_BLOB_PART_SCHEMA,
            schema_version=metadata.schema_version,
            on_conflict=metadata.on_conflict,
        )

    @staticmethod
    def _blob_rows(name: str, data: bytes, metadata: Mapping[str, object] | None) -> tuple[dict, list[dict]]:
        descriptor: dict[str, object] = {
            BlobColumns.NAME: name,
            BlobColumns.SIZE: len(data),
            BlobColumns.METADATA: json.dumps(dict(metadata or {})),
        }
        if len(data) <= INLINE_BLOB_BYTES:
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

    def transaction(
        self,
        *,
        max_bytes: int | None = None,
        base: ReadView | None = None,
        message: str | None = None,
        author: str | None = None,
    ) -> Transaction:
        """Create a transaction with an Arrow buffer budget and an optional pinned base.

        With a ``message``, the commit this transaction publishes records it as ``commit_info``,
        together with ``author``, the commit's sequence and its publication time. ``author``
        defaults to the launching user (``rigging.provenance.launch_user``).
        On a non-empty archive, a transaction with only read guards returns its checked commit.
        """
        self.raise_if_failed()
        transaction_limit = self._max_buffer_bytes if max_bytes is None else max_bytes
        if transaction_limit <= 0:
            raise ValueError("max_bytes must be positive")
        return Transaction(self, max_bytes=transaction_limit, base=base, attribution=_attribution(author, message))

    def unbounded_transaction(self, *, message: str | None = None, author: str | None = None) -> Transaction:
        """Create a transaction without an Arrow buffer byte limit; ``message`` and ``author`` as for ``transaction``."""
        self.raise_if_failed()
        return Transaction(self, max_bytes=None, attribution=_attribution(author, message))

    def update(
        self,
        table: str,
        keys: Sequence[tuple],
        edit: Callable[[dict], dict],
        *,
        message: str,
        author: str | None = None,
        attempts: int = DEFAULT_UPDATE_ATTEMPTS,
    ) -> CommitToken:
        """Replace each keyed row with ``edit(row)`` in one guarded commit.

        The rows are read from a fresh snapshot and guarded. When another writer commits any of
        them first, ``edit`` runs again on the newer rows after a backoff, so it may run more than
        once. After ``attempts`` conflicting attempts the last ``RowConflict`` is raised. A missing
        key raises ``KeyError``, and an edit that changes the primary key raises ``ValueError``.
        The table must use ``OnConflict.SUPERSEDE``, since an update writes a newer row for a key.
        """
        if attempts < 1:
            raise ValueError("attempts must be positive")
        registered = self._tables.get(table)
        if registered is None:
            raise KeyError(f"table {table!r} must be registered before an update")
        if registered.on_conflict != OnConflict.SUPERSEDE:
            raise ValueError(f"update writes newer rows, so table {table!r} must use OnConflict.SUPERSEDE")

        def attempt() -> CommitToken:
            transaction = self.unbounded_transaction(message=message, author=author)
            rows = transaction.read(table, keys)
            missing = [key for key in keys if tuple(key) not in rows]
            if missing:
                raise KeyError(f"table {table!r} has no rows for keys {missing[:_SHOWN_KEYS]}")
            primary_key = registered.primary_key
            edited = []
            for key, row in rows.items():
                new = edit(dict(row))
                if tuple(new.get(name) for name in primary_key) != key:
                    raise ValueError(f"edit changed the primary key of {key!r} in table {table!r}")
                edited.append(new)
            transaction.table(table).extend(edited)
            return transaction.commit()

        return retry_with_backoff(
            attempt,
            retryable=lambda exc: isinstance(exc, RowConflict),
            max_attempts=attempts,
            backoff=_UPDATE_BACKOFF,
            operation=f"FineStore update of {table!r} at {self.root}",
        )

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
                    table.name: TableAddition(table.metadata_path, (table._write(pending),), table.incarnation)
                    for table, pending in claimed.items()
                }
                return self._commits.commit(_addition_delta(additions))
            except BaseException:
                for table, pending in claimed.items():
                    table._restore(pending)
                raise

    def _commit_transaction(
        self,
        rows: dict[str, _TransactionRows],
        delta: CommitDelta,
    ) -> CommitToken:
        with self._commit_lock, ExitStack() as retire:
            retiring = []
            if delta.reset_tables:
                retire.enter_context(self._register_lock)
                for name in delta.reset_tables:
                    table = self._tables.get(name)
                    if table is not None:
                        retire.enter_context(table._retire_on_commit())
                        retiring.append(table)

            additions = {}
            for name, buffered in rows.items():
                table = buffered.writer
                if self._tables.get(name) is not table:
                    raise CommitConflict(f"table {name!r} was dropped or replaced after the writer registered it")
                pending_rows = buffered.rows
                shard = table._write_rows(pending_rows, nbytes=sum(row.table.nbytes for row in pending_rows))
                additions[name] = TableAddition(table.metadata_path, (shard,), table.incarnation)
            if additions:
                delta = _addition_delta(additions, delta=delta)
            if additions or delta.reset_tables:
                token = self._commits.commit(delta)
                for table in retiring:
                    del self._tables[table.name]
                return token
            snapshot = self._commits.snapshot()
            if snapshot.token is None:
                return self._commits.commit(delta)
            check_preconditions(snapshot.manifest, delta)
            return snapshot.token

    def compact(self, table: str, *, generation: int | None = None) -> CompactionResult:
        """Logically compact one table against the current manifest."""
        with self._commit_lock:
            return compact_table(
                self.root,
                table,
                generation=generation,
                coordinator=self._commits,
                target_bytes=self._compaction_target_bytes,
            )

    def _acquire_maintenance(self) -> bool:
        """Hold the archive's maintenance lease, announcing a new epoch when first acquired.

        One writer per archive compacts at a time. Acquisition publishes a manifest with the
        next ``maintainer_epoch``; compactions prepared under an older epoch are then rejected
        at commit, so a holder whose lease silently lapsed cannot publish stale work.
        """
        if self._lease_epoch is not None:
            if time.monotonic() - self._lease_refreshed < HEARTBEAT_INTERVAL:
                return True
            try:
                self._lease.refresh()
            except LeaseLostError:
                self._lease_epoch = None
                _LEASE_LOST.add()
                logger.warning("FineStore maintenance lease for %s was taken over", self.root)
                return False
            self._lease_refreshed = time.monotonic()
            return True
        if not self._lease.try_acquire():
            return False
        self._lease_refreshed = time.monotonic()
        snapshot = self._commits.snapshot()
        epoch = snapshot.manifest.maintainer_epoch + 1
        try:
            self._commits.commit(CommitDelta(epoch_update=epoch), base=snapshot)
        except CommitConflict:
            # Another maintainer announced a newer epoch between the stale-lease takeover and ours.
            self._lease.release()
            return False
        self._lease_epoch = epoch
        _LEASE_ACQUIRED.add()
        return True

    @contextmanager
    def _maintenance(self) -> Iterator[bool]:
        """Try to hold the maintenance lease for the block, refreshing it on a heartbeat while held.

        Yields whether the lease is held. The lease stays held after the block so a maintainer
        keeps its role across passes; ``_release_maintenance`` gives it up.
        """
        with self._maintenance_lock:
            held = self._acquire_maintenance()
            with lease_refresh(self._lease) if held else nullcontext():
                yield held

    def _release_maintenance(self) -> None:
        with self._maintenance_lock:
            if self._lease_epoch is None:
                return
            self._lease_epoch = None
            self._lease.release()

    def maintain(self) -> CommitToken | None:
        """Flush pending rows and, while holding the maintenance lease, compact each overfull level once.

        The lease is refreshed on a heartbeat while compactions run, so a pass longer than the
        lease timeout is not taken over mid-way.
        """
        token = self.flush()
        with self._maintenance() as held:
            if not held:
                return token
            view = ReadView(self.root, self._commits.snapshot())
            for name in view.table_names():
                for generation, shard_limit in enumerate(self._compaction_levels):
                    level_shards = [shard for shard in view.list_shards(name) if shard.generation == generation]
                    if len(level_shards) <= shard_limit:
                        continue
                    self.compact(name, generation=generation)
                    view = ReadView(self.root, self._commits.last_snapshot())
                    token = view.token
        return token

    def seal(self) -> CommitToken:
        """Flush, compact active tables, and publish sealed state in the manifest.

        Sealing compacts without the maintenance lease when another writer holds it; the epoch
        check still rejects its compactions if that holder is replaced meanwhile.
        """
        self.flush()
        superseded = {}
        with self._maintenance() as held:
            if not held:
                logger.info("FineStore sealing %s while another writer holds the maintenance lease", self.root)
            snapshot = self._commits.snapshot()
            for name in ReadView(self.root, snapshot).table_names():
                result = self.compact(name)
                if result.superseded:
                    superseded[name] = result.superseded
        return self._commits.commit(
            CommitDelta(
                seal_update=SealMarker(writer=self.writer_id, superseded=superseded),
                expected_incarnations=snapshot.manifest.table_incarnations,
            )
        )

    def close(self) -> None:
        """Flush pending writes and, as the maintainer, merge bounded groups of level-zero shards.

        The merge is best effort. A closer that holds the maintenance lease, or can take a free
        or stale one, folds every writer's level-zero shards within the close budget; a closer
        that finds another writer maintaining the archive leaves its shards to that writer.
        """
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        self._thread.join()
        try:
            self.flush()
            if not self._has_level_zero_groups():
                return
            with self._maintenance() as held:
                if not held:
                    logger.info(
                        "FineStore writer %s left level-zero shards in %s to the maintainer", self.writer_id, self.root
                    )
                    return
                with self._commit_lock:
                    compact_level_zero_tables(
                        self.root,
                        list(self._tables),
                        max_input_bytes=_CLOSE_COMPACTION_MAX_INPUT_BYTES,
                        max_input_rows=_CLOSE_COMPACTION_MAX_INPUT_ROWS,
                        target_bytes=self._compaction_target_bytes,
                        coordinator=self._commits,
                    )
        finally:
            self._release_maintenance()

    def _has_level_zero_groups(self) -> bool:
        """Whether some registered table holds at least two level-zero shards, from any writer."""
        view = ReadView(self.root, self._commits.last_snapshot())
        return any(
            sum(1 for shard in view.list_shards(name) if shard.generation == 0) >= 2 for name in list(self._tables)
        )

    def request_stop(self) -> None:
        """Request maintenance shutdown without waiting or performing storage I/O.

        Use ``close()`` for normal shutdown that flushes pending writes.
        """
        self._stop.set()
        self._wake.set()

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

    def __init__(self, transaction: Transaction, writer: DataTable) -> None:
        self._transaction = transaction
        self._writer = writer

    def add(self, row: dict) -> None:
        self._transaction._add_rows({self._writer: [row]})

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)


class StagedTable:
    """A transaction-private table whose Arrow batches are written as they arrive."""

    def __init__(self, transaction: Transaction, writer: DataTable, max_bytes: int) -> None:
        self._transaction = transaction
        self._writer = writer
        self._max_bytes = max_bytes
        self._shards: list[Shard] = []

    def add(self, row: dict) -> None:
        self.extend((row,))

    def extend(self, rows: Iterable[dict]) -> None:
        """Write bounded batches. A staging failure aborts the transaction."""
        self._transaction._require_open()
        if self not in self._transaction._table_changes.values():
            raise RuntimeError("staged table is no longer part of the transaction")
        try:
            self._shards.extend(self._writer._stage_shards(rows, max_bytes=self._max_bytes))
        except BaseException:
            self._transaction._closed = True
            raise

    def _manifest_table(self) -> ManifestTable:
        return ManifestTable(metadata_path=self._writer.metadata_path, shards=tuple(self._shards))


@dataclass(frozen=True)
class _PublishedTable:
    source: str


class Transaction:
    """An isolated set of table changes and writes published through one commit token."""

    def __init__(
        self,
        store: DataStore,
        *,
        max_bytes: int | None,
        attribution: Attribution | None,
        base: ReadView | None = None,
    ) -> None:
        if base is not None and base.root != store.root:
            raise ValueError("transaction base must come from the same archive")
        self._store = store
        self._base = store.read_view() if base is None else base
        self._rows: dict[str, _TransactionRows] = {}
        self._table_changes: dict[str, StagedTable | _PublishedTable | None] = {}
        self._dependencies: set[str] = set()
        self._staging_id: str | None = None
        self._objects: dict[str, bytes] = {}
        self._guards: dict[str, set[tuple]] = {}
        self._attribution = attribution
        self._max_bytes = max_bytes
        self._buffer_bytes = 0
        self._closed = False
        self.token: CommitToken | None = None

    def read(self, table: str, keys: Sequence[tuple]) -> dict[tuple, dict]:
        """Read keyed rows from the pinned snapshot and guard them until commit.

        Returns the present rows by primary-key tuple, without FineStore's system columns.
        The commit raises ``RowConflict`` if another writer commits any of these keys,
        present or absent, after this transaction's snapshot.
        """
        self._require_open()
        if table not in self._store._tables:
            raise KeyError(f"table {table!r} must be registered before the transaction")
        if table in self._table_changes:
            raise ValueError(f"table {table!r} was created, renamed or dropped in this transaction")
        keys = [tuple(key) for key in keys]
        if any(value is None for key in keys for value in key):
            raise ValueError(f"read keys for table {table!r} must not contain null values")
        self._guards.setdefault(table, set()).update(keys)
        if not keys or self._base.table_metadata_path(table) is None:
            return {}
        width = len(self._base.primary_key(table))
        if any(len(key) != width for key in keys):
            raise ValueError(f"keys for table {table!r} must have {width} columns")
        rows = self._base.rows_by_key(table, keys)
        return {
            key: {name: value for name, value in row.items() if name not in RESERVED_COLUMNS}
            for key, row in rows.items()
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("transaction is closed")

    def _visible_table(self, name: str) -> StagedTable | _PublishedTable | None:
        if name in self._table_changes:
            return self._table_changes[name]
        if self._base.table_metadata_path(name) is not None or name in self._rows:
            return _PublishedTable(name)
        return None

    def _can_append(self, source: str) -> bool:
        return source not in self._table_changes or _PublishedTable(source) in self._table_changes.values()

    def table(self, name: str) -> TransactionTable | StagedTable:
        self._require_open()
        table = self._visible_table(name)
        if isinstance(table, StagedTable):
            return table
        if name in self._table_changes and table is None:
            raise KeyError(f"table {name!r} is absent from the transaction")
        source = table.source if isinstance(table, _PublishedTable) else name
        writer = self._store._tables.get(source)
        if writer is None:
            raise KeyError(f"table {name!r} must be registered before the transaction")
        return TransactionTable(self, writer)

    def _add_rows(self, rows: dict[DataTable, list[dict]]) -> None:
        self._require_open()
        buffered = {}
        for table, table_rows in rows.items():
            name = table.name
            if not self._can_append(name):
                raise KeyError(f"table {name!r} is absent from the transaction")
            existing = self._rows.get(name)
            if existing is not None and existing.writer is not table:
                raise CommitConflict(f"table {name!r} was dropped or replaced after the writer registered it")
            buffered[table] = table._arrow_rows(table_rows)
        added_bytes = sum(row.table.nbytes for table_rows in buffered.values() for row in table_rows)
        if self._max_bytes is not None and self._buffer_bytes + added_bytes > self._max_bytes:
            raise TransactionTooLarge(f"transaction payload exceeds {self._max_bytes} bytes; commit a smaller batch")
        self._buffer_bytes += added_bytes
        for table, table_rows in buffered.items():
            self._rows.setdefault(table.name, _TransactionRows(table, [])).rows.extend(table_rows)

    def create_table(
        self,
        name: str,
        *,
        metadata: TableMetadata,
        schema: pa.Schema | None = None,
    ) -> StagedTable:
        """Create a private table that can stream rows before publication."""
        self._require_open()
        _require_user_table(name)
        if self._visible_table(name) is not None:
            raise ValueError(f"table {name!r} already exists")
        if self._staging_id is None:
            self._staging_id = uuid.uuid4().hex
        storage_table = prefix_join(prefix_join(STAGED_TABLE_DIR, self._staging_id), uuid.uuid4().hex)
        writer = self._store._table_writer(
            name,
            metadata=metadata,
            schema=schema,
            incarnation=INITIAL_TABLE_INCARNATION,
            start_seq=0,
            storage_table=storage_table,
        )
        batch_limit = self._store._max_buffer_bytes if self._max_bytes is None else self._max_bytes
        table = StagedTable(self, writer, batch_limit)
        self._table_changes[name] = table
        return table

    def rename_table(
        self, source: str, destination: str, *, on_conflict: TableNameConflict = TableNameConflict.ERROR
    ) -> None:
        """Move a table in the transaction's namespace."""
        self._require_open()
        _require_user_table(source)
        _require_user_table(destination)
        table = self._visible_table(source)
        if table is None:
            raise KeyError(f"table {source!r} is absent from the transaction")
        if source == destination:
            return
        if self._visible_table(destination) is not None and on_conflict != TableNameConflict.REPLACE:
            raise ValueError(f"table {destination!r} already exists")
        self._table_changes[source] = None
        self._table_changes[destination] = table

    def drop_table(self, name: str) -> None:
        """Remove a table from the transaction if it is present."""
        self._require_open()
        _require_user_table(name)
        if self._visible_table(name) is not None:
            self._table_changes[name] = None
        elif name not in self._table_changes:
            self._dependencies.add(name)

    def require_unchanged(self, *tables: str) -> None:
        """Reject publication if any of these tables changed since the base view."""
        self._require_open()
        self._dependencies.update(tables)

    def _commit_delta(self, rows: dict[str, _TransactionRows]) -> CommitDelta:
        resets = set()
        creations = {}
        moves = {}
        for name, table in self._table_changes.items():
            if table == _PublishedTable(name):
                continue
            if table is None and self._base.table_metadata_path(name) is None and name not in rows:
                continue
            resets.add(name)
            if isinstance(table, StagedTable):
                creations[name] = table._manifest_table()
            elif isinstance(table, _PublishedTable):
                moves[name] = table.source
        return CommitDelta(
            reset_tables=frozenset(resets),
            table_creations=creations,
            table_moves=moves,
            expected_contents={name: self._base.table_content_version(name) for name in resets | self._dependencies},
            attribution=self._attribution,
            precondition=self._row_guards(),
        )

    def _row_guards(self) -> Precondition | None:
        if not self._guards:
            return None
        return _rows_unchanged_since(self._base, {table: frozenset(keys) for table, keys in self._guards.items()})

    def write_object(self, name: str, data: bytes, metadata: Mapping[str, object] | None = None) -> str:
        """Buffer one named byte object and return its ``finestore://`` URI."""
        self._require_open()
        blob_table = self._store._blob_table()
        descriptor, parts = self._store._blob_rows(name, data, metadata)
        rows = {blob_table: [descriptor]}
        if parts:
            part_table = self._store._blob_part_table()
            rows[part_table] = parts
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
        rows = {name: pending for name, pending in self._rows.items() if self._can_append(name)}
        self.token = self._store._commit_transaction(rows, self._commit_delta(rows))
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
        incarnation: int,
        max_buffer_bytes: int,
        scheduler: FlushScheduler,
        primary_key: tuple[str, ...],
        schema: pa.Schema | None,
        schema_version: int,
        on_conflict: OnConflict = OnConflict.ERROR,
        start_seq: int = 0,
        storage_table: str | None = None,
    ) -> None:
        self.name = name
        self.primary_key = primary_key
        self.schema_version = schema_version
        self.on_conflict = on_conflict
        self.metadata_path = metadata_path
        self.incarnation = incarnation
        self._writer_id = writer_id
        self._layout = layout
        self._storage_table = name if storage_table is None else storage_table
        self._max_buffer_bytes = max_buffer_bytes
        self._scheduler = scheduler
        self._schema = _with_stamp_columns(schema) if schema is not None else None
        self._pending: list[ArrowRow] = []
        self._pending_bytes = 0
        self._next_seq = start_seq
        self._lock = threading.RLock()
        self._retired = False
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
            if self._retired:
                raise CommitConflict(f"table {self.name!r} was dropped or replaced after the writer registered it")
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
        arrow_rows = self._arrow_rows(rows)
        with self._lock:
            buffered = self._stamp_arrow_rows(arrow_rows)
            self._pending.extend(buffered)
            self._pending_bytes += sum(row.table.nbytes for row in buffered)
            over_cap = self._pending_bytes >= self._max_buffer_bytes
        if over_cap:
            self._scheduler.request_flush()

    @contextmanager
    def _retire_on_commit(self) -> Iterator[None]:
        # An append either reaches the pending buffer before publication or sees the retired handle.
        with self._lock:
            if self._pending:
                raise CommitConflict(f"flush buffered appends to table {self.name!r} before changing it")
            yield
            self._retired = True

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

    def _stage_shards(self, rows: Iterable[dict], *, max_bytes: int) -> Iterator[Shard]:
        batch = []
        batch_bytes = 0
        for row in rows:
            (arrow_row,) = self._arrow_rows((row,))
            size = arrow_row.table.nbytes
            if batch and batch_bytes + size > max_bytes:
                yield self._write_rows(batch, nbytes=batch_bytes)
                batch = []
                batch_bytes = 0
            batch.append(arrow_row)
            batch_bytes += size
        if batch:
            yield self._write_rows(batch, nbytes=batch_bytes)

    def _write_rows(self, rows: Sequence[ArrowRow], *, nbytes: int) -> Shard:
        return self._write(PendingRows(tuple(self._stamp_arrow_rows(rows)), nbytes))

    def _write(self, pending: PendingRows) -> Shard:
        rows = list(pending.rows)
        primary_key_sorted = all(row.sort_key is not None for row in rows)
        if primary_key_sorted:
            rows.sort(key=lambda row: row.sort_key or ())
        assert all(row.sequence is not None for row in rows)
        min_seq = min(row.sequence for row in rows if row.sequence is not None)
        max_seq = max(row.sequence for row in rows if row.sequence is not None)
        path = self._layout.shard_path(self._storage_table, self._writer_id, 0, min_seq, uuid.uuid4().hex[:8])
        schema = self._schema or pa.unify_schemas([row.table.schema for row in rows], promote_options="permissive")
        schema = _drop_empty_struct_fields(schema)
        arrow_rows = (_align_arrow_row(row.table, schema) for row in rows)
        with ShardWriter(path, schema) as writer:
            for group in table_row_groups(self.name, arrow_rows):
                writer.write_table(group)
        written = writer.result
        bounds = KeyRange.of(row.key for row in rows) if primary_key_sorted else None
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
            min_key=None if bounds is None else bounds.lowest,
            max_key=None if bounds is None else bounds.highest,
        )

    def flush(self) -> CommitToken | None:
        return self._scheduler.flush_table(self)
