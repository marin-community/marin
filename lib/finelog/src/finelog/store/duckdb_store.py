# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Namespace registry over the DuckDB-backed log store.

:class:`DuckDBLogStore` holds the global locks (insertion mutex +
query-visibility rwlock) and the shared connection pool, and routes RPCs to
per-namespace storage (:class:`LogNamespaceProtocol`). Schemas persist via
:class:`Catalog` and are rehydrated on startup. The ``log`` namespace is
upserted on first boot for back-compat with deployments whose registry DB
pre-dates the namespace registry.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import duckdb
import pyarrow as pa
import pyarrow.ipc as paipc

from finelog.rpc import logging_pb2
from finelog.store.catalog import Catalog, NamespaceStats
from finelog.store.compactor import CompactionConfig
from finelog.store.layout_migration import LOG_NAMESPACE_DIR
from finelog.store.log_namespace import (
    LOG_REGISTERED_SCHEMA,
    DiskLogNamespace,
    LogNamespaceProtocol,
    MemoryLogNamespace,
)
from finelog.store.rwlock import RWLock
from finelog.store.schema import (
    MAX_WRITE_ROWS_BYTES,
    MAX_WRITE_ROWS_ROWS,
    InvalidNamespaceError,
    NamespaceNotFoundError,
    Schema,
    SchemaValidationError,
    duckdb_type_for,
    merge_schemas,
    resolve_key_column,
    validate_and_align_batch,
    with_implicit_seq,
)
from finelog.store.sql_escape import quote_ident, quote_literal
from finelog.types import LogReadResult

logger = logging.getLogger(__name__)

LOG_NAMESPACE_NAME = "log"

SEGMENT_TARGET_BYTES = 100 * 1024 * 1024
DEFAULT_FLUSH_INTERVAL_SEC = 60.0

# 4GB was the previous default but in practice finelog's read pattern
# rarely needs more than tens of MB; the high cap mostly let mimalloc/DuckDB
# retain pages indefinitely. 512MB on the read pool is plenty against 5
# segments x ~50MB + zstd decompression scratch. Compaction tier-merges can
# spill larger sort buffers, so it gets its own (still bounded) limit.
_DEFAULT_DUCKDB_MEMORY_LIMIT = "2GB"
# Sized for an L1 merge of ~256 MiB segments: DuckDB's working set during
# COPY (... ORDER BY ...) is several x the output size, and the prod
# 1 GB cap was OOMing.
_DEFAULT_DUCKDB_COMPACTION_MEMORY_LIMIT = "4GB"
_DEFAULT_DUCKDB_THREADS = "4"

# Embedded mode (iris controller's bundled log-server) keeps VMS small so the
# parent doesn't trip Linux's overcommit heuristic when forking subprocesses.
EMBEDDED_DUCKDB_MEMORY_LIMIT = "128MB"
EMBEDDED_DUCKDB_THREADS = "2"

# Namespace names: lowercase ASCII alphanumerics + ._-, starting with a
# letter, max 64 chars. Restrictive enough to be safe as both a directory
# name and a double-quoted DuckDB identifier without further escaping.
_NAMESPACE_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


_cursor_counter = 0
_cursor_counter_lock = Lock()


def _next_cursor_id() -> int:
    global _cursor_counter
    with _cursor_counter_lock:
        _cursor_counter += 1
        return _cursor_counter


_DEFAULT_POOL_RECYCLE_SEC = 600.0


class ConnectionPool:
    """Single DuckDB read connection shared across all read paths.

    ``enable_object_cache`` keeps parquet footer / row-group stats hot
    across queries. The connection is recycled periodically (default
    10 min) so DuckDB-internal accounting (spill counters, arena bloat)
    cannot accumulate without bound.

    All access goes through :meth:`cursor`, which serializes callers,
    recycles if stale, and manages table registration / cleanup.
    """

    def __init__(
        self,
        memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
        threads: str = _DEFAULT_DUCKDB_THREADS,
        temp_directory: Path | None = None,
        recycle_sec: float = _DEFAULT_POOL_RECYCLE_SEC,
    ):
        self._config: dict[str, str] = {"memory_limit": memory_limit, "threads": threads}
        if temp_directory is not None:
            temp_directory.mkdir(parents=True, exist_ok=True)
            self._config["temp_directory"] = str(temp_directory)
        self._recycle_sec = recycle_sec
        self._conn = self._new_conn()
        self._conn_born = time.monotonic()
        self._lock = Lock()

    def _new_conn(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(config=dict(self._config))
        conn.execute("SET enable_object_cache=true")
        return conn

    def _maybe_recycle(self) -> None:
        if time.monotonic() - self._conn_born < self._recycle_sec:
            return
        old = self._conn
        self._conn = self._new_conn()
        self._conn_born = time.monotonic()
        old.close()
        logger.debug("ConnectionPool recycled read connection")

    @contextmanager
    def cursor(
        self,
        buffers: dict[str, list[pa.Table]] | None = None,
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a cursor under the pool lock.

        Recycles the underlying connection if stale. ``buffers`` maps
        view names to lists of Arrow tables; each entry becomes a
        ``CREATE VIEW <name> AS SELECT * FROM ... UNION ALL ...``
        so the caller can reference the tables by the view name it
        chose. Everything is torn down on exit.
        """
        with self._lock:
            self._maybe_recycle()
            cid = _next_cursor_id()
            cur = self._conn.cursor()
            registered: list[str] = []
            views: list[str] = []
            try:
                for view_name, tables in (buffers or {}).items():
                    parts: list[str] = []
                    for table in tables:
                        reg = f"_reg_{cid}_{len(registered)}"
                        cur.register(reg, table)
                        registered.append(reg)
                        parts.append(f"SELECT * FROM {reg}")
                    cur.execute(f"CREATE VIEW {view_name} AS {' UNION ALL '.join(parts)}")
                    views.append(view_name)
                yield cur
            finally:
                for v in views:
                    cur.execute(f"DROP VIEW IF EXISTS {v}")
                for r in registered:
                    cur.unregister(r)
                cur.close()

    def close(self) -> None:
        self._conn.close()


def _validate_namespace_name(name: str, data_dir: Path | None) -> Path | None:
    """Validate ``name`` and return its on-disk subdirectory (or ``None``)."""
    if not _NAMESPACE_NAME_RE.match(name):
        raise InvalidNamespaceError(f"namespace {name!r} does not match {_NAMESPACE_NAME_RE.pattern}")
    if data_dir is None:
        return None
    target = (data_dir / name).resolve()
    base = data_dir.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise InvalidNamespaceError(
            f"namespace {name!r} resolves to {target} which is not strictly inside {base}"
        ) from exc
    if target == base:
        raise InvalidNamespaceError(f"namespace {name!r} resolves to the data dir itself")
    return target


class DuckDBLogStore:
    """Namespace registry routing log + stats RPCs to per-namespace storage.

    Concurrency: the registry owns the global insertion mutex (held by every
    write and every compaction's slow phase) and the query-visibility rwlock
    (queries hold read for their full duration; commits and drops briefly
    hold write).

    Layout: per-namespace under ``{log_dir}/{name}/``; schema sidecar at
    ``{log_dir}/_finelog_registry.duckdb``. ``log_dir=None`` selects
    in-memory mode (no segmentation, no remote copy, state vanishes on close).
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        compaction_config: CompactionConfig = CompactionConfig(),
        segment_target_bytes: int = SEGMENT_TARGET_BYTES,
        duckdb_memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
        duckdb_compaction_memory_limit: str = _DEFAULT_DUCKDB_COMPACTION_MEMORY_LIMIT,
        duckdb_threads: str = _DEFAULT_DUCKDB_THREADS,
    ):
        self._data_dir: Path | None = log_dir
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        self._insertion_lock = Lock()
        self._query_visibility_lock = RWLock()
        # Spill into the data dir, not CWD — the prod container runs with a
        # read-only working directory.
        pool_tmp = (log_dir / ".duckdb_tmp_read") if log_dir is not None else None
        self._pool = ConnectionPool(
            memory_limit=duckdb_memory_limit,
            threads=duckdb_threads,
            temp_directory=pool_tmp,
        )
        self._catalog = Catalog(self._data_dir)

        self._namespace_registered_at: dict[str, int] = {}
        self._namespaces: dict[str, LogNamespaceProtocol] = {}
        # Names currently being dropped. ``drop_table`` reserves the name
        # here under ``_insertion_lock`` so a concurrent ``register_table``
        # cannot recreate the namespace in the window between popping
        # ``_namespaces`` and the late ``catalog.delete`` /
        # ``remove_local_storage``. Without this, the late steps would
        # delete the freshly registered namespace's catalog row and wipe
        # its on-disk directory.
        self._dropping: set[str] = set()

        # Disk-only kwargs; ignored by memory namespaces. ``duckdb_memory_limit``
        # in this dict feeds the per-namespace compaction connection in
        # ``DiskLogNamespace``; we route the dedicated compaction cap here so
        # tier-merges don't share the read pool's tighter ceiling. Each disk
        # namespace owns its own remote sync — empty ``remote_log_dir``
        # disables it.
        self._disk_namespace_kwargs = dict(
            remote_log_dir=remote_log_dir,
            flush_interval_sec=flush_interval_sec,
            compaction_config=compaction_config,
            segment_target_bytes=segment_target_bytes,
            duckdb_memory_limit=duckdb_compaction_memory_limit,
        )

        self._rehydrate_from_registry()
        self._ensure_log_namespace_registered()

    def _make_namespace(self, name: str, schema: Schema, namespace_dir: Path | None) -> LogNamespaceProtocol:
        if self._data_dir is None:
            return MemoryLogNamespace(
                name=name,
                schema=schema,
                insertion_lock=self._insertion_lock,
                query_visibility_lock=self._query_visibility_lock,
                read_pool=self._pool,
            )
        assert namespace_dir is not None, "disk mode requires a namespace dir"
        namespace_dir.mkdir(parents=True, exist_ok=True)
        return DiskLogNamespace(
            name=name,
            schema=schema,
            data_dir=namespace_dir,
            insertion_lock=self._insertion_lock,
            query_visibility_lock=self._query_visibility_lock,
            read_pool=self._pool,
            catalog=self._catalog,
            **self._disk_namespace_kwargs,
        )

    def _rehydrate_from_registry(self) -> None:
        for name, schema in self._catalog.list_all().items():
            namespace_dir = self._namespace_dir(name)
            self._namespaces[name] = self._make_namespace(name, schema, namespace_dir)
            # Monotonic counter as the eviction tiebreak.
            self._namespace_registered_at[name] = len(self._namespace_registered_at)

    def _ensure_log_namespace_registered(self) -> None:
        """First-boot fixup: materialize the ``log`` registry row if missing."""
        if LOG_NAMESPACE_NAME in self._namespaces:
            return
        log_dir = self._data_dir / LOG_NAMESPACE_DIR if self._data_dir is not None else None
        resolve_key_column(LOG_REGISTERED_SCHEMA)
        stored_schema = with_implicit_seq(LOG_REGISTERED_SCHEMA)
        with self._insertion_lock:
            self._catalog.upsert(LOG_NAMESPACE_NAME, stored_schema)
            self._namespaces[LOG_NAMESPACE_NAME] = self._make_namespace(LOG_NAMESPACE_NAME, stored_schema, log_dir)
            self._namespace_registered_at.setdefault(LOG_NAMESPACE_NAME, len(self._namespace_registered_at))

    def _namespace_dir(self, name: str) -> Path | None:
        if self._data_dir is None:
            # Still enforce the name regex so in-memory stores match the
            # on-disk naming contract.
            _validate_namespace_name(name, None)
            return None
        if name == LOG_NAMESPACE_NAME:
            return self._data_dir / LOG_NAMESPACE_DIR
        return _validate_namespace_name(name, self._data_dir)

    def register_table(self, name: str, schema: Schema) -> Schema:
        """Register or evolve ``name`` to ``schema``; return the effective schema.

        Implicit ``seq`` is stamped at this boundary so the on-disk layout
        is uniform across namespaces.
        """
        namespace_dir = self._namespace_dir(name)
        resolve_key_column(schema)
        stored_schema = with_implicit_seq(schema)

        with self._insertion_lock:
            if name in self._dropping:
                # A concurrent ``drop_table`` has already removed the
                # namespace from ``_namespaces`` but has not yet finished
                # its catalog/storage cleanup. Recreating the namespace
                # now would let that cleanup wipe the new state. Surface
                # this as a transient validation error; the caller can
                # retry once the drop completes.
                raise InvalidNamespaceError(
                    f"namespace {name!r} is currently being dropped; retry once drop_table completes"
                )
            existing_ns = self._namespaces.get(name)
            if existing_ns is None:
                self._catalog.upsert(name, stored_schema)
                self._namespaces[name] = self._make_namespace(name, stored_schema, namespace_dir)
                self._namespace_registered_at[name] = len(self._namespace_registered_at)
                return stored_schema

            # merge_schemas raises SchemaConflictError on non-additive change.
            effective = merge_schemas(existing_ns.schema, stored_schema)
            if effective != existing_ns.schema:
                self._catalog.upsert(name, effective)
                existing_ns.update_schema(effective)
            return effective

    def list_namespaces(self) -> list[tuple[str, Schema]]:
        with self._insertion_lock:
            items = sorted(
                self._namespaces.items(),
                key=lambda kv: self._namespace_registered_at.get(kv[0], 0),
            )
            return [(name, ns.schema) for name, ns in items]

    def list_namespaces_with_stats(self) -> list[tuple[str, Schema, NamespaceStats]]:
        """Like :meth:`list_namespaces`, but also returns per-namespace stats.

        Each entry's stats are read from the namespace's in-memory state,
        which is held in lockstep with the on-disk segment catalog. This is
        the read path that backs ``StatsService.ListNamespaces`` — the
        dashboard relies on it to render the namespace summary table without
        issuing per-namespace ``count(*)`` queries against parquet.
        """
        with self._insertion_lock:
            items = sorted(
                self._namespaces.items(),
                key=lambda kv: self._namespace_registered_at.get(kv[0], 0),
            )
            namespaces = [(name, ns) for name, ns in items]
        return [(name, ns.schema, ns.stats()) for name, ns in namespaces]

    def get_table_schema(self, name: str) -> Schema:
        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            return ns.schema

    def memory_summary(self) -> dict[str, int]:
        """Aggregate ram_bytes / chunk_count across namespaces, for diagnostics.

        Used by the periodic pool-diagnostics logger in the standalone server.
        ``MemoryLogNamespace`` reports zeros (no in-RAM segmented buffer).
        """
        total_ram_bytes = 0
        total_chunks = 0
        with self._insertion_lock:
            for ns in self._namespaces.values():
                total_ram_bytes += ns.ram_bytes()
                total_chunks += ns.chunk_count()
            return {
                "namespaces": len(self._namespaces),
                "ram_bytes": total_ram_bytes,
                "chunks": total_chunks,
            }

    def write_rows(self, name: str, arrow_ipc_bytes: bytes) -> int:
        """Validate ``arrow_ipc_bytes`` and append the rows to ``name``.

        ``arrow_ipc_bytes`` carries exactly one RecordBatch. Returns the
        number of rows appended.
        """
        if len(arrow_ipc_bytes) > MAX_WRITE_ROWS_BYTES:
            raise SchemaValidationError(
                f"WriteRows body {len(arrow_ipc_bytes)} bytes exceeds {MAX_WRITE_ROWS_BYTES} limit"
            )

        batch = _decode_single_record_batch(arrow_ipc_bytes)
        if batch.num_rows > MAX_WRITE_ROWS_ROWS:
            raise SchemaValidationError(f"WriteRows batch {batch.num_rows} rows exceeds {MAX_WRITE_ROWS_ROWS} limit")

        # The lookup must happen under the insertion mutex: drop_table
        # removes the namespace from the dict under the same mutex, so
        # any write that observes a namespace here is guaranteed to see
        # it survive the subsequent append (which retakes the mutex).
        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            aligned = validate_and_align_batch(batch, ns.schema)
        ns.append_record_batch(aligned)
        return aligned.num_rows

    def query(self, sql: str) -> pa.Table:
        """Execute ``sql`` against a DuckDB view of every registered namespace.

        The query-visibility read lock is held across the whole call: DuckDB
        opens Parquet files lazily during execution, so dropping the lock
        before fetch would let compaction unlink files mid-scan.

        Unknown namespaces in the FROM clause surface as DuckDB
        ``CatalogException`` (the view doesn't exist).
        """
        view_names: list[str] = []
        self._query_visibility_lock.read_acquire()
        try:
            with self._pool.cursor() as cursor:
                # Snapshot under the insertion lock so a concurrent drop_table
                # can't trigger "dictionary changed size during iteration".
                with self._insertion_lock:
                    ns_snapshot = list(self._namespaces.items())

                extra_registered: list[str] = []
                try:
                    for ns_name, ns in ns_snapshot:
                        ns_quoted = quote_ident(ns_name)
                        view_names.append(ns_quoted)
                        segments, ram_tables = ns.query_snapshot()
                        if not segments and not ram_tables:
                            cols_sql = ", ".join(
                                f"NULL::{duckdb_type_for(c)} AS {quote_ident(c.name)}" for c in ns.schema.columns
                            )
                            cursor.execute(f"CREATE OR REPLACE VIEW {ns_quoted} AS SELECT {cols_sql} WHERE FALSE")
                            continue

                        parts: list[str] = []
                        if segments:
                            paths_literal = "[" + ", ".join(quote_literal(s.path) for s in segments) + "]"
                            parts.append(f"SELECT * FROM read_parquet({paths_literal}, union_by_name=true)")
                        for table in ram_tables:
                            reg_name = f"_q{_next_cursor_id()}_seg_{len(extra_registered)}"
                            cursor.register(reg_name, table)
                            extra_registered.append(reg_name)
                            parts.append(f"SELECT * FROM {reg_name}")
                        cursor.execute(f"CREATE OR REPLACE VIEW {ns_quoted} AS {' UNION ALL BY NAME '.join(parts)}")
                    return cursor.execute(sql).fetch_arrow_table()
                finally:
                    for name in extra_registered:
                        cursor.unregister(name)
                    for vname in view_names:
                        cursor.execute(f"DROP VIEW IF EXISTS {vname}")
        finally:
            self._query_visibility_lock.read_release()

    def drop_table(self, name: str) -> None:
        """Remove ``name`` from the registry and delete its local segments.

        We can't hold the insertion mutex end-to-end because the bg flush
        thread itself takes it every iteration; joining under the mutex
        would deadlock. The order is:

          1. Under the mutex: remove from the registry *and* mark ``name``
             as ``_dropping`` (new ops fail fast — including a concurrent
             ``register_table``, which would otherwise recreate the
             namespace and have its state wiped by the cleanup steps
             below).
          2. Stop and join the bg thread — *before* dropping catalog rows,
             because ``_sync_step`` would otherwise see an empty catalog
             plus a populated bucket and ``fs.rm`` every remote file as
             an orphan.
          3. Drop the catalog rows now that no concurrent reader can act
             on them.
          4. Take the rwlock write side and delete the segment directory.
          5. Clear the ``_dropping`` reservation under the mutex; the name
             is now free to be re-registered.

        GCS-archived data is intentionally preserved; the bucket is the
        caller's to clean up.
        """
        if name == LOG_NAMESPACE_NAME:
            raise InvalidNamespaceError(f"namespace {name!r} is privileged and cannot be dropped via DropTable")

        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            del self._namespaces[name]
            self._namespace_registered_at.pop(name, None)
            self._dropping.add(name)

        try:
            ns.stop_and_join()

            with self._insertion_lock:
                self._catalog.delete(name)

            self._query_visibility_lock.write_acquire()
            try:
                ns.remove_local_storage()
            finally:
                self._query_visibility_lock.write_release()
        finally:
            with self._insertion_lock:
                self._dropping.discard(name)

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        self.append_batch([(key, entries)])

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        self._namespaces[LOG_NAMESPACE_NAME].append_log_batch(items)

    def get_logs(
        self,
        key: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        return self._namespaces[LOG_NAMESPACE_NAME].get_logs(
            key,
            match_scope=match_scope,
            since_ms=since_ms,
            cursor=cursor,
            substring_filter=substring_filter,
            max_lines=max_lines,
            tail=tail,
            min_level=min_level,
        )

    def has_logs(self, key: str) -> bool:
        result = self.get_logs(key, max_lines=1)
        return len(result.entries) > 0

    def cursor(self, key: str):
        from finelog.store import LogCursor

        return LogCursor(self, key)

    def close(self) -> None:
        for ns in self._namespaces.values():
            ns.close()
        self._pool.close()
        self._catalog.close()

    # Test hooks below; forward to the registered "log" namespace.

    @property
    def _log_namespace(self) -> DiskLogNamespace:
        ns = self._namespaces[LOG_NAMESPACE_NAME]
        assert isinstance(ns, DiskLogNamespace), "test hook called on memory-mode store"
        return ns

    def _force_flush(self) -> None:
        self._log_namespace._flush_step()

    def _wait_for_flush(self, timeout: float = 10.0) -> None:
        ns = self._log_namespace
        start_gen = ns._flush_generation
        deadline = time.monotonic() + timeout
        with ns._flush_generation_cond:
            while ns._flush_generation == start_gen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for flush")
                ns._flush_generation_cond.wait(timeout=remaining)

    def _force_compaction(self) -> None:
        self._log_namespace._force_compact_l0()

    def _wait_for_compaction(self, timeout: float = 10.0) -> None:
        ns = self._log_namespace
        start_gen = ns._compaction_generation
        deadline = time.monotonic() + timeout
        with ns._compaction_generation_cond:
            while ns._compaction_generation == start_gen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for compaction")
                ns._compaction_generation_cond.wait(timeout=remaining)


def _decode_single_record_batch(arrow_ipc_bytes: bytes) -> pa.RecordBatch:
    """Decode a single-batch IPC stream.

    Uses ``read_next_batch`` rather than ``list(reader)`` so the EOS check
    doesn't build an intermediate Python list (this path was 1.15M allocs/30s
    on prod). Note: the returned ``RecordBatch`` is a zero-copy view into
    ``arrow_ipc_bytes`` and keeps it alive — see ARROW-7305 in the design
    notes for why we may want a hard copy here later.
    """
    reader = paipc.open_stream(pa.BufferReader(arrow_ipc_bytes))
    try:
        batch = reader.read_next_batch()
    except StopIteration:
        raise SchemaValidationError("WriteRows: expected exactly one RecordBatch in IPC stream, got 0") from None
    try:
        reader.read_next_batch()
    except StopIteration:
        return batch
    raise SchemaValidationError("WriteRows: expected exactly one RecordBatch in IPC stream, got >1")
