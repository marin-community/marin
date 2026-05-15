# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Service-implementation layer above :class:`Catalog`.

:class:`DuckDBLogStore` is what the RPC handlers sit on top of. Catalog
owns the namespace state — schemas, segment rows, the live
``LogNamespaceProtocol`` dict, the in-flight drop set — behind a single
mutex. This class owns the I/O machinery and the cross-component
orchestration around it:

* The read-side DuckDB :class:`ConnectionPool` that backs ``query()``.
* The query-visibility :class:`RWLock`, held in read across queries
  (DuckDB opens parquet files lazily during scan) and in write briefly
  during ``drop_table``'s unlink phase.
* The wire-format pipeline for ``write_rows`` (Arrow IPC decode, size
  checks, schema validation).
* The SQL-execution path in ``query`` (snapshot live namespaces, build
  DuckDB views over their parquet + RAM Arrow tables, run the user's
  SQL, tear views down).
* The five-step ``drop_table`` sequence (Catalog ``begin_drop`` → bg
  thread join → catalog row delete → rwlock-write + ``remove_local_storage``
  → ``finish_drop``).
* The factory that picks ``DiskLogNamespace`` vs ``MemoryLogNamespace``
  and wires their dependencies.

The ``log`` namespace is upserted on first boot for back-compat with
deployments whose registry DB pre-dates the namespace registry.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Timer

import duckdb
import pyarrow as pa
import pyarrow.ipc as paipc

from finelog.rpc import logging_pb2
from finelog.store.catalog import Catalog
from finelog.store.compactor import CompactionConfig
from finelog.store.layout_migration import LOG_NAMESPACE_DIR
from finelog.store.log_namespace import (
    LOG_REGISTERED_SCHEMA,
    DiskLogNamespace,
    MemoryLogNamespace,
)
from finelog.store.rwlock import RWLock
from finelog.store.schema import (
    MAX_WRITE_ROWS_BYTES,
    MAX_WRITE_ROWS_ROWS,
    InvalidNamespaceError,
    Schema,
    SchemaValidationError,
    duckdb_type_for,
    merge_schemas,
    resolve_key_column,
    validate_and_align_batch,
    with_implicit_seq,
)
from finelog.store.sql_escape import quote_ident, quote_literal
from finelog.store.types import LogNamespaceProtocol, NamespaceStats
from finelog.types import LogReadResult

logger = logging.getLogger(__name__)

LOG_NAMESPACE_NAME = "log"

SEGMENT_TARGET_BYTES = 100 * 1024 * 1024
# Bounds the worst-case "in RAM but not on disk" window. Service handlers
# block writers until their rows are persisted by polling
# ``DiskLogNamespace.max_persisted_seq()``; the bg loop's wake cadence
# combined with this interval determine the floor on observable WriteRows
# latency under low load.
DEFAULT_FLUSH_INTERVAL_SEC = 5.0

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
# Read-pool thread cap. Sized so a single query can't monopolize a small VM
# and starve the namespace bg threads (flush + compaction). On a 4-vCPU host
# we have 4 namespace bg threads + RPC handlers + per-namespace compaction
# (each capped at 2 threads); leaving the read pool at host cpu_count
# created multi-hundred-ms tail latency on parquet encode under flush bursts
# because the read pool's threads competed for the same cores. 2 threads is
# enough to keep large list_recent / range scans parallel without
# co-monopolizing the box.
_DEFAULT_DUCKDB_THREADS = "2"

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

# Wall-clock cap on a single read-pool cursor scope. DuckDB has no
# statement_timeout pragma; we enforce it by arming a watchdog that calls
# ``connection.interrupt()`` if the scope is still active when the timer
# fires. Sized to be longer than any legitimate dashboard / list_recent
# query but short enough that a runaway scan cannot pin the read pool's
# single connection and starve other readers.
_DEFAULT_DUCKDB_QUERY_TIMEOUT_SEC = 10.0


class _QueryWatchdog:
    """One-shot timer that calls ``cursor.interrupt()`` if not disarmed in time.

    The interrupt target must be the cursor actually running the query —
    ``parent_conn.interrupt()`` does not propagate down to a cursor obtained
    via ``parent_conn.cursor()``.

    ``threading.Timer.cancel`` does not prevent an already-running callback,
    so we additionally guard the interrupt with a flag+lock: ``disarm`` marks
    the watchdog dead, and the callback drops out if it sees that flag.
    This keeps a late-firing timer from interrupting the cleanup queries
    that run after the user's ``with cursor()`` block exits normally.
    """

    def __init__(self, cursor: duckdb.DuckDBPyConnection, timeout_sec: float) -> None:
        self._cursor = cursor
        self._lock = Lock()
        self._done = False
        self._timer = Timer(timeout_sec, self._fire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if self._done:
                return
            self._cursor.interrupt()

    def disarm(self) -> None:
        self._timer.cancel()
        with self._lock:
            self._done = True
        self._timer.join()


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
        query_timeout_sec: float | None = _DEFAULT_DUCKDB_QUERY_TIMEOUT_SEC,
    ):
        self._config: dict[str, str] = {"memory_limit": memory_limit, "threads": threads}
        if temp_directory is not None:
            temp_directory.mkdir(parents=True, exist_ok=True)
            self._config["temp_directory"] = str(temp_directory)
        self._recycle_sec = recycle_sec
        self._query_timeout_sec = query_timeout_sec
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
            watchdog: _QueryWatchdog | None = None
            if self._query_timeout_sec is not None and self._query_timeout_sec > 0:
                watchdog = _QueryWatchdog(cur, self._query_timeout_sec)
                watchdog.start()
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
                # Disarm before running cleanup so a late-firing watchdog
                # cannot interrupt the DROP VIEW / unregister calls below.
                if watchdog is not None:
                    watchdog.disarm()
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
    """RPC-handler-facing store wrapping :class:`Catalog` and namespaces.

    Catalog is the data-and-coordination layer (persistent rows + live
    registry under one mutex). This class wires it to the outside world:
    the read-side connection pool, the query-visibility rwlock, the IPC
    decoder, the namespace factory, and the multi-step drop sequence.
    See the module docstring for the full breakdown.

    The ``catalog`` attribute is part of this class's public surface — RPC
    handlers and tests reach through it for live-registry introspection.
    Per-namespace insertion mutexes live on each namespace and serialize
    append/flush/compaction for that namespace.

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
        duckdb_query_timeout_sec: float | None = _DEFAULT_DUCKDB_QUERY_TIMEOUT_SEC,
    ):
        self._data_dir: Path | None = log_dir
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        self._query_visibility_lock = RWLock()
        # Spill into the data dir, not CWD — the prod container runs with a
        # read-only working directory.
        pool_tmp = (log_dir / ".duckdb_tmp_read") if log_dir is not None else None
        self._pool = ConnectionPool(
            memory_limit=duckdb_memory_limit,
            threads=duckdb_threads,
            temp_directory=pool_tmp,
            query_timeout_sec=duckdb_query_timeout_sec,
        )
        self.catalog = Catalog(self._data_dir)

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
                query_visibility_lock=self._query_visibility_lock,
                read_pool=self._pool,
            )
        assert namespace_dir is not None, "disk mode requires a namespace dir"
        namespace_dir.mkdir(parents=True, exist_ok=True)
        return DiskLogNamespace(
            name=name,
            schema=schema,
            data_dir=namespace_dir,
            query_visibility_lock=self._query_visibility_lock,
            read_pool=self._pool,
            catalog=self.catalog,
            **self._disk_namespace_kwargs,
        )

    def _rehydrate_from_registry(self) -> None:
        for name, schema in self.catalog.list_all().items():
            namespace_dir = self._namespace_dir(name)
            ns = self._make_namespace(name, schema, namespace_dir)
            self.catalog.insert_live(name, ns)

    def _ensure_log_namespace_registered(self) -> None:
        """First-boot fixup: materialize the ``log`` registry row if missing."""
        if LOG_NAMESPACE_NAME in self.catalog:
            return
        log_dir = self._data_dir / LOG_NAMESPACE_DIR if self._data_dir is not None else None
        resolve_key_column(LOG_REGISTERED_SCHEMA)
        stored_schema = with_implicit_seq(LOG_REGISTERED_SCHEMA)
        self.catalog.register_or_evolve(
            LOG_NAMESPACE_NAME,
            stored_schema,
            lambda schema: self._make_namespace(LOG_NAMESPACE_NAME, schema, log_dir),
            on_existing=lambda existing: existing.schema,
        )

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

        def on_existing(existing_ns: LogNamespaceProtocol) -> Schema:
            # merge_schemas raises SchemaConflictError on non-additive change.
            effective = merge_schemas(existing_ns.schema, stored_schema)
            if effective != existing_ns.schema:
                self.catalog.upsert(name, effective)
                existing_ns.update_schema(effective)
            return effective

        return self.catalog.register_or_evolve(
            name,
            stored_schema,
            lambda effective_schema: self._make_namespace(name, effective_schema, namespace_dir),
            on_existing=on_existing,
        )

    def list_namespaces_with_stats(self) -> list[tuple[str, Schema, NamespaceStats]]:
        """Return ``(name, schema, stats)`` for every live namespace.

        Backs ``StatsService.ListNamespaces`` — the dashboard relies on
        it to render the summary table without issuing per-namespace
        ``count(*)`` queries against parquet. Stats are read from the
        namespace's in-memory state (kept in lockstep with the on-disk
        segment catalog).
        """
        # ns.stats() takes the per-namespace insertion lock; we want the
        # registry snapshot to release the catalog lock before any stats()
        # call so a slow namespace can't stall this call for every other.
        namespaces = self.catalog.snapshot_live()
        return [(name, ns.schema, ns.stats()) for name, ns in namespaces]

    def get_table_schema(self, name: str) -> Schema:
        return self.catalog.require_live(name).schema

    def memory_summary(self) -> dict[str, int]:
        """Aggregate ram_bytes / chunk_count across namespaces, for diagnostics.

        Used by the periodic pool-diagnostics logger in the standalone server.
        ``MemoryLogNamespace`` reports zeros (no in-RAM segmented buffer).
        """
        namespaces = self.catalog.live_values()
        total_ram_bytes = 0
        total_chunks = 0
        for ns in namespaces:
            total_ram_bytes += ns.ram_bytes()
            total_chunks += ns.chunk_count()
        return {
            "namespaces": len(namespaces),
            "ram_bytes": total_ram_bytes,
            "chunks": total_chunks,
        }

    def write_rows(self, name: str, arrow_ipc_bytes: bytes) -> tuple[int, int]:
        """Validate ``arrow_ipc_bytes`` and append the rows to ``name``.

        ``arrow_ipc_bytes`` carries exactly one RecordBatch. Returns
        ``(rows_written, last_seq)`` where ``last_seq`` is the durability
        target the caller should wait on (``-1`` if the batch was empty).
        """
        if len(arrow_ipc_bytes) > MAX_WRITE_ROWS_BYTES:
            raise SchemaValidationError(
                f"WriteRows body {len(arrow_ipc_bytes)} bytes exceeds {MAX_WRITE_ROWS_BYTES} limit"
            )

        batch = _decode_single_record_batch(arrow_ipc_bytes)
        if batch.num_rows > MAX_WRITE_ROWS_ROWS:
            raise SchemaValidationError(f"WriteRows batch {batch.num_rows} rows exceeds {MAX_WRITE_ROWS_ROWS} limit")

        # Resolve the namespace + schema, then validate outside any lock:
        # validate_and_align_batch is the bulk of the CPU work per WriteRows,
        # and pinning it under the catalog lock would serialize every writer
        # across every namespace. Schema is monotonic-additive — if it
        # evolves between snapshot and append, the stamp step in the
        # namespace NULL-fills the new columns at projection time.
        ns = self.catalog.require_live(name)
        schema = ns.schema
        aligned = validate_and_align_batch(batch, schema)
        last_seq = ns.append_aligned_batch(aligned)
        return aligned.num_rows, last_seq

    def max_persisted_seq(self, name: str) -> int:
        """Highest seq durably persisted in namespace ``name``.

        Service handlers poll this on the event loop to gate WriteRows /
        PushLogs replies until the bg flush thread has written an L0
        parquet segment covering the caller's rows.
        """
        return self.catalog.require_live(name).max_persisted_seq()

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
                # Snapshot the live registry under the catalog lock so a
                # concurrent drop_table can't trigger "dictionary changed
                # size during iteration".
                ns_snapshot = self.catalog.snapshot_live()

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

        The order is:

          1. ``Catalog.begin_drop``: atomically pop from the live registry
             and reserve the name in ``_dropping`` so new ops fail fast
             (a concurrent ``register_table`` would otherwise recreate
             the namespace and have its state wiped by the cleanup steps
             below).
          2. Stop and join the bg thread — *before* dropping catalog rows,
             because ``_sync_step`` would otherwise see an empty catalog
             plus a populated bucket and ``fs.rm`` every remote file as
             an orphan.
          3. Drop the catalog rows now that no concurrent reader can act
             on them.
          4. Take the rwlock write side and delete the segment directory.
          5. ``Catalog.finish_drop`` clears the reservation; the name is
             now free to be re-registered.

        GCS-archived data is intentionally preserved; the bucket is the
        caller's to clean up.
        """
        if name == LOG_NAMESPACE_NAME:
            raise InvalidNamespaceError(f"namespace {name!r} is privileged and cannot be dropped via DropTable")

        ns = self.catalog.begin_drop(name)
        try:
            ns.stop_and_join()
            self.catalog.delete(name)

            self._query_visibility_lock.write_acquire()
            try:
                ns.remove_local_storage()
            finally:
                self._query_visibility_lock.write_release()
        finally:
            self.catalog.finish_drop(name)

    def append(self, key: str, entries: list) -> int:
        if not entries:
            return -1
        return self.append_batch([(key, entries)])

    def append_batch(self, items: list[tuple[str, list]]) -> int:
        """Append log-namespace items; returns the last seq allocated (or -1)."""
        return self.catalog[LOG_NAMESPACE_NAME].append_log_batch(items)

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
        return self.catalog[LOG_NAMESPACE_NAME].get_logs(
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
        from finelog.store import LogCursor  # circular import: duckdb_store -> store.__init__ -> duckdb_store

        return LogCursor(self, key)

    def close(self) -> None:
        for ns in self.catalog.live_values():
            ns.close()
        self._pool.close()
        self.catalog.close()

    # Test hooks below; forward to the registered "log" namespace.

    @property
    def _log_namespace(self) -> DiskLogNamespace:
        ns = self.catalog[LOG_NAMESPACE_NAME]
        assert isinstance(ns, DiskLogNamespace), "test hook called on memory-mode store"
        return ns

    def _force_flush(self) -> None:
        self._log_namespace._flush_step()

    def _force_compaction(self) -> None:
        self._log_namespace._force_compact_l0()

    def _wait_persisted(self, name: str, target_seq: int, timeout: float = 10.0) -> None:
        """Test helper: spin until ``max_persisted_seq(name) >= target_seq``.

        Mirrors the async polling loop in the service handlers without
        requiring an event loop in sync tests.
        """
        ns = self.catalog.require_live(name)
        deadline = time.monotonic() + timeout
        while ns.max_persisted_seq() < target_seq:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for ns={name!r} persisted_seq>={target_seq} "
                    f"(current={ns.max_persisted_seq()})"
                )
            time.sleep(0.02)


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
