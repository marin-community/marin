# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Namespace registry over the DuckDB-backed log store.

The registry backs both the log RPC and the ``StatsService``
Register/WriteRows/Query/DropTable surface. Concretely:

- :class:`finelog.store.log_namespace.LogNamespace` — owns all per-namespace
  state: pending rows, Arrow chunks, sealed segments, the flush thread,
  compaction state, the GCS offload state, and the registered Arrow schema.
- :class:`finelog.store.registry_db.RegistryDB` — sidecar DuckDB DB at
  ``{data_dir}/_finelog_registry.duckdb`` that persists each namespace's
  Schema. Rehydrated on startup.
- :class:`DuckDBLogStore` (this module) — thin :class:`NamespaceRegistry`
  that holds the global locks (insertion mutex + query-visibility rwlock),
  the shared connection pool, and routes RPCs by namespace.

The ``log`` namespace is eagerly upserted into the registry on first boot
so older deployments whose registry DB pre-dates the namespace registry
get a clean fixup path on startup.
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

from finelog.store.layout_migration import LOG_NAMESPACE_DIR
from finelog.store.log_namespace import LOG_REGISTERED_SCHEMA, LogNamespace, _is_tmp_path
from finelog.store.registry_db import RegistryDB
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
)
from finelog.store.sql_escape import quote_ident, quote_literal
from finelog.types import LogReadResult

logger = logging.getLogger(__name__)

# Name of the namespace registered eagerly on first boot. Arbitrary
# additional namespaces are admitted via RegisterTable.
LOG_NAMESPACE_NAME = "log"

# Heuristic thresholds for local Parquet retention.

# Target size for a single Parquet segment on disk. New data is concatenated
# onto the latest segment until it reaches this size, then a new file starts.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

# Background step cadences.
DEFAULT_FLUSH_INTERVAL_SEC = 60.0
DEFAULT_COMPACTION_INTERVAL_SEC = 600.0  # merge small tmp parquets into one archive

# Trigger compaction when this many tmp segments accumulate, even before the
# time-based interval fires. Keeps per-read fanout bounded under high ingest.
DEFAULT_MAX_TMP_SEGMENTS_BEFORE_COMPACT = 10

# Default caps for local Parquet retention.
# The read path has no remote fallback yet (see module docstring TODO): once a
# parquet is GC'd locally, its rows are unreachable via FetchLogs even though
# they're durable on GCS. Sized to keep ~2 weeks of the production `marin`
# cluster's ingest (~6-7 GB/day) fully local.
DEFAULT_MAX_LOCAL_SEGMENTS = 1000
DEFAULT_MAX_LOCAL_BYTES = 100 * 1024**3  # 100 GB

# Per-connection memory ceiling for DuckDB. Tight limits (e.g. 256MB) caused
# spill-to-disk loops under concurrent tail reads over large row groups,
# wedging the controller. 4GB is generous against realistic working sets
# (5 segments x 500MB + zstd decompression scratch).
_DEFAULT_DUCKDB_MEMORY_LIMIT = "4GB"
_DEFAULT_DUCKDB_THREADS = "4"

# Tuning for embedded use (e.g. iris controller's bundled log-server). The
# working set is tiny — short retention, single-digit namespaces, no large
# row groups — so a small memory ceiling and minimal thread pool keep VMS
# down without hurting throughput. Lower numbers also matter because the
# embedding process forks subprocesses; large parent VMS trips Linux's
# overcommit heuristic and fails with ENOMEM at fork().
EMBEDDED_DUCKDB_MEMORY_LIMIT = "128MB"
EMBEDDED_DUCKDB_THREADS = "2"


# Namespace name validation. Lowercase ASCII alphanumerics + ._-, starting
# with a letter, max 64 chars. The regex is restrictive enough to be safe as
# both a directory name and a double-quoted DuckDB identifier without further
# escaping; path-containment alone is not sufficient.
_NAMESPACE_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


_cursor_counter = 0
_cursor_counter_lock = Lock()


def _next_cursor_id() -> int:
    global _cursor_counter
    with _cursor_counter_lock:
        _cursor_counter += 1
        return _cursor_counter


class _ConnectionPool:
    """Two DuckDB databases: one for reads, one for compaction.

    Reads share a single ``duckdb.connect()`` with ``enable_object_cache`` so
    parquet footer / row-group stats are cached across queries. Callers get
    cursors via ``conn.cursor()`` which share that connection's thread pool
    and buffer pool.

    Compaction runs on a second, isolated connection so its sort cost does
    not starve reads (a shared connection wedged the controller under
    concurrent tail reads).
    """

    def __init__(
        self,
        memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
        threads: str = _DEFAULT_DUCKDB_THREADS,
    ):
        config = {"memory_limit": memory_limit, "threads": threads}
        self._conn = duckdb.connect(config=config)
        self._conn.execute("SET enable_object_cache=true")
        self._compaction_conn = duckdb.connect(config=config)

    @contextmanager
    def checkout(self, buffer_tables: list[pa.Table]) -> Iterator[tuple[duckdb.DuckDBPyConnection, list[str]]]:
        """Create a cursor and register each RAM table under a unique name.

        Yields ``(cursor, list_of_table_names)`` so callers can UNION ALL the
        names into their SQL without a ``pa.concat_tables`` copy.
        """
        cid = _next_cursor_id()
        cursor = self._conn.cursor()
        names: list[str] = []
        try:
            for i, table in enumerate(buffer_tables):
                name = f"_ram_{cid}_{i}"
                cursor.register(name, table)
                names.append(name)
            yield cursor, names
        finally:
            for name in names:
                cursor.unregister(name)
            cursor.close()

    @property
    def compaction_conn(self) -> duckdb.DuckDBPyConnection:
        return self._compaction_conn

    def close(self) -> None:
        self._conn.close()
        self._compaction_conn.close()


def _validate_namespace_name(name: str, data_dir: Path | None) -> Path | None:
    """Validate ``name`` and return its on-disk subdirectory (or ``None``).

    In-memory mode (``data_dir is None``) only enforces the regex; there
    is no path-containment check because there is no filesystem to escape.

    Raises:
        InvalidNamespaceError: name fails the regex or its resolved path
            escapes ``data_dir``.
    """
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

    Stats surface (in addition to the log API):

    - :meth:`register_table` — register or evolve a namespace's Schema.
    - :meth:`write_rows` — validate an Arrow IPC RecordBatch against a
      registered Schema and append it to the matching namespace.
    - :meth:`query` — execute SQL against DuckDB views over every
      registered namespace's sealed Parquet segments.
    - :meth:`drop_table` — remove a namespace and delete its local
      segments (refuses the privileged ``log`` namespace).

    Concurrency: the registry owns the global insertion mutex (held by every
    write and every compaction's slow phase) and the query-visibility rwlock
    (queries hold read for their full duration; commits and drops briefly
    hold write). Each :class:`LogNamespace` shares both via constructor
    references.

    Layout: callers pass ``log_dir`` as the finelog data directory. Per
    namespace: ``{log_dir}/{name}/``. Schema sidecar:
    ``{log_dir}/_finelog_registry.duckdb``.

    ``log_dir=None`` selects in-memory mode: no tempdir, no parquet
    files, no sidecar registry file. Segments live as Arrow tables on
    the namespace and the registry DB runs in ``:memory:``. GCS offload
    is disabled. State vanishes on ``close()``.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        max_local_segments: int = DEFAULT_MAX_LOCAL_SEGMENTS,
        max_local_bytes: int = DEFAULT_MAX_LOCAL_BYTES,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        compaction_interval_sec: float = DEFAULT_COMPACTION_INTERVAL_SEC,
        max_tmp_segments_before_compact: int = DEFAULT_MAX_TMP_SEGMENTS_BEFORE_COMPACT,
        segment_target_bytes: int = SEGMENT_TARGET_BYTES,
        duckdb_memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
        duckdb_threads: str = _DEFAULT_DUCKDB_THREADS,
    ):
        self._data_dir: Path | None = log_dir
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        self._insertion_lock = Lock()
        self._query_visibility_lock = RWLock()
        self._pool = _ConnectionPool(memory_limit=duckdb_memory_limit, threads=duckdb_threads)
        self._registry_db = RegistryDB(self._data_dir)

        # Global storage caps. Eviction picks the oldest sealed segment
        # across all namespaces (oldest-first by ``min_seq`` with namespace
        # registration time as the tiebreak). Per-namespace quotas are
        # deferred until evidence of starvation forces them.
        self._max_local_segments = max_local_segments
        self._max_local_bytes = max_local_bytes
        self._namespace_registered_at: dict[str, int] = {}

        self._namespaces: dict[str, LogNamespace] = {}

        # Per-namespace constructor kwargs that don't depend on the schema.
        self._namespace_kwargs = dict(
            remote_log_dir=remote_log_dir,
            flush_interval_sec=flush_interval_sec,
            compaction_interval_sec=compaction_interval_sec,
            max_tmp_segments_before_compact=max_tmp_segments_before_compact,
            segment_target_bytes=segment_target_bytes,
            insertion_lock=self._insertion_lock,
            query_visibility_lock=self._query_visibility_lock,
            compaction_conn=self._pool.compaction_conn,
            read_pool=self._pool,
            evict_hook=self._evict_globally,
        )

        # Rehydrate every previously-registered namespace from the sidecar
        # DB. The "log" namespace is upserted afterwards as a defensive
        # fixup for older deployments whose registry DB doesn't yet
        # contain its row.
        self._rehydrate_from_registry()
        self._ensure_log_namespace_registered()

    # ------------------------------------------------------------------
    # Registry rehydration
    # ------------------------------------------------------------------

    def _rehydrate_from_registry(self) -> None:
        """Instantiate a LogNamespace per row in the sidecar registry DB.

        In-memory mode: list_all() returns an empty dict (the registry DB
        is fresh ``:memory:``) so this loop is a no-op.
        """
        for name, schema in self._registry_db.list_all().items():
            namespace_dir = self._namespace_dir(name)
            if namespace_dir is not None:
                namespace_dir.mkdir(parents=True, exist_ok=True)
            self._namespaces[name] = LogNamespace(
                name=name,
                schema=schema,
                data_dir=namespace_dir,
                **self._namespace_kwargs,
            )
            # Use a monotonic counter as the in-process tiebreak so eviction
            # is deterministic even if two namespaces share registered_at_ms.
            self._namespace_registered_at[name] = len(self._namespace_registered_at)

    def _ensure_log_namespace_registered(self) -> None:
        """First-boot fixup: materialize the ``log`` registry row if missing.

        Older deployments wrote the ``log/`` directory via the layout
        migration but had no registry DB. First-boot only — after the
        first boot the registry has the row and the rehydration path
        skips this.
        """
        if LOG_NAMESPACE_NAME in self._namespaces:
            return
        if self._data_dir is not None:
            log_dir = self._data_dir / LOG_NAMESPACE_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = None
        with self._insertion_lock:
            self._registry_db.upsert(
                LOG_NAMESPACE_NAME,
                LOG_REGISTERED_SCHEMA,
                resolve_key_column(LOG_REGISTERED_SCHEMA),
            )
            self._namespaces[LOG_NAMESPACE_NAME] = LogNamespace(
                name=LOG_NAMESPACE_NAME,
                schema=LOG_REGISTERED_SCHEMA,
                data_dir=log_dir,
                **self._namespace_kwargs,
            )
            self._namespace_registered_at.setdefault(LOG_NAMESPACE_NAME, len(self._namespace_registered_at))

    def _namespace_dir(self, name: str) -> Path | None:
        if self._data_dir is None:
            # Still validate the name regex so in-memory stores match the
            # on-disk store's naming contract.
            _validate_namespace_name(name, None)
            return None
        if name == LOG_NAMESPACE_NAME:
            return self._data_dir / LOG_NAMESPACE_DIR
        return _validate_namespace_name(name, self._data_dir)

    # ------------------------------------------------------------------
    # Public Stats API
    # ------------------------------------------------------------------

    def register_table(self, name: str, schema: Schema) -> Schema:
        """Register or evolve ``name`` to ``schema``.

        Returns the *effective* schema after additive merge with any prior
        registration. See ``design.md`` "Schema evolution" for the full
        rule set.
        """
        # Validate name (regex + path containment) before anything else.
        namespace_dir = self._namespace_dir(name)
        # Validate the schema declares a valid ordering key. Re-raises
        # SchemaValidationError on missing/wrong-typed key column.
        resolved_key = resolve_key_column(schema)

        with self._insertion_lock:
            existing_ns = self._namespaces.get(name)
            if existing_ns is None:
                effective = schema
                if namespace_dir is not None:
                    namespace_dir.mkdir(parents=True, exist_ok=True)
                self._registry_db.upsert(name, effective, resolved_key)
                self._namespaces[name] = LogNamespace(
                    name=name,
                    schema=effective,
                    data_dir=namespace_dir,
                    **self._namespace_kwargs,
                )
                self._namespace_registered_at[name] = len(self._namespace_registered_at)
                return effective

            # Evolve-by-default: merge_schemas raises SchemaConflictError on
            # non-additive change. Identical / subset returns the existing
            # schema unchanged; a true extension returns the union.
            effective = merge_schemas(existing_ns.schema, schema)
            if effective != existing_ns.schema:
                self._registry_db.upsert(name, effective, resolved_key)
                existing_ns.update_schema(effective)
            return effective

    def list_namespaces(self) -> list[tuple[str, Schema]]:
        """Snapshot every registered namespace with its current schema.

        Returns ``(name, schema)`` tuples in registration order. The
        privileged ``log`` namespace is included so callers can introspect
        it like any other surface. Snapshot is taken under the insertion
        mutex so the result is consistent with concurrent register/drop.
        """
        with self._insertion_lock:
            items = sorted(
                self._namespaces.items(),
                key=lambda kv: self._namespace_registered_at.get(kv[0], 0),
            )
            return [(name, ns.schema) for name, ns in items]

    def get_table_schema(self, name: str) -> Schema:
        """Return the registered schema for ``name``.

        Raises ``NamespaceNotFoundError`` for unknown namespaces.
        """
        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            return ns.schema

    def write_rows(self, name: str, arrow_ipc_bytes: bytes) -> int:
        """Validate ``arrow_ipc_bytes`` and append the rows to ``name``.

        ``arrow_ipc_bytes`` is the body of one ``WriteRowsRequest.arrow_ipc``
        — an Arrow IPC stream containing exactly one RecordBatch. Returns
        the number of rows appended.

        Raises:
            SchemaValidationError: oversized batch, missing non-nullable
                column, unknown column, type mismatch, or nested type.
            NamespaceNotFoundError: ``name`` is not registered.
        """
        if len(arrow_ipc_bytes) > MAX_WRITE_ROWS_BYTES:
            raise SchemaValidationError(
                f"WriteRows body {len(arrow_ipc_bytes)} bytes exceeds {MAX_WRITE_ROWS_BYTES} limit"
            )

        batch = _decode_single_record_batch(arrow_ipc_bytes)
        if batch.num_rows > MAX_WRITE_ROWS_ROWS:
            raise SchemaValidationError(f"WriteRows batch {batch.num_rows} rows exceeds {MAX_WRITE_ROWS_ROWS} limit")

        # Look up the namespace and align the batch under the insertion
        # mutex. Holding the mutex around the lookup is the load-bearing
        # invariant against the DropTable race: drop_table removes the
        # namespace from the dict under the same mutex, so any write that
        # observes a namespace here is guaranteed to see it survive the
        # subsequent append (which retakes the mutex).
        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            aligned = validate_and_align_batch(batch, ns.schema)
        # The actual append takes the insertion lock again internally; doing
        # the pyarrow work outside the lock keeps the critical section short.
        ns.append_record_batch(aligned)
        return aligned.num_rows

    def query(self, sql: str) -> pa.Table:
        """Execute ``sql`` against a DuckDB view of every registered namespace.

        Each namespace gets a view named after the namespace. The view
        source is built per-segment: parquet-backed sealed segments come
        in via ``read_parquet([…])``; in-memory sealed segments are
        registered on the connection and referenced by name, then
        ``UNION ALL``ed together. Namespaces with zero sealed segments
        get a typed empty view so user queries against them return zero
        rows instead of erroring.

        The query-visibility read lock is held across the whole call
        because DuckDB opens Parquet files lazily during execution:
        dropping the lock before fetch would let compaction unlink files
        mid-scan. (In-memory segments are immune to that race but we
        keep one lock discipline for both backings.)

        ``DropTable``-style errors (unknown namespace referenced in the
        SQL) surface as DuckDB's own ``CatalogException`` — the view
        doesn't exist, so the user's ``FROM "ns.unknown"`` resolves to
        nothing.
        """
        con = duckdb.connect()
        registered_names: list[str] = []
        self._query_visibility_lock.read_acquire()
        try:
            # Snapshot the namespace dict under the insertion lock so a
            # concurrent ``drop_table`` (which mutates ``self._namespaces``
            # under the same insertion lock) can't make the iteration below
            # raise ``RuntimeError: dictionary changed size during iteration``.
            # The insertion lock is held only for the snapshot; the read
            # side of the rwlock keeps the segment files alive across fetch.
            with self._insertion_lock:
                ns_snapshot = list(self._namespaces.items())
            for ns_name, ns in ns_snapshot:
                ns_quoted = quote_ident(ns_name)
                segments, ram_tables = ns.query_snapshot()
                if not segments and not ram_tables:
                    cols_sql = ", ".join(
                        f"NULL::{duckdb_type_for(c)} AS {quote_ident(c.name)}" for c in ns.schema.columns
                    )
                    con.execute(f"CREATE VIEW {ns_quoted} AS SELECT {cols_sql} WHERE FALSE")
                    continue

                parts: list[str] = []
                parquet_paths = [s.path for s in segments if s.table is None]
                if parquet_paths:
                    paths_literal = "[" + ", ".join(quote_literal(p) for p in parquet_paths) + "]"
                    parts.append(f"SELECT * FROM read_parquet({paths_literal}, union_by_name=true)")
                arrow_tables = [s.table for s in segments if s.table is not None] + ram_tables
                for table in arrow_tables:
                    reg_name = f"_seg_{len(registered_names)}"
                    con.register(reg_name, table)
                    registered_names.append(reg_name)
                    parts.append(f"SELECT * FROM {reg_name}")
                con.execute(f"CREATE VIEW {ns_quoted} AS {' UNION ALL BY NAME '.join(parts)}")
            return con.execute(sql).fetch_arrow_table()
        finally:
            for name in registered_names:
                try:
                    con.unregister(name)
                except Exception:
                    pass
            self._query_visibility_lock.read_release()
            con.close()

    def drop_table(self, name: str) -> None:
        """Remove ``name`` from the registry and delete its local segments.

        Refuses to drop the privileged ``log`` namespace. Unknown
        namespaces raise ``NamespaceNotFoundError``. GCS-archived data is
        intentionally left in place — the contract is "drop frees local
        capacity; the bucket is the caller's to clean up if they want it
        gone."

        Lock ordering. We can't hold the insertion mutex end-to-end because
        the namespace's bg flush thread itself takes the insertion mutex
        every iteration; joining it under the lock would deadlock. Instead:

        1. Under the insertion mutex, look up the namespace, remove it from
           the registry row + in-process map. Subsequent ``write_rows``
           calls under the same mutex see no namespace and raise.
        2. Release the insertion mutex. Signal the bg thread to stop and
           join it. The just-removed namespace can no longer be reached
           by new writes.
        3. Acquire the query-visibility write lock and delete the local
           segment directory. Holding the write lock guarantees no
           in-flight query can observe a half-deleted segment list.
        """
        if name == LOG_NAMESPACE_NAME:
            raise InvalidNamespaceError(f"namespace {name!r} is privileged and cannot be dropped via DropTable")

        with self._insertion_lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            self._registry_db.delete(name)
            del self._namespaces[name]
            self._namespace_registered_at.pop(name, None)

        # Outside the insertion lock: stop and join the bg thread. The
        # namespace is already unreachable via the registry, so no new
        # writes can find it. Stop & join is a no-op-on-stop pattern.
        ns.stop_and_join()

        self._query_visibility_lock.write_acquire()
        try:
            ns.remove_local_storage()
        finally:
            self._query_visibility_lock.write_release()

    def _evict_globally(self) -> None:
        """Evict the globally-oldest sealed segments until under the caps.

        Called by every namespace after a flush or compaction. Walks every
        namespace's segment list, picks the oldest-by-(min_seq,
        registration_order) sealed segment, and unlinks it. Tmp segments
        are not eligible (they may be in flight). The query-visibility
        write lock is held briefly across each unlink so a concurrent
        query never observes a torn segment list.
        """
        # Enumerate eviction candidates under the insertion lock: a
        # consistent snapshot of every sealed segment across all namespaces.
        # Tmp (in-flight) segments are excluded from *both* the count and
        # the candidate list — eviction can only act on sealed segments,
        # so counting tmp toward ``total_segments``/``total_bytes`` would
        # falsely classify a namespace with N tmp segments as "over cap"
        # and evict its sealed segments prematurely.
        candidates: list[tuple[int, int, str, int, str]] = []
        with self._insertion_lock:
            total_segments = 0
            total_bytes = 0
            for ns_name, ns in self._namespaces.items():
                reg_order = self._namespace_registered_at.get(ns_name, 0)
                for seg in ns.all_segments_unlocked():
                    if _is_tmp_path(seg.path):
                        continue
                    total_segments += 1
                    total_bytes += seg.size_bytes
                    candidates.append((seg.min_seq, reg_order, ns_name, seg.size_bytes, seg.path))

        if total_segments <= self._max_local_segments and total_bytes <= self._max_local_bytes:
            return

        # Oldest first: smallest min_seq wins; ties broken by registration
        # order then namespace name (for total ordering determinism).
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))

        evictions: list[tuple[str, str, int]] = []  # (ns_name, path, size)
        remaining_segments = total_segments
        remaining_bytes = total_bytes
        for _min_seq, _reg, ns_name, size, path in candidates:
            if remaining_segments <= self._max_local_segments and remaining_bytes <= self._max_local_bytes:
                break
            evictions.append((ns_name, path, size))
            remaining_segments -= 1
            remaining_bytes -= size

        if not evictions:
            return

        self._query_visibility_lock.write_acquire()
        try:
            freed_bytes = 0
            for ns_name, path, _size in evictions:
                ns = self._namespaces.get(ns_name)
                if ns is None:
                    continue
                freed_bytes += ns.evict_segment(path)
        finally:
            self._query_visibility_lock.write_release()

        logger.info(
            "Globally evicted %d segment(s), freed=%d bytes, remaining=%d segments / %d bytes",
            len(evictions),
            freed_bytes,
            remaining_segments,
            remaining_bytes,
        )

    # ------------------------------------------------------------------
    # Public LogStore API — routes to the "log" namespace.
    # ------------------------------------------------------------------

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
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        return self._namespaces[LOG_NAMESPACE_NAME].get_logs(
            key,
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
        self._registry_db.close()

    # ------------------------------------------------------------------
    # Test hooks. Forward to the single registered "log" namespace.
    # ------------------------------------------------------------------

    @property
    def _log_namespace(self) -> LogNamespace:
        return self._namespaces[LOG_NAMESPACE_NAME]

    def _compact_step(self) -> None:
        self._log_namespace._compact_step()

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
        self._log_namespace._compaction_step()

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
    """Decode an Arrow IPC stream containing exactly one RecordBatch."""
    reader = paipc.open_stream(pa.BufferReader(arrow_ipc_bytes))
    batches = list(reader)
    if len(batches) != 1:
        raise SchemaValidationError(f"WriteRows: expected exactly one RecordBatch in IPC stream, got {len(batches)}")
    return batches[0]
