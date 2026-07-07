# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy-backed controller database wrapper.

Hosts the SA ``Engine`` factories, the ``Tx`` wrapper, and the two
transaction context managers (``write_transaction`` / ``read_snapshot``).

The engine is split into a **write engine** and two **read engines**:

* The write engine uses pool size 1 so writes are funneled through a
  single connection. Serialization between writers is enforced by an
  external ``threading.RLock`` passed into ``write_transaction``.
* The shared read engine uses ``QueuePool(pool_size=4, max_overflow=4)``
  with ``PRAGMA query_only = ON`` **pinned at connect time** and backs every
  RPC-handler read (``read_snapshot``). Pinning avoids toggling the pragma on
  every call. The pool caps in-flight readers: SQLite's WAL admits concurrent
  readers but each contends on the WAL-index header lock when establishing a
  snapshot, so queueing surplus readers at the SA pool (FIFO) beats spinning
  inside SQLite. A total of 8 is the controller benchmark's measured knee — it
  lets the high-volume fast reads (e.g. ``GetJobState``) bypass the handful of
  slow ones (``ListJobs``, ``GetSchedulerState``) that would otherwise occupy
  every connection and head-of-line block the dashboard; widening past 8 only
  adds WAL-index/GIL contention with no throughput gain.
* The control read engine is a **dedicated** ``QueuePool(pool_size=2,
  max_overflow=2)`` used only by the single control-loop thread's per-tick
  snapshot (``control_read_snapshot``). Isolating it guarantees the
  schedule/reconcile/autoscale tick never waits behind a slow dashboard read
  for a connection — the pool-checkout wait visible as ``QueuePool.get`` in
  controller stacks. Its connections sit idle between ticks (~1 read/s), so
  they add negligible WAL-index contention.

All engines use ``isolation_level="AUTOCOMMIT"`` so callers issue
``BEGIN`` / ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` explicitly.

Post-commit hooks registered via ``Tx.register`` fire *under the write
lock*, after ``COMMIT``. A concurrent thread cannot observe the
SQL-committed-but-cache-not-yet-updated window because the lock is held
until every hook has run.
"""

import importlib.util
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import fsspec.core
from rigging.timing import Timestamp
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.controller.caches import CacheRegistry
from iris.cluster.controller.schema import auth_metadata, metadata, schema_migrations_table

logger = logging.getLogger(__name__)


def _install_pragmas(dbapi_conn, auth_path_str: str | None) -> None:
    """Run startup PRAGMAs and ATTACH the auth DB if provided."""
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode = WAL")
        cur.execute("PRAGMA synchronous = NORMAL")
        cur.execute("PRAGMA busy_timeout = 5000")
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute("PRAGMA cache_size = -65536")
        if auth_path_str is not None:
            cur.execute("ATTACH DATABASE ? AS auth", (auth_path_str,))
    finally:
        cur.close()


def _make_engine(
    db_path: Path,
    *,
    read_only: bool,
    pool_size: int,
    max_overflow: int,
    auth_db_path: Path | None = None,
) -> Engine:
    """Build a SA engine for ``db_path`` (see the module docstring for the
    write/read pool split and pragma rationale)."""
    auth_path_str = str(auth_db_path) if auth_db_path is not None else None
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5.0},
        pool_size=pool_size,
        max_overflow=max_overflow,
        isolation_level="AUTOCOMMIT",
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pyrefly: ignore  # event hook
        _install_pragmas(dbapi_conn, auth_path_str)
        if read_only:
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA query_only = ON")
            finally:
                cur.close()

    return engine


# Read-pool sizing. See the module docstring for the rationale; the totals are
# the controller benchmark's measured knee.
SHARED_READ_POOL_SIZE = 4
SHARED_READ_MAX_OVERFLOW = 4
CONTROL_READ_POOL_SIZE = 2
CONTROL_READ_MAX_OVERFLOW = 2
# Auth reads are low-volume (token/key lookups) against a separate WAL; keep
# the pool small and pinned rather than inheriting the shared default.
AUTH_READ_POOL_SIZE = 2
AUTH_READ_MAX_OVERFLOW = 2


def _make_write_engine(db_path: Path, auth_db_path: Path | None) -> Engine:
    return _make_engine(db_path, read_only=False, pool_size=1, max_overflow=0, auth_db_path=auth_db_path)


def _make_read_engine(
    db_path: Path,
    auth_db_path: Path | None,
    *,
    pool_size: int = SHARED_READ_POOL_SIZE,
    max_overflow: int = SHARED_READ_MAX_OVERFLOW,
) -> Engine:
    return _make_engine(
        db_path, read_only=True, pool_size=pool_size, max_overflow=max_overflow, auth_db_path=auth_db_path
    )


class Tx:
    """Canonical write/read transaction context for the Iris controller.

    Wraps a SQLAlchemy ``Connection``. Accepts only SA Core constructs
    (``insert``/``update``/``delete``/``select``/``text``). Raw strings are
    rejected — use ``sqlalchemy.text()`` if you need to pass literal SQL.

    Post-commit hooks registered via :meth:`register` fire after ``COMMIT``,
    while the write lock is still held (see ``write_transaction``).
    """

    def __init__(self, conn: Connection, caches: CacheRegistry, seq: int):
        self.conn = conn
        self._hooks: list[Callable[[], None]] = []
        # The DB commit sequence sampled just BEFORE this cursor's snapshot was
        # established (``BEGIN``). It is a conservative lower bound on what the
        # snapshot sees — any commit that ticked ``commit_seq`` after this sample
        # either lands in the snapshot or not, but is never counted as seen when it
        # isn't. Lazy-fill guards compare it against a per-key invalidation seq to
        # reject a fill computed from a pre-invalidation snapshot (the stale set).
        self.seq = seq
        # Per-transaction extension slot: a write helper may attach one typed cache
        # object here to memoize a lookup across calls within one transaction (e.g.
        # the federation changelog gate resolving a job's requester once per root).
        # Never persists past the transaction, so a cached value can never go stale.
        self.memo: dict[str, object] = {}
        # The owning DB's per-controller cache registry (see ``caches.py``), shared
        # by every Tx the DB mints. Persists across transactions, so a write sink
        # holding only this cursor reaches a memo without it being threaded in.
        self.caches = caches

    def execute(self, stmt, params=None) -> CursorResult:
        """Execute a SA Core construct. Returns a ``CursorResult``.

        ``stmt`` must be a SQLAlchemy expression (``Select``, ``Insert``,
        ``Update``, ``Delete``, ``TextClause``, CTE, etc.). Raw strings are
        rejected — use ``sqlalchemy.text()`` if you really need a string.
        """
        if isinstance(stmt, str):
            raise TypeError(
                "Tx.execute does not accept raw SQL strings. "
                "Pass a SQLAlchemy construct (select/insert/update/delete/text)."
            )
        return self.conn.execute(stmt, params or {})

    def register(self, hook: Callable[[], None]) -> None:
        """Register a post-commit hook.

        Hook fires once after commit, under the write lock. Write-tx only;
        ``read_snapshot`` never fires hooks.
        """
        self._hooks.append(hook)

    def _fire_hooks(self) -> None:
        for hook in self._hooks:
            hook()


@contextmanager
def write_transaction(
    write_engine: Engine,
    write_lock: threading.RLock,
    caches: CacheRegistry,
) -> Iterator[Tx]:
    """Open a write transaction backed by ``write_engine``.

    Acquires ``write_lock``, checks out a connection, emits
    ``BEGIN IMMEDIATE``, yields a ``Tx``, and commits on clean exit.
    Post-commit hooks registered via ``Tx.register`` fire **while the
    lock is still held** so in-memory caches stay consistent with the DB.

    ``caches`` is the owning DB's per-controller cache registry, mirrored onto
    the yielded ``Tx`` as ``tx.caches`` so write sinks can reach a memo through
    the cursor (see :mod:`iris.cluster.controller.caches`).
    """
    write_lock.acquire()
    conn: Connection | None = None
    try:
        conn = write_engine.connect()
        # Sample the commit sequence BEFORE opening the snapshot (conservative).
        seq = caches.commit_seq
        conn.execute(text("BEGIN IMMEDIATE"))
        tx = Tx(conn, caches, seq)
        try:
            yield tx
        except Exception:
            conn.execute(text("ROLLBACK"))
            raise
        conn.execute(text("COMMIT"))
        # Tick the commit sequence under the still-held write lock, before hooks
        # fire, so an invalidation hook stamps the post-commit seq and a concurrent
        # reader's guard sees this commit as either fully applied or not at all.
        caches.tick()
        tx._fire_hooks()
    finally:
        if conn is not None:
            conn.close()
        write_lock.release()


@contextmanager
def read_snapshot(read_engine: Engine, caches: CacheRegistry) -> Iterator[Tx]:
    """Open a read-only snapshot against ``read_engine``.

    ``query_only`` is pinned at connect time on the read engine, so this
    path only pays for the BEGIN/ROLLBACK round-trips per call. Yields a
    ``Tx`` over a pooled connection and rolls back on exit so the
    snapshot does not leak into the next checkout from the pool.

    ``caches`` is mirrored onto the yielded ``Tx`` as ``tx.caches`` (see
    :func:`write_transaction`); a read handler that consults a derived-count memo
    reaches it there. Post-commit hooks never fire on a read snapshot, so the
    invalidation path is inert here — only the memo *read* path is used.
    """
    conn = read_engine.connect()
    try:
        # Sample the commit sequence BEFORE opening the snapshot (conservative).
        seq = caches.commit_seq
        conn.execute(text("BEGIN"))
        try:
            yield Tx(conn, caches, seq)
        finally:
            conn.execute(text("ROLLBACK"))
    finally:
        conn.close()


class ControllerDB:
    """Thread-safe SQLite wrapper with typed query and migration helpers."""

    DB_FILENAME = "controller.sqlite3"
    AUTH_DB_FILENAME = "auth.sqlite3"
    BASELINE_MIGRATION = "0001_baseline.py"

    def __init__(self, db_dir: Path):
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / self.DB_FILENAME
        self._auth_db_path = self._db_dir / self.AUTH_DB_FILENAME
        self._lock = RLock()
        self._reopen_hooks: list[Callable[[], None]] = []
        # Per-controller cache registry, mirrored onto every Tx this DB mints as
        # ``tx.caches``. Built before the engines so no cursor is ever minted
        # without it. Populated by higher layers (each per-controller memo
        # registers itself on construction) — the raw layer stays cache-agnostic.
        self._caches = CacheRegistry()

        # Build SA engines first so apply_migrations can use raw_connection().
        t0 = time.monotonic()
        self._sa_write_engine: Engine = _make_write_engine(self._db_path, self._auth_db_path)
        # Read connections must not see auth tables — pass None so auth is not ATTACHed.
        self._sa_read_engine: Engine = _make_read_engine(self._db_path, None)
        # Dedicated read engine for the control-loop tick, isolated from the
        # shared RPC pool so scheduling never queues behind a slow dashboard read.
        self._sa_control_read_engine: Engine = _make_read_engine(
            self._db_path, None, pool_size=CONTROL_READ_POOL_SIZE, max_overflow=CONTROL_READ_MAX_OVERFLOW
        )
        # Dedicated read engine backed by the auth DB file directly so auth
        # read functions do not go through the write connection.
        self._sa_auth_read_engine: Engine = _make_read_engine(
            self._auth_db_path, None, pool_size=AUTH_READ_POOL_SIZE, max_overflow=AUTH_READ_MAX_OVERFLOW
        )
        logger.info("SA engines initialized in %.2fs", time.monotonic() - t0)

        t0 = time.monotonic()
        self.apply_migrations()
        logger.info("Migrations applied in %.2fs", time.monotonic() - t0)

        # Populate sqlite_stat1 so the query planner picks good join orders.
        # Without this, queries like running_tasks_by_worker scan thousands of
        # rows instead of using the narrower index path.
        t0 = time.monotonic()
        raw_conn = self._sa_write_engine.raw_connection()
        try:
            raw_conn.execute("ANALYZE")
        finally:
            raw_conn.close()
        logger.info("ANALYZE completed in %.2fs", time.monotonic() - t0)

    def register_reopen_hook(self, hook: Callable[[], None]) -> None:
        """Register a no-arg callable to run at the end of ``replace_from``."""
        self._reopen_hooks.append(hook)

    @property
    def sa_read_engine(self) -> Engine:
        """Shared SA Core read engine (RPC-handler reads)."""
        return self._sa_read_engine

    @property
    def sa_control_read_engine(self) -> Engine:
        """Dedicated SA Core read engine for the control-loop tick."""
        return self._sa_control_read_engine

    @property
    def sa_write_engine(self) -> Engine:
        """SA Core write engine."""
        return self._sa_write_engine

    @property
    def caches(self) -> CacheRegistry:
        """The per-controller cache registry (also reachable via any ``Tx.caches``)."""
        return self._caches

    @property
    def commit_seq(self) -> int:
        """Monotonic write-commit counter (ticked per commit and per DB-file swap)."""
        return self._caches.commit_seq

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def auth_db_path(self) -> Path:
        return self._auth_db_path

    def optimize(self) -> None:
        """Run PRAGMA optimize to refresh statistics for tables with stale data.

        Lightweight operation that SQLite recommends running periodically or on
        connection close. Only re-analyzes tables whose stats have drifted.
        """
        with self._lock:
            raw_conn = self._sa_write_engine.raw_connection()
            try:
                raw_conn.execute("PRAGMA optimize")
            finally:
                raw_conn.close()

    def wal_checkpoint(self) -> tuple[int, int, int]:
        """Reclaim freelist pages, flush WAL into the main DB, and truncate it.

        Left unchecked, the WAL grows unbounded under continuous write load and
        makes every reader walk more frames to assemble a snapshot. The preceding
        ``PRAGMA incremental_vacuum`` (enabled via the auto_vacuum=INCREMENTAL
        migration) writes frames describing the shortened file; the subsequent
        TRUNCATE checkpoint flushes those frames and physically truncates both
        the main DB and WAL on disk. ``executescript`` drains the pragma so every
        available freelist page is reclaimed (it yields one row per freed page).

        Returns ``(busy, log_frames, checkpointed_frames)`` exactly as SQLite does.
        """
        # Pin to the main schema so the attached auth/profiles DBs (which may
        # not even be in WAL mode) cannot raise SQLITE_LOCKED here.
        with self._lock:
            raw_conn = self._sa_write_engine.raw_connection()
            try:
                raw_conn.executescript("PRAGMA main.incremental_vacuum")
                row = raw_conn.execute("PRAGMA main.wal_checkpoint(TRUNCATE)").fetchone()
            finally:
                raw_conn.close()
        return (int(row[0]), int(row[1]), int(row[2]))

    def close(self) -> None:
        self._sa_write_engine.dispose()
        self._sa_read_engine.dispose()
        self._sa_control_read_engine.dispose()
        self._sa_auth_read_engine.dispose()

    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        """Open an IMMEDIATE write transaction and yield a cursor.

        On successful commit, any hooks registered via ``Tx.register``
        fire while the write lock is still held — keeping in-memory caches
        in sync with the DB without exposing a torn snapshot to concurrent
        readers. The yielded cursor carries this DB's cache registry as
        ``tx.caches`` so write sinks reach per-controller memos through it.
        """
        with write_transaction(self._sa_write_engine, self._lock, self._caches) as tx:
            yield tx

    @contextmanager
    def read_snapshot(self) -> Iterator[Tx]:
        """Read-only snapshot that does NOT acquire the write lock.

        Uses a pooled read-only connection with WAL isolation. Safe for
        concurrent use from dashboard/RPC threads while the scheduling
        loop holds the write lock.
        """
        with read_snapshot(self._sa_read_engine, self._caches) as tx:
            yield tx

    @contextmanager
    def control_read_snapshot(self) -> Iterator[Tx]:
        """Read-only snapshot for the control loop, backed by a dedicated engine.

        Identical to :meth:`read_snapshot` but checks out from the control-only
        pool, so the schedule/reconcile/autoscale tick never queues behind
        RPC-handler reads for a connection. Use only from control-plane threads
        (the single control-loop thread, or the scheduling/autoscaler loops on
        the legacy path).
        """
        with read_snapshot(self._sa_control_read_engine, self._caches) as tx:
            yield tx

    @contextmanager
    def auth_read_snapshot(self) -> Iterator[Tx]:
        """Read-only snapshot backed by the auth DB (auth.sqlite3) directly.

        Auth tables live in a separate SQLite file. Read connections from
        ``read_snapshot`` do not ATTACH that file (by design — main-DB readers
        must not see auth tables). Use this context manager for auth-only
        read queries so they remain non-blocking while the write lock is free.
        """
        with read_snapshot(self._sa_auth_read_engine, self._caches) as tx:
            yield tx

    def apply_migrations(self) -> None:
        """Bring the DB to the current schema, then apply any delta migrations.

        The current schema is materialized declaratively from ``schema.py``'s
        ``metadata`` / ``auth_metadata`` via ``Table.create_all`` — a single
        ``0001_baseline`` step that runs once per DB. Pre-baseline history
        (the original ``0001_init`` through the last pre-baseline migration)
        is no longer carried as files; any prod DB seeded under that scheme
        already has the schema, and we detect that case and self-heal by
        recording the baseline marker without recreating anything.

        Anything in ``migrations/`` after baseline is a delta — a small Python
        module exposing ``migrate(raw_conn)`` — applied in lexicographic order
        and recorded in ``schema_migrations`` by stem. Stems already recorded
        are skipped (including legacy pre-baseline stems on upgraded prod DBs).
        Deltas must be idempotent under ``IF [NOT] EXISTS`` so a crash mid-run
        is safe to retry.
        """
        baseline_stem = Path(self.BASELINE_MIGRATION).stem

        # ``Table.create`` checks out its own connection from the write engine,
        # so run it before we hold a raw connection (the pool_size=1 pool can
        # only hand out one at a time).
        self._ensure_schema_migrations_table()

        # Baseline step. ``metadata.create_all`` checks out its own connection
        # from the write engine, which collides with the pool_size=1 pool if we
        # hold one ourselves — so scope each raw-connection use tightly.
        raw_conn = self._sa_write_engine.raw_connection()
        try:
            applied_stems = self._applied_migration_stems(raw_conn)
            needs_baseline = baseline_stem not in applied_stems
            has_user_tables = self._has_user_tables(raw_conn) if needs_baseline else False
        finally:
            raw_conn.close()

        if needs_baseline:
            if not has_user_tables:
                t0 = time.monotonic()
                metadata.create_all(self._sa_write_engine)
                # auth_metadata's Tables don't carry a schema= so create_all
                # would target the engine's main DB. Open a one-shot engine
                # pointing at auth.sqlite3 so the tables land there.
                auth_write = _make_write_engine(self._auth_db_path, None)
                try:
                    auth_metadata.create_all(auth_write)
                finally:
                    auth_write.dispose()
                logger.info("Baseline schema created in %.2fs", time.monotonic() - t0)
            else:
                logger.info("Legacy DB detected; recording baseline marker without recreating schema")
            self._record_migration(self.BASELINE_MIGRATION)
            applied_stems.add(baseline_stem)

        # Delta migrations.
        raw_conn = self._sa_write_engine.raw_connection()
        try:
            self._apply_delta_migrations(raw_conn, applied_stems)
        finally:
            raw_conn.close()

        # Migrations may have churned the WAL; reclaim and truncate.
        # wal_checkpoint() takes its own raw connection, so do it after
        # releasing ours back to the pool_size=1 write pool.
        busy, log_frames, checkpointed = self.wal_checkpoint()
        logger.info(
            "Post-migration wal_checkpoint(TRUNCATE): busy=%d log_frames=%d checkpointed=%d",
            busy,
            log_frames,
            checkpointed,
        )

    def _ensure_schema_migrations_table(self) -> None:
        # Single source of truth: emit the DDL from the SA Table definition
        # (checkfirst => CREATE TABLE IF NOT EXISTS) rather than hand-written SQL.
        schema_migrations_table.create(self._sa_write_engine, checkfirst=True)

    @staticmethod
    def _applied_migration_stems(raw_conn) -> set[str]:
        rows = raw_conn.execute("SELECT name FROM schema_migrations").fetchall()
        return {Path(row[0]).stem for row in rows}

    @staticmethod
    def _has_user_tables(raw_conn) -> bool:
        return (
            raw_conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'schema_migrations' "
                "LIMIT 1"
            ).fetchone()
            is not None
        )

    def _record_migration(self, name: str) -> None:
        raw_conn = self._sa_write_engine.raw_connection()
        try:
            raw_conn.execute(
                "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                (name, Timestamp.now().epoch_ms()),
            )
            raw_conn.commit()
        finally:
            raw_conn.close()

    def _apply_delta_migrations(self, raw_conn, applied_stems: set[str]) -> None:
        migrations_dir = Path(__file__).with_name("migrations")
        if not migrations_dir.exists():
            return

        pending = [
            path
            for path in sorted(migrations_dir.glob("*.py"))
            if not path.name.startswith("__") and path.stem not in applied_stems
        ]
        if not pending:
            return

        logger.info("Applying %d pending migration(s): %s", len(pending), [p.name for p in pending])

        raw_conn.execute("PRAGMA synchronous=OFF")
        # journal_mode returns a row; consume so the cursor closes and cannot
        # hold a statement-level lock that would block wal_checkpoint.
        raw_conn.execute("PRAGMA journal_mode=MEMORY").fetchall()
        raw_conn.execute("PRAGMA temp_store=MEMORY")
        try:
            for path in pending:
                t0 = time.monotonic()
                spec = importlib.util.spec_from_file_location(path.stem, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.migrate(raw_conn)
                # Commit any implicit transaction left open by migrate() so
                # the next BEGIN IMMEDIATE succeeds.
                raw_conn.commit()
                logger.info("Migration %s applied in %.2fs", path.name, time.monotonic() - t0)

                raw_conn.execute("BEGIN IMMEDIATE")
                try:
                    raw_conn.execute(
                        "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                        (path.name, Timestamp.now().epoch_ms()),
                    )
                    raw_conn.commit()
                except Exception:
                    raw_conn.execute("ROLLBACK")
                    raise
        finally:
            raw_conn.commit()
            raw_conn.execute("PRAGMA synchronous=NORMAL")
            raw_conn.execute("PRAGMA journal_mode=WAL").fetchall()

    def backup_to(self, destination: Path) -> None:
        """Create a hot backup to ``destination`` using SQLite backup API.

        The source DB uses WAL journal mode, but the backup API copies
        the WAL flag into the destination header.  We switch the
        destination to DELETE mode so the result is a single
        self-contained file (no -wal/-shm sidecars) that survives
        compression and remote upload without corruption.

        We also set ``auto_vacuum=INCREMENTAL`` on the backup and run one
        incremental vacuum pass so controllers restoring from this
        checkpoint start in incremental mode without needing a full
        VACUUM at boot.  This is a single-pass operation against the
        already-written backup file -- no redundant copy is required.

        The backup runs through a dedicated read-only source connection,
        so writers on the write engine proceed concurrently under SQLite's
        WAL semantics -- no controller-level lock is held for the
        duration of the copy.  Batched page copying (``pages=500``)
        yields between steps so a sustained write stream cannot starve
        the backup.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(str(self._db_path), check_same_thread=False)
        try:
            src.execute("PRAGMA journal_mode = WAL")
            src.execute("PRAGMA synchronous = NORMAL")
            src.execute("PRAGMA busy_timeout = 5000")
            src.execute("PRAGMA foreign_keys = ON")
            src.execute("PRAGMA cache_size = -65536")
            src.execute("PRAGMA query_only = ON")
            dest = sqlite3.connect(str(destination))
            try:
                src.backup(dest, pages=500, sleep=0)
                dest.execute("PRAGMA journal_mode = DELETE")
                dest.execute("PRAGMA auto_vacuum = INCREMENTAL")
                dest.execute("PRAGMA incremental_vacuum")
                dest.commit()
            finally:
                dest.close()
        finally:
            src.close()

    @staticmethod
    def _sidecar_paths(path: Path) -> tuple[Path, Path]:
        return (path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm"))

    @staticmethod
    def _remove_sidecars(path: Path) -> None:
        for sidecar in ControllerDB._sidecar_paths(path):
            sidecar.unlink(missing_ok=True)

    def replace_from(self, source_dir: str | Path) -> None:
        """Replace current DB files from ``source_dir`` and reopen connection.

        ``source_dir`` is a directory (local or remote) containing
        ``controller.sqlite3`` and optionally ``auth.sqlite3``. Files are
        downloaded via fsspec so remote paths (e.g. ``gs://...``) work.
        Only called at startup before concurrent access begins.
        """
        source_dir_str = str(source_dir).rstrip("/")

        with self._lock:
            # Dispose existing SA pools before swapping files.
            self._sa_write_engine.dispose()
            self._sa_read_engine.dispose()
            self._sa_control_read_engine.dispose()

            # Download main DB
            main_source = f"{source_dir_str}/{self.DB_FILENAME}"
            tmp_path = self._db_path.with_suffix(".tmp")
            with fsspec.core.open(main_source, "rb") as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
            self._remove_sidecars(self._db_path)
            tmp_path.rename(self._db_path)

            # Download auth DB if present in source
            auth_source = f"{source_dir_str}/{self.AUTH_DB_FILENAME}"
            fs, fs_path = fsspec.core.url_to_fs(auth_source)
            if fs.exists(fs_path):
                auth_tmp = self._auth_db_path.with_suffix(".tmp")
                with fsspec.core.open(auth_source, "rb") as src, open(auth_tmp, "wb") as dst:
                    dst.write(src.read())
                self._remove_sidecars(self._auth_db_path)
                auth_tmp.rename(self._auth_db_path)

            # Rebuild SA engines against the freshly-installed DB.
            self._sa_write_engine = _make_write_engine(self._db_path, self._auth_db_path)
            # Read connections must not see auth tables — pass None so auth is not ATTACHed.
            self._sa_read_engine = _make_read_engine(self._db_path, None)
            self._sa_control_read_engine = _make_read_engine(
                self._db_path, None, pool_size=CONTROL_READ_POOL_SIZE, max_overflow=CONTROL_READ_MAX_OVERFLOW
            )
            self._sa_auth_read_engine = _make_read_engine(
                self._auth_db_path, None, pool_size=AUTH_READ_POOL_SIZE, max_overflow=AUTH_READ_MAX_OVERFLOW
            )

        self.apply_migrations()
        # The DB file was swapped: every open snapshot's seq now predates a file
        # that shares no history with the new one. Tick so a lazy guard's floor
        # (set by the reopen ``clear`` hook below) rejects any pre-restore fill.
        self._caches.tick()
        for hook in self._reopen_hooks:
            hook()

    # Read access is through ``read_snapshot()`` and typed table metadata at module scope.
