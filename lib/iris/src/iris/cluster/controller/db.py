# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy-backed controller database wrapper.

Hosts the SA ``Engine`` factories, the ``Tx`` wrapper, and the two
transaction context managers (``write_transaction`` / ``read_snapshot``).

The engine is split into a **write engine** and a **read engine**:

* The write engine uses pool size 1 so writes are funneled through a
  single connection. Serialization between writers is enforced by an
  external ``threading.RLock`` passed into ``write_transaction``.
* The read engine uses ``QueuePool(pool_size=32, max_overflow=4)`` with
  ``PRAGMA query_only = ON`` **pinned at connect time**. Pinning avoids
  toggling the pragma on every ``read_snapshot`` call.

Both engines use ``isolation_level="AUTOCOMMIT"`` so callers issue
``BEGIN`` / ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` explicitly.

Post-commit hooks registered via ``Tx.register`` fire *under the write
lock*, after ``COMMIT``. A concurrent thread cannot observe the
SQL-committed-but-cache-not-yet-updated window because the lock is held
until every hook has run.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import fsspec.core
from rigging.timing import Deadline, Duration, Timestamp
from sqlalchemy import Engine, create_engine, event, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.controller.schema import (
    meta_table,
    user_budgets_table,
    users_table,
)
from iris.cluster.types import TERMINAL_TASK_STATES, JobName
from iris.rpc import job_pb2

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


def _make_write_engine(db_path: Path, auth_db_path: Path | None) -> Engine:
    """Build the controller's SA **write** engine.

    Pool size 1: writes are serialized by an external ``RLock``.
    ``isolation_level="AUTOCOMMIT"`` disables SA's autoBEGIN behaviour so
    callers issue ``BEGIN IMMEDIATE`` explicitly. ``query_only`` is
    **not** set — writes need to mutate the DB.
    """
    auth_path_str = str(auth_db_path) if auth_db_path is not None else None

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5.0},
        pool_size=1,
        max_overflow=0,
        isolation_level="AUTOCOMMIT",
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pyrefly: ignore  # event hook
        _install_pragmas(dbapi_conn, auth_path_str)

    return engine


def _make_read_engine(db_path: Path, auth_db_path: Path | None) -> Engine:
    """Build the controller's SA **read** engine.

    ``QueuePool(pool_size=32, max_overflow=4)``.
    ``PRAGMA query_only = ON`` is pinned at **connect time** so that
    accidental writes raise without paying a pragma round-trip per call.
    ``isolation_level="AUTOCOMMIT"`` lets callers open / close transactions
    with explicit BEGIN/ROLLBACK.
    """
    auth_path_str = str(auth_db_path) if auth_db_path is not None else None

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5.0},
        pool_size=32,
        max_overflow=4,
        isolation_level="AUTOCOMMIT",
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pyrefly: ignore  # event hook
        _install_pragmas(dbapi_conn, auth_path_str)
        # Pin query_only at connect time so accidental writes raise
        # without paying a pragma round-trip per snapshot.
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA query_only = ON")
        finally:
            cur.close()

    return engine


class Tx:
    """Canonical write/read transaction context for the Iris controller.

    Wraps a SQLAlchemy ``Connection``. Accepts only SA Core constructs
    (``insert``/``update``/``delete``/``select``/``text``). Raw strings are
    rejected — use ``sqlalchemy.text()`` if you need to pass literal SQL.

    Post-commit hooks registered via :meth:`register` fire after ``COMMIT``,
    while the write lock is still held (see ``write_transaction``). The
    :attr:`on_commit` attribute is an alias for :meth:`register`; both names
    are first-class and used at different call sites.
    """

    def __init__(self, conn: Connection):
        self.conn = conn
        self._hooks: list[Callable[[], None]] = []

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

    def executemany(self, stmt, params_list) -> CursorResult:
        """Execute ``stmt`` repeatedly against ``params_list``.

        ``stmt`` must be a SA Core construct (see :meth:`execute`).
        """
        if isinstance(stmt, str):
            raise TypeError(
                "Tx.executemany does not accept raw SQL strings. "
                "Pass a SQLAlchemy construct (insert/update/delete/text)."
            )
        return self.conn.execute(stmt, params_list)

    def fetchone(self, stmt, params=None):
        """Execute and return the first row, or ``None``."""
        return self.execute(stmt, params).first()

    def fetchall(self, stmt, params=None) -> list:
        """Execute and return all rows as a list."""
        return list(self.execute(stmt, params).all())

    def scalar(self, stmt, params=None):
        """Execute and return the first column of the first row, or ``None``."""
        return self.execute(stmt, params).scalar()

    def register(self, hook: Callable[[], None]) -> None:
        """Register a post-commit hook.

        Hook fires once after commit, under the write lock. Write-tx only;
        ``read_snapshot`` never fires hooks.
        """
        self._hooks.append(hook)

    # Both names are first-class; ``on_commit`` is used by projection write
    # helpers, ``register`` by inline call sites in writes/*.py.
    on_commit = register

    def _fire_hooks(self) -> None:
        for hook in self._hooks:
            hook()


@contextmanager
def write_transaction(write_engine: Engine, write_lock: threading.RLock) -> Iterator[Tx]:
    """Open a write transaction backed by ``write_engine``.

    Acquires ``write_lock``, checks out a connection, emits
    ``BEGIN IMMEDIATE``, yields a ``Tx``, and commits on clean exit.
    Post-commit hooks registered via ``Tx.register`` fire **while the
    lock is still held** so in-memory caches stay consistent with the DB.
    """
    write_lock.acquire()
    conn: Connection | None = None
    try:
        conn = write_engine.connect()
        conn.execute(text("BEGIN IMMEDIATE"))
        tx = Tx(conn)
        try:
            yield tx
        except Exception:
            conn.execute(text("ROLLBACK"))
            raise
        conn.execute(text("COMMIT"))
        tx._fire_hooks()
    finally:
        if conn is not None:
            conn.close()
        write_lock.release()


@contextmanager
def read_snapshot(read_engine: Engine) -> Iterator[Tx]:
    """Open a read-only snapshot against ``read_engine``.

    ``query_only`` is pinned at connect time on the read engine, so this
    path only pays for the BEGIN/ROLLBACK round-trips per call. Yields a
    ``Tx`` over a pooled connection and rolls back on exit so the
    snapshot does not leak into the next checkout from the pool.
    """
    conn = read_engine.connect()
    try:
        conn.execute(text("BEGIN"))
        try:
            yield Tx(conn)
        finally:
            conn.execute(text("ROLLBACK"))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Shared predicate functions operating on SA Row objects from the tasks
# and workers tables.
# ---------------------------------------------------------------------------


def task_is_finished(
    state: int, failure_count: int, max_retries_failure: int, preemption_count: int, max_retries_preemption: int
) -> bool:
    """Whether a task has reached a terminal state with no remaining retries."""
    if state == job_pb2.TASK_STATE_SUCCEEDED:
        return True
    if state in (job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_UNSCHEDULABLE, job_pb2.TASK_STATE_COSCHED_FAILED):
        return True
    if state == job_pb2.TASK_STATE_FAILED:
        return failure_count > max_retries_failure
    if state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED):
        return preemption_count > max_retries_preemption
    return False


def task_row_is_finished(task: Any) -> bool:
    return task_is_finished(
        task.state, task.failure_count, task.max_retries_failure, task.preemption_count, task.max_retries_preemption
    )


def task_row_can_be_scheduled(task: Any) -> bool:
    if task.state != job_pb2.TASK_STATE_PENDING:
        return False
    return task.current_attempt_id < 0 or not task_is_finished(
        task.state, task.failure_count, task.max_retries_failure, task.preemption_count, task.max_retries_preemption
    )


# TERMINAL_TASK_STATES and TERMINAL_JOB_STATES are imported from iris.cluster.types.

ACTIVE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_ASSIGNED,
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    }
)

# Tasks executing on a worker (subset of ACTIVE that excludes ASSIGNED).
EXECUTING_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    }
)

# All non-terminal task states (ACTIVE plus PENDING). Complement of TERMINAL_TASK_STATES.
NON_TERMINAL_TASK_STATES: frozenset[int] = ACTIVE_TASK_STATES | {job_pb2.TASK_STATE_PENDING}

# Failure states that trigger coscheduled sibling cascades.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)


# job_is_finished is imported from iris.cluster.types (canonical definition).


def job_scheduling_deadline(scheduling_deadline_epoch_ms: int | None) -> Deadline | None:
    """Compute scheduling deadline from epoch ms."""
    if scheduling_deadline_epoch_ms is None:
        return None
    return Deadline.after(Timestamp.from_ms(scheduling_deadline_epoch_ms), Duration.from_ms(0))


def attempt_is_terminal(state: int) -> bool:
    """Check if an attempt is in a terminal state."""
    return state in TERMINAL_TASK_STATES


def attempt_is_worker_failure(state: int) -> bool:
    """Check if an attempt is a worker failure or preemption."""
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskJobSummary:
    job_id: JobName
    task_count: int = 0
    completed_count: int = 0
    failure_count: int = 0
    preemption_count: int = 0
    task_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UserBudget:
    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


class ControllerDB:
    """Thread-safe SQLite wrapper with typed query and migration helpers."""

    DB_FILENAME = "controller.sqlite3"
    AUTH_DB_FILENAME = "auth.sqlite3"

    def __init__(self, db_dir: Path):
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / self.DB_FILENAME
        self._auth_db_path = self._db_dir / self.AUTH_DB_FILENAME
        self._lock = RLock()

        # Build SA engines first so apply_migrations can use raw_connection().
        t0 = time.monotonic()
        self._sa_write_engine: Engine = _make_write_engine(self._db_path, self._auth_db_path)
        # Read connections must not see auth tables — pass None so auth is not ATTACHed.
        self._sa_read_engine: Engine = _make_read_engine(self._db_path, None)
        # Dedicated read engine backed by the auth DB file directly so auth
        # read functions do not go through the write connection.
        self._sa_auth_read_engine: Engine = _make_read_engine(self._auth_db_path, None)
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

        # Callables invoked at the end of ``replace_from`` so callers with
        # caches over DB contents (e.g. projections) can reload them
        # after a checkpoint restore. Registered via ``register_reopen_hook``.
        self._reopen_hooks: list[Callable[[], None]] = []

        # Enforce the @writes_to invariant. Importing the writes package
        # re-exports every entity module so REGISTERED_WRITE_FUNCTIONS is
        # fully populated. Projection classes load via the projections
        # package re-export; instances appear in PROJECTIONS only after
        # construction. The check is safe pre-instantiation — owned will be
        # empty and no violations can fire — and is re-runnable after
        # projections are built.
        from iris.cluster.controller import projections, writes  # noqa: F401

        projections.assert_owned_tables_not_externally_written()

    def register_reopen_hook(self, hook: Callable[[], None]) -> None:
        """Register a no-arg callable to run at the end of ``replace_from``."""
        self._reopen_hooks.append(hook)

    @property
    def sa_read_engine(self) -> Engine:
        """SA Core read engine."""
        return self._sa_read_engine

    @property
    def sa_write_engine(self) -> Engine:
        """SA Core write engine."""
        return self._sa_write_engine

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
        self._sa_auth_read_engine.dispose()

    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        """Open an IMMEDIATE write transaction and yield a ``Tx``.

        On successful commit, any hooks registered via ``Tx.register`` or
        ``Tx.on_commit`` fire while the write lock is still held — keeping
        in-memory caches in sync with the DB without exposing a torn
        snapshot to concurrent readers.
        """
        with write_transaction(self._sa_write_engine, self._lock) as tx:
            yield tx

    @contextmanager
    def read_snapshot(self) -> Iterator[Tx]:
        """Read-only snapshot that does NOT acquire the write lock.

        Uses a pooled read-only connection with WAL isolation. Safe for
        concurrent use from dashboard/RPC threads while the scheduling
        loop holds the write lock.
        """
        with read_snapshot(self._sa_read_engine) as tx:
            yield tx

    @contextmanager
    def auth_read_snapshot(self) -> Iterator[Tx]:
        """Read-only snapshot backed by the auth DB (auth.sqlite3) directly.

        Auth tables live in a separate SQLite file. Read connections from
        ``read_snapshot`` do not ATTACH that file (by design — main-DB readers
        must not see auth tables). Use this context manager for auth-only
        read queries so they remain non-blocking while the write lock is free.
        """
        with read_snapshot(self._sa_auth_read_engine) as tx:
            yield tx

    def apply_migrations(self) -> None:
        """Apply pending migrations from the migrations/ directory.

        Supports Python migration files that define a ``migrate(conn)``
        function. Migration names are matched by stem so a migration recorded
        under its ``.sql`` name is not re-run after conversion to ``.py``.

        Migrations run outside a transaction because executescript() implicitly
        commits. This is fine: migrations only run at startup before any
        concurrent access. Each migration is applied then recorded; if the
        process crashes mid-migration the partially-applied file won't be in
        schema_migrations and the next startup will re-run it (migrations must
        be idempotent via IF NOT EXISTS / IF EXISTS guards).
        """
        migrations_dir = Path(__file__).with_name("migrations")
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Use a raw sqlite3 connection so we can call executescript() and flip
        # PRAGMAs outside any SA transaction context. The pool_size=1 write
        # engine is used; we hold this connection for the entire migration run
        # and close it before returning so the pool is free afterwards.
        raw_conn = self._sa_write_engine.raw_connection()
        try:
            raw_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at_ms INTEGER NOT NULL
                )
                """
            )
            raw_conn.commit()
            applied = {row[0] for row in raw_conn.execute("SELECT name FROM schema_migrations ORDER BY name").fetchall()}

            # Match by stem: a migration recorded under its .sql name is not
            # re-run after conversion to .py.
            applied_stems = {Path(name).stem for name in applied}

            pending = []
            for path in sorted(migrations_dir.glob("*.py")):
                if path.name.startswith("__"):
                    continue
                if path.stem in applied_stems:
                    continue
                pending.append(path)

            if not pending:
                return

            logger.info("Applying %d pending migration(s): %s", len(pending), [p.name for p in pending])

            # Flip to fast-mode PRAGMAs for the duration of the migration loop.
            # Safe: migrations run at startup before any concurrent access, and a
            # crash re-runs the migration from schema_migrations. journal_mode
            # cannot change inside a transaction, so commit first and restore at
            # the end.
            raw_conn.commit()
            raw_conn.execute("PRAGMA synchronous=OFF")
            # journal_mode returns a row; consume it so the cursor is closed and
            # cannot hold a statement-level lock that would block wal_checkpoint.
            raw_conn.execute("PRAGMA journal_mode=MEMORY").fetchall()
            raw_conn.execute("PRAGMA temp_store=MEMORY")
            # Migrations 0005/0014/0020/0023 reference `profiles.task_profiles`.
            # Attach profiles.sqlite3 for the migration loop; 0046 and the
            # finally block below detach and unlink it.
            profiles_path = self._db_dir / "profiles.sqlite3"
            raw_conn.execute("ATTACH DATABASE ? AS profiles", (str(profiles_path),))
            try:
                for path in pending:
                    t0 = time.monotonic()
                    spec = importlib.util.spec_from_file_location(path.stem, path)
                    assert spec is not None and spec.loader is not None
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module.migrate(raw_conn)
                    # Commit any implicit transaction left open by migrate() (e.g.
                    # row-by-row UPDATEs in 0008) so the next BEGIN IMMEDIATE succeeds.
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
                # Detach + unlink profiles.sqlite3. Idempotent — 0046 may
                # already have detached and unlinked it.
                try:
                    raw_conn.execute("DETACH DATABASE profiles")
                except sqlite3.OperationalError:
                    pass
                try:
                    profiles_path.unlink()
                except FileNotFoundError:
                    pass
                # Checkpoint inline: must release raw_conn back to the pool first
                # because wal_checkpoint() acquires a new raw_connection() and the
                # write engine pool_size=1 would deadlock if we held raw_conn here.
                raw_conn.close()
                raw_conn = None
                busy, log_frames, checkpointed = self.wal_checkpoint()
                logger.info(
                    "Post-migration wal_checkpoint(TRUNCATE): busy=%d log_frames=%d checkpointed=%d",
                    busy,
                    log_frames,
                    checkpointed,
                )
        finally:
            if raw_conn is not None:
                raw_conn.close()

    @property
    def api_keys_table(self) -> str:
        return "auth.api_keys"

    @property
    def secrets_table(self) -> str:
        return "auth.controller_secrets"

    def ensure_user(self, user_id: str, now: Timestamp, role: str = "user") -> None:
        """Create user if not exists. Does not update role for existing users."""
        stmt = sqlite_insert(users_table).values(
            user_id=user_id,
            created_at_ms=now,
            role=role,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["user_id"])
        with self.transaction() as tx:
            tx.execute(stmt)

    def set_user_role(self, user_id: str, role: str) -> None:
        """Update the role for an existing user."""
        stmt = update(users_table).where(users_table.c.user_id == user_id).values(role=role)
        with self.transaction() as tx:
            tx.execute(stmt)

    def get_user_role(self, user_id: str) -> str:
        """Get a user's role. Returns 'user' if not found."""
        stmt = select(users_table.c.role).where(users_table.c.user_id == user_id)
        with self.read_snapshot() as tx:
            row = tx.fetchone(stmt)
        return row[0] if row is not None else "user"

    def next_sequence(self, key: str, *, cur: Tx) -> int:
        stmt = select(meta_table.c.value).where(meta_table.c.key == key)
        row = cur.fetchone(stmt)
        if row is None:
            cur.execute(sqlite_insert(meta_table).values(key=key, value=1))
            return 1
        value = int(row[0]) + 1
        cur.execute(update(meta_table).where(meta_table.c.key == key).values(value=value))
        return value

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
            self._sa_auth_read_engine = _make_read_engine(self._auth_db_path, None)

        self.apply_migrations()
        for hook in self._reopen_hooks:
            hook()

    # Read access is through ``read_snapshot()`` and typed table metadata at module scope.

    # -- User budget accessors --------------------------------------------------

    def set_user_budget(self, user_id: str, budget_limit: int, max_band: int, now: Timestamp) -> None:
        """Insert or update a user's budget configuration."""
        stmt = sqlite_insert(user_budgets_table).values(
            user_id=user_id,
            budget_limit=budget_limit,
            max_band=max_band,
            updated_at_ms=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id"],
            set_={"budget_limit": budget_limit, "max_band": max_band, "updated_at_ms": now},
        )
        with self.transaction() as tx:
            tx.execute(stmt)

    def get_user_budget(self, user_id: str) -> UserBudget | None:
        """Get budget config for a user. Returns None if user has no budget row."""
        stmt = select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        ).where(user_budgets_table.c.user_id == user_id)
        with self.read_snapshot() as tx:
            row = tx.fetchone(stmt)
        if row is None:
            return None
        return UserBudget(
            user_id=row[0],
            budget_limit=row[1],
            max_band=row[2],
            updated_at=row[3],
        )

    def list_user_budgets(self) -> list[UserBudget]:
        """List all user budgets."""
        stmt = select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        )
        with self.read_snapshot() as tx:
            rows = tx.fetchall(stmt)
        return [
            UserBudget(
                user_id=row[0],
                budget_limit=row[1],
                max_band=row[2],
                updated_at=row[3],
            )
            for row in rows
        ]

    def get_all_user_budget_limits(self) -> dict[str, int]:
        """Return ``{user_id: budget_limit}`` for every user with a budget row."""
        rows = self.list_user_budgets()
        return {row.user_id: row.budget_limit for row in rows}
