# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite access layer and typed query models for controller state."""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import fsspec.core
from rigging.timing import Deadline, Duration, Timestamp
from sqlalchemy import Engine, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db_v2 import (
    Tx,
    _make_read_engine,
    _make_write_engine,
    write_transaction,
)
from iris.cluster.controller.db_v2 import (
    read_snapshot as _read_snapshot,
)
from iris.cluster.controller.schema import (
    TASK_DETAIL_PROJECTION,
    WORKER_ROW_PROJECTION,
    decode_worker_id,
)
from iris.cluster.controller.schema_v2 import (
    meta_table,
    user_budgets_table,
    users_table,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import TERMINAL_TASK_STATES, JobName, WorkerId
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy types preserved for import compatibility.
#
# ``TransactionCursor``, ``QuerySnapshot``, and ``Row`` are still imported by
# ``transitions.py``, ``service.py``, ``auth.py``, and test conftest files.
# Removing them from the module would break those imports.
#
# M5/M6/M11 will update all call sites to use ``Tx``; after that M12 deletes
# these classes. Until then, keep them here so imports succeed at module load
# time even though ``ControllerDB.transaction()`` now yields ``Tx`` (not
# ``TransactionCursor``) and ``ControllerDB.read_snapshot()`` yields ``Tx``
# (not ``QuerySnapshot``).  Runtime calls that relied on the old behaviour
# (e.g. ``cur.execute("raw SQL?")``) will raise ``TypeError``; that is the
# expected M11 breakage.
# ---------------------------------------------------------------------------


class Row:
    """Lightweight result row with attribute access for raw query results.

    Kept for import compatibility. M8 deletes this once all call sites migrate.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Row has no column {name!r}") from None

    def __repr__(self) -> str:
        return f"Row({self._data!r})"


def _sa_stmt_to_sql(stmt: Any) -> str:
    """Lower a SA ``text(...)``/``select(...)`` construct to a SQL string.

    Used by the legacy ``TransactionCursor`` and ``QuerySnapshot``. Kept for
    import compatibility; deleted in M12 once those classes are gone.
    """
    text_attr = getattr(stmt, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    compiled = stmt.compile(compile_kwargs={"literal_binds": False})
    return str(compiled)


class QuerySnapshot:
    """Read-only snapshot over the controller DB.

    Kept for import compatibility. ``ControllerDB.read_snapshot()`` now yields
    ``Tx``; this class remains so ``from iris.cluster.controller.db import
    QuerySnapshot`` in ``transitions.py`` / ``service.py`` / ``auth.py`` does
    not raise ``ImportError``. M5/M6/M11 migrate the call sites; M12 deletes
    this class.
    """

    def __init__(self, conn: sqlite3.Connection, lock: RLock | None):
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> QuerySnapshot:
        if self._lock is not None:
            self._lock.acquire()
        self._conn.execute("BEGIN")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self._conn.rollback()
        finally:
            if self._lock is not None:
                self._lock.release()

    def execute_sql(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def execute(self, sql, params=None) -> sqlite3.Cursor:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return self._conn.execute(sql, params if params is not None else ())

    def fetchall(self, sql, params=None) -> list[sqlite3.Row]:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return list(self._conn.execute(sql, params if params is not None else ()).fetchall())

    def fetchone(self, sql, params=None) -> sqlite3.Row | None:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return self._conn.execute(sql, params if params is not None else ()).fetchone()

    def _fetchall(self, sql: str, params: Sequence[object]) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def raw(
        self,
        sql: str,
        params: tuple = (),
        decoders: dict[str, Callable] | None = None,
    ) -> list[Row]:
        cursor = self._conn.execute(sql, params)
        col_names = [desc[0] for desc in cursor.description]
        active_decoders = decoders or {}
        rows = []
        for raw_row in cursor.fetchall():
            data = {
                name: active_decoders[name](raw_row[name]) if name in active_decoders else raw_row[name]
                for name in col_names
            }
            rows.append(Row(data))
        return rows


class TransactionCursor:
    """Wraps a raw sqlite3.Cursor for use within controller transactions.

    Kept for import compatibility. ``ControllerDB.transaction()`` now yields
    ``Tx``; this class remains so ``from iris.cluster.controller.db import
    TransactionCursor`` in ``transitions.py`` / ``stores.py`` / ``budget.py``
    does not raise ``ImportError``. M5/M6/M11 migrate the call sites; M12
    deletes this class.
    """

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor
        self._commit_hooks: list[Callable[[], None]] = []

    def execute(self, sql, params=None) -> sqlite3.Cursor:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return self._cursor.execute(sql, params if params is not None else ())

    def executemany(self, sql, params: Iterable[tuple | Mapping[str, object]]) -> sqlite3.Cursor:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return self._cursor.executemany(sql, params)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        return self._cursor.executescript(sql)

    def fetchall(self, sql, params=None) -> list[sqlite3.Row]:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return list(self._cursor.execute(sql, params if params is not None else ()).fetchall())

    def fetchone(self, sql, params=None) -> sqlite3.Row | None:
        if not isinstance(sql, str):
            sql = _sa_stmt_to_sql(sql)
        return self._cursor.execute(sql, params if params is not None else ()).fetchone()

    def on_commit(self, hook: Callable[[], None]) -> None:
        self._commit_hooks.append(hook)

    def register(self, hook: Callable[[], None]) -> None:
        self._commit_hooks.append(hook)

    def _run_commit_hooks(self) -> None:
        for hook in self._commit_hooks:
            hook()

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class _LegacySnapshot(QuerySnapshot):
    """``QuerySnapshot`` adapter backed by a SA raw connection fairy.

    Closes the fairy on ``__exit__`` so the pooled connection is returned.
    Used only by ``ControllerDB.snapshot()`` for backward compatibility.
    Deleted in M12.
    """

    def __init__(self, fairy: Any, raw_conn: sqlite3.Connection):
        super().__init__(raw_conn, lock=None)
        self._fairy = fairy

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            super().__exit__(exc_type, exc, tb)
        finally:
            self._fairy.close()


def _decode_attribute_rows(rows: Sequence[Any]) -> dict[WorkerId, dict[str, AttributeValue]]:
    """Decode worker attribute rows into a nested dict.

    Kept for import compatibility with conftest.py and stores.py. M7/M12
    delete this once the legacy Projection-based read path is gone.
    """
    attrs_by_worker: dict[WorkerId, dict[str, AttributeValue]] = {}
    for row in rows:
        worker_attrs = attrs_by_worker.setdefault(row.worker_id, {})
        if row.value_type == "int":
            worker_attrs[row.key] = AttributeValue(int(row.int_value))
        elif row.value_type == "float":
            worker_attrs[row.key] = AttributeValue(float(row.float_value))
        else:
            worker_attrs[row.key] = AttributeValue(str(row.str_value or ""))
    return attrs_by_worker


# ---------------------------------------------------------------------------
# Shared predicate functions for Task/TaskRow and Worker/WorkerRow.
# Placed above the class definitions so both full and lightweight models
# can delegate to the same logic without duplication.
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
        self._sa_read_engine: Engine = _make_read_engine(self._db_path, self._auth_db_path)
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
        # caches over DB contents (e.g. ``ControllerStore``) can reload them
        # after a checkpoint restore. Registered via ``register_reopen_hook``.
        self._reopen_hooks: list[Callable[[], None]] = []

        # Stage 12: enforce the @writes_to invariant. Importing the writes
        # package re-exports every entity module so REGISTERED_WRITE_FUNCTIONS
        # is fully populated. Projection classes load via the projections
        # package re-export; instances appear in PROJECTIONS only after
        # ControllerStore (or a test) constructs them. The check is safe
        # to run pre-instantiation — owned will be empty and no violations
        # can fire — and is re-runnable from any caller after projections
        # are built.
        from iris.cluster.controller import projections, writes  # noqa: F401

        projections.assert_owned_tables_not_externally_written()

    def register_reopen_hook(self, hook: Callable[[], None]) -> None:
        """Register a no-arg callable to run at the end of ``replace_from``."""
        self._reopen_hooks.append(hook)

    @property
    def sa_read_engine(self) -> Engine:
        """SA Core read engine for the v2 data layer."""
        return self._sa_read_engine

    @property
    def sa_write_engine(self) -> Engine:
        """SA Core write engine for the v2 data layer."""
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

    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        """Open an IMMEDIATE transaction and yield a ``Tx``.

        On successful commit, any hooks registered via ``Tx.register`` or
        ``Tx.on_commit`` fire while the write lock is still held — keeping
        in-memory caches (e.g. ``EndpointsProjection``) in sync with the DB
        without exposing a torn snapshot to concurrent readers.
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
        with _read_snapshot(self._sa_read_engine) as tx:
            yield tx

    def snapshot(self) -> QuerySnapshot:
        """Return a legacy ``QuerySnapshot`` over a raw read connection.

        Preserved for import compatibility with ``auth.py`` and ``service.py``.
        M5/M6 will migrate those call sites to ``read_snapshot()``; this method
        will be removed in M12.

        The returned snapshot uses a fresh raw sqlite3 connection checked out
        of the write engine (no query_only pin). Callers must use it as a
        context manager (``with db.snapshot() as q:``).
        """
        raw_fairy = self._sa_write_engine.raw_connection()
        raw_conn = raw_fairy.driver_connection
        raw_conn.row_factory = sqlite3.Row
        # Wrap in a QuerySnapshot; we don't pass a lock so the snapshot does
        # not block writers. The raw_fairy is closed in __exit__ below via
        # a thin adapter.
        snap = _LegacySnapshot(raw_fairy, raw_conn)
        return snap

    def fetchall(self, query: str, params: tuple | list = ()) -> list:
        """Execute raw SQL and return all rows.

        Accepts ``?``-placeholder SQL for call sites not yet migrated to SA
        Core. The raw connection bypasses ``Tx`` intentionally — this method
        is a compatibility escape hatch used by the legacy layer and some
        tests. Prefer ``read_snapshot()`` + SA Core for new code.
        """
        raw_fairy = self._sa_write_engine.raw_connection()
        try:
            raw_fairy.driver_connection.row_factory = sqlite3.Row
            return list(raw_fairy.execute(query, params).fetchall())
        finally:
            raw_fairy.close()

    def fetchone(self, query: str, params: tuple | list = ()):
        """Execute raw SQL and return the first row, or ``None``."""
        raw_fairy = self._sa_write_engine.raw_connection()
        try:
            raw_fairy.driver_connection.row_factory = sqlite3.Row
            return raw_fairy.execute(query, params).fetchone()
        finally:
            raw_fairy.close()

    def execute(self, query: str, params: tuple | list = ()) -> None:
        """Execute raw SQL in a transaction.

        Accepts ``?``-placeholder SQL for call sites not yet migrated to SA
        Core. Prefer ``transaction()`` + SA Core for new code.
        """
        raw_fairy = self._sa_write_engine.raw_connection()
        try:
            raw_fairy.execute("BEGIN IMMEDIATE")
            try:
                raw_fairy.execute(query, params)
                raw_fairy.execute("COMMIT")
            except Exception:
                raw_fairy.execute("ROLLBACK")
                raise
        finally:
            raw_fairy.close()

    @staticmethod
    def decode_task(row: sqlite3.Row):
        return TASK_DETAIL_PROJECTION.decode_one([row])

    def apply_migrations(self) -> None:
        """Apply pending migrations from the migrations/ directory.

        Supports Python migration files that define a ``migrate(conn)``
        function. Migration names are matched by stem so that a migration
        previously applied as .sql is not re-run when converted to .py.

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

            # Match by stem so a migration previously recorded as .sql is not
            # re-run after conversion to .py.
            applied_stems = {Path(name).stem for name in applied}

<<<<<<< HEAD
        pending = []
        for path in sorted(migrations_dir.glob("*.py")):
            if path.name.startswith("__"):
                continue
            if path.stem in applied_stems:
                continue
            pending.append(path)
||||||| parent of 2d270e409 ([iris] M1-M4: rewrite reads/writes/projections to pure SA Core; tighten Tx)
        import time

        pending = []
        for path in sorted(migrations_dir.glob("*.py")):
            if path.name.startswith("__"):
                continue
            if path.stem in applied_stems:
                continue
            pending.append(path)
=======
            pending = []
            for path in sorted(migrations_dir.glob("*.py")):
                if path.name.startswith("__"):
                    continue
                if path.stem in applied_stems:
                    continue
                pending.append(path)

            if not pending:
                return
>>>>>>> 2d270e409 ([iris] M1-M4: rewrite reads/writes/projections to pure SA Core; tighten Tx)

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
            # Legacy migrations 0005/0014/0020/0023 reference `profiles.task_profiles`,
            # so attach the legacy file for the migration loop. 0046 + the finally
            # block below detach and unlink it.
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
                # Detach + unlink the legacy profiles DB. Idempotent — 0046 may
                # already have detached and unlinked.
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
            self._sa_read_engine = _make_read_engine(self._db_path, self._auth_db_path)

        self.apply_migrations()
        for hook in self._reopen_hooks:
            hook()

    # SQL-canonical read access is exposed through ``read_snapshot()`` and typed table
    # metadata at module scope. Legacy list/get/count helper methods were removed
    # to keep relation assembly explicit in controller/service/state query flows.

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


# ---------------------------------------------------------------------------
# Shared read-only query helpers
#
# Pure DB reads that are used by both controller.py and service.py.
# Each takes a ControllerDB and returns domain objects.
# ---------------------------------------------------------------------------


def running_tasks_by_worker(db: ControllerDB, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker."""
    if not worker_ids:
        return {}
    worker_id_strs = [str(wid) for wid in worker_ids]
    active_states = sorted(ACTIVE_TASK_STATES)
    placeholders_w = ",".join(f":w{i}" for i in range(len(worker_id_strs)))
    placeholders_s = ",".join(f":s{i}" for i in range(len(active_states)))
    params: dict[str, Any] = {}
    for i, wid in enumerate(worker_id_strs):
        params[f"w{i}"] = wid
    for i, s in enumerate(active_states):
        params[f"s{i}"] = s
    sql = text(
        f"SELECT t.current_worker_id AS worker_id, t.task_id FROM tasks t "
        f"WHERE t.current_worker_id IN ({placeholders_w}) AND t.state IN ({placeholders_s})"
    )
    with db.read_snapshot() as tx:
        rows = tx.fetchall(sql, params)
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        wid = decode_worker_id(row[0])
        task_id = JobName.from_wire(row[1])
        running[wid].add(task_id)
    return running


@dataclass(frozen=True, slots=True)
class TimedOutTask:
    """A running task that has exceeded its execution timeout."""

    task_id: JobName
    worker_id: WorkerId | None


def timed_out_executing_tasks(db: ControllerDB, now: Timestamp) -> list[TimedOutTask]:
    """Find executing tasks whose current attempt has exceeded the job's execution timeout.

    Reads the timeout from job_config.timeout_ms. Uses the current attempt's
    started_at_ms so that retried tasks get a fresh timeout budget per attempt.
    """
    now_ms = now.epoch_ms()
    executing_states = sorted(EXECUTING_TASK_STATES)
    placeholders = ",".join(f":s{i}" for i in range(len(executing_states)))
    params: dict[str, Any] = {f"s{i}": s for i, s in enumerate(executing_states)}
    params["now_ms"] = now_ms
    sql = text(
        f"SELECT t.task_id, t.current_worker_id AS worker_id, "
        f"ta.started_at_ms AS attempt_started_at_ms, jc.timeout_ms "
        f"FROM tasks t "
        f"JOIN job_config jc ON jc.job_id = t.job_id "
        f"JOIN task_attempts ta ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id "
        f"WHERE t.state IN ({placeholders}) "
        f"AND jc.timeout_ms IS NOT NULL AND jc.timeout_ms > 0 "
        f"AND ta.started_at_ms IS NOT NULL"
    )
    with db.read_snapshot() as tx:
        rows = tx.fetchall(sql, params)
    result: list[TimedOutTask] = []
    for row in rows:
        attempt_started_at_ms = int(row[2])
        timeout_ms = int(row[3])
        if attempt_started_at_ms + timeout_ms <= now_ms:
            task_id = JobName.from_wire(row[0])
            worker_id = WorkerId(row[1]) if row[1] is not None else None
            result.append(TimedOutTask(task_id=task_id, worker_id=worker_id))
    return result


def _worker_row_select() -> str:
    """Return WORKER_ROW_PROJECTION.select_clause()."""
    return WORKER_ROW_PROJECTION.select_clause()


class WorkerAttrsSource(Protocol):
    """Read-only view over the worker_attributes cache.

    Declared as a ``Protocol`` so :func:`healthy_active_workers_with_attributes`
    can stay in ``db.py`` without importing the concrete
    :class:`WorkerAttrsProjection` (which itself imports from ``db.py``).
    """

    def all(self) -> dict[WorkerId, dict[str, AttributeValue]]: ...


@dataclass(frozen=True, slots=True)
class SchedulableWorker:
    """Worker shape consumed by the scheduler.

    Field names mirror the :class:`scheduler.WorkerSnapshot` protocol so
    instances flow into ``Scheduler.create_scheduling_context`` without
    an adapter.
    """

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict[str, AttributeValue]


def healthy_active_workers_with_attributes(
    db: ControllerDB,
    health: WorkerHealthTracker,
    attrs: WorkerAttrsSource,
) -> list[SchedulableWorker]:
    """Return healthy + active workers with attributes.

    ``attrs`` is the live :class:`WorkerAttrsProjection` (declared as a
    ``Protocol`` to break the db → projection import cycle).
    """
    liveness = health.all()
    healthy_active = {wid for wid, l in liveness.items() if l.healthy and l.active}
    if not healthy_active:
        return []
    worker_id_strs = [str(wid) for wid in healthy_active]
    placeholders = ",".join(f":w{i}" for i in range(len(worker_id_strs)))
    params = {f"w{i}": wid for i, wid in enumerate(worker_id_strs)}
    sql = text(f"SELECT {_worker_row_select()} FROM workers w WHERE w.worker_id IN ({placeholders})")
    with db.read_snapshot() as tx:
        rows = WORKER_ROW_PROJECTION.decode(tx.fetchall(sql, params))
    if not rows:
        return []
    attrs_by_worker = attrs.all()
    out: list[SchedulableWorker] = []
    for w in rows:
        out.append(
            SchedulableWorker(
                worker_id=w.worker_id,
                address=w.address,
                total_cpu_millicores=w.total_cpu_millicores,
                total_memory_bytes=w.total_memory_bytes,
                total_gpu_count=w.total_gpu_count,
                total_tpu_count=w.total_tpu_count,
                device_type=w.device_type,
                device_variant=w.device_variant,
                attributes=attrs_by_worker.get(w.worker_id, {}),
            )
        )
    return out
