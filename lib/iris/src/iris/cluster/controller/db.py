# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite access layer and typed query models for controller state."""

from __future__ import annotations

import logging
import queue
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.schema import decode_timestamp_ms, decode_worker_id
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from rigging.timing import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


class Row:
    """Lightweight result row with attribute access for raw query results."""

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


class QuerySnapshot:
    """Read-only snapshot over the controller DB."""

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
        """Execute raw SQL and return the cursor for result inspection."""
        return self._conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute SQL and return all rows."""
        return self._fetchall(sql, list(params))

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute SQL and return the first row, or None."""
        return self._conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: Sequence[object]) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def raw(
        self,
        sql: str,
        params: tuple = (),
        decoders: dict[str, Callable] | None = None,
    ) -> list[Row]:
        """Execute raw SQL and return decoded rows with attribute access.

        Each key in `decoders` maps a column name to a decoder function.
        Columns without decoders are returned as-is from SQLite.
        """
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
    if state in (job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_UNSCHEDULABLE):
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


TERMINAL_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)

TERMINAL_JOB_STATES: frozenset[int] = frozenset(
    {
        job_pb2.JOB_STATE_SUCCEEDED,
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_WORKER_FAILED,
        job_pb2.JOB_STATE_UNSCHEDULABLE,
    }
)

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

# Failure states that trigger coscheduled sibling cascades.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)


def job_is_finished(state: int) -> bool:
    """Check if a job is in a terminal state."""
    return state in TERMINAL_JOB_STATES


def job_scheduling_deadline(scheduling_deadline_epoch_ms: int | None) -> Deadline | None:
    """Compute scheduling deadline from epoch ms."""
    if scheduling_deadline_epoch_ms is None:
        return None
    return Deadline.after(Timestamp.from_ms(scheduling_deadline_epoch_ms), Duration.from_ms(0))


def task_is_live(state: int) -> bool:
    """Check if a task is in a non-terminal state."""
    return state not in TERMINAL_TASK_STATES


def task_is_dead(state: int) -> bool:
    """Check if a task is in a terminal state."""
    return state in TERMINAL_TASK_STATES


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
    job_ids: tuple[JobName, ...] = ()
    job_id: JobName | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


def _decode_attribute_rows(rows: Sequence[Any]) -> dict[WorkerId, dict[str, AttributeValue]]:
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


def endpoint_query_sql(query: EndpointQuery) -> tuple[str, list[object]]:
    """Build SQL query for endpoint lookups."""
    from_clause = "SELECT e.* FROM endpoints e"
    conditions: list[str] = []
    params: list[object] = []

    if query.task_ids:
        from_clause += " JOIN endpoints et ON e.endpoint_id = et.endpoint_id"
        placeholders = ",".join("?" for _ in query.task_ids)
        conditions.append(f"et.task_id IN ({placeholders})")
        params.extend(tid.to_wire() for tid in query.task_ids)

    if query.endpoint_ids:
        placeholders = ",".join("?" for _ in query.endpoint_ids)
        conditions.append(f"e.endpoint_id IN ({placeholders})")
        params.extend(query.endpoint_ids)

    if query.name_prefix:
        conditions.append("e.name LIKE ?")
        params.append(f"{query.name_prefix}%")

    if query.exact_name:
        conditions.append("e.name = ?")
        params.append(query.exact_name)

    job_ids = list(query.job_ids)
    if query.job_id is not None:
        job_ids.append(query.job_id)
    if job_ids:
        placeholders = ",".join("?" for _ in job_ids)
        conditions.append(f"e.job_id IN ({placeholders})")
        params.extend(jid.to_wire() for jid in job_ids)

    sql = from_clause
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if query.limit is not None:
        sql += " LIMIT ?"
        params.append(query.limit)
    return sql, params


class TransactionCursor:
    """Wraps a raw sqlite3.Cursor for use within controller transactions."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Raw SQL escape hatch."""
        return self._cursor.execute(sql, params)

    def executemany(self, sql: str, params: Iterable[tuple]) -> sqlite3.Cursor:
        """Raw SQL batch escape hatch."""
        return self._cursor.executemany(sql, params)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        """Raw SQL script escape hatch."""
        return self._cursor.executescript(sql)

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class ControllerDB:
    """Thread-safe SQLite wrapper with typed query and migration helpers."""

    _READ_POOL_SIZE = 8
    DB_FILENAME = "controller.sqlite3"
    AUTH_DB_FILENAME = "auth.sqlite3"
    PROFILES_DB_FILENAME = "profiles.sqlite3"

    def __init__(self, db_dir: Path):
        import time

        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / self.DB_FILENAME
        self._auth_db_path = self._db_dir / self.AUTH_DB_FILENAME
        self._profiles_db_path = self._db_dir / self.PROFILES_DB_FILENAME
        self._lock = RLock()

        t0 = time.monotonic()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure(self._conn)
        self._conn.execute("ATTACH DATABASE ? AS auth", (str(self._auth_db_path),))
        self._conn.execute("ATTACH DATABASE ? AS profiles", (str(self._profiles_db_path),))
        logger.info("DB opened in %.2fs (path=%s)", time.monotonic() - t0, self._db_path)

        t0 = time.monotonic()
        self.apply_migrations()
        logger.info("Migrations applied in %.2fs", time.monotonic() - t0)

        # Populate sqlite_stat1 so the query planner picks good join orders.
        # Without this, queries like running_tasks_by_worker scan thousands of
        # rows instead of using the narrower index path.
        t0 = time.monotonic()
        self._conn.execute("ANALYZE")
        logger.info("ANALYZE completed in %.2fs", time.monotonic() - t0)

        t0 = time.monotonic()
        self._read_pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._init_read_pool()
        logger.info("Read pool initialized in %.2fs", time.monotonic() - t0)
        # Lazily populated cache of worker attributes, keyed by worker_id.
        # Eliminates the per-cycle attribute SQL query from the scheduling hot path.
        self._attr_cache: dict[WorkerId, dict[str, AttributeValue]] | None = None
        self._attr_cache_lock = Lock()

    def _populate_attr_cache(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Load all worker attributes from the DB into the cache.

        Called once on cold start (first access). The caller must NOT hold
        _attr_cache_lock when calling this, because the DB read can be slow.
        """
        with self.read_snapshot() as q:
            rows = q.raw(
                "SELECT worker_id, key, value_type, str_value, int_value, float_value FROM worker_attributes",
            )
        return _decode_attribute_rows(rows)

    def get_worker_attributes(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Return cached worker attributes, populating from DB on first call."""
        cache = self._attr_cache
        if cache is not None:
            return cache
        fresh = self._populate_attr_cache()
        with self._attr_cache_lock:
            if self._attr_cache is None:
                self._attr_cache = fresh
            return self._attr_cache

    def set_worker_attributes(self, worker_id: WorkerId, attrs: dict[str, AttributeValue]) -> None:
        """Update the cached attributes for a single worker after registration."""
        with self._attr_cache_lock:
            if self._attr_cache is None:
                return
            self._attr_cache[worker_id] = attrs

    def remove_worker_from_attr_cache(self, worker_id: WorkerId) -> None:
        """Remove a single worker from the attribute cache."""
        with self._attr_cache_lock:
            if self._attr_cache is None:
                return
            self._attr_cache.pop(worker_id, None)

    def _init_read_pool(self) -> None:
        """Create (or recreate) the read-only connection pool."""
        while True:
            try:
                self._read_pool.get_nowait().close()
            except queue.Empty:
                break
        for _ in range(self._READ_POOL_SIZE):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._configure(conn)
            conn.execute("ATTACH DATABASE ? AS profiles", (str(self._profiles_db_path),))
            conn.execute("PRAGMA query_only = ON")
            self._read_pool.put(conn)

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def auth_db_path(self) -> Path:
        return self._auth_db_path

    @property
    def profiles_db_path(self) -> Path:
        return self._profiles_db_path

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")

    def optimize(self) -> None:
        """Run PRAGMA optimize to refresh statistics for tables with stale data.

        Lightweight operation that SQLite recommends running periodically or on
        connection close. Only re-analyzes tables whose stats have drifted.
        """
        with self._lock:
            self._conn.execute("PRAGMA optimize")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        for _ in range(self._READ_POOL_SIZE):
            try:
                self._read_pool.get(timeout=1).close()
            except queue.Empty:
                break

    @contextmanager
    def transaction(self):
        """Open an IMMEDIATE transaction and yield a TransactionCursor."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield TransactionCursor(cur)
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def fetchall(self, query: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(query, params).fetchall())

    def fetchone(self, query: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def execute(self, query: str, params: tuple | list = ()) -> None:
        with self.transaction() as cur:
            cur.execute(query, params)

    def snapshot(self) -> QuerySnapshot:
        return QuerySnapshot(self._conn, self._lock)

    @contextmanager
    def read_snapshot(self) -> Iterator[QuerySnapshot]:
        """Read-only snapshot that does NOT acquire the write lock.

        Uses a pooled read-only connection with WAL isolation. Safe for
        concurrent use from dashboard/RPC threads while the scheduling
        loop holds the write lock.
        """
        conn = self._read_pool.get()
        try:
            conn.execute("BEGIN")
            yield QuerySnapshot(conn, lock=None)
        finally:
            try:
                conn.rollback()
            except sqlite3.OperationalError:
                logging.getLogger(__name__).warning("read_snapshot rollback failed", exc_info=True)
            self._read_pool.put(conn)

    @staticmethod
    def decode_task(row: sqlite3.Row):
        from iris.cluster.controller.schema import TASK_DETAIL_PROJECTION

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
        import importlib.util

        migrations_dir = Path(__file__).with_name("migrations")
        migrations_dir.mkdir(parents=True, exist_ok=True)

        with self.transaction() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at_ms INTEGER NOT NULL
                )
                """
            )
            applied = {row[0] for row in cur.execute("SELECT name FROM schema_migrations ORDER BY name").fetchall()}

        # Match by stem so a migration previously recorded as .sql is not
        # re-run after conversion to .py.
        applied_stems = {Path(name).stem for name in applied}

        import time

        pending = []
        for path in sorted(migrations_dir.glob("*.py")):
            if path.name.startswith("__"):
                continue
            if path.stem in applied_stems:
                continue
            pending.append(path)

        if pending:
            logger.info("Applying %d pending migration(s): %s", len(pending), [p.name for p in pending])

        for path in pending:
            t0 = time.monotonic()
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.migrate(self._conn)
            # Commit any implicit transaction left open by migrate() (e.g.
            # row-by-row UPDATEs in 0008) so the next BEGIN IMMEDIATE succeeds.
            self._conn.commit()
            logger.info("Migration %s applied in %.2fs", path.name, time.monotonic() - t0)

            with self.transaction() as cur:
                cur.execute(
                    "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                    (path.name, Timestamp.now().epoch_ms()),
                )

    @property
    def api_keys_table(self) -> str:
        return "auth.api_keys"

    @property
    def secrets_table(self) -> str:
        return "auth.controller_secrets"

    @property
    def task_profiles_table(self) -> str:
        return "profiles.task_profiles"

    def ensure_user(self, user_id: str, now: Timestamp, role: str = "user") -> None:
        """Create user if not exists. Does not update role for existing users."""
        self.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at_ms, role) VALUES (?, ?, ?)",
            (user_id, now.epoch_ms(), role),
        )

    def set_user_role(self, user_id: str, role: str) -> None:
        """Update the role for an existing user."""
        self.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))

    def get_user_role(self, user_id: str) -> str:
        """Get a user's role. Returns 'user' if not found."""
        with self.read_snapshot() as q:
            rows = q.raw(
                "SELECT role FROM users WHERE user_id = ?",
                (user_id,),
                decoders={"role": str},
            )
            return rows[0].role if rows else "user"

    def next_sequence(self, key: str, *, cur: TransactionCursor) -> int:
        row = cur.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            cur.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, 1))
            return 1
        value = int(row[0]) + 1
        cur.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
        return value

    def backup_to(self, destination: Path) -> None:
        """Create a hot backup to ``destination`` using SQLite backup API.

        The source DB uses WAL journal mode, but the backup API copies
        the WAL flag into the destination header.  We switch the
        destination to DELETE mode so the result is a single
        self-contained file (no -wal/-shm sidecars) that survives
        compression and remote upload without corruption.

        The backup is also VACUUMed with auto_vacuum=INCREMENTAL so that
        controllers restoring from this checkpoint start in incremental
        mode without needing a full VACUUM at boot.
        """
        import time

        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(destination))
            try:
                self._conn.backup(dest)
                dest.execute("PRAGMA journal_mode = DELETE")
                dest.commit()
            finally:
                dest.close()

        # VACUUM INTO a compacted copy with incremental auto_vacuum enabled.
        # Runs outside the lock since it operates on the already-written backup.
        t0 = time.monotonic()
        vacuumed = destination.with_suffix(".vacuumed.sqlite3")
        conn = sqlite3.connect(str(destination))
        try:
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            conn.execute(f"VACUUM INTO '{vacuumed}'")
        finally:
            conn.close()
        vacuumed.rename(destination)
        logger.info("Checkpoint vacuumed in %.1fs", time.monotonic() - t0)

    def replace_from(self, source_dir: str | Path) -> None:
        """Replace current DB files from ``source_dir`` and reopen connection.

        ``source_dir`` is a directory (local or remote) containing
        ``controller.sqlite3`` and optionally ``auth.sqlite3``. Files are
        downloaded via fsspec so remote paths (e.g. ``gs://...``) work.
        Only called at startup before concurrent access begins.
        """
        import fsspec.core

        source_dir_str = str(source_dir).rstrip("/")

        with self._lock:
            # Download main DB
            main_source = f"{source_dir_str}/{self.DB_FILENAME}"
            tmp_path = self._db_path.with_suffix(".tmp")
            with fsspec.core.open(main_source, "rb") as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
            self._conn.close()
            tmp_path.rename(self._db_path)

            # Download auth DB if present in source
            auth_source = f"{source_dir_str}/{self.AUTH_DB_FILENAME}"
            fs, fs_path = fsspec.core.url_to_fs(auth_source)
            if fs.exists(fs_path):
                auth_tmp = self._auth_db_path.with_suffix(".tmp")
                with fsspec.core.open(auth_source, "rb") as src, open(auth_tmp, "wb") as dst:
                    dst.write(src.read())
                auth_tmp.rename(self._auth_db_path)

            # Download profiles DB if present in source
            profiles_source = f"{source_dir_str}/{self.PROFILES_DB_FILENAME}"
            fs2, fs_path2 = fsspec.core.url_to_fs(profiles_source)
            if fs2.exists(fs_path2):
                profiles_tmp = self._profiles_db_path.with_suffix(".tmp")
                with fsspec.core.open(profiles_source, "rb") as src, open(profiles_tmp, "wb") as dst:
                    dst.write(src.read())
                profiles_tmp.rename(self._profiles_db_path)

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._configure(self._conn)
            self._conn.execute("ATTACH DATABASE ? AS auth", (str(self._auth_db_path),))
            self._conn.execute("ATTACH DATABASE ? AS profiles", (str(self._profiles_db_path),))
            self._init_read_pool()
        self.apply_migrations()

    # SQL-canonical read access is exposed through ``snapshot()`` and typed table
    # metadata at module scope. Legacy list/get/count helper methods were removed
    # to keep relation assembly explicit in controller/service/state query flows.

    def delete_endpoint(self, endpoint_id: str):
        from iris.cluster.controller.schema import ENDPOINT_PROJECTION

        with self.transaction() as cur:
            row = cur.execute(
                "SELECT endpoint_id, name, address, job_id, metadata_json, registered_at_ms "
                "FROM endpoints WHERE endpoint_id = ?",
                (endpoint_id,),
            ).fetchone()
            if row is None:
                return None
            cur.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))
            return ENDPOINT_PROJECTION.decode_one([row])

    def delete_endpoints(self, endpoint_ids: Sequence[str]) -> None:
        if not endpoint_ids:
            return
        placeholders = ",".join("?" for _ in endpoint_ids)
        self.execute(f"DELETE FROM endpoints WHERE endpoint_id IN ({placeholders})", tuple(endpoint_ids))

    # -- User budget accessors --------------------------------------------------

    def set_user_budget(self, user_id: str, budget_limit: int, max_band: int, now: Timestamp) -> None:
        """Insert or update a user's budget configuration."""
        self.execute(
            "INSERT INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET budget_limit=?, max_band=?, updated_at_ms=?",
            (user_id, budget_limit, max_band, now.epoch_ms(), budget_limit, max_band, now.epoch_ms()),
        )

    def get_user_budget(self, user_id: str) -> UserBudget | None:
        """Get budget config for a user. Returns None if user has no budget row."""
        with self.read_snapshot() as q:
            row = q.fetchone(
                "SELECT user_id, budget_limit, max_band, updated_at_ms FROM user_budgets WHERE user_id = ?",
                (user_id,),
            )
        if row is None:
            return None
        return UserBudget(
            user_id=row["user_id"],
            budget_limit=row["budget_limit"],
            max_band=row["max_band"],
            updated_at=Timestamp.from_ms(row["updated_at_ms"]),
        )

    def list_user_budgets(self) -> list[UserBudget]:
        """List all user budgets."""
        with self.read_snapshot() as q:
            rows = q.fetchall("SELECT user_id, budget_limit, max_band, updated_at_ms FROM user_budgets", ())
        return [
            UserBudget(
                user_id=row["user_id"],
                budget_limit=row["budget_limit"],
                max_band=row["max_band"],
                updated_at=Timestamp.from_ms(row["updated_at_ms"]),
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
    """Return the set of currently-running task IDs for each worker.

    Uses the denormalized current_worker_id column instead of joining task_attempts.
    """
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT t.current_worker_id AS worker_id, t.task_id FROM tasks t "
            f"WHERE t.current_worker_id IN ({placeholders}) AND t.state IN (?, ?, ?)",
            (*[str(wid) for wid in worker_ids], *ACTIVE_TASK_STATES),
            decoders={"worker_id": decode_worker_id, "task_id": JobName.from_wire},
        )
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
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
    executing_states = tuple(sorted(EXECUTING_TASK_STATES))
    placeholders = ",".join("?" for _ in executing_states)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT t.task_id, t.current_worker_id AS worker_id, "
            f"ta.started_at_ms AS attempt_started_at_ms, jc.timeout_ms "
            f"FROM tasks t "
            f"JOIN job_config jc ON jc.job_id = t.job_id "
            f"JOIN task_attempts ta ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id "
            f"WHERE t.state IN ({placeholders}) "
            f"AND jc.timeout_ms IS NOT NULL AND jc.timeout_ms > 0 "
            f"AND ta.started_at_ms IS NOT NULL",
            (*executing_states,),
            decoders={
                "task_id": JobName.from_wire,
                "worker_id": lambda v: WorkerId(v) if v is not None else None,
                "attempt_started_at_ms": int,
                "timeout_ms": int,
            },
        )
    result: list[TimedOutTask] = []
    for row in rows:
        if row.attempt_started_at_ms + row.timeout_ms <= now_ms:
            result.append(TimedOutTask(task_id=row.task_id, worker_id=row.worker_id))
    return result


def tasks_for_job_with_attempts(db: ControllerDB, job_id: JobName) -> list:
    """Fetch all tasks for a job with their attempt history."""
    from iris.cluster.controller.schema import ATTEMPT_PROJECTION, TASK_DETAIL_PROJECTION, tasks_with_attempts

    with db.read_snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall(
                "SELECT * FROM tasks WHERE job_id = ? ORDER BY task_index, task_id",
                (job_id.to_wire(),),
            ),
        )
        if not tasks:
            return []
        placeholders = ",".join("?" for _ in tasks)
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT * FROM task_attempts WHERE task_id IN ({placeholders}) ORDER BY task_id, attempt_id",
                tuple(t.task_id.to_wire() for t in tasks),
            ),
        )
    return tasks_with_attempts(tasks, attempts)


def _worker_row_select() -> str:
    """Lazily resolve WORKER_ROW_PROJECTION.select_clause() to break the db -> schema cycle."""
    from iris.cluster.controller.schema import WORKER_ROW_PROJECTION

    return WORKER_ROW_PROJECTION.select_clause()


def healthy_active_workers_with_attributes(db: ControllerDB) -> list:
    """Fetch all healthy, active workers with their attributes populated.

    Returns WorkerRow (scalar-only) so the scheduling loop avoids loading metadata columns.
    Uses the in-memory attribute cache to avoid a per-cycle SQL join.
    """
    from iris.cluster.controller.schema import WORKER_ROW_PROJECTION

    with db.read_snapshot() as q:
        workers = WORKER_ROW_PROJECTION.decode(
            q.fetchall(f"SELECT {_worker_row_select()} FROM workers w WHERE w.healthy = 1 AND w.active = 1"),
        )
        if not workers:
            return []
    attrs_by_worker = db.get_worker_attributes()
    return [
        dc_replace(
            w,
            attributes=attrs_by_worker.get(w.worker_id, {}),
            available_cpu_millicores=w.total_cpu_millicores - w.committed_cpu_millicores,
            available_memory=w.total_memory_bytes - w.committed_mem,
            available_gpus=w.total_gpu_count - w.committed_gpu,
            available_tpus=w.total_tpu_count - w.committed_tpu,
        )
        for w in workers
    ]


def insert_task_profile(
    db: ControllerDB, task_id: str, profile_data: bytes, captured_at: Timestamp, profile_kind: str = "cpu"
) -> None:
    """Insert a captured profile snapshot for a task.

    The DB trigger caps profiles at 10 per (task_id, profile_kind), evicting the oldest automatically.
    """
    db.execute(
        "INSERT INTO profiles.task_profiles (task_id, profile_data, captured_at_ms, profile_kind) VALUES (?, ?, ?, ?)",
        (task_id, profile_data, captured_at.epoch_ms(), profile_kind),
    )


def get_task_profiles(
    db: ControllerDB, task_id: str, profile_kind: str | None = None
) -> list[tuple[bytes, Timestamp, str]]:
    """Return stored profile snapshots for a task, newest first.

    Args:
        db: Controller database.
        task_id: Task wire string.
        profile_kind: If set, filter to this kind (e.g. "cpu", "memory"). Returns all kinds when None.
    """
    if profile_kind is not None:
        query = (
            "SELECT profile_data, captured_at_ms, profile_kind FROM profiles.task_profiles"
            " WHERE task_id = ? AND profile_kind = ? ORDER BY id DESC"
        )
        params: tuple[str, ...] = (task_id, profile_kind)
    else:
        query = (
            "SELECT profile_data, captured_at_ms, profile_kind FROM profiles.task_profiles"
            " WHERE task_id = ? ORDER BY id DESC"
        )
        params = (task_id,)
    with db.read_snapshot() as q:
        rows = q.raw(query, params, decoders={"captured_at_ms": decode_timestamp_ms})
    return [(row.profile_data, row.captured_at_ms, row.profile_kind) for row in rows]
