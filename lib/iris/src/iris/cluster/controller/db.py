# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite access layer and typed query models for controller state."""

from __future__ import annotations

import logging
import queue
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from iris.cluster.controller.store import UserBudget
from iris.rpc import job_pb2
from rigging.timing import Timestamp

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

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
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


# Failure states that trigger coscheduled sibling cascades.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
    }
)


class TransactionCursor:
    """Wraps a raw sqlite3.Cursor for use within controller transactions.

    Post-commit hooks registered via :meth:`on_commit` run after the wrapping
    ``ControllerDB.transaction()`` block commits successfully. They are used
    by caches (e.g. ``EndpointStore``) to update in-memory state atomically
    with the DB write: rollback suppresses the hook so memory never drifts
    from disk.
    """

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor
        self._commit_hooks: list[Callable[[], None]] = []

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Raw SQL escape hatch."""
        return self._cursor.execute(sql, params)

    def executemany(self, sql: str, params: Iterable[tuple]) -> sqlite3.Cursor:
        """Raw SQL batch escape hatch."""
        return self._cursor.executemany(sql, params)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        """Raw SQL script escape hatch."""
        return self._cursor.executescript(sql)

    def on_commit(self, hook: Callable[[], None]) -> None:
        """Register ``hook`` to run after the transaction commits successfully."""
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


class ControllerDB:
    """Thread-safe SQLite wrapper with typed query and migration helpers."""

    _READ_POOL_SIZE = 32
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
            self._conn.executescript("PRAGMA main.incremental_vacuum")
            row = self._conn.execute("PRAGMA main.wal_checkpoint(TRUNCATE)").fetchone()
        return (int(row[0]), int(row[1]), int(row[2]))

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
        """Open an IMMEDIATE transaction and yield a TransactionCursor.

        On successful commit, any hooks registered via ``TransactionCursor.on_commit``
        fire while the write lock is still held — keeping in-memory caches
        (e.g. ``EndpointStore``) in sync with the DB without exposing a
        torn snapshot to concurrent readers.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            tx_cur = TransactionCursor(cur)
            try:
                yield tx_cur
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
                tx_cur._run_commit_hooks()

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

        if not pending:
            return

        logger.info("Applying %d pending migration(s): %s", len(pending), [p.name for p in pending])

        # Flip to fast-mode PRAGMAs for the duration of the migration loop.
        # Safe: migrations run at startup before any concurrent access, and a
        # crash re-runs the migration from schema_migrations. journal_mode
        # cannot change inside a transaction, so commit first and restore at
        # the end.
        self._conn.commit()
        self._conn.execute("PRAGMA synchronous=OFF")
        # journal_mode returns a row; consume it so the cursor is closed and
        # cannot hold a statement-level lock that would block wal_checkpoint.
        self._conn.execute("PRAGMA journal_mode=MEMORY").fetchall()
        self._conn.execute("PRAGMA temp_store=MEMORY")
        try:
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
        finally:
            self._conn.commit()
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA journal_mode=WAL").fetchall()
            # Checkpoint and truncate the WAL so the migration's write volume
            # does not linger as a giant WAL file that every subsequent reader
            # must walk to build a snapshot.
            busy, log_frames, checkpointed = self.wal_checkpoint()
            logger.info(
                "Post-migration wal_checkpoint(TRUNCATE): busy=%d log_frames=%d checkpointed=%d",
                busy,
                log_frames,
                checkpointed,
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

    def get_counter(self, key: str, cur: TransactionCursor) -> int:
        """Read an integer counter from meta. Returns 0 if unset."""
        row = cur.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return 0
        return int(row[0])

    def set_counter(self, key: str, value: int, cur: TransactionCursor) -> None:
        """Write an integer counter to meta inside the given transaction."""
        cur.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) " "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

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
        so writers on ``self._conn`` proceed concurrently under SQLite's
        WAL semantics -- no controller-level lock is held for the
        duration of the copy.  Batched page copying (``pages=500``)
        yields between steps so a sustained write stream cannot starve
        the backup.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(str(self._db_path), check_same_thread=False)
        try:
            self._configure(src)
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

    def _close_read_pool_connections(self) -> None:
        while True:
            try:
                self._read_pool.get_nowait().close()
            except queue.Empty:
                break

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
            self._close_read_pool_connections()
            self._conn.close()

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

            # Download profiles DB if present in source
            profiles_source = f"{source_dir_str}/{self.PROFILES_DB_FILENAME}"
            fs2, fs_path2 = fsspec.core.url_to_fs(profiles_source)
            if fs2.exists(fs_path2):
                profiles_tmp = self._profiles_db_path.with_suffix(".tmp")
                with fsspec.core.open(profiles_source, "rb") as src, open(profiles_tmp, "wb") as dst:
                    dst.write(src.read())
                self._remove_sidecars(self._profiles_db_path)
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


def batch_delete(
    db: ControllerDB,
    sql: str,
    params: tuple[object, ...],
    stopped: Callable[[], bool],
    pause_between_s: float,
) -> int:
    """Delete rows in batches, sleeping between transactions.

    Returns the total number of rows deleted.
    """
    total = 0
    while not stopped():
        with db.transaction() as cur:
            batch = cur.execute(sql, params).rowcount
        if batch == 0:
            break
        total += batch
        time.sleep(pause_between_s)
    return total
