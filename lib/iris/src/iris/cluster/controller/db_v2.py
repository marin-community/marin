# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy-backed controller database wrapper.

Hosts the SA ``Engine`` factory, the ``Tx`` wrapper, and the two transaction
context managers (``write_transaction`` / ``read_snapshot``) used by the
v2 data layer. The engine is pooled (``QueuePool(32+4)``) and shared
between readers and writers; serialization between writers is enforced by
an external ``threading.RLock`` supplied by the caller, mirroring today's
``ControllerDB._lock`` discipline.

Post-commit hooks registered via ``Tx.register`` fire *under the write
lock*, after ``COMMIT``. That keeps the atomicity contract from today's
``ControllerDB.transaction()`` byte-for-byte: a concurrent thread cannot
observe the SQL-committed-but-cache-not-yet-updated window because the
lock is held until every hook has run.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.cursor import CursorResult


def _make_engine(db_path: Path, auth_db_path: Path | None) -> Engine:
    """Build the controller's SA engine.

    ``isolation_level="AUTOCOMMIT"`` disables SA's autoBEGIN behaviour so
    callers can issue ``BEGIN`` / ``BEGIN IMMEDIATE`` explicitly — matching
    today's ``ControllerDB.transaction()`` discipline. (SA 2.0 uses the
    string ``"AUTOCOMMIT"`` here; ``None`` is not the same and leaves SA
    issuing implicit ROLLBACKs that conflict with our manual BEGIN.)

    A single ``QueuePool`` (32 + 4 overflow) serves both readers and
    writers; concurrent writes are gated by an external ``RLock`` passed
    into ``write_transaction``.

    The ``connect`` listener installs the four PRAGMAs from
    ``ControllerDB._configure`` and, when ``auth_db_path`` is provided,
    attaches the auth database as ``auth`` (matches today's startup at
    ``db.py:311`` and per-read-conn setup at ``db.py:392``).
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5.0},
        pool_size=32,
        max_overflow=4,
        isolation_level="AUTOCOMMIT",
        future=True,
    )

    auth_path_str = str(auth_db_path) if auth_db_path is not None else None

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pyrefly: ignore  # event hook
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

    return engine


class Tx:
    """Wraps a SA ``Connection`` and accumulates post-commit hooks.

    Read paths use ``Tx.execute`` / ``Tx.executemany`` exactly the same as
    write paths; ``register`` is only meaningful inside ``write_transaction``
    (hooks fire after ``COMMIT`` while the write lock is still held).
    """

    def __init__(self, conn: Connection):
        self.conn = conn
        self._hooks: list[Callable[[], None]] = []

    def execute(self, stmt, params=None) -> CursorResult:
        """Execute ``stmt``. ``params`` may be a dict or omitted."""
        return self.conn.execute(stmt, params or {})

    def executemany(self, stmt, params_list) -> CursorResult:
        """Execute ``stmt`` repeatedly against ``params_list``."""
        return self.conn.execute(stmt, params_list)

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
def write_transaction(engine: Engine, write_lock: threading.RLock) -> Iterator[Tx]:
    """Open a write transaction backed by ``engine``.

    Acquires ``write_lock`` (matching today's ``ControllerDB._lock``),
    checks out a connection, emits ``BEGIN IMMEDIATE``, yields a ``Tx``,
    and commits on clean exit. Post-commit hooks registered via
    ``Tx.register`` fire **while the lock is still held**, preserving the
    atomicity contract from today's ``ControllerDB.transaction()`` at
    ``db.py:476``.
    """
    write_lock.acquire()
    conn: Connection | None = None
    try:
        conn = engine.connect()
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
def read_snapshot(engine: Engine) -> Iterator[Tx]:
    """Open a read-only snapshot.

    Sets ``PRAGMA query_only = ON`` on the connection (so accidental writes
    raise), opens a ``BEGIN`` transaction for snapshot isolation, yields a
    ``Tx``, and rolls back on exit. ``query_only`` is cleared before the
    connection returns to the pool.
    """
    conn = engine.connect()
    try:
        conn.execute(text("PRAGMA query_only = ON"))
        conn.execute(text("BEGIN"))
        try:
            yield Tx(conn)
        finally:
            conn.execute(text("ROLLBACK"))
            conn.execute(text("PRAGMA query_only = OFF"))
    finally:
        conn.close()
