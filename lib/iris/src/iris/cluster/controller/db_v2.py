# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy-backed controller database wrapper.

Hosts the SA ``Engine`` factories, the ``Tx`` wrapper, and the two
transaction context managers (``write_transaction`` / ``read_snapshot``)
used by the v2 data layer.

The engine is split into a **write engine** and a **read engine**,
matching today's ``ControllerDB._init_read_pool`` design exactly:

* The write engine uses a tiny pool (size 1) so writes are funneled
  through a single connection. Serialization between writers is enforced
  by an external ``threading.RLock`` passed into ``write_transaction``.
* The read engine uses ``QueuePool(pool_size=32, max_overflow=4)`` with
  ``PRAGMA query_only = ON`` **pinned at connect time**. Pinning avoids
  toggling the pragma on every ``read_snapshot`` call — those pragma
  round-trips dominate the per-call cost of the slim reservation read.

Both engines use ``isolation_level="AUTOCOMMIT"`` so callers issue
``BEGIN`` / ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` explicitly,
mirroring today's ``ControllerDB`` discipline.

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


def _install_pragmas(dbapi_conn, auth_path_str: str | None) -> None:
    """Run the four startup PRAGMAs and ATTACH the auth DB if provided.

    Mirrors ``ControllerDB._configure`` + the per-conn ATTACH at
    ``db.py:313`` / ``db.py:399``.
    """
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

    Pool size 1: writes are serialized by an external ``RLock`` so the
    extra connections from the previous shared pool were unused.
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

    ``QueuePool(pool_size=32, max_overflow=4)`` matches today's
    ``ControllerDB._READ_POOL_SIZE = 32`` plus a small overflow.
    ``PRAGMA query_only = ON`` is pinned at **connect time** so that
    accidental writes raise, but ``read_snapshot`` does not pay the
    pragma round-trip cost on every call. ``isolation_level="AUTOCOMMIT"``
    lets callers open / close transactions with explicit BEGIN/ROLLBACK.
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
def write_transaction(write_engine: Engine, write_lock: threading.RLock) -> Iterator[Tx]:
    """Open a write transaction backed by ``write_engine``.

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
