# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime authority storage and replica coordination for grant services."""

import hashlib
import ssl
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.pool import NullPool


def _lock_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], signed=True)


class GrantDatabase:
    """Serialize decisions and mutations, and coordinate lifecycle workers.

    PostgreSQL transaction locks provide a single order for grant creation,
    ancestor checks, revocation and idempotency. Session locks coordinate work
    that includes remote calls without holding a database transaction open.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self._local_lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _lock_key("marin-auth-ledger")})
            yield connection

    @contextmanager
    def serialize(self, name: str) -> Iterator[None]:
        """Hold a named operation lock, including across remote API calls."""
        if self.engine.dialect.name != "postgresql":
            with self._local_lock:
                yield
            return
        with self.engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _lock_key(name)})
            connection.commit()
            try:
                yield
            finally:
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _lock_key(name)})
                connection.commit()

    @contextmanager
    def leadership(self, name: str) -> Iterator[bool]:
        """Attempt one lifecycle pass; a disconnected worker loses leadership."""
        if self.engine.dialect.name != "postgresql":
            acquired = self._local_lock.acquire(blocking=False)
            try:
                yield acquired
            finally:
                if acquired:
                    self._local_lock.release()
            return
        with self.engine.connect() as connection:
            acquired = connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": _lock_key(name)}
            ).scalar_one()
            connection.commit()
            try:
                yield bool(acquired)
            finally:
                if acquired:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _lock_key(name)})
                    connection.commit()

    def close(self) -> None:
        self.engine.dispose()


class SqliteGrantDatabase(GrantDatabase):
    """Local single-replica storage; SQLite still serializes separate writers."""

    def __init__(self, path: Path):
        self.path = path
        engine = create_engine(f"sqlite:///{path}", poolclass=NullPool, connect_args={"timeout": 30})
        event.listen(engine, "connect", _configure_sqlite)
        event.listen(engine, "begin", _begin_sqlite)
        super().__init__(engine)


def _configure_sqlite(connection, _record) -> None:
    connection.isolation_level = None
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")


def _begin_sqlite(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE")


class PostgresGrantDatabase(GrantDatabase):
    """Shared storage with transaction and session locks across broker replicas."""

    def __init__(self, url: str, *, ssl_context: ssl.SSLContext | None):
        engine = create_engine(
            url, isolation_level="READ COMMITTED", poolclass=NullPool, connect_args={"ssl_context": ssl_context}
        )
        if engine.dialect.name != "postgresql" or engine.dialect.driver != "pg8000":
            raise ValueError("The grant database requires a postgresql+pg8000 URL")
        super().__init__(engine)
