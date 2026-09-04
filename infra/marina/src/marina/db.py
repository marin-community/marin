# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One Postgres for every app, one schema per app.

The kernel knows one database. Each app gets its own engine on it whose every connection
has ``search_path`` set to the app's schema (then ``public``, where extensions such as
pgvector live), so an app's tables, migration ledger, and queries are unqualified and
land in its own schema. In production the database is
reached through the Cloud SQL connector with IAM authentication as the service's own
account; locally and in tests it is a plain SQLAlchemy URL.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

POOL_SIZE = 5
GRANT_LOCK_TIMEOUT = "5s"
DATABASE_URL_ENV = "MARINA_DATABASE_URL"
CLOUDSQL_CONNECTION_ENV = "CLOUDSQL_CONNECTION"
PGDATABASE_ENV = "PGDATABASE"
PGUSER_ENV = "PGUSER"
RUNNER_LOCK_PREFIX = "marina-runner:"


@dataclass(frozen=True)
class CloudSqlDatabase:
    """A Cloud SQL Postgres reached through the connector with IAM authentication."""

    connection_name: str
    database: str
    user: str


@dataclass(frozen=True)
class UrlDatabase:
    """A Postgres reached by SQLAlchemy URL (local development and tests)."""

    url: str


DatabaseSpec = CloudSqlDatabase | UrlDatabase


def database_from_env(environ: Mapping[str, str]) -> DatabaseSpec | None:
    """The database the environment names, or None when it names none.

    ``MARINA_DATABASE_URL`` wins; otherwise ``CLOUDSQL_CONNECTION``, ``PGDATABASE`` and
    ``PGUSER`` together name a Cloud SQL instance reached with IAM authentication.
    """
    url = environ.get(DATABASE_URL_ENV)
    if url:
        return UrlDatabase(url=url)
    connection = environ.get(CLOUDSQL_CONNECTION_ENV)
    if not connection:
        return None
    missing = [name for name in (PGDATABASE_ENV, PGUSER_ENV) if not environ.get(name)]
    if missing:
        raise ValueError(f"{CLOUDSQL_CONNECTION_ENV} is set but {missing} are not")
    return CloudSqlDatabase(connection_name=connection, database=environ[PGDATABASE_ENV], user=environ[PGUSER_ENV])


def _bare_engine(spec: DatabaseSpec) -> Engine:
    match spec:
        case UrlDatabase(url=url):
            return sqlalchemy.create_engine(url, pool_size=POOL_SIZE, pool_pre_ping=True)
        case CloudSqlDatabase(connection_name=name, database=database, user=user):
            connector = Connector(refresh_strategy="lazy")

            def connect():
                return connector.connect(name, "pg8000", user=user, db=database, enable_iam_auth=True)

            engine = sqlalchemy.create_engine(
                "postgresql+pg8000://", creator=connect, pool_size=POOL_SIZE, pool_pre_ping=True
            )
            event.listens_for(engine, "engine_disposed")(lambda _engine: connector.close())
            return engine
    raise ValueError(f"unknown database spec {spec!r}")


def schema_name(app: str) -> str:
    """The Postgres schema for an app: hyphens become underscores."""
    return app.replace("-", "_")


def grant_read(engine: Engine, schema: str, role: str) -> None:
    """Let ``role`` read ``schema``, including tables an app adds later.

    People hold a Cloud SQL group login for queries the apps do not expose; every write
    still goes through an app as the service account.

    ``GRANT`` takes each table's exclusive lock, which would otherwise queue behind a long
    write and hold every reader behind it, so the grant gives up rather than wait.
    """
    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL lock_timeout = '{GRANT_LOCK_TIMEOUT}'"))
        conn.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        conn.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        conn.execute(text(f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" GRANT SELECT ON TABLES TO "{role}"'))


def engine_for(spec: DatabaseSpec, app: str) -> Engine:
    """An engine with its own pool whose connections search the app's schema first.

    The schema is created if it does not exist, so the first app to run owns no setup step.
    """
    schema = schema_name(app)
    engine = _bare_engine(spec)

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        # Valid before the schema exists: Postgres resolves search_path at lookup time.
        cursor.execute(f'SET search_path TO "{schema}", public')
        cursor.close()
        # pg8000 opens a transaction implicitly; without a commit the pool's reset-on-return
        # rollback undoes the SET and the connection falls back to public.
        dbapi_connection.commit()

    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    return engine


@contextmanager
def runner_lock(spec: DatabaseSpec, runner: str, *, wait: bool) -> Iterator[bool]:
    """Hold a database-wide advisory lock for one Marina runner execution.

    Scheduled executions use a non-blocking lock and skip overlap. A migration-only deploy
    execution waits so the new service revision cannot start before its migrations finish.
    Non-Postgres test databases have no cross-process advisory-lock primitive and always enter.
    """
    engine = _bare_engine(spec)
    try:
        with engine.connect() as conn:
            if conn.dialect.name != "postgresql":
                yield True
                return
            operation = "pg_advisory_lock" if wait else "pg_try_advisory_lock"
            result = conn.execute(
                text(f"SELECT {operation}(hashtext(:name))"),
                {"name": f"{RUNNER_LOCK_PREFIX}{runner}"},
            ).scalar()
            acquired = True if wait else bool(result)
            conn.commit()
            try:
                yield acquired
            finally:
                if acquired:
                    conn.execute(
                        text("SELECT pg_advisory_unlock(hashtext(:name))"),
                        {"name": f"{RUNNER_LOCK_PREFIX}{runner}"},
                    )
                    conn.commit()
    finally:
        engine.dispose()
