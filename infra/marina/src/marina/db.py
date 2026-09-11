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
from enum import StrEnum

import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy import event, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

POOL_SIZE = 5
GRANT_LOCK_TIMEOUT = "5s"
DATABASE_URL_ENV = "MARINA_DATABASE_URL"
CLOUDSQL_CONNECTION_ENV = "CLOUDSQL_CONNECTION"
PGDATABASE_ENV = "PGDATABASE"
PGUSER_ENV = "PGUSER"
RUNNER_LOCK_PREFIX = "marina-runner:"
MIGRATION_LOCK_NAME = "marina-migrations"


class LockAcquisition(StrEnum):
    BLOCKING = "pg_advisory_lock"
    NONBLOCKING = "pg_try_advisory_lock"


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
        grant_read_on_connection(conn, schema, role)


def grant_read_on_connection(connection: Connection, schema: str, role: str) -> None:
    """Let ``role`` read existing and future relations in ``schema`` within a caller transaction."""
    connection.execute(text(f"SET LOCAL lock_timeout = '{GRANT_LOCK_TIMEOUT}'"))
    connection.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
    connection.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
    connection.execute(text(f'GRANT SELECT ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{role}"'))
    connection.execute(text(f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" GRANT SELECT ON TABLES TO "{role}"'))
    connection.execute(text(f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" GRANT SELECT ON SEQUENCES TO "{role}"'))


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


def engine_for_role(spec: DatabaseSpec, schema: str, role: str) -> Engine:
    """An engine whose connections assume a pre-provisioned role and schema."""
    engine = _bare_engine(spec)

    @event.listens_for(engine, "connect")
    def _set_role_and_search_path(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET ROLE "{role}"')
        cursor.execute(f'SET search_path TO "{schema}", public')
        cursor.close()
        dbapi_connection.commit()

    return engine


@contextmanager
def _advisory_lock(spec: DatabaseSpec, name: str, acquisition: LockAcquisition) -> Iterator[bool]:
    engine = _bare_engine(spec)
    try:
        with engine.connect() as conn:
            if conn.dialect.name != "postgresql":
                yield True
                return
            result = conn.execute(
                text(f"SELECT {acquisition.value}(hashtext(:name))"),
                {"name": name},
            ).scalar()
            acquired = acquisition is LockAcquisition.BLOCKING or bool(result)
            conn.commit()
            try:
                yield acquired
            finally:
                if acquired:
                    conn.execute(
                        text("SELECT pg_advisory_unlock(hashtext(:name))"),
                        {"name": name},
                    )
                    conn.commit()
    finally:
        engine.dispose()


@contextmanager
def runner_lock(spec: DatabaseSpec, runner: str) -> Iterator[bool]:
    """Try to lease a scheduled runner, returning false when another execution owns it."""
    with _advisory_lock(spec, f"{RUNNER_LOCK_PREFIX}{runner}", LockAcquisition.NONBLOCKING) as acquired:
        yield acquired


@contextmanager
def deployment_runner_lock(spec: DatabaseSpec, runner: str) -> Iterator[bool]:
    """Wait to lease the runner used for deployment migrations."""
    with _advisory_lock(spec, f"{RUNNER_LOCK_PREFIX}{runner}", LockAcquisition.BLOCKING) as acquired:
        yield acquired


@contextmanager
def migration_lock(spec: DatabaseSpec) -> Iterator[None]:
    """Serialize schema migrations across every runner and deployment."""
    with _advisory_lock(spec, MIGRATION_LOCK_NAME, LockAcquisition.BLOCKING) as acquired:
        assert acquired
        yield
