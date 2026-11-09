# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A throwaway Postgres for tests and journeys.

``MARINA_TEST_DATABASE_URL`` names a database that is already up. Otherwise a pgvector
container is started on a free port for the session and removed at exit, so two runs
coexist and nothing is left behind.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import sqlalchemy

from marina.applets import validate_query
from marina.client import AppletQuery
from marina.db import DatabaseSpec, engine_for
from marina.local_postgres import docker_database

TEST_DATABASE_URL_ENV = "MARINA_TEST_DATABASE_URL"


@dataclass(frozen=True)
class SourceServices:
    """Backend services for unit tests that manage their own schema."""

    name: str
    data_url: str
    database: DatabaseSpec | None

    def engine(self) -> sqlalchemy.Engine:
        if self.database is None:
            raise RuntimeError(f"app {self.name} needs a database")
        return engine_for(self.database, self.name)


@dataclass(frozen=True)
class EngineServices:
    """Backend services for a unit test that hands the app an engine it already built."""

    database_engine: sqlalchemy.Engine

    def engine(self) -> sqlalchemy.Engine:
        return self.database_engine


@contextmanager
def migrated_schema(
    database_url: str, prefix: str, migrate: Callable[[sqlalchemy.Connection], None]
) -> Iterator[sqlalchemy.Engine]:
    """Yield an engine whose connections use a fresh schema that ``migrate`` has set up."""
    admin = sqlalchemy.create_engine(database_url)
    schema = f"{prefix}_{uuid.uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(sqlalchemy.text(f"CREATE SCHEMA {schema}"))
    engine = sqlalchemy.create_engine(database_url)

    @sqlalchemy.event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")

    try:
        with engine.begin() as connection:
            migrate(connection)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(sqlalchemy.text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def engine_query(engine: sqlalchemy.Engine) -> AppletQuery:
    """Stand in for an app's SQL endpoint: validate each statement, then run it on ``engine``."""

    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        validate_query(sql)
        with engine.begin() as connection:
            result = connection.execute(sqlalchemy.text(sql), parameters)
            return {
                "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                "row_count": result.rowcount,
            }

    return query


@contextmanager
def isolated_database(server_url: str) -> Iterator[str]:
    """Create and remove one database on an existing local test server."""
    name = f"marina_test_{uuid.uuid4().hex[:8]}"
    admin = sqlalchemy.create_engine(server_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(sqlalchemy.text(f'CREATE DATABASE "{name}"'))
        try:
            yield sqlalchemy.make_url(server_url).set(database=name).render_as_string(hide_password=False)
        finally:
            with admin.connect() as conn:
                conn.execute(sqlalchemy.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


@contextmanager
def test_database() -> Iterator[str]:
    """Yield a SQLAlchemy URL for a fresh database this process may freely write to.

    With ``MARINA_TEST_DATABASE_URL`` set, a throwaway database is created on that server
    and dropped afterwards; otherwise a container is started for the run.
    """
    given = os.environ.get(TEST_DATABASE_URL_ENV)
    if given:
        with isolated_database(given) as url:
            yield url
        return
    with docker_database() as url:
        yield url
