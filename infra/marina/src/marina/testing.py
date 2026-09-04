# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A throwaway Postgres for tests and journeys.

``MARINA_TEST_DATABASE_URL`` names a database that is already up. Otherwise a pgvector
container is started on a free port for the session and removed at exit, so two runs
coexist and nothing is left behind.
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy

from marina.local_postgres import docker_database

TEST_DATABASE_URL_ENV = "MARINA_TEST_DATABASE_URL"


@contextmanager
def _fresh_database(server_url: str) -> Iterator[str]:
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
        with _fresh_database(given) as url:
            yield url
        return
    with docker_database() as url:
        yield url
