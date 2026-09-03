# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A throwaway Postgres for tests and journeys.

``MARINA_TEST_DATABASE_URL`` names a database that is already up. Otherwise a pgvector
container is started on a free port for the session and removed at exit, so two runs
coexist and nothing is left behind.
"""

import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy

from marina.journeys import free_port

TEST_DATABASE_URL_ENV = "MARINA_TEST_DATABASE_URL"
POSTGRES_IMAGE = "pgvector/pgvector:0.8.5-pg16-bookworm"
POSTGRES_PASSWORD = "marina"
START_TIMEOUT = 60.0


def _wait_ready(url: str) -> None:
    deadline = time.monotonic() + START_TIMEOUT
    engine = sqlalchemy.create_engine(url)
    refused: Exception | None = None
    try:
        while time.monotonic() < deadline:
            try:
                with engine.connect() as conn:
                    conn.execute(sqlalchemy.text("SELECT 1"))
                return
            except sqlalchemy.exc.SQLAlchemyError as error:
                # A container that is still starting refuses the connection; anything else
                # (bad credentials, bad URL) refuses it just as often, so keep the last one
                # to report instead of a bare timeout.
                refused = error
                time.sleep(0.3)
        raise RuntimeError(f"postgres at {url} did not become ready within {START_TIMEOUT}s") from refused
    finally:
        engine.dispose()


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
    port = free_port()
    name = f"marina-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "-e",
            f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
            "-e",
            "POSTGRES_DB=marina",
            "-p",
            f"127.0.0.1:{port}:5432",
            POSTGRES_IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    url = f"postgresql+pg8000://postgres:{POSTGRES_PASSWORD}@127.0.0.1:{port}/marina"
    try:
        _wait_ready(url)
        yield url
    finally:
        subprocess.run(["docker", "rm", "--force", name], check=False, capture_output=True)
