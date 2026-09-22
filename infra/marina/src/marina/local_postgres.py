# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disposable Postgres for local Marina processes."""

import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy

from marina.journeys import free_port

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
            except (sqlalchemy.exc.SQLAlchemyError, OSError) as error:
                # A container that is still starting may refuse the connection or
                # reset it mid-handshake. Preserve the last error for a useful timeout.
                refused = error
                time.sleep(0.3)
        raise RuntimeError(f"postgres at {url} did not become ready within {START_TIMEOUT}s") from refused
    finally:
        engine.dispose()


@contextmanager
def docker_database() -> Iterator[str]:
    """Run disposable Postgres in Docker and yield its SQLAlchemy URL."""
    port = free_port()
    name = f"marina-postgres-{uuid.uuid4().hex[:8]}"
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
