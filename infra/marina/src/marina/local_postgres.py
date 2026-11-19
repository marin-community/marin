# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disposable Postgres for local Marina processes."""

import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import sqlalchemy

from marina.journeys import free_port

# Keep the Hub name so CI's Docker registry mirror can fall back on a cache miss.
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


def _terminate(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def _cleanup_on_termination() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _remove_container(name: str) -> None:
    removal = subprocess.run(
        ["docker", "rm", "--force", "--volumes", name], capture_output=True, check=False, timeout=30
    )
    if removal.returncode:
        # --rm can already have removed a container that exited during startup.
        # A daemon/cleanup failure must still fail the run when removal is unproven.
        remaining = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        if remaining.stdout.strip():
            removal.check_returncode()


@contextmanager
def docker_database() -> Iterator[str]:
    """Remove the owned Postgres container and volume on context exit.

    Main-thread callers also handle SIGTERM; worker threads keep the process's
    existing signal handling.
    """
    port = free_port()
    name = f"marina-postgres-{uuid.uuid4().hex[:8]}"
    with _cleanup_on_termination():
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--rm",
                    "--name",
                    name,
                    "--label",
                    f"marina.local-postgres.owner-pid={os.getpid()}",
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
            _wait_ready(url)
            yield url
        finally:
            _remove_container(name)
