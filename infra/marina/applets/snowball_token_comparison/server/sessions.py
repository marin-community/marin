# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped API session keys, issued and revoked by the run owner."""

import hashlib
from contextvars import ContextVar

from sqlalchemy import text
from sqlalchemy.engine import Engine

SESSION_HEADER = "X-Lab-Session-Key"
session_run: ContextVar[str | None] = ContextVar("lab_session_run", default=None)


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def session_run_id(engine: Engine, key: str) -> str | None:
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT run_id FROM runs WHERE worker_session_hash=:hash"), {"hash": key_hash(key)}
        ).scalar()
