# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0048_endpoints_drop_job_fk``.

Builds a pre-migration ``endpoints`` table carrying the foreign keys to
``jobs``/``tasks`` and asserts the migration rebuilds it without them — preserving
rows and every index — so a federation parent can store an endpoint absorbed from a
child (whose job/task rows do not exist locally). Also asserts idempotency and a
no-op on a DB already without the FKs.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0048_endpoints_drop_job_fk.py"

_ENDPOINT_COLUMNS = (
    "endpoint_id, name, address, job_id, task_id, metadata_json, registered_at_ms, lease_deadline_ms, access, peer_id"
)

# Pre-migration shape: endpoints.job_id/task_id reference jobs/tasks with CASCADE.
_OLD_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY);
CREATE TABLE tasks (task_id VARCHAR PRIMARY KEY);
CREATE TABLE endpoints (
    endpoint_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    address VARCHAR NOT NULL,
    job_id VARCHAR NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    task_id VARCHAR REFERENCES tasks (task_id) ON DELETE CASCADE,
    metadata_json VARCHAR NOT NULL,
    registered_at_ms INTEGER NOT NULL,
    lease_deadline_ms INTEGER,
    access INTEGER,
    peer_id VARCHAR,
    PRIMARY KEY (endpoint_id)
);
CREATE INDEX idx_endpoints_name ON endpoints (name);
CREATE INDEX idx_endpoints_task ON endpoints (task_id);
CREATE INDEX idx_endpoints_job_id ON endpoints (job_id);
CREATE INDEX idx_endpoints_peer_id ON endpoints (peer_id);
"""

# Current shape: same columns and indexes, no FKs (how a fresh baseline builds it).
_CURRENT_SCHEMA = """
CREATE TABLE endpoints (
    endpoint_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    address VARCHAR NOT NULL,
    job_id VARCHAR NOT NULL,
    task_id VARCHAR,
    metadata_json VARCHAR NOT NULL,
    registered_at_ms INTEGER NOT NULL,
    lease_deadline_ms INTEGER,
    access INTEGER,
    peer_id VARCHAR,
    PRIMARY KEY (endpoint_id)
);
CREATE INDEX idx_endpoints_name ON endpoints (name);
CREATE INDEX idx_endpoints_task ON endpoints (task_id);
CREATE INDEX idx_endpoints_job_id ON endpoints (job_id);
CREATE INDEX idx_endpoints_peer_id ON endpoints (peer_id);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0048", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fk_targets(conn: sqlite3.Connection) -> set[str]:
    return {row[2] for row in conn.execute("PRAGMA foreign_key_list(endpoints)")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='endpoints' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _insert_endpoint(conn: sqlite3.Connection, endpoint_id: str, name: str, job_id: str, task_id: str, peer_id) -> None:
    conn.execute(
        f"INSERT INTO endpoints ({_ENDPOINT_COLUMNS}) VALUES (?, ?, 'h:1', ?, ?, '{{}}', 1, NULL, 1, ?)",
        (endpoint_id, name, job_id, task_id, peer_id),
    )


def test_migration_0048_drops_the_job_fk_and_keeps_rows_and_indexes():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id) VALUES ('/u/a')")
    conn.execute("INSERT INTO tasks (task_id) VALUES ('/u/a/0')")
    _insert_endpoint(conn, "e1", "/serve/foo", "/u/a", "/u/a/0", None)
    conn.commit()
    assert _fk_targets(conn) == {"jobs", "tasks"}

    _load_migration().migrate(conn)
    conn.commit()

    assert _fk_targets(conn) == set()
    assert conn.execute("SELECT name FROM endpoints WHERE endpoint_id='e1'").fetchone()[0] == "/serve/foo"
    assert _indexes(conn) == {
        "idx_endpoints_name",
        "idx_endpoints_task",
        "idx_endpoints_job_id",
        "idx_endpoints_peer_id",
    }

    # An endpoint absorbed from a child — job/task ids that name no local row — now
    # inserts even with FK enforcement on, which the old FK would have rejected.
    _insert_endpoint(conn, "e2", "/serve/remote", "/child/job", "/child/job/0", "cw")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM endpoints").fetchone()[0] == 2
    conn.close()


def test_migration_0048_is_idempotent_on_repeat():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_OLD_SCHEMA)
    conn.commit()

    migration = _load_migration()
    migration.migrate(conn)
    conn.commit()
    migration.migrate(conn)  # second run must not error
    conn.commit()

    assert _fk_targets(conn) == set()
    conn.close()


def _schema_objects(conn: sqlite3.Connection) -> dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}


def test_migration_0048_is_noop_on_current_schema():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_CURRENT_SCHEMA)
    conn.commit()
    before = _schema_objects(conn)

    _load_migration().migrate(conn)
    conn.commit()

    assert _schema_objects(conn) == before
    conn.close()
