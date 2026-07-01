# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0029_drop_reservations``.

Builds a representative pre-migration DB (reservation columns/indexes/CHECK, a
holder job, a real reservation job, a plain job, an "auto"-variant GPU reservation,
a FK-referenced task) and asserts the migration: deletes holders (cascading to
their tasks), converts real reservations into hard ``availability:<variant>``
constraints while skipping the unschedulable ``auto`` variant, strips the schema,
and leaves data + FK integrity intact — and is idempotent on re-run.
"""

import importlib.util
import json
import sqlite3
from pathlib import Path

from iris.cluster.constraints import ConstraintOp, availability_key
from iris.cluster.controller.codec import constraints_from_json
from iris.rpc import job_pb2

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0029_drop_reservations.py"

_OLD_SCHEMA = """
CREATE TABLE users (user_id VARCHAR PRIMARY KEY, created_at_ms INTEGER NOT NULL, role VARCHAR NOT NULL DEFAULT 'user');
CREATE TABLE jobs (
    job_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    parent_job_id VARCHAR,
    root_job_id VARCHAR NOT NULL,
    depth INTEGER NOT NULL,
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    root_submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    scheduling_deadline_epoch_ms INTEGER,
    error VARCHAR,
    exit_code INTEGER,
    num_tasks INTEGER NOT NULL,
    is_reservation_holder INTEGER NOT NULL,
    name VARCHAR DEFAULT '' NOT NULL,
    has_reservation INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT jobs_is_reservation_holder_check CHECK (is_reservation_holder IN (0, 1)),
    PRIMARY KEY (job_id),
    FOREIGN KEY(user_id) REFERENCES users (user_id),
    FOREIGN KEY(parent_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
);
CREATE INDEX idx_jobs_parent ON jobs (parent_job_id);
CREATE INDEX idx_jobs_has_reservation ON jobs (has_reservation, state) WHERE has_reservation = 1;
CREATE INDEX idx_jobs_reservation_holder ON jobs (job_id) WHERE is_reservation_holder = 1;
CREATE TABLE job_config (
    job_id VARCHAR NOT NULL,
    name VARCHAR DEFAULT '' NOT NULL,
    has_reservation INTEGER NOT NULL DEFAULT 0,
    constraints_json VARCHAR,
    reservation_json VARCHAR,
    PRIMARY KEY (job_id),
    FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
);
CREATE INDEX idx_job_config_has_reservation ON job_config (has_reservation, job_id) WHERE has_reservation = 1;
CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    state INTEGER NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
);
CREATE TABLE reservation_claims (worker_id VARCHAR PRIMARY KEY, job_id VARCHAR NOT NULL, entry_idx INTEGER NOT NULL);
"""


def _reservation_json(variant: str) -> str:
    """Build a ``reservation_json`` blob the way the old (now-removed) codec stored it."""
    return json.dumps({"entries": [{"resources": {"device": {"tpu": {"variant": variant}}}}]})


def _gpu_reservation_json(variant: str) -> str:
    """Same as ``_reservation_json`` but for a GPU device entry."""
    return json.dumps({"entries": [{"resources": {"device": {"gpu": {"variant": variant}}}}]})


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0029", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO users VALUES ('u1', 1, 'user')")

    def add_job(job_id, *, holder, has_reservation, name):
        conn.execute(
            "INSERT INTO jobs (job_id,user_id,parent_job_id,root_job_id,depth,state,submitted_at_ms,"
            "root_submitted_at_ms,num_tasks,is_reservation_holder,name,has_reservation) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "u1", None, job_id, 0, job_pb2.JOB_STATE_RUNNING, 10, 10, 1, holder, name, has_reservation),
        )

    # Holder job with a RUNNING task — must be deleted, cascading to the task.
    add_job("/u1/holder", holder=1, has_reservation=0, name="holder")
    conn.execute(
        "INSERT INTO job_config (job_id,name,has_reservation,constraints_json,reservation_json) VALUES (?,?,?,?,?)",
        ("/u1/holder", "holder", 0, "[]", _reservation_json("v5p-8")),
    )
    conn.execute("INSERT INTO tasks VALUES ('/u1/holder/0','/u1/holder',?)", (job_pb2.TASK_STATE_RUNNING,))

    # Real reservation job — its reservation converts to a hard availability constraint.
    add_job("/u1/real", holder=0, has_reservation=1, name="real")
    conn.execute(
        "INSERT INTO job_config (job_id,name,has_reservation,constraints_json,reservation_json) VALUES (?,?,?,?,?)",
        ("/u1/real", "real", 1, "[]", _reservation_json("v5litepod-16")),
    )

    # Plain job — untouched.
    add_job("/u1/plain", holder=0, has_reservation=0, name="plain")
    conn.execute(
        "INSERT INTO job_config (job_id,name,has_reservation,constraints_json,reservation_json) VALUES (?,?,?,?,?)",
        ("/u1/plain", "plain", 0, "[]", None),
    )

    # GPU reservation left at the "auto" (any-GPU) variant — must NOT become a hard
    # availability:auto constraint, since no worker/group ever advertises one.
    add_job("/u1/auto-gpu", holder=0, has_reservation=1, name="auto-gpu")
    conn.execute(
        "INSERT INTO job_config (job_id,name,has_reservation,constraints_json,reservation_json) VALUES (?,?,?,?,?)",
        ("/u1/auto-gpu", "auto-gpu", 1, "[]", _gpu_reservation_json("auto")),
    )

    conn.execute("INSERT INTO reservation_claims VALUES ('w1','/u1/real',0)")
    conn.commit()


def test_migration_0029_drops_reservations_and_converts_in_flight():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)
    migration.migrate(conn)  # idempotent re-run

    # Holder job + its task are gone (cascade).
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE job_id='/u1/holder'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE job_id='/u1/holder'").fetchone()[0] == 0

    # Real reservation became exactly one hard availability:<variant> constraint.
    real_json = conn.execute("SELECT constraints_json FROM job_config WHERE job_id='/u1/real'").fetchone()[0]
    constraints = constraints_from_json(real_json)
    avail = [c for c in constraints if c.key == availability_key("v5litepod-16")]
    assert len(avail) == 1, constraints
    assert avail[0].op == ConstraintOp.EXISTS
    assert avail[0].mode == job_pb2.CONSTRAINT_MODE_REQUIRED

    # Plain job's constraints are untouched.
    assert conn.execute("SELECT constraints_json FROM job_config WHERE job_id='/u1/plain'").fetchone()[0] == "[]"

    # "auto" GPU variant is skipped, not folded into an unschedulable availability:auto.
    auto_json = conn.execute("SELECT constraints_json FROM job_config WHERE job_id='/u1/auto-gpu'").fetchone()[0]
    assert constraints_from_json(auto_json) == []

    # Schema stripped: columns, indexes, table all gone.
    jcols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)")]
    ccols = [r[1] for r in conn.execute("PRAGMA table_info(job_config)")]
    tbls = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "is_reservation_holder" not in jcols and "has_reservation" not in jcols
    assert "has_reservation" not in ccols and "reservation_json" not in ccols
    assert "reservation_claims" not in tbls

    # FK integrity intact after the rebuild.
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()
