# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sqlite3
from pathlib import Path


def test_locality_migration_preserves_existing_jobs_and_attempts():
    path = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0055_coscheduling_groups.py"
    spec = importlib.util.spec_from_file_location("m0055", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE job_config (job_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO job_config VALUES ('/user/job')")
        conn.execute("CREATE TABLE task_attempts (task_id TEXT, attempt_id INTEGER)")
        conn.execute("INSERT INTO task_attempts VALUES ('/user/job/0', 0)")
        migration.migrate(conn)
        migration.migrate(conn)
        assert conn.execute("SELECT * FROM job_config").fetchall() == [("/user/job", "[]")]
        assert conn.execute("SELECT * FROM task_attempts").fetchall() == [("/user/job/0", 0, "{}")]
