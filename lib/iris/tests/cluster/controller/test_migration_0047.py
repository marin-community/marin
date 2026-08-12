# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0047_bundle_init_image``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0047_bundle_init_image.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("migration_0047", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0047_preserves_existing_jobs_and_is_idempotent():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE job_config (job_id TEXT PRIMARY KEY, task_image TEXT NOT NULL DEFAULT '')")
    connection.execute("INSERT INTO job_config (job_id, task_image) VALUES ('/user/job', 'task:tag')")
    migration = _load_migration()

    migration.migrate(connection)
    migration.migrate(connection)

    assert connection.execute("SELECT job_id, task_image, bundle_init_image FROM job_config").fetchall() == [
        ("/user/job", "task:tag", "")
    ]
