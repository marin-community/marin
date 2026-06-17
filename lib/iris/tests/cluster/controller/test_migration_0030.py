# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0030_index_workers_slice_id``.

Builds a pre-migration ``workers`` table without the index and asserts the
migration creates ``idx_workers_slice_id`` and is idempotent on re-run.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0030_index_workers_slice_id.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0030", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_migration_0030_adds_workers_slice_id_index():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE workers (worker_id VARCHAR PRIMARY KEY, slice_id VARCHAR NOT NULL DEFAULT '')")
    assert "idx_workers_slice_id" not in _indexes(conn)

    migration = _load_migration()
    migration.migrate(conn)
    migration.migrate(conn)  # idempotent re-run

    assert "idx_workers_slice_id" in _indexes(conn)
    conn.close()
