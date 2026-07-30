# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0050_drop_controller_secrets``.

The runner's convergence tests only ever migrate a DB whose baseline never
created the table, so they cover the no-op path alone. These cover the forward
work: a legacy DB that still carries ``controller_secrets`` in its attached
``auth`` database loses it, and a re-run after a crash is safe.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0050_drop_controller_secrets.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0050", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_db(tmp_path: Path) -> sqlite3.Connection:
    """A main DB with the auth sidecar attached, holding a populated secrets table."""
    conn = sqlite3.connect(tmp_path / "controller.sqlite3")
    conn.execute("ATTACH DATABASE ? AS auth", (str(tmp_path / "auth.sqlite3"),))
    conn.execute(
        "CREATE TABLE auth.controller_secrets ("
        "key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL, created_at_ms INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO auth.controller_secrets (key, value, created_at_ms) VALUES ('jwt_signing_key', 'hmac', 1000)"
    )
    conn.commit()
    return conn


def _auth_tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM auth.sqlite_master WHERE type = 'table'")}


def test_migration_0050_drops_the_secrets_table_from_the_auth_db(tmp_path: Path):
    conn = _legacy_db(tmp_path)

    _load_migration().migrate(conn)
    conn.commit()

    assert _auth_tables(conn) == set()
    conn.close()


def test_migration_0050_is_idempotent_on_repeat(tmp_path: Path):
    conn = _legacy_db(tmp_path)
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)
    conn.commit()

    assert _auth_tables(conn) == set()
    conn.close()
