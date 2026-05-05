# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations runner for the registry DuckDB sidecar.

Migration files live next to this module as ``NNNN_name.py`` and define a
``migrate(conn)`` function. The runner discovers them in filename order,
filters out any already recorded in the ``schema_migrations`` table, and
applies the remainder. Each migration runs inside a transaction that also
inserts its ``schema_migrations`` row — a crash mid-migration leaves no
half-applied state.

Migrations must be idempotent (``CREATE TABLE IF NOT EXISTS``,
``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``, etc.). Existing deployments
may already have tables created by an earlier release that pre-dates this
runner; the first migration just records that v1 has been observed.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


@contextmanager
def transactional(conn: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    """Run a block inside a DuckDB ``BEGIN``/``COMMIT`` transaction.

    Any exception triggers ``ROLLBACK`` and propagates; a clean exit
    ``COMMIT``s. Useful for callers that want all-or-nothing semantics
    over multiple statements (compaction's segment swap, the migration
    runner's apply-and-record pair).
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _discover_migrations(migrations_dir: Path) -> list[Path]:
    """Migration files in numeric order. Anything starting with ``_`` (e.g.
    ``_runner.py``, ``__init__.py``) is skipped."""
    return [p for p in sorted(migrations_dir.glob("*.py")) if not p.name.startswith("_")]


def _ensure_schema_migrations_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name          TEXT PRIMARY KEY,
            applied_at_ms BIGINT NOT NULL
        )
        """
    )


def _load_migration(path: Path):
    """Load a migration module from a file path; return its ``migrate``."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"failed to load migration spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "migrate"):
        raise RuntimeError(f"migration {path.name} missing required ``migrate(conn)`` callable")
    return module.migrate


def apply_migrations(
    conn: duckdb.DuckDBPyConnection,
    migrations_dir: Path | None = None,
) -> None:
    """Apply pending migrations from ``migrations_dir`` to ``conn``.

    ``migrations_dir`` defaults to this package's directory so callers from
    ``RegistryDB`` don't need to know where the files live.
    """
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent

    _ensure_schema_migrations_table(conn)
    applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
    applied_stems = {Path(name).stem for name in applied}

    pending = [p for p in _discover_migrations(migrations_dir) if p.stem not in applied_stems]
    if not pending:
        return

    logger.info("Applying %d pending finelog registry migration(s): %s", len(pending), [p.name for p in pending])
    for path in pending:
        t0 = time.monotonic()
        migrate = _load_migration(path)
        with transactional(conn):
            migrate(conn)
            conn.execute(
                "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                [path.name, int(time.time() * 1000)],
            )
        logger.info("finelog migration %s applied in %.3fs", path.name, time.monotonic() - t0)
