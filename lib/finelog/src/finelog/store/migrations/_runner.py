# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations runner for the registry DuckDB sidecar.

Migration files live next to this module as ``NNNN_name.py`` and define a
``migrate(conn, *, data_dir)`` function. The runner discovers them in
filename order, filters out any already recorded in the
``schema_migrations`` table, and applies the remainder. After ``migrate``
returns the runner inserts the ``schema_migrations`` row.

The runner does **not** wrap ``migrate()`` in an outer transaction. DuckDB
rejects several useful sequences inside a single transaction — most
notably multiple schema-altering DDLs interleaved with DML on the same
table — so each migration manages its own atomicity. Two consequences:

* Migrations must be idempotent (``CREATE TABLE IF NOT EXISTS``,
  ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``, ``DROP COLUMN IF EXISTS``,
  conditional UPDATE filters). A crash mid-migration may leave partial
  side effects on disk; on the next open, the migration runs again and
  the idempotent statements converge.
* When a migration needs multi-statement atomicity for a block that
  DuckDB *does* accept (e.g. compaction's swap of input/output rows),
  it can call :func:`transactional` itself. The helper is exported for
  exactly this case.

The ``schema_migrations`` row insert lands only if ``migrate()`` returned
without raising, so a failed migration is naturally re-run on the next
open.
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
    over multiple statements (compaction's segment swap, a migration's
    multi-row backfill where partial visibility would be wrong).
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
    *,
    data_dir: Path | None = None,
) -> None:
    """Apply pending migrations from ``migrations_dir`` to ``conn``.

    ``migrations_dir`` defaults to this package's directory. ``data_dir``
    is the parent directory of the registry DB and is forwarded to every
    migration's ``migrate(conn, *, data_dir=...)`` signature; migrations
    that need to rename on-disk parquet files use it.

    Each ``migrate`` runs without an enclosing transaction (see module
    docstring). If it raises, the ``schema_migrations`` row is not
    inserted, so the migration will re-run on the next open — and must
    therefore be idempotent.
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
        migrate(conn, data_dir=data_dir)
        conn.execute(
            "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
            [path.name, int(time.time() * 1000)],
        )
        logger.info("finelog migration %s applied in %.3fs", path.name, time.monotonic() - t0)
