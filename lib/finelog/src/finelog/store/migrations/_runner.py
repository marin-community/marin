# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations runner for the registry DuckDB sidecar.

Migration files live next to this module as ``NNNN_name.py`` and define
a ``migrate(conn, *, data_dir)`` function. The runner applies them in
filename order, skipping any already recorded in ``schema_migrations``,
and inserts the ``schema_migrations`` row only after ``migrate`` returns
without raising.

Migrations run with no enclosing transaction — DuckDB rejects several
useful sequences (notably multiple DDLs + DML on the same table) inside
one — so each migration is responsible for atomicity and must be
idempotent. Migrations that want a multi-statement transaction can call
:func:`transactional` themselves.
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
    """``BEGIN``/``COMMIT`` around a block; ``ROLLBACK`` on exception."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _discover_migrations(migrations_dir: Path) -> list[Path]:
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
    """Apply pending migrations. ``data_dir`` is forwarded to migrations
    that need to rename on-disk parquet files."""
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
