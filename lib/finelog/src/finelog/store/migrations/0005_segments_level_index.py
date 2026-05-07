# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create the ``(namespace, level, min_seq)`` index on ``segments``.

Split out from :mod:`0003_segment_level` because DuckDB rejects
``CREATE INDEX`` in a transaction that has outstanding ``UPDATE``s on
the same table, and rejects ``DROP COLUMN`` for a column positioned
before any indexed column. Migration 0003 handles the schema rewrite
and backfill; this migration runs in a fresh transaction with no
pending updates, so ``CREATE INDEX`` succeeds.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    conn.execute("CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq)")
