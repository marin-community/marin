# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add per-segment ``key_column`` value bounds to the catalog.

Each registered ``Schema`` may declare a ``key_column`` — the column the
segment is sorted by inside its parquet file (see
``compaction_sort_keys`` in ``compactor.py``).
Tracking that column's min/max in the catalog lets future read paths
prune the segment list before opening any parquet footer.

Both columns are nullable: namespaces that don't declare a ``key_column``
(e.g. iris worker stats) leave them ``NULL``. ``reconcile_segments``
populates existing rows on the next process boot from the parquet
footers, so this migration does not backfill itself.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS min_key_value TEXT")
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS max_key_value TEXT")
