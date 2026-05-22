# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add per-segment ``min_key_value`` / ``max_key_value`` bounds.

Tracks the min/max of each segment's ``key_column`` so reads can prune
without opening parquet footers. Nullable for namespaces that don't
declare a key column; ``reconcile_segments`` backfills existing rows on
the next boot.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from finelog.store.migrations._runner import transactional


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    with transactional(conn):
        conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS min_key_value TEXT")
        conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS max_key_value TEXT")
