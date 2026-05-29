# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace storage retention overrides.

One row per namespace that wants to override the cluster-wide eviction
caps. ``max_segments`` / ``max_bytes`` shadow the corresponding
:class:`finelog.store.compactor.CompactionConfig` fields;
``max_age_seconds`` has no default-config analogue and enables a new
age-based eviction path. NULL in any column means "inherit / disabled".

A namespace with no row in this table behaves exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from finelog.store.migrations._runner import transactional


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    with transactional(conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS storage_policies (
                namespace        TEXT PRIMARY KEY,
                max_segments     INTEGER,
                max_bytes        BIGINT,
                max_age_seconds  BIGINT
            )
            """
        )
