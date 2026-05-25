# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline registry schema: ``namespaces`` and ``segments`` tables.

``IF NOT EXISTS`` so older deployments whose tables pre-date this runner
adopt cleanly.
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
            CREATE TABLE IF NOT EXISTS namespaces (
                namespace        TEXT PRIMARY KEY,
                schema_json      TEXT NOT NULL,
                registered_at_ms BIGINT NOT NULL,
                last_modified_ms BIGINT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                namespace     TEXT   NOT NULL,
                path          TEXT   NOT NULL,
                state         TEXT   NOT NULL,
                min_seq       BIGINT NOT NULL,
                max_seq       BIGINT NOT NULL,
                row_count     BIGINT NOT NULL,
                byte_size     BIGINT NOT NULL,
                created_at_ms BIGINT NOT NULL,
                PRIMARY KEY (namespace, path)
            )
            """
        )
