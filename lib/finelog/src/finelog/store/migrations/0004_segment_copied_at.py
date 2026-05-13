# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``segments.copied_at_ms`` to gate eviction on remote durability.

Stamped by the copy worker on upload; eviction filters on NOT NULL so a
freshly-compacted segment isn't deleted before its remote copy lands.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS copied_at_ms BIGINT")
