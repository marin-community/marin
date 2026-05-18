# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create the ``(namespace, level, min_seq)`` index on ``segments``.

Split from 0003 because DuckDB can't combine CREATE INDEX with the
DROP COLUMN there in a single transaction.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    conn.execute("CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq)")
