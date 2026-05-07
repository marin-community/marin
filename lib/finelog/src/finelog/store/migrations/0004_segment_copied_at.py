# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``segments.copied_at_ms`` to gate eviction on remote durability.

Eviction picks the oldest L_>=1 segment in each namespace; without this
column, a freshly-compacted segment could be deleted in the window
between rename and copy completion. The :class:`CopyWorker`
stamps ``copied_at_ms`` when the upload finishes, and the eviction
query in :class:`Catalog.select_eviction_candidate` filters on it.

Existing rows from before this migration get ``NULL`` and are therefore
not evictable until a subsequent boot's reconciliation either re-uploads
them or marks them locally-only. In practice the production registry
runs migration 0003 and 0004 together as part of the leveled compaction
ship; the operator restarts the service, the copy worker stamps the
existing finalized segments on first compaction round, and eviction
catches up.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS copied_at_ms BIGINT")
