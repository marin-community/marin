# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``egress_allow_json`` to ``job_config``.

``egress_allow_json`` holds a job's own egress rules as a JSON list
of rule strings (``iris.cluster.egress`` syntax). Existing rows get an
empty list.

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "job_config", "egress_allow_json"):
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN egress_allow_json TEXT NOT NULL DEFAULT '[]'")
