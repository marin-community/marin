# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``sandbox_egress`` to ``job_config``.

``sandbox_egress`` is the resolved network egress of a ``CONTAINER_PROFILE_SANDBOX``
job (a ``SandboxEgress`` proto enum int). ``0`` is ``UNSPECIFIED``, which every
non-sandbox job stores. The controller copies it onto each ``RunTaskRequest``.

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "job_config", "sandbox_egress"):
        return
    raw_conn.execute("ALTER TABLE job_config ADD COLUMN sandbox_egress INTEGER NOT NULL DEFAULT 0")
