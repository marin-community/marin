# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``egress_policy`` to ``job_config`` and retire the GVISOR container profile.

``egress_policy`` is the job's resolved network egress (a ``job_pb2.EgressPolicy``
int), which the controller copies onto each ``RunTaskRequest``. Existing rows
get the default for their profile: INTERNET for SANDBOX, CLUSTER otherwise.

This migration converts existing ``CONTAINER_PROFILE_GVISOR`` (5) rows to
SANDBOX (6), so a converted job dispatched after this migration runs without
cluster env, credentials or shared caches, and with internet-only egress.

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""

_RETIRED_GVISOR_PROFILE = 5
_SANDBOX_PROFILE = 6
_EGRESS_UNSPECIFIED = 0
_EGRESS_CLUSTER = 1
_EGRESS_INTERNET = 2


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "job_config", "egress_policy"):
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN egress_policy INTEGER NOT NULL DEFAULT 0")
    raw_conn.execute(
        "UPDATE job_config SET container_profile = ? WHERE container_profile = ?",
        (_SANDBOX_PROFILE, _RETIRED_GVISOR_PROFILE),
    )
    raw_conn.execute(
        "UPDATE job_config SET egress_policy = CASE container_profile WHEN ? THEN ? ELSE ? END WHERE egress_policy = ?",
        (_SANDBOX_PROFILE, _EGRESS_INTERNET, _EGRESS_CLUSTER, _EGRESS_UNSPECIFIED),
    )
