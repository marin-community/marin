# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``container_profile`` to ``job_config``.

``container_profile`` is the job's requested container security profile (a
``ContainerProfile`` proto enum int). ``0`` is ``UNSPECIFIED``, which resolves
to ``DEFAULT`` at dispatch — so the column defaults to ``0`` and existing jobs
keep today's behavior. The controller reads it from the immutable job_config row
when building each ``RunTaskRequest``; it never varies per attempt, so it lives
only on ``job_config`` (unlike ``priority_band``, which the scheduler also
snapshots on ``tasks``).

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""


def _columns(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA table_info({table})").fetchall()


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in _columns(raw_conn, table))


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "job_config", "container_profile"):
        return
    raw_conn.execute("ALTER TABLE job_config ADD COLUMN container_profile INTEGER NOT NULL DEFAULT 0")
