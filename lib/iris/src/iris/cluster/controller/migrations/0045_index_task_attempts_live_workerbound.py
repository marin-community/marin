# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add partial indexes that make the per-tick scheduling reads cheap.

``TaskAttemptStore.resource_usage_by_worker`` runs every scheduling tick
and reads exactly the rows where ``worker_id IS NOT NULL`` and
``finished_at_ms IS NULL``. Without an index SQLite picks ``jobs`` as
the driving table and scans all ~24k rows; with the partial index here
the planner drives from ``task_attempts`` and the query drops from
~380 ms to ~3 ms on a production-scale DB.

The reservation-holder lookup that the same method uses to filter
holder jobs out of the result is also walked every tick. Without an
index it scans ``jobs`` (~24k rows on production) for the ~200 rows
with ``is_reservation_holder = 1``. The partial index here makes that
lookup ~0.1 ms.

Partial-index predicates must be canonical ``IS NULL`` / ``IS NOT NULL``
so the planner can match the WHERE clause shape exactly.

The controller runs ``ANALYZE`` separately at startup, which is what
populates ``sqlite_stat1`` for plan selection. We deliberately don't
ANALYZE inside the migration so the schema-parity test (which diffs
the migration-built DB against the schema registry) doesn't see the
``sqlite_stat1`` system table as an unexpected delta.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_attempts_live_workerbound "
        "ON task_attempts(worker_id) "
        "WHERE worker_id IS NOT NULL AND finished_at_ms IS NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_reservation_holder " "ON jobs(job_id) WHERE is_reservation_holder = 1"
    )
