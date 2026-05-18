# Sub-doc: DB Migrations

Companion to `spec.md` §6 and §7. Draft 2. The schema work for Operation Puffin Bill: one additive migration for UID columns in Phase C, one PK-swap migration in Phase D, optional cleanup.

## Changes from Draft 1

- Phase numbering aligned with v2 spec (Phases A–D, not 1–6).
- `task_uid` is gone (per `identity.md` v2). Only `jobs.job_uid` and `task_attempts.attempt_uid` are added.
- Phase D explicitly framed as "forward-fix, not revert" for application-level rollback.

## Migration list

| Phase | Migration | Effect | Reversible |
|---|---|---|---|
| C | `0027_attempt_uids.py` | Add UID columns + backfill + secondary indexes | Yes (drop columns) |
| D | `0028_uid_primary_keys.py` | Promote `attempt_uid` to PK; demote `(task_id, attempt_id)` to unique secondary | Technically yes; operationally forward-fix |
| (optional) | `0029_drop_legacy_attempt_index.py` | Drop `(task_id, attempt_id)` unique index if all dashboards migrated | Don't bother; keep the index forever |

## 0027 — Add UID columns (Phase C)

### DDL

```python
# lib/iris/src/iris/cluster/controller/migrations/0027_attempt_uids.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add server-minted UIDs to jobs and task_attempts.

attempt_uid is the only UID that appears on the wire (see sub/identity.md).
job_uid is DB-only — used for incarnation disambiguation in joins but never
travels in RPC messages. tasks does NOT get a UID column; tasks are
addressed via attempt_uid -> task_attempts.task_id.

Backfill assigns random UIDs to every existing row in batches to avoid
holding the write lock too long.
"""

import secrets
import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def _new_uid() -> str:
    return secrets.token_hex(8)


def migrate(conn: sqlite3.Connection) -> None:
    # Additive: every column gains a NOT NULL DEFAULT '' so existing inserts
    # without the column keep working until the schema rolls forward.
    if not _has_column(conn, "jobs", "job_uid"):
        conn.execute("ALTER TABLE jobs ADD COLUMN job_uid TEXT NOT NULL DEFAULT ''")
    if not _has_column(conn, "task_attempts", "attempt_uid"):
        conn.execute("ALTER TABLE task_attempts ADD COLUMN attempt_uid TEXT NOT NULL DEFAULT ''")

    # Backfill jobs in batches of 1000. Each batch is its own implicit txn;
    # SQLite holds the writer lock per-statement, so chunked UPDATEs let
    # other writers in between.
    _backfill(conn, "jobs", "job_uid", "job_id")
    _backfill(conn, "task_attempts", "attempt_uid", "rowid")

    # Secondary unique indexes. These are NOT primary keys yet — Phase D
    # migration (0028) promotes attempt_uid to PK.
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_uid ON jobs(job_uid)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attempts_uid ON task_attempts(attempt_uid)")


def _backfill(conn: sqlite3.Connection, table: str, uid_column: str, pk_column: str) -> None:
    """Backfill an empty UID column in batches. Idempotent."""
    BATCH = 1000
    while True:
        rows = conn.execute(
            f"SELECT {pk_column} FROM {table} WHERE {uid_column} = '' LIMIT {BATCH}"
        ).fetchall()
        if not rows:
            return
        updates = [(_new_uid(), row[0]) for row in rows]
        conn.executemany(
            f"UPDATE {table} SET {uid_column} = ? WHERE {pk_column} = ?",
            updates,
        )
```

### Backfill performance

For a production-shape DB (200k jobs, 600k tasks, 1.2M attempts) the migration must complete in a single restart window. Performance test plan:

1. **Capture a recent prod snapshot.** Anonymize, scrub user data, copy to a dev machine.
2. **Time the migration cold.** Expected: backfill at ~50k rows/sec on local SSD with batches of 1000. So 1.2M attempts ≈ 25 seconds.
3. **Time it with concurrent writers.** The write lock is yielded between batches (each `executemany` is one statement). Concurrent submit_job / heartbeat traffic should see <100ms tail latency stalls during the migration.
4. **Failure mid-migration.** The migration is idempotent — if it crashes after partial backfill, re-running picks up where it left off (the `WHERE uid = ''` filter).

If the times are unacceptable (>5 min for prod-shape), fall back to **offline migration**: stop controller, run migration on the DB file, start controller. Don't try to do online migration with a multi-minute lock window.

### Rollback

```python
# 0027r_attempt_uids_rollback.py — only used if Phase C needs to roll back

def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_jobs_uid")
    conn.execute("DROP INDEX IF EXISTS idx_attempts_uid")
    # SQLite doesn't support DROP COLUMN before 3.35. All Iris-supported
    # versions are >=3.35, but the recreate-table workaround is available.
    for table, col in [("jobs", "job_uid"), ("task_attempts", "attempt_uid")]:
        try:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        except sqlite3.OperationalError:
            # Recreate-table fallback elided; see 0024_normalize_resource_usage.py for the pattern.
            raise
```

We do NOT auto-run the rollback as a separate migration step; it's available on demand if Phase C needs to back out. In practice Phase C is rolled back by flipping `IRIS_UID_PRIMARY_KEY=false`, not by dropping the columns.

### Verification

After migration:

```sql
-- Every row has a UID
SELECT count(*) FROM jobs WHERE job_uid = '';                  -- should be 0
SELECT count(*) FROM task_attempts WHERE attempt_uid = '';     -- should be 0

-- UIDs are unique (the indexes enforce this; double-check)
SELECT job_uid, count(*) FROM jobs GROUP BY job_uid HAVING count(*) > 1;       -- 0 rows
SELECT attempt_uid, count(*) FROM task_attempts GROUP BY attempt_uid HAVING count(*) > 1;   -- 0 rows
```

Add these as assertions in a `tests/cluster/test_migration_0027.py` integration test that runs the migration against a synthetic 10k-row DB.

### Schema declaration update

The migrations are the source of truth at apply time, but `schema.py` (the declarative DDL used for fresh DB creation) must mirror them. Update `schema.py`:

- `JOBS` table: add `Column("job_uid", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default="")`.
- `TASK_ATTEMPTS` table: add `Column("attempt_uid", "TEXT", "NOT NULL DEFAULT ''", ...)`.
- Indexes: add the two CREATE INDEX statements to each table's `indexes` tuple.
- `TASKS` does NOT get a UID column (per `identity.md` v2 — tasks are addressed via `attempt_uid → task_attempts.task_id`).

`stores.py` then exposes:

```python
class TaskAttemptStore:
    def get_by_uid(self, snap, attempt_uid: str) -> TaskAttemptRow | None: ...
    def resolve_uid(self, snap, task_id: JobName, attempt_id: int) -> str | None: ...
    def insert(self, cur, attempt_uid: str, ...) -> None: ...  # gains attempt_uid param
```

## 0028 — Promote UID to primary key (Phase D)

This is the harder migration. SQLite cannot ALTER PRIMARY KEY in place. The standard workaround: create new table, copy data, swap. Reference: how migration `0024_normalize_resource_usage.py` does table rebuilds.

```python
# 0028_uid_primary_keys.py — skeleton

def migrate(conn: sqlite3.Connection) -> None:
    # 1. Create new task_attempts schema with attempt_uid as PK.
    conn.execute("""
        CREATE TABLE task_attempts_new (
            attempt_uid TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            attempt_id INTEGER NOT NULL,
            worker_id TEXT REFERENCES workers(worker_id) ON DELETE SET NULL,
            state INTEGER NOT NULL,
            ...
            UNIQUE (task_id, attempt_id)
        )
    """)

    # 2. Copy all rows. Order by rowid for deterministic insert order.
    conn.execute("""
        INSERT INTO task_attempts_new (attempt_uid, task_id, attempt_id, worker_id, state, ...)
        SELECT attempt_uid, task_id, attempt_id, worker_id, state, ...
        FROM task_attempts
        ORDER BY rowid
    """)

    # 3. Drop old, rename new.
    conn.execute("DROP TABLE task_attempts")
    conn.execute("ALTER TABLE task_attempts_new RENAME TO task_attempts")

    # 4. Recreate indexes (the old ones were dropped with the table).
    conn.execute("CREATE INDEX idx_task_attempts_worker_task ON task_attempts(worker_id, task_id, attempt_id)")
    conn.execute("CREATE INDEX idx_task_attempts_live_workerbound ...")
    # (etc — copy from schema.py)
```

### Why this is more careful than 0027

- **Foreign key references to `task_attempts.(task_id, attempt_id)`**: there shouldn't be any (the schema search confirms task_attempts is referenced by nothing as FK source), but verify with `PRAGMA foreign_key_list` before running.
- **Index recreation is load-bearing**: missing the `idx_task_attempts_live_workerbound` index would tank `reconcile_rows_for_workers` performance (migration 0045's note: "without this the planner falls back to scanning ~24k jobs"). Re-create every index.
- **Holding the write lock for the whole CREATE/INSERT/DROP/RENAME sequence** means no other writer can interleave. For 1.2M attempts and SSD, that's ~30 seconds of locked write. Acceptable during a maintenance window, not for online migration.

### Verification

```sql
SELECT count(*) FROM task_attempts;   -- matches pre-migration count
SELECT count(DISTINCT attempt_uid) FROM task_attempts;  -- == count above
```

### Rollback

```sql
-- Symmetric: swap back to composite-key PK.
CREATE TABLE task_attempts_legacy (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    attempt_id INTEGER NOT NULL,
    attempt_uid TEXT NOT NULL,    -- now a regular column, no longer PK
    ...
    PRIMARY KEY (task_id, attempt_id)
);
INSERT INTO task_attempts_legacy SELECT task_id, attempt_id, attempt_uid, ... FROM task_attempts;
DROP TABLE task_attempts;
ALTER TABLE task_attempts_legacy RENAME TO task_attempts;
```

Same risk profile as forward.

## 0029 — Drop legacy index (optional, probably never)

The (task_id, attempt_id) unique index is cheap to maintain and dashboards/queries that join by name keep using it. Likely outcome: we never run this migration. Documented for completeness.

## Test plan

`tests/cluster/test_migrations.py` already exercises every migration on a fresh DB. Add:

1. `test_migration_0027_idempotent`: run twice; second is no-op.
2. `test_migration_0027_partial_failure`: kill mid-backfill; re-run completes.
3. `test_migration_0027_perf`: assert backfill completes for 100k attempts in <5s on CI hardware. Sentinel for regressions.
4. `test_migration_0028_preserves_data`: insert known rows, run migration, verify all attributes round-trip.
5. `test_migration_0028_index_present`: assert `idx_task_attempts_live_workerbound` exists post-migration (regression catch).

## Schema drift detection

The `schema.py` declarative DDL is used for fresh DB creation; migrations are used for upgrades. They can drift. Existing test `tests/cluster/test_schema_drift.py` (or equivalent) should be extended to assert: a fresh DB built from `schema.py` has the same `PRAGMA table_info` and `PRAGMA index_list` as a migrated DB. If this test doesn't exist, add it as part of Phase C.
