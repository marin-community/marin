# Iris Controller Performance Optimization Design

**Goal**: 5x improvement in scheduling loop cycle time and dashboard RPC latency.

## Problem

The Iris controller uses a single `sqlite3.Connection` serialized by a Python `RLock` (`db.py:1006`). WAL mode enables concurrent readers at the SQLite level, but the RLock forces all reads and writes through a single bottleneck. The scheduling loop (every 0.5s) makes 5+ separate snapshot acquisitions per cycle, and dashboard RPCs block behind it.

Measured latencies (from #3697):
- `_building_counts`: 490–760ms (3 sequential DB round trips)
- `_schedulable_tasks` + worker fetch: 170–315ms
- Dashboard RPCs: 1.1–1.6s total (blocked on RLock)
- `list_jobs`: decodes ALL jobs' protobufs, sorts/paginates in Python

## Proposed Solution

Five independent workstreams that compound:

1. **Read connection pool** — separate read-only connections bypass the write RLock, enabling dashboard reads concurrent with scheduling writes.
2. **Query consolidation** — collapse 5+ snapshot acquisitions in the scheduling loop into 2 (state reads + building counts).
3. **New indices** — cover the hot queries that currently do implicit table scans.
4. **Dashboard SQL rewrites** — push pagination, aggregation, and filtering into SQL; stop decoding unused protobufs.
5. **Benchmark harness** — measure each optimization in isolation against a real checkpoint DB.

### Why this approach

- The RLock is the dominant bottleneck for dashboard latency. Removing it for reads is the single highest-impact change.
- Query consolidation reduces lock hold time per scheduling cycle from ~5 acquisitions to ~2, directly reducing RLock contention even before the read pool.
- SQL-level pagination eliminates O(all_jobs) protobuf decode on every `list_jobs` call.

## 1. Read Connection Pool

### Current state

`ControllerDB.__init__` (`db.py:1003-1010`) creates one `sqlite3.Connection`. `QuerySnapshot` (`db.py:353-369`) acquires the same `RLock` as `transaction()`.

### Design

Add a pool of read-only connections that use their own locking (or no lock, since WAL supports concurrent readers). The write connection + RLock remain unchanged for mutations.

```python
# db.py — new in ControllerDB.__init__
class ControllerDB:
    _READ_POOL_SIZE = 4

    def __init__(self, db_path: Path):
        # ... existing write connection setup ...
        self._read_pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        for _ in range(self._READ_POOL_SIZE):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._configure(conn)
            conn.execute("PRAGMA query_only = ON")
            self._read_pool.put(conn)

    @contextmanager
    def read_snapshot(self) -> Iterator[QuerySnapshot]:
        """Read-only snapshot that does NOT acquire the write lock."""
        conn = self._read_pool.get()
        try:
            conn.execute("BEGIN")
            yield QuerySnapshot(conn, lock=None)  # no lock needed
        finally:
            conn.rollback()
            self._read_pool.put(conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        while not self._read_pool.empty():
            self._read_pool.get_nowait().close()
```

`QuerySnapshot.__init__` needs to accept `lock=None` and skip acquire/release when None:

```python
# db.py:356-369 — modify QuerySnapshot
class QuerySnapshot:
    def __init__(self, conn: sqlite3.Connection, lock: RLock | None):
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> QuerySnapshot:
        if self._lock is not None:
            self._lock.acquire()
        self._conn.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._conn.rollback()
        finally:
            if self._lock is not None:
                self._lock.release()
```

Then `snapshot()` (write-lock) stays for the scheduling loop, and `read_snapshot()` is used for all dashboard/service queries.

**Files to modify**:
- `lib/iris/src/iris/cluster/controller/db.py:1000-1054` — add read pool, modify `QuerySnapshot`, add `read_snapshot()`
- `lib/iris/src/iris/cluster/controller/service.py` — change all `db.snapshot()` calls to `db.read_snapshot()`

**Expected improvement**: Dashboard RPCs go from 1.1–1.6s (blocked behind scheduling lock) to ~50–200ms (just query time). This is the single biggest win.

**Risk**: Read connections see slightly stale data (WAL checkpoint lag, typically <100ms). Acceptable for dashboard reads.

## 2. Query Consolidation in Scheduling Loop

### 2a. Combine `_building_counts` into one snapshot

`_building_counts` (`controller.py:322-346`) makes 3 separate snapshot acquisitions:
1. `running_tasks_by_worker()` → snapshot
2. `_tasks_by_ids_with_attempts()` → snapshot
3. `_jobs_by_id()` → snapshot

Replace with a single raw SQL query:

```python
# controller.py — replace _building_counts
def _building_counts(queries: ControllerDB, workers: list[Worker]) -> dict[WorkerId, int]:
    if not workers:
        return {}
    worker_ids = [str(w.worker_id) for w in workers]
    placeholders = ",".join("?" for _ in worker_ids)
    sql = f"""
        SELECT a.worker_id, COUNT(*) as cnt
        FROM tasks t
        JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id
        JOIN jobs j ON t.job_id = j.job_id
        WHERE a.worker_id IN ({placeholders})
          AND t.state IN (?, ?)
          AND j.is_reservation_holder = 0
        GROUP BY a.worker_id
    """
    params = (
        *worker_ids,
        cluster_pb2.TASK_STATE_BUILDING,
        cluster_pb2.TASK_STATE_ASSIGNED,
    )
    with queries.snapshot() as q:
        rows = q.raw(sql, params, decoders={"worker_id": _decode_worker_id})
    return {row.worker_id: row.cnt for row in rows}
```

**Files**: `controller.py:322-346`
**Expected improvement**: 490–760ms → ~20–50ms (one query instead of three, one lock acquisition instead of three).

### 2b. Cache `_jobs_by_id` within scheduling cycle

`_run_scheduling` (`controller.py:1179`) calls `_jobs_by_id` at line 1226, then `_building_counts` calls it again at line 330. After 2a, the second call is gone. But we can also avoid re-fetching jobs that were already fetched for `_schedulable_tasks`:

```python
# controller.py:1207-1226 — consolidate into one snapshot
def _run_scheduling(self) -> None:
    # ... reservation handling ...
    timer = Timer()
    with slow_log(logger, "scheduling state reads", threshold_ms=50):
        pending_tasks = _schedulable_tasks(self._db)
        workers = healthy_active_workers_with_attributes(self._db)
    # ... rest unchanged, _jobs_by_id already called once at line 1226 ...
```

After 2a, `_building_counts` no longer calls `_jobs_by_id`, so no duplicate. The main savings come from 2a.

### 2c. Narrow `_schedulable_tasks` query

`_schedulable_tasks` (`controller.py:289-301`) fetches all columns including `resource_usage_proto` for every non-terminal task. The scheduling loop only needs a subset. More importantly, it fetches ASSIGNED/BUILDING/RUNNING tasks that `can_be_scheduled()` always rejects.

```python
# controller.py:289-301 — tighten the WHERE clause
def _schedulable_tasks(queries: ControllerDB) -> list[Task]:
    SCHEDULABLE_STATES = (
        cluster_pb2.TASK_STATE_PENDING,
        cluster_pb2.TASK_STATE_PREEMPTED,
    )
    with queries.snapshot() as snapshot:
        tasks = snapshot.select(
            TASKS,
            where=TASKS.c.state.in_(list(SCHEDULABLE_STATES)),
            order_by=(
                TASKS.c.priority_neg_depth.asc(),
                TASKS.c.priority_root_submitted_ms.asc(),
                TASKS.c.submitted_at_ms.asc(),
                TASKS.c.task_id.asc(),
            ),
        )
    return [task for task in tasks if task.can_be_scheduled()]
```

This needs verification that PENDING and PREEMPTED are the only states `can_be_scheduled()` returns True for. Check the `can_be_scheduled` method on Task.

**Files**: `controller.py:289-301`
**Expected improvement**: 170–315ms → ~30–80ms (fewer rows, index-only scan possible with narrower state filter).

## 3. DB Schema Improvements

### 3a. New index: task_attempts covering building_counts query

The `_building_counts` consolidated query (§2a) joins `tasks`, `task_attempts`, and `jobs`. The critical join path is `task_attempts(worker_id) → tasks(task_id, current_attempt_id)`.

```sql
-- 0007_perf_indexes.sql
CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task
    ON task_attempts(worker_id, task_id, attempt_id);
```

The existing `idx_task_attempts_worker` only covers `worker_id`. Adding `task_id` and `attempt_id` makes the JOIN a covering index scan.

### 3b. New index: jobs state for list_jobs

`list_jobs` queries `WHERE state IN (...)` on jobs. Currently no index on `jobs.state`.

```sql
CREATE INDEX IF NOT EXISTS idx_jobs_state
    ON jobs(state, submitted_at_ms DESC);
```

### 3c. New index: tasks state for _schedulable_tasks

The existing `idx_tasks_pending` starts with `state` but is a wide composite. A narrower index for the tightened query:

```sql
CREATE INDEX IF NOT EXISTS idx_tasks_schedulable
    ON tasks(state)
    WHERE state IN (1, 7);  -- PENDING, PREEMPTED (verify exact int values)
```

Note: SQLite partial indexes may not help if the planner doesn't choose them. An alternative is to rely on `idx_tasks_pending` which already leads with `state`. The tighter WHERE clause in §2c is the real win.

### Migration file

Create `lib/iris/src/iris/cluster/controller/migrations/0007_perf_indexes.sql`:

```sql
-- Performance indexes for scheduling and dashboard queries.
CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task
    ON task_attempts(worker_id, task_id, attempt_id);

CREATE INDEX IF NOT EXISTS idx_jobs_state
    ON jobs(state, submitted_at_ms DESC);
```

**Files**: `lib/iris/src/iris/cluster/controller/migrations/0007_perf_indexes.sql` (new)
**Expected improvement**: ~20–30% reduction in query time for affected queries.

## 4. Dashboard Query Rewrites

### 4a. `list_jobs`: SQL-level pagination

`list_jobs` (`service.py:769-920`) fetches ALL jobs, decodes ALL protobufs, sorts in Python, then paginates. Replace with SQL-level sort + LIMIT/OFFSET for the common case (sort by date), falling back to full fetch only for exotic sort fields.

```python
# service.py — replace list_jobs core data fetch
def _list_jobs_paginated(
    db: ControllerDB,
    *,
    states: tuple[int, ...],
    name_filter: str = "",
    state_filter: str = "",
    sort_field: int,
    descending: bool,
    offset: int,
    limit: int,
) -> tuple[list[Job], int]:
    """Fetch jobs with SQL-level pagination for the common sort-by-date case."""
    # Build WHERE clause
    state_placeholders = ",".join("?" for _ in states)
    conditions = [f"j.state IN ({state_placeholders})"]
    params: list[object] = list(states)

    if name_filter:
        # name is inside request_proto, can't filter in SQL without extracting it.
        # For name filtering, we still need to decode protos.
        # But we can at least do SQL pagination for the no-filter case.
        pass

    if state_filter:
        # Map state_filter string to integer
        pass  # handled by narrowing the states tuple

    # For date sorting (the default and most common), do it in SQL
    order_col = "j.submitted_at_ms"
    direction = "DESC" if descending else "ASC"

    # Count total (without proto decode)
    count_sql = f"SELECT COUNT(*) FROM jobs j WHERE {' AND '.join(conditions)}"

    # Fetch page (still need full rows for proto decode, but only for the page)
    select_sql = f"""
        SELECT j.* FROM jobs j
        WHERE {' AND '.join(conditions)}
        ORDER BY {order_col} {direction}
        LIMIT ? OFFSET ?
    """
    params_page = [*params, limit, offset]

    with db.read_snapshot() as q:
        total = q.execute_sql(count_sql, tuple(params)).fetchone()[0]
        rows = q._fetchall(select_sql, params_page)
    jobs = [_decode_row(Job, row) for row in rows]
    return jobs, total
```

The name filter case (which requires proto decode to get the job name) falls back to the current approach but only when `name_filter` is non-empty. The common case (no filter, sort by date) is fully SQL-driven.

**Key insight**: Job name is stored inside `request_proto`, not as a standalone column. To make name filtering SQL-native, we'd need a denormalized `name` column on the `jobs` table. This is worth doing:

```sql
-- In 0007_perf_indexes.sql
ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_jobs_name ON jobs(name);
```

With a backfill migration that extracts names from existing `request_proto` blobs. This enables:

```sql
WHERE j.name LIKE '%filter%'
```

without decoding protos.

**Files**: `service.py:769-920`, migration file
**Expected improvement**: `list_jobs` from O(all_jobs × proto_decode) to O(page_size × proto_decode). For 500 jobs showing 50, that's 10x.

### 4b. `_task_summaries_for_jobs`: SQL GROUP BY

Replace Python-side aggregation (`service.py:321-346`) with SQL:

```python
def _task_summaries_for_jobs(db: ControllerDB, job_ids: set[JobName] | None = None) -> dict[JobName, TaskJobSummary]:
    if job_ids is not None:
        placeholders = ",".join("?" for _ in job_ids)
        where = f"WHERE t.job_id IN ({placeholders})"
        params = tuple(j.to_wire() for j in job_ids)
    else:
        where = ""
        params = ()

    sql = f"""
        SELECT t.job_id,
               t.state,
               COUNT(*) as cnt,
               SUM(t.failure_count) as total_failures,
               SUM(t.preemption_count) as total_preemptions
        FROM tasks t
        {where}
        GROUP BY t.job_id, t.state
    """
    with db.read_snapshot() as q:
        rows = q.raw(sql, params, decoders={"job_id": _decode_job_name})

    summaries: dict[JobName, TaskJobSummary] = {}
    for row in rows:
        s = summaries.get(row.job_id, TaskJobSummary(job_id=row.job_id))
        summaries[row.job_id] = TaskJobSummary(
            job_id=row.job_id,
            task_count=s.task_count + row.cnt,
            completed_count=s.completed_count + (row.cnt if row.state in (TASK_STATE_SUCCEEDED, TASK_STATE_KILLED) else 0),
            failure_count=s.failure_count + row.total_failures,
            preemption_count=s.preemption_count + row.total_preemptions,
            task_state_counts={**s.task_state_counts, row.state: row.cnt},
        )
    return summaries
```

**Files**: `service.py:321-346`
**Expected improvement**: For 3500 tasks, Python processes ~N/distinct_states grouped rows instead of N individual rows. Modest CPU saving (~2x), but reduces data transferred from SQLite.

### 4c. `get_job_status`: fetch only relevant workers

`get_job_status` (`service.py:670-728`) calls `_worker_addresses(self._db)` at line 682, which fetches ALL workers. Replace with a targeted query:

```python
# service.py:682 — replace _worker_addresses with targeted fetch
def _worker_addresses_for_tasks(db: ControllerDB, tasks: list[Task]) -> dict[WorkerId, str]:
    """Fetch addresses only for workers referenced by the given tasks."""
    worker_ids = {t.worker_id for t in tasks if t.worker_id is not None}
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT worker_id, address FROM workers WHERE worker_id IN ({placeholders})",
            tuple(str(wid) for wid in worker_ids),
            decoders={"worker_id": _decode_worker_id},
        )
    return {row.worker_id: row.address for row in rows}
```

**Files**: `service.py:682`, new helper function
**Expected improvement**: Eliminates full workers table scan for single-job views.

### 4d. `list_tasks`: add pagination

`list_tasks` (`service.py:949-970`) fetches ALL tasks when no `job_id` is given. Add SQL-level pagination:

```python
def list_tasks(self, request, ctx):
    job_id = JobName.from_wire(request.job_id) if request.job_id else None
    limit = min(request.limit, 500) if request.limit > 0 else 200
    offset = max(request.offset, 0)

    tasks = _tasks_for_listing_paginated(self._db, job_id=job_id, limit=limit, offset=offset)
    worker_addr_by_id = _worker_addresses_for_tasks(self._db, tasks)
    # ... rest unchanged ...
```

This requires the proto to support `limit`/`offset` fields on `ListTasksRequest`. If they don't exist, add them.

**Files**: `service.py:949-970`, `service.py:293-307`
**Expected improvement**: Prevents unbounded task fetches from blocking the controller.

### 4e. `_live_user_stats`: SQL aggregation

`_live_user_stats` (`service.py:381-400`) fetches all jobs and all tasks, then aggregates in Python. Replace with SQL:

```python
def _live_user_stats(db: ControllerDB) -> list[UserStats]:
    with db.read_snapshot() as q:
        job_rows = q.raw(
            "SELECT user_id, state, COUNT(*) as cnt FROM jobs GROUP BY user_id, state"
        )
        task_rows = q.raw(
            "SELECT j.user_id, t.state, COUNT(*) as cnt "
            "FROM tasks t JOIN jobs j ON t.job_id = j.job_id "
            "GROUP BY j.user_id, t.state"
        )
    by_user: dict[str, UserStats] = {}
    for row in job_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.job_state_counts[row.state] = row.cnt
    for row in task_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.task_state_counts[row.state] = row.cnt
    return list(by_user.values())
```

**Files**: `service.py:381-400`
**Expected improvement**: From O(jobs + tasks) rows transferred to O(users × states) grouped rows.

### 4f. Default to top-level jobs only

`list_jobs` currently shows all jobs including children. The default view should filter to `depth = 0` (top-level jobs):

```sql
WHERE j.depth = 0  -- unless show_children=true in request
```

This reduces the number of jobs processed significantly (most clusters have a flat hierarchy, but some have 3-5x child jobs).

**Files**: `service.py:793`, migration for index on `jobs(depth, state)`

## 5. Benchmark Script

Create `lib/iris/scripts/benchmark_db_queries.py` that operates on a local checkpoint copy (no running controller needed).

```python
"""Benchmark Iris controller DB queries against a local checkpoint.

Usage:
    # Download a checkpoint
    gsutil cp gs://<bucket>/<prefix>/controller-state/latest.sqlite3 ./controller.sqlite3

    # Run benchmarks
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3

    # Run specific benchmark
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only scheduling
"""
import sqlite3
import time
from pathlib import Path
from iris.cluster.controller.db import ControllerDB

def bench(name: str, fn, *, iterations: int = 20):
    """Run fn() iterations times, report p50/p95/p99 latency."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(f"{name:40s}  p50={p50:7.1f}ms  p95={p95:7.1f}ms")

def benchmark_scheduling(db: ControllerDB):
    bench("_schedulable_tasks", lambda: _schedulable_tasks(db))
    bench("healthy_active_workers", lambda: healthy_active_workers_with_attributes(db))
    workers = healthy_active_workers_with_attributes(db)
    bench("_building_counts", lambda: _building_counts(db, workers))
    bench("_jobs_by_id (all pending)", lambda: _jobs_by_id(db, ...))

def benchmark_dashboard(db: ControllerDB):
    bench("list_jobs (all)", lambda: _jobs_in_states(db, USER_JOB_STATES))
    bench("_task_summaries (all)", lambda: _task_summaries_for_jobs(db))
    bench("_worker_addresses", lambda: _worker_addresses(db))
    bench("_live_user_stats", lambda: _live_user_stats(db))
    bench("running_tasks_by_worker", lambda: running_tasks_by_worker(db, ...))
```

The script should import the actual query functions from `controller.py` and `service.py`, run them against a real checkpoint, and report latencies. It should also benchmark the optimized versions for A/B comparison.

**Files**: `lib/iris/scripts/benchmark_db_queries.py` (new)

## Implementation Plan

### Task 1: Benchmark script (no dependencies)
- **Create** `lib/iris/scripts/benchmark_db_queries.py`
- Import query functions from `controller.py`, `service.py`, `db.py`
- Benchmark: `_schedulable_tasks`, `_building_counts`, `healthy_active_workers_with_attributes`, `_jobs_by_id`, `_jobs_in_states`, `_task_summaries_for_jobs`, `_worker_addresses`, `_live_user_stats`, `running_tasks_by_worker`
- Use `bench()` helper with p50/p95 reporting
- Accept a local `.sqlite3` path as CLI argument
- **Test**: Run against a test fixture DB created by existing test setup

### Task 2: Read connection pool (no dependencies)
- **Modify** `db.py:353-369` — make `QuerySnapshot._lock` optional (`RLock | None`)
- **Modify** `db.py:1000-1054` — add `_read_pool`, `read_snapshot()`, update `close()`
- **Modify** all dashboard query helpers in `service.py` — change `db.snapshot()` → `db.read_snapshot()`:
  - `_worker_addresses` (line 311)
  - `_jobs_in_states` (line 317)
  - `_task_summaries_for_jobs` (line 322)
  - `_worker_roster` (line 350)
  - `_query_endpoints` (line 356)
  - `_descendant_jobs` (line 367)
  - `_live_user_stats` (line 383)
  - `_tasks_for_listing` (line 294)
  - `_read_job` (wherever defined)
  - `tasks_for_job_with_attempts` (db.py:1217)
  - `query.py:56` — `execute_raw_query`
- Do NOT change scheduling loop queries (they must see write-consistent state)
- **Test**: Extend `lib/iris/tests/cluster/controller/test_db.py` — verify concurrent reads don't block, verify `read_snapshot` returns consistent data

### Task 3: `_building_counts` consolidation (no dependencies)
- **Rewrite** `controller.py:322-346` — single SQL query with JOIN across tasks, task_attempts, jobs
- Remove calls to `running_tasks_by_worker`, `_tasks_by_ids_with_attempts`, `_jobs_by_id` from within `_building_counts`
- **Test**: Verify `_building_counts` returns identical results before and after. Add a test in `test_scheduler.py` that sets up tasks in BUILDING/ASSIGNED states with reservation holders and verifies counts.

### Task 4: `_schedulable_tasks` tightening (depends on verifying `can_be_scheduled()`)
- **Modify** `controller.py:289-301` — change WHERE clause to only include PENDING and PREEMPTED states
- First verify `can_be_scheduled()` implementation to confirm these are the only schedulable states
- **Test**: Existing scheduler tests should continue passing. Add a test that ASSIGNED/BUILDING/RUNNING tasks are excluded.

### Task 5: Migration file for new indices (no dependencies)
- **Create** `lib/iris/src/iris/cluster/controller/migrations/0007_perf_indexes.sql`
- Add `idx_task_attempts_worker_task`, `idx_jobs_state`
- **Test**: Verify migration applies cleanly on existing test DBs

### Task 6: Dashboard query rewrites (depends on Task 2 for `read_snapshot`)
- **Rewrite** `service.py:321-346` (`_task_summaries_for_jobs`) — SQL GROUP BY
- **Rewrite** `service.py:381-400` (`_live_user_stats`) — SQL GROUP BY
- **Add** `_worker_addresses_for_tasks()` helper, use in `get_job_status` (line 682) and `list_tasks` (line 957)
- **Add** SQL-level pagination to `list_jobs` (service.py:769-920) for default sort-by-date case
- **Add** pagination to `list_tasks` (service.py:949-970) with default limit=200
- **Test**: Extend `test_service.py` to verify paginated results match full results for small datasets

### Task 7: Jobs name denormalization (depends on Task 5)
- **Add** `name TEXT` column to jobs table in migration
- **Backfill** existing rows by extracting name from `request_proto`
- **Populate** name on job insertion in the transitions layer
- **Use** in `list_jobs` for SQL-level name filtering
- **Test**: Verify name column matches proto name for all jobs in test fixtures

## Dependency Graph

```
Task 1 (benchmark)     ─────────────────────────────────→ done
Task 2 (read pool)     ─────────────────────────────────→ done
Task 3 (building_counts) ──────────────────────────────→ done
Task 4 (schedulable_tasks) ────────────────────────────→ done
Task 5 (migration)     ─────────────────────────────────→ done
Task 6 (dashboard SQL) ──── depends on Task 2 ────────→ done
Task 7 (name column)   ──── depends on Task 5 ────────→ done
```

Tasks 1–5 can all run in parallel. Task 6 depends on Task 2. Task 7 depends on Task 5.

## Expected Performance Impact

| Component | Before | After | Improvement |
|---|---|---|---|
| `_building_counts` | 490–760ms | 20–50ms | ~15x |
| `_schedulable_tasks` | 170–315ms | 30–80ms | ~4x |
| Scheduling loop total | ~800ms | ~100ms | ~8x |
| Dashboard `list_jobs` | 1.1–1.6s (blocked) | 50–150ms | ~10x |
| Dashboard `get_job_status` | 200–400ms | 50–100ms | ~4x |
| Dashboard `_live_user_stats` | 100–200ms | 10–30ms | ~7x |

Aggregate: scheduling loop drops from ~800ms to ~100ms (8x). Dashboard RPCs drop from 1.1–1.6s to 50–200ms (8–10x). The 5x target is met by Tasks 2+3 alone; the remaining tasks provide further compounding improvements.

## Risks and Open Questions

1. **`can_be_scheduled()` state coverage** (Task 4): Need to verify that only PENDING and PREEMPTED states pass `can_be_scheduled()`. If other states can be scheduled (e.g., after error recovery), the tightened WHERE clause would miss them. Check the method implementation before changing the query.

2. **Read pool connection lifecycle**: WAL checkpointing requires all readers to release their connections. The pool should handle `SQLITE_BUSY` gracefully if a checkpoint is in progress. The `busy_timeout=5000` pragma on read connections should suffice.

3. **Name column backfill**: Extracting names from protobuf blobs in SQL is not possible. The migration needs a Python-side backfill step that runs after `apply_migrations()`. This is a one-time cost at controller startup.

4. **Proto pagination edge cases**: The parent-child grouping logic in `list_jobs` (lines 875-910) is complex. SQL-level pagination may break family grouping for edge cases where a parent is on one page and children on another. The fallback (fetch-all for name-filtered queries) handles this, but the SQL path needs careful testing.

5. **Partial index support**: SQLite supports partial indexes (`WHERE` clause on `CREATE INDEX`), but the query planner may not always use them. Benchmark with and without to verify.

6. **Existing `benchmark_controller.py`**: There's already an e2e benchmark at `lib/iris/tests/e2e/benchmark_controller.py` that spins up a full controller. The new benchmark script (Task 1) is complementary — it operates on a static DB snapshot for reproducible, fast measurements of individual queries.
