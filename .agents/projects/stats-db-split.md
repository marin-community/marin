# StatsDB: Splitting High-Volume Telemetry Out of the Controller DB

## Context

The Iris controller's `controller.sqlite3` is now the hot spot for write contention. Heartbeat dispatch, resource history, and workdir-file blobs share the single writer lock with authoritative job/task/worker state. The 0023 migration already split `task_profiles` into a sibling DB via ATTACH; experience shows ATTACH does not actually free the writer lock — SQLite serializes writers across all attached schemas. We want a separate connection/process-local writer for ephemeral stats.

## 1. Scope

Tables moving to `stats.sqlite3`:

| Table | Read pattern | Write volume | Size | Criticality |
|---|---|---|---|---|
| `task_resource_history` | dashboard, `/tasks/{id}` detail, profile UI | ~N_tasks × heartbeat_hz (hundreds/s at scale) | Largest table by bytes | Lossy-OK |
| `worker_resource_history` | dashboard worker drill-down | ~N_workers × heartbeat_hz | Large | Lossy-OK |
| `worker_task_history` | audit / job tree view | bounded per assignment | Medium | Nice-to-have |
| `logs` | unused in UI (see open Q2) | append on transition | Medium | Dead? |
| `job_workdir_files` | read once at dispatch, GC'd later | write-once per job submission, potentially MB blobs | Large, bursty | Must survive crash between write and dispatch |

Kept in `controller.sqlite3`:
- `jobs`, `tasks`, `workers`, `worker_assignments`, `endpoints`, `api_keys`, `budgets`, `reservations`, all scheduling state. These need cross-table ACID with FKs and are the authoritative source of truth.

`task_profiles` stays in `profiles.sqlite3` (already split via 0023 but ATTACHed — we will detach it as part of this work).

`job_workdir_files` is a judgement call. Arguments for including:
- Write volume is bursty but coincides with job submission, which is exactly when the controller is busy.
- Blobs are multi-MB and bloat controller.sqlite3 on-disk, slowing backups and VACUUM.
Arguments for a sibling `bundles.sqlite3`:
- Different lifecycle (read-once then delete), different corruption blast radius, different backup policy (don't need to back up at all).
- Cleaner separation; StatsDB stays purely append-only telemetry.

**Recommendation**: Put bundles in a sibling `bundles.sqlite3`, not StatsDB. See Open Question 1.

## 2. Interface

```python
# lib/iris/src/iris/cluster/controller/stats_db.py
class StatsDB:
    def __init__(self, path: Path): ...
    def close(self) -> None: ...

    # Writers — all async-dispatched onto a dedicated writer thread.
    def record_task_resources(self, entries: Sequence[TaskResourceEntry]) -> None: ...
    def record_worker_resources(self, entries: Sequence[WorkerResourceEntry]) -> None: ...
    def record_worker_task_assignment(self, worker_id: WorkerId, task_id: str, ts_ms: int) -> None: ...
    def append_log(self, job_id: str, level: str, message: str, ts_ms: int) -> None: ...

    # Readers — synchronous, use a separate read-only connection.
    def get_task_history(self, task_id: str, limit: int = 500) -> list[TaskResourceRow]: ...
    def get_worker_history(self, worker_id: WorkerId, since_ms: int) -> list[WorkerResourceRow]: ...
    def get_job_history(self, job_id: str) -> list[WorkerTaskRow]: ...

    # Housekeeping — called from prune loop.
    def prune_before(self, cutoff_ms: int) -> PruneStats: ...
    def sweep_orphans(self, live_task_ids: set[str], live_worker_ids: set[WorkerId]) -> int: ...
```

Narrow, typed, no raw SQL escape hatch on the writer side. Readers may still expose a `QuerySnapshot`-style context for dashboard queries that JOIN across resource tables.

## 3. Connection + Concurrency Model

**Separate file, separate connection, not ATTACH.** ATTACH was tried in 0023; under load the writer lock is still global across the process because SQLite holds a single write transaction per connection regardless of how many schemas are attached. Splitting files without splitting connections buys nothing; splitting connections without splitting files (shared WAL) hits the same lock.

**Dedicated writer thread**: a single thread owns the sole write connection, drains a bounded `queue.Queue[WriteOp]`, and batches ops within ~50 ms into one transaction. Callers enqueue and return immediately. This mirrors the pattern `ControllerDB` already uses for its write lock but relaxes the back-pressure — stats drops are acceptable.

```python
class _StatsWriter(threading.Thread):
    def __init__(self, path: Path, q: queue.Queue[WriteOp], max_batch: int = 1024, max_wait_ms: int = 50): ...
    def run(self) -> None:
        while not self._stopped:
            batch = self._drain_batch()
            with self._conn:  # implicit BEGIN/COMMIT
                for op in batch:
                    op.apply(self._conn)
```

Readers get their own `sqlite3.Connection` with `PRAGMA query_only=1`. WAL mode for concurrency; `synchronous=NORMAL` (not FULL) — we don't need fsync-on-every-commit for telemetry.

Queue is bounded; on overflow we drop oldest resource samples (per-task coalescing: keep latest). We do **not** drop workdir-file writes — those go through a synchronous path if we do include them.

## 4. Consistency Story

We lose cross-DB ACID. Concrete consequences:

1. **Bounded stats loss on crash.** Up to `max_wait_ms` of batched writes vanish on hard kill. Acceptable — current dashboards already tolerate heartbeat gaps.
2. **Orphan task_id/worker_id rows.** Today `ON DELETE CASCADE` on the FK to `tasks`/`workers` cleans history when a task/worker is deleted. Post-split, stats rows outlive their parents. Fix: the existing prune loop (`transitions.py` around 2765-2820) gets a new step that calls `stats_db.sweep_orphans(live_ids)`.
3. **Bundle-before-dispatch ordering.** If we move `job_workdir_files`, we must fsync the bundle write before the controller commits the `jobs.state='pending'` transition. Otherwise a crash between the two leaves a scheduled job whose bundle never existed. Implement as: bundle write completes (synchronous), then the controller transaction flips job state. Recovery on startup: any job in pending/dispatching state whose bundle is missing is moved to failed. This is the strongest argument for keeping bundles out of StatsDB's async-writer pattern.
4. **No FK enforcement across files.** Tests that rely on FK cascades need updates; orphan sweep assertions replace them.

## 5. Migration Plan

**Hard cutover, single release.** Dual-write has real cost: doubled write volume during the cutover window, two prune paths, and a decision point about which copy is authoritative for reads. For ephemeral telemetry, a one-shot migration is fine — at worst we lose a few minutes of history during controller restart.

Steps:
1. Land `StatsDB` class + writer thread behind a feature flag, default off. No call-site changes.
2. Write migration `0026_split_stats_db.py`: on startup, if `stats.sqlite3` doesn't exist, create it, `INSERT INTO stats.X SELECT * FROM main.X` for each moved table inside one transaction, then `DROP TABLE main.X`. Since this is a one-time copy of tables that are already bounded (we prune aggressively), wall-clock cost should be seconds on a live controller.
3. Flip the flag, update call sites (§6).
4. Follow-up release: remove the flag.

We do **not** support rollback after step 2. If the split is bad we roll forward.

## 6. Call-Site Audit

Legend: R = read, W = write.

| File:line | Current | After |
|---|---|---|
| `transitions.py` 840-860 (heartbeat ingest) | W `task_resource_history` | `stats_db.record_task_resources(...)` |
| `transitions.py` 1190-1200 (worker heartbeat) | W `worker_resource_history` | `stats_db.record_worker_resources(...)` |
| `transitions.py` 1615-1640 (assignment) | W `worker_task_history` | `stats_db.record_worker_task_assignment(...)` |
| `transitions.py` 1735-1750 (logs) | W `logs` | `stats_db.append_log(...)` — or deleted if Q2 resolves to dead |
| `transitions.py` 2014-2022 (dispatch read) | R `job_workdir_files` | `bundles_db.get(...)` |
| `transitions.py` 2765-2820 (prune) | DELETE across tables in one txn | Two-phase: controller prune commits, then `stats_db.prune_before()` + `sweep_orphans()` |
| `transitions.py` 2918-2925, 3246-3256, 3366-3382 | R stats tables for dashboard | go through `StatsDB` readers |
| `controller.py` 1322-1360 (workdir upload) | W `job_workdir_files` | `bundles_db.put(...)` synchronous, before job commit |
| `service.py` 487-497 (task detail endpoint) | R `task_resource_history` | `stats_db.get_task_history` |
| `service.py` 864-871 (worker detail) | R `worker_resource_history` | `stats_db.get_worker_history` |
| `service.py` 1262-1275, 1463-1475, 1520-1534 | R mixed stats | `stats_db` readers |
| `checkpoint.py` | snapshots entire DB file | now snapshots three files; document backup ordering |
| `schema.py` 675-690 | defines moved tables | moves to `stats_schema.py` |
| `schema.py` 954-1046, 1220-1232 | helpers touching moved tables | move or delete |
| `migrations/0023_separate_profiles_db.py` | ATTACH profiles | precedent to undo; profiles joins StatsDB pattern |

### Representative rewrites

**Heartbeat ingest** (`transitions.py` ~840):

```python
# before
with self._db.transaction() as tx:
    tx.executemany(
        "INSERT INTO task_resource_history(task_id, ts_ms, cpu, mem, gpu) VALUES (?,?,?,?,?)",
        rows,
    )
    tx.execute("UPDATE tasks SET last_heartbeat_ms=? WHERE id=?", (now_ms, task_id))

# after
self._stats.record_task_resources([TaskResourceEntry(task_id, *r) for r in rows])
with self._db.transaction() as tx:
    tx.execute("UPDATE tasks SET last_heartbeat_ms=? WHERE id=?", (now_ms, task_id))
```

Two side effects, two transactions, two stores. Stats loss on crash is bounded by the writer-thread flush interval.

**Dashboard task detail** (`service.py` ~487):

```python
# before
with self._db.snapshot() as s:
    rows = s.fetchall(
        "SELECT ts_ms, cpu, mem FROM task_resource_history WHERE task_id=? ORDER BY ts_ms DESC LIMIT ?",
        (task_id, limit),
    )

# after
rows = self._stats.get_task_history(task_id, limit=limit)
```

**Prune loop** (`transitions.py` ~2765):

```python
# before (single transaction cascades)
with self._db.transaction() as tx:
    tx.execute("DELETE FROM tasks WHERE terminal_at_ms < ?", (cutoff,))
    # FK ON DELETE CASCADE cleans task_resource_history

# after (controller first, stats second, orphan sweep catches leaks)
with self._db.transaction() as tx:
    deleted_ids = tx.fetchall("SELECT id FROM tasks WHERE terminal_at_ms < ?", (cutoff,))
    tx.execute("DELETE FROM tasks WHERE terminal_at_ms < ?", (cutoff,))
self._stats.prune_before(cutoff)
# every N prune cycles:
self._stats.sweep_orphans(live_task_ids=self._db.all_task_ids(), live_worker_ids=...)
```

## 7. Benchmark Plan

Run `benchmark_db_queries.py` (lives in `lib/iris/tests/bench/`) scenarios before and after, on a controller-sized dataset (10k tasks, 200 workers, 1 week history):

- `bench_heartbeat_ingest` — primary target. Expect 3-5x throughput win once stats writes leave the controller writer.
- `bench_dashboard_task_detail` — cross-DB read path; want no regression.
- `bench_prune_loop` — now two transactions + sweep; must stay under current wall-clock budget.
- `bench_worker_query` — mixed read; sanity check.
- `bench_concurrent_rw` — the load test that motivated this change. Measure p99 write latency under concurrent dashboard reads.

Record results in the PR description.

## 8. Risks & Non-Goals

**Non-goals.** This does not fix: controller-DB write contention between job state and task scheduling; `task_profiles` already-split performance; any issue inside the controller's own transaction graph.

**Risks.**
- **Separate-file corruption.** One DB going bad no longer takes everything down, but recovery logic must tolerate a missing/corrupt `stats.sqlite3` — on startup, if it fails to open, rename and recreate empty. Never block controller boot on stats.
- **Backup complexity.** `checkpoint.py` must snapshot three files atomically-enough. Use `VACUUM INTO` per file; acknowledge these are not mutually consistent (stats can be slightly newer than controller). Document that stats is not authoritative.
- **Bundle ordering bug.** The single biggest correctness hazard. Mitigated by (a) keeping bundles out of StatsDB and (b) explicit startup recovery for jobs whose bundle is missing.
- **Writer thread stall.** If the stats writer hangs (disk full), queue fills and we start dropping. Alert on queue depth; never block the controller on stats writes.
- **Test fallout.** Tests asserting FK cascades will fail. Replace with orphan-sweep assertions.

## Open Questions

1. **Bundles: StatsDB, sibling `bundles.sqlite3`, or filesystem blob?** Leaning sibling DB for transactional put/get but no backup; filesystem is tempting but reintroduces the two-phase-commit problem. Need a decision before implementation.
2. **Is `logs` truly dead?** Grep shows no UI/API reader; confirm before migrating (or just drop the table).
3. **Dual-write vs hard cutover?** Proposed hard cutover above. Worth confirming we're okay with a ~minute of blank history on the rollout deploy.
