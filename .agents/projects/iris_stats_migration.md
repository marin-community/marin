# Iris controller-stats migration

## 1. Context

PR #5290 (the parent of this branch) lands the per-namespace finelog Stats backend and wires `StatsServiceImpl` into the controller's bundled log-server. Iris does not yet *consume* it: every per-tick / per-attempt resource snapshot is INSERT-ed into the controller SQLite DB via `worker_resource_history` and `task_resource_history`, then read back by `GetWorkerStatus` and pruned by a 600s background loop. That blurs two concerns the controller should keep separate — *decisions* (liveness verdicts, scheduling, registry) belong in the controller DB; *measurements* (cpu/mem/disk samples, per-task usage) are time-series and belong in finelog stats namespaces. This migration deletes the two history tables, removes their pruning machinery, repoints the dashboard at finelog stats, and shrinks `GetWorkerStatusResponse`. `ListWorkers` stays — it is the canonical roster.

## 2. Cut-line

| Concern | Owner | Where |
|---|---|---|
| Worker registry, address, attributes | Controller DB | `workers`, `worker_attributes` |
| Liveness verdict (`healthy`, `consecutive_failures`, `last_heartbeat_ms`, `status_message`) | Controller DB | `workers` columns |
| Task ↔ worker assignments, attempts | Controller DB | `tasks`, `task_attempts`, `worker_task_history` |
| Scheduling state (committed resources, dispatch queue, slices, reservations) | Controller DB | `workers`, `dispatch_queue`, `scaling_groups`, `slices` |
| Per-tick host utilization (cpu, mem, disk, running tasks, net) | Stats namespace | `iris.worker` |
| Per-attempt task resource snapshots | Stats namespace | `iris.task` |
| Latest snapshot for "current resources" panel | Stats `SELECT … ORDER BY ts DESC LIMIT 1` | `iris.worker` |

The `snapshot_*` columns on `workers` (cached "latest" copy from `apply_snapshots`) are also removed — `current_resources` becomes a stats query like `resource_history`.

## 3. New stats namespaces

Both schemas use `key_column = "ts"` (TIMESTAMP_MS). Both register eagerly at worker start via `LogClient.get_table(<namespace>, <dataclass>)` so schema mismatches surface in tests and on first ping rather than silently dropping rows.

### 3.1 `iris.worker` — per-heartbeat host utilization

Reusing the dataclass shape from the backup branch (`origin/claude/brave-ride-b06270-backup-iris-stats-cutover:lib/iris/src/iris/cluster/worker/stats.py`), with one change: **drop `healthy`** — that is a controller decision, not a measurement. Add `net_*_bps` (already in `WorkerResourceSnapshot`, already rendered by the dashboard).

```python
@dataclass
class IrisWorkerStat:
    key_column: ClassVar[str] = "ts"
    # identity
    worker_id: str
    ts: datetime           # tz-naive UTC
    status: str            # WorkerStatus enum: IDLE | RUNNING (worker self-report)
    address: str
    # per-tick utilization
    cpu_pct: float
    mem_bytes: int
    mem_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    running_task_count: int
    total_process_count: int
    net_recv_bps: int
    net_sent_bps: int
    # static metadata (replicated each tick — keeps tables self-contained)
    device_type: str
    device_variant: str
    cpu_count: int
    memory_bytes: int
    tpu_name: str
    gce_instance_name: str
    zone: str
```

Retention: leave to finelog defaults; document the operator knob in `lib/iris/OPS.md`. Sparkline window is the last ~50 rows.

### 3.2 `iris.task` — per-attempt resource snapshots

```python
@dataclass
class IrisTaskStat:
    key_column: ClassVar[str] = "ts"
    task_id: str            # JobName.to_wire()
    attempt_id: int
    worker_id: str
    ts: datetime
    cpu_millicores: int
    memory_mb: int
    disk_mb: int
    memory_peak_mb: int
    # accelerator placeholders (nullable — fill later when worker collector exposes them)
    accelerator_util_pct: float | None = None
    accelerator_mem_bytes: int | None = None
```

Replaces `task_resource_history`. Retention: per-namespace TTL on the finelog side; no SQL pruning loop.

## 4. Worker emission

### 4.1 `iris.worker` (per Ping)

Reuse the backup-branch wiring in `lib/iris/src/iris/cluster/worker/worker.py`:

- Hold a `_log_client: LogClient | None` and `_stats_table: Table | None`. Build the LogClient before adopt; register the stats table eagerly after `_controller_client` exists (so the resolver works).
- In `handle_ping` (after `check_worker_health`), call `_emit_worker_stat(snapshot)` (no `healthy` argument). `healthy` stays in the *Ping reply* — controller still consumes it for liveness.
- `_emit_worker_stat` swallows `(StatsError, ConnectError, ConnectionError, OSError, TimeoutError)` with one log line; lets schema-validation `TypeError` propagate (fail fast in tests). Same shape as the backup branch.
- Drain on shutdown: `LogClient.close()` flushes the bg worker; null out `_stats_table` after.

### 4.2 `iris.task` (per attempt resource update)

`lib/iris/src/iris/cluster/worker/task_attempt.py:511` already builds `job_pb2.ResourceUsage`. Add an emit sibling there: every time `task_attempt` reports `resource_usage` in a `WorkerTaskStatus`, also write one `IrisTaskStat` row. This decouples the stats write from the controller heartbeat ingest path — there is no "controller writes stats on the worker's behalf" anymore.

The TaskAttempt receives the LogClient via the existing `log_client=` plumbing the backup branch already proved out.

### 4.3 Controller side: do nothing

The Ping reply still carries `WorkerResourceSnapshot` (the controller may still log it), but it is NOT persisted to a history table. The cached `snapshot_*` columns on `workers` are dropped (see §5).

## 5. Controller DB shrink

### Files / lines to delete

`lib/iris/src/iris/cluster/controller/schema.py`:
- `WORKER_RESOURCE_HISTORY` table + indexes: `:992-1022`.
- `TASK_RESOURCE_HISTORY` table + indexes: `:1024-1047`.
- `MAIN_TABLES` entries `WORKER_RESOURCE_HISTORY` / `TASK_RESOURCE_HISTORY`: `:1314-1315`.
- `snapshot_*` columns on `WORKERS` table. Migration must `ALTER TABLE workers DROP COLUMN ...` for each (modern SQLite supports it; fall back to the `CREATE TABLE workers_new / INSERT … / DROP / RENAME` recipe used by `0023_separate_profiles_db.py` if older SQLite is targeted).

`lib/iris/src/iris/cluster/controller/stores.py`:
- `WORKER_RESOURCE_HISTORY_RETENTION` (`:70-71`), `TASK_RESOURCE_HISTORY_RETENTION` / `TTL` / `DELETE_CHUNK` (`:73-80`).
- `ResourceUsageInsertParams` dataclass (`:460`) — drop entirely.
- `insert_resource_usage` (`:1375-1389`), `insert_resource_usage_many` (`:1391-1414`).
- Task `prune_resource_history` (`:1416-1473`).
- Worker `apply_snapshots` history INSERT block (`:1912-1925`) — keep the `last_heartbeat_ms` UPDATE; drop the `snapshot_*` UPDATE clause and the history INSERT.
- Worker `prune_resource_history` (`:2016-2020`).

`lib/iris/src/iris/cluster/controller/transitions.py`:
- `worker_resource_snapshot` field on `HeartbeatApplyRequest` (`:223-229`): keep the field (it is the wire payload); `_update_worker_health` (`:1326-1340`) only uses it to update last_heartbeat. `apply_snapshots` no longer writes history, so semantically a `None` and a populated snapshot are equivalent on the DB side.
- Drop `ResourceUsageInsertParams` import (`:52`).
- `insert_resource_usage` call sites at `:1408`, `:2546` and the batched-history block `:1653-1706`. Replace with: nothing — the worker writes the row to `iris.task` directly.

`lib/iris/src/iris/cluster/controller/controller.py`:
- Prune scheduler entries `:1518-1526` (the two `prune_resource_history` calls plus the 600s `resource_history_limiter`). Keep the limiter for `prune_task_history` (worker_task_history) which is unchanged; drop only the two resource-history blocks.

## 6. Migration mechanism

Pattern: numbered file under `lib/iris/src/iris/cluster/controller/migrations/` with `def migrate(conn: sqlite3.Connection) -> None`. Schema version bookkeeping is via `SCHEMA_MIGRATIONS` (schema.py:483); each migration runs once. See `0035_drop_dead_logs_table.py` and `0037_drop_txn_log_and_txn_actions.py` for the canonical drop-table form.

New migration: `0040_drop_resource_history_tables.py` (next free number after `0039_requeue_split_coscheduled_jobs.py`):

```python
def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_worker_resource_history_worker")
    conn.execute("DROP INDEX IF EXISTS idx_worker_resource_history_ts")
    conn.execute("DROP TABLE  IF EXISTS worker_resource_history")
    conn.execute("DROP INDEX IF EXISTS idx_task_resource_history_task_attempt")
    conn.execute("DROP TABLE  IF EXISTS task_resource_history")
    for col in (
        "snapshot_host_cpu_percent", "snapshot_memory_used_bytes",
        "snapshot_memory_total_bytes", "snapshot_disk_used_bytes",
        "snapshot_disk_total_bytes",  "snapshot_running_task_count",
        "snapshot_total_process_count","snapshot_net_recv_bps",
        "snapshot_net_sent_bps",
    ):
        conn.execute(f"ALTER TABLE workers DROP COLUMN {col}")
```

`MAIN_TABLES` in schema.py is updated in the same commit so a freshly-initialized DB never creates the dropped tables.

## 7. Proto changes

`lib/iris/src/iris/rpc/controller.proto:381-407` — `GetWorkerStatusResponse`:

- Delete fields 7 (`current_resources`) and 8 (`resource_history`). Replace with `reserved 7; reserved 8;` to prevent number reuse.
- Per `lib/iris/AGENTS.md`: NO BACKWARD COMPATIBILITY. All callers updated in the same commit. Regenerate stubs via the project's proto build (the same path used by `git show c97272cc8 -- lib/iris/src/iris/rpc/controller_pb2.py` — `buf generate` from `lib/iris/`, or check `lib/iris/scripts/generate_protos.py` if present).

`WorkerResourceSnapshot` (in `job.proto`) stays — the worker still puts it on the Ping reply for `WorkerTaskStatus`, even though the controller no longer persists history.

## 8. Dashboard repointing

`lib/iris/dashboard/src/components/controller/WorkerDetail.vue`:
- Remove `data.value?.resourceHistory` consumption (`:39, :60-67, :281-339`). Replace with a `useStatsRpc` query (composable already exists at `composables/useRpc.ts:103`).
- New query, gated on a ListNamespaces check (the FleetTab cold-start pattern from commit `a30ff7ec6`):

```sql
SELECT ts, cpu_pct, mem_bytes, mem_total_bytes,
       disk_used_bytes, disk_total_bytes,
       net_recv_bps, net_sent_bps, running_task_count
FROM "iris.worker"
WHERE worker_id = '<id>'
ORDER BY ts DESC
LIMIT 50
```

Reverse client-side for the Sparkline (oldest → newest), then derive `currentResources` as the `ts DESC LIMIT 1` row. Reuse the Arrow IPC decoding helper from finelog's dashboard composable.

For task-level views (any consumer of `task_resource_history` reads — none today in `WorkerDetail.vue`, but check `JobDetail.vue` / `TaskDetail.vue` / `RpcStatsPanel.vue`): same pattern against `iris.task` keyed on `task_id`.

## 9. Documentation

Add a section to `lib/iris/AGENTS.md` (after the existing "Code Conventions" block, near the proto-regen note):

> **Decisions vs measurements.** The controller SQLite DB stores the *registry and decisions*: worker liveness verdict, task↔worker assignments, scheduling state. Time-series *measurements* (per-tick utilization, per-attempt resource snapshots) live in the finelog stats namespaces (`iris.worker`, `iris.task`) and are queried via the controller-bundled StatsService. New columns that record measurements should be added as stats namespaces, not controller tables.

Update `OPS.md` with the retention knob for the two namespaces and a query recipe for "show me utilization for worker X over the last hour".

## 10. Commit ordering

Each commit must leave tests green and is independently revertable.

1. **`[iris] introduce iris.worker / iris.task stats schemas`**
   New `lib/iris/src/iris/cluster/worker/stats.py` (port from backup branch, drop `healthy`, add `net_*_bps`); register both namespaces eagerly in `Worker.start()`; emit on Ping (`iris.worker`) and on attempt resource_usage (`iris.task`). No reads change. Controller still writes to its tables.

2. **`[iris-dashboard] repoint WorkerDetail to iris.worker stats`**
   Switch sparkline to a stats query. Drop the `resource_history` consumer. `GetWorkerStatusResponse` still has `current_resources` / `resource_history` populated — dashboard just stops reading them.

3. **`[iris] shrink GetWorkerStatusResponse + stop populating history`**
   Remove `current_resources` / `resource_history` from `controller.proto:381-407` (reserve 7,8). Update `service.py:2117-2126` to stop building them. Drop `resource_history` from the `_read_worker_detail` projection. Delete `insert_resource_usage{,_many}` call sites in `transitions.py:1408, 1653-1706, 2546`. Stop the history INSERT inside `stores.py:apply_snapshots`. Remove prune scheduler entries `controller.py:1518-1526`. Tests updated to match.

4. **`[iris] drop worker_resource_history / task_resource_history tables`**
   Migration `0040_drop_resource_history_tables.py`. Remove table defs from `schema.py:992-1047`, drop `MAIN_TABLES` entries, drop `snapshot_*` columns from `workers`. Delete `prune_resource_history` (`stores.py:1416, 2016`), `ResourceUsageInsertParams` (`stores.py:460`), retention constants (`:70-80`).

5. **`[iris] document decisions-vs-measurements split`**
   Update `lib/iris/AGENTS.md` and `lib/iris/OPS.md`.

## 11. Test plan

Per step:

1. **Stats emission**:
   - New unit: `tests/cluster/worker/test_stats.py` — `build_worker_stat` returns expected dataclass shape; numeric fields cast.
   - Integration: extend `tests/cluster/worker/test_worker.py` — fake `LogClient` records `Table.write` calls; `handle_ping` produces one row per call; transport errors swallowed; schema TypeError propagates.
   - Round-trip: `tests/cluster/test_stats_roundtrip.py` — boot a real bundled finelog server, register `iris.worker`, write a row, query it back via SQL.

2. **Dashboard repoint**:
   - Vue component test (vue-tsc) confirms removed types compile; manual check on local cluster.

3. **Proto shrink**:
   - Update `tests/cluster/controller/test_service.py` `GetWorkerStatus` cases to expect no `current_resources` / `resource_history`.
   - Update `tests/cluster/controller/test_transitions.py` heartbeat-with-resource-snapshot cases — the snapshot still drives `last_heartbeat_ms` reset but no rows land in any table. Add an assertion that `worker_resource_history` (still existing pre-migration) gets zero new rows.

4. **Table drops**:
   - Migration test under `tests/cluster/controller/migrations/test_0040_drop_resource_history_tables.py`: build a v0039 DB with rows in both tables, run migration, assert tables are gone. Pattern from `test_0035_*` / `test_0037_*`.
   - Post-migration smoke: `pragma_table_info('worker_resource_history')` returns empty; same for `task_resource_history`. Confirm `workers` no longer has `snapshot_*` columns.

5. **Docs**: no test, just CI lint pass.

## 12. Risks / open questions

- **Stats service unavailable** → dashboard sparklines empty; `currentResources` unavailable. Acceptable: the worker page header (health, address, recent attempts) renders independently from stats. The stats query is gated behind `ListNamespaces` (already established pattern from PR #5290) so a cold-start finelog doesn't error out the page.
- **Finelog retention vs dashboard window**: Dashboard shows ~50 ticks (~ a few minutes). Finelog default retention must cover the longest sparkline window we ship. Document the operator setting; default is fine for current dashboard but pin it explicitly in `OPS.md`.
- **Clock skew**: `ts` is the worker's wall clock. Cross-worker comparisons skew if hosts disagree. Acceptable — same risk we already accept for `last_heartbeat_ms`.
- **No backfill**: existing `worker_resource_history` / `task_resource_history` rows are dropped, not migrated. Per AGENTS.md NO BACKWARD COMPATIBILITY. Operators who care can keep a snapshot of their pre-migration `iris.db`.

### Open questions for the user
1. Keep `WorkerResourceSnapshot` on the Ping reply once the controller stops persisting it, or drop it in a follow-up PR? (It is currently the only consumer.)
2. Should `iris.task` include accelerator fields (`accelerator_util_pct`, `accelerator_mem_bytes`) as nullable placeholders now, or wait until the worker collector actually populates them? (Today's `task_attempt.py:511` does not have them.)
3. Confirm modern-SQLite `ALTER TABLE … DROP COLUMN` is acceptable, or do we need the `CREATE TABLE / COPY / RENAME` recipe used by `0023_separate_profiles_db.py`?

### Critical files for implementation
- `lib/iris/src/iris/cluster/worker/worker.py`
- `lib/iris/src/iris/cluster/worker/task_attempt.py`
- `lib/iris/src/iris/cluster/controller/schema.py`
- `lib/iris/src/iris/cluster/controller/stores.py`
- `lib/iris/src/iris/cluster/controller/transitions.py`
- `lib/iris/src/iris/cluster/controller/service.py`
- `lib/iris/src/iris/rpc/controller.proto`
- `lib/iris/dashboard/src/components/controller/WorkerDetail.vue`
- `lib/iris/src/iris/cluster/controller/migrations/0040_drop_resource_history_tables.py` (new)
