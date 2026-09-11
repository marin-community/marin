# Research — controller_tick_snapshot

Notes gathered before drafting the design. References pinned to `origin/main` at the time the design branch was cut (`0d440a1b2`).

## Motivating profile

`2026-05-13_17-28-25_profile-_system_controller.out` (3.5 s wall, 101 threads). The four control loops collectively repeat very similar worker+task reads ~12/sec, costing **~2 s of CPU per 3.5 s window (~57% of one core)**. Estimated savings from sharing the worker/task read: **~0.45–0.5 s of CPU per wall-second**.

Hot frames (inclusive):

| ms | function | owner |
|---:|---|---|
| 2600 | `_read_scheduling_state` (controller.py:1809) | scheduler |
| 2100 | `compute_demand_entries` (controller.py:272) | autoscaler |
| 1750 | `_reconcile_worker_batch` / heartbeat processing | polling |
| 1700 | `apply_heartbeats_batch` (transitions.py:1700–1749) | polling (write) |
| 1300 | `_schedulable_tasks` decode loop (controller.py:667) | **scheduler + autoscaler** |
| 1150 | task-id `from_string`/`from_wire` decode | scheduler |
|  800 | `_get_active_worker_addresses` (controller.py:2325) | ping |
|  750 | `healthy_active_workers_with_attributes` (db.py:915) | ping |
| 4800 | `list_jobs` (service.py:1510), 3450 inside `_query_jobs` | gRPC TPE |
|  650 | `httpx` → `tpu_describe` / `_describe_cloud` | autoscaler |

Key redundancies measured in the window:

- `_schedulable_tasks` runs in **scheduler and autoscaler** in the same 3.5 s window, ~2.45 s combined; identical query, decoded twice. Sharing saves ~1.1–1.3 s.
- `healthy_active_workers_with_attributes` runs in **ping + scheduler + autoscaler**, ~1.3 s combined. Sharing saves ~0.5 s.
- Both `_read_scheduling_state` and `compute_demand_entries` independently fetch `_jobs_by_id` (~450 ms in autoscaler alone).
- Task-ID decode (`from_string`/`from_wire`) is ~1 s in scheduler+autoscaler combined; sharing the snapshot eliminates the duplicate decode automatically.

Surprises:

1. `get_config` (stores.py:716) shows ~450 ms of N+1 reads inside the polling loop's heartbeat path.
2. Autoscaler spends ~650 ms in `httpx` calling GCE `tpu_describe`. Independent of the snapshot work — worth caching separately.
3. `read_snapshot` BEGIN/ROLLBACK + SA compile is only ~550 ms total. Not the bottleneck; the cost is in row materialization and decode.
4. gRPC handler `list_jobs` alone is ~4.8 s in the ThreadPoolExecutor (3.45 s in `_query_jobs`). Service-layer cost dwarfs control loops; out of scope here but a candidate for a follow-up (process-split, query simplification, or cache).

---

## Loop topology (current)

Four loops, each spawned via `_threads.spawn` (controller.py:1428–1472).

| Loop | Entry | Cadence | Wake | Notes |
|---|---|---:|---|---|
| Scheduler | `_run_scheduling_loop` (controller.py:1554) | 1–10 s (ExponentialBackoff) | `_scheduling_wake` Event | Set by job submit + capacity-freeing heartbeats |
| Polling/reconcile | `_run_polling_loop` (controller.py:1590) | 1 s fixed | `_polling_wake` Event | Set by job submit + task state changes |
| Autoscaler | `_run_autoscaler_loop` (controller.py:1648); work in `_run_autoscaler_once` (controller.py:2596) | 10–30 s (RateLimiter, configurable) | none | Optional thread |
| Ping | `_run_ping_loop` (controller.py:2503) | 5 s fixed (RateLimiter) | none | Sends RPCs to workers; uses in-memory `WorkerHealthTracker` |

Loops are independent today. Coordination is by wake events; over-waking is explicitly acceptable. The write lock (`ControllerDB._lock`, db.py:221) is taken only during `db.transaction()`; reads do not hold it.

---

## DB reads per loop (call sites)

### Scheduler (controller.py:1911 — `_read_scheduling_state`)
- `_pending_tasks_with_jobs(snap)` — tasks ⨝ jobs ⨝ job_config, filtered by `task_row_can_be_scheduled`.
- `healthy_active_workers_with_attributes(snap, health, attrs)` — workers ⨝ worker_attributes filtered by health tracker.
- `resource_usage_by_worker(snap)` — aggregate held resources from active task_attempts.
- `get_priority_bands` (via `_compute_scheduling_order`) — per-job priority band from job_config.

### Polling (controller.py:2329–2373)
- `list_active_healthy_workers(snap, health)` — worker_id → address map.
- Task/attempt snapshot join — task_attempts ⨝ tasks for unfinished worker-bound rows.
- `run_request_template(snap, job_id)` per job — serialized RunTaskRequest template.

### Autoscaler (controller.py:2596–2618)
- `_build_worker_status_map` (controller.py:2620) — health tracker + `running_tasks_by_worker`.
- `healthy_active_workers_with_attributes(snap, ...)` — same query as scheduler.
- `compute_demand_entries` (controller.py:272) — internally calls `_pending_tasks_with_jobs`, `_reserved_job_ids`, `_building_counts`, `resource_usage_by_worker`.

### Ping (controller.py:2517)
- `_get_active_worker_addresses` — reads in-memory health tracker only (no DB).
- Subsequent `update_worker_pings` is a write.

---

## DB writes per loop

### Scheduler
- `queue_assignments` (controller.py:2140)
- `preempt_task` via `_apply_preemptions` (controller.py:2174)
- `replace_reservation_claims` (controller.py:1763, 1811, 1908)
- `cancel_tasks_for_timeout` (controller.py:2271)
- `mark_task_unschedulable` (controller.py:2287)

### Polling
- `apply_heartbeats_batch` (controller.py:2561) — the largest write hotspot in the profile (~1.7 s inclusive; most is the read-for-update path).
- `fail_workers_batch` via `_terminate_workers` (controller.py:2571, 2582).

### Autoscaler
- No direct DB writes. Issues cloud-API calls (`Autoscaler.execute`); subsequent worker registrations/failures flow through other loops.

### Ping
- `update_worker_pings` (controller.py:2528) — `workers.last_heartbeat_ms`.
- `fail_workers_batch` if ping threshold exceeded.

---

## Existing snapshot-shaped types

- `_SchedulingStateRead` (controller.py:210) — `@dataclass(frozen=True)` with `pending_tasks: list[PendingTask]`, `workers: list[WorkerSnapshot]`, `state_read_ms: int`. **This is the template the new ControlTick should generalize.**
- `WorkerResourceUsage` (reads.py:102) — frozen dataclass, `cpu_millicores`, `memory_bytes`, `gpu_count`, `tpu_count`.
- `WorkerSnapshot` (scheduler.py protocol) — per-worker capacity view.
- `PendingTask` (reads.py, via `_row_to_pending_task`) — task + job config bundled.
- `_GatedCandidates`, `_SchedulingOrder` (controller.py:219, 228) — intermediate frozen results within one scheduling cycle.
- `WorkerLiveness` (worker_health.py:35) — per-worker transient state, frozen slots=True.

Pattern is well-established; the design extends it rather than introducing a new abstraction.

---

## Related design docs in `.agents/projects/`

- `iris-autoscaler-refactor-plan.md` — splits autoscaler.py into demand_router / scaling_plan / runtime. Introduces frozen `ScalePlan`. Compatible: shared snapshot is the input the new autoscaler runtime would consume.
- `20260310_iris_sql_canonical.md` — documents "every DB access is serialized by ControllerDB._lock with multi-statement mutations wrapped in BEGIN IMMEDIATE". Snapshot refactor must respect this — no reads outside `read_snapshot()`.
- `20260223_iris_resultectomy.md` — removed earlier shared-result-queue pattern between loops. Each loop now applies its own writes. The snapshot refactor replaces the cross-loop *read* duplication; it deliberately does not reinstate cross-loop write coordination (that's the optional Phase later).

---

## Related GitHub issues

- **#5470** — gangs dispatched in same tick see stale worker capacity (scheduling + polling concurrency bug). **Shared snapshot directly addresses this race.**
- **#5574** — proposes consolidating task status updates to the polling reconcile path. Compatible; reinforces polling-loop centrality.
- **#4822** — autoscaler idle tracking should include ASSIGNED/BUILDING. Demand calculation must read those states; the shared snapshot already includes them.
- **#3042** — autoscaler/controller worker-state staleness. Snapshot becomes the single source of truth per tick.

---

## Reusable utilities

- Frozen dataclass pattern (slots=True). Already pervasive.
- `_read_scheduling_state` — the seed: takes one snap, builds `_SchedulingStateRead`. The refactor generalizes its callers and broadens its payload.
- `rigging.timing.RateLimiter`, `ExponentialBackoff` — used by every loop; no change.
- `Event` — `_scheduling_wake`, `_polling_wake`. Still useful for opportunistic wakes; the snapshot adds a publish/subscribe layer on top.
- `ManagedThread` (controller.py:1305+) — thread wrapper used today; the refactor does not introduce a new thread primitive.

---

## Invariants the design must preserve

1. **Single write lock** (`ControllerDB._lock`) serializes all mutations. Read snapshots never hold it. The refactor does not change this.
2. **Frozen-dataclass snapshots** are the established pattern. Adopt, don't invent.
3. **Reads are independent today**; introducing a shared snapshot is a refinement, not a coupling — loops still run in their own threads with their own cadences.
4. **`transitions.*` owns all writes** and returns frozen result dataclasses. Phase 1–2 of the refactor leaves this untouched; the optional Phase 3 (bulk write apply) is the only one that touches the write surface.
5. **Over-waking is acceptable.** The design does not need to add a tick-aligned barrier across loops.
