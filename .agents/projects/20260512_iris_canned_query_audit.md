# Iris Canned Query Audit — 2026-05-12

## Background

The SA Core migration produced two patterns the owner wants to clean up:

1. **Module-level `*_QUERY = select(...)` constants** — an artifact of the
   old hand-rolled SQLite layer where every query was pre-compiled once.
   With SA Core, the per-call `select()` is short and there is no material
   warm-up cost to justify hoisting it.

2. **Convenience wrapper dataclasses** that serve callers in the same package
   only (`ActiveTaskRow`, `PendingDispatchRow`, `WorkerResourceUsage`,
   `ReconcileRow`, `JobRecomputeBasis`, `TaskScope`, `WorkerAttributeParams`).

The owner's preferred direction: remove canned constants; use Protocols when
the row shape crosses module boundaries without carrying behaviour; keep
dataclasses when row copies make sense (cross-module + schema-stable shapes,
or when the constructor does useful decoding).

---

## Section 1 — Inventory of Canned Queries

### 1.1 `reads/scheduler.py`

| Constant | File:line | Tables touched | JOIN shape | TypeDecorators |
|---|---|---|---|---|
| `JOBS_WITH_RESERVATIONS_QUERY` | scheduler.py:53 | `jobs`, `job_config` | INNER JOIN on `job_id` | `JobNameType` on `job_id` |
| `HOLDER_JOBS_QUERY` | scheduler.py:80 | `jobs` | single table scan | `JobNameType` on `job_id` |
| `RESOURCE_USAGE_QUERY` | scheduler.py:82 | `task_attempts`, `tasks`, `job_config` | two INNER JOINs | `WorkerIdType` on `worker_id`, `JobNameType` on `job_id` |
| `RECONCILE_ROWS_QUERY` | scheduler.py:151 | `task_attempts`, `tasks` | INNER JOIN with compound predicate `task_id AND current_attempt_id == attempt_id` | `WorkerIdType`, `JobNameType` |
| `RUNNING_TASKS_QUERY` | scheduler.py:210 | `tasks` | single table, IN on `worker_ids` | `WorkerIdType`, `JobNameType` |
| `TIMED_OUT_QUERY` | scheduler.py:239 | `tasks`, `job_config`, `task_attempts` | two JOINs + compound predicate | `TimestampMsType` on `started_at_ms`, `JobNameType`, `WorkerIdType` |

**Callers:**

| Constant | Direct callers in `reads/scheduler.py` | External callers |
|---|---|---|
| `JOBS_WITH_RESERVATIONS_QUERY` | `jobs_with_reservations()` (line 68) | None (wrapped only) |
| `HOLDER_JOBS_QUERY` | `resource_usage_by_worker()` (line 108) | None (internal two-step) |
| `RESOURCE_USAGE_QUERY` | `resource_usage_by_worker()` (line 111) | None (wrapped only) |
| `RECONCILE_ROWS_QUERY` | `reconcile_rows_for_workers()` (line 185) | None (wrapped only) |
| `RUNNING_TASKS_QUERY` | `running_tasks_by_worker()` (line 225) | None (wrapped only) |
| `TIMED_OUT_QUERY` | `timed_out_executing_tasks()` (line 279) | None (wrapped only) |

The wrappers themselves are called externally:

| Wrapper function | Callers | Columns consumed by caller |
|---|---|---|
| `jobs_with_reservations()` | `controller.py::_jobs_with_reservations()` (line 549), `controller.py::_claim_workers_for_reservations` (line 1763) | `job_id`, `reservation_json` (2 of 2 projected) |
| `resource_usage_by_worker()` | `controller.py` (lines 435, 1898, 2242), tests (3 sites) | full `WorkerResourceUsage` dict (passed to `worker_snapshot_from_row`) |
| `reconcile_rows_for_workers()` | `controller.py::_run_reconcile_loop` (line 2278) | all 6 fields of `ReconcileRow` |
| `running_tasks_by_worker()` | `controller.py` (line 2536), `service.py` (lines 1758, 1929) | `worker_id`, `task_id` (full dict) |
| `timed_out_executing_tasks()` | `controller.py::_run_scheduling_loop` (line 2211) | `task_id`, `worker_id` (both fields of `TimedOutTask`) |

### 1.2 `reads/tasks.py`

| Constant | File:line | Tables | JOIN | TypeDecorators |
|---|---|---|---|---|
| `GET_TASK_DETAIL_QUERY` | tasks.py:64 | `tasks` | none | `JobNameType`, `WorkerIdType`, `TimestampMsType` |
| `BULK_TASK_DETAIL_QUERY` | tasks.py:66 | `tasks` | none | same |
| `GET_JOB_ID_QUERY` | tasks.py:99 | `tasks` | none | `JobNameType` |
| `GET_CURRENT_ATTEMPT_QUERY` | tasks.py:101 | `tasks` | none | `JobNameType` |
| `GET_PRIORITY_BAND_FOR_JOB_QUERY` | tasks.py:103 | `tasks` | none | `JobNameType` |
| `STATE_COUNTS_FOR_JOB_QUERY` | tasks.py:107 | `tasks` | none, GROUP BY | `JobNameType` |
| `FIRST_ERROR_FOR_JOB_QUERY` | tasks.py:113 | `tasks` | none | `JobNameType` |
| `GET_WITH_RESOURCES_QUERY` | tasks.py:267 | `tasks`, `jobs`, `job_config` | two JOINs | `JobNameType`, `WorkerIdType` |
| `LIST_PENDING_DISPATCH_QUERY` | tasks.py:308 | `tasks`, `jobs`, `job_config` | two JOINs | `JobNameType` |
| `LIST_ASSIGNED_NULL_WORKER_DISPATCH_QUERY` | tasks.py:318 | `tasks`, `jobs`, `job_config` | two JOINs | `JobNameType` |

All constants are used only inside `reads/tasks.py` by their wrapper functions.
No external code imports any of these constants.

Key wrappers and their column usage:

| Wrapper | Called from | Columns used by callers |
|---|---|---|
| `get_detail()` | `transitions.py` (≈8 sites), `service.py` (≈3 sites) | all 17 task detail columns |
| `bulk_get_detail()` | `transitions.py::apply_heartbeats_batch` (line 1663), `apply_direct_provider_updates` (line 2516) | all 17 columns (state-machine uses most) |
| `get_job_id()` | `transitions.py::mark_task_unschedulable` (line 1892) | `job_id` only |
| `get_current_attempt_id()` | `controller.py` (line 2713) | `current_attempt_id` only |
| `state_counts_for_job()` | `transitions.py::_recompute_job_state` (line 848) | `{state: count}` dict |
| `first_error_for_job()` | `transitions.py::_recompute_job_state` (line 888) | `error` string |
| `list_active()` | `transitions.py` (7 call sites), `controller.py` (2 sites) | all `ActiveTaskRow` fields; see §2.2 |
| `get_with_resources()` | `transitions.py::preempt_task` (line 1919) | all `ActiveTaskRow` fields |
| `list_pending_for_direct_provider()` | `transitions.py::drain_for_direct_provider` (line 2427) | all `PendingDispatchRow` fields |
| `list_assigned_null_worker_for_direct_provider()` | `transitions.py::drain_for_direct_provider` (line 2425) | all `PendingDispatchRow` fields |

### 1.3 `reads/jobs.py`

| Constant | File:line | Tables | JOIN | TypeDecorators |
|---|---|---|---|---|
| `GET_STATE_QUERY` | jobs.py:50 | `jobs` | none | `JobNameType` |
| `GET_ROOT_SUBMITTED_AT_QUERY` | jobs.py:52 | `jobs` | none | `TimestampMsType`, `JobNameType` |
| `GET_PREEMPTION_INFO_QUERY` | jobs.py:54 | `jobs`, `job_config` | INNER JOIN | `JobNameType` |
| `GET_RECOMPUTE_BASIS_QUERY` | jobs.py:60 | `jobs`, `job_config` | INNER JOIN | `JobNameType`, `TimestampMsType` |
| `JOB_DETAIL_QUERY` | jobs.py:110 | `jobs`, `job_config` | INNER JOIN | `JobNameType`, `TimestampMsType`, `BoolIntType` (multiple) |
| `GET_CONFIG_QUERY` | jobs.py:167 | `job_config` | none | `JobNameType`, `BoolIntType` |
| `FIND_PRUNABLE_QUERY` | jobs.py:276 | `jobs` | none | `TimestampMsType`, `JobNameType` |
| `GET_WORKDIR_FILES_QUERY` | jobs.py:301 | `job_workdir_files` | none | `JobNameType` |

All constants are used only by their wrappers within `reads/jobs.py`.
No external code imports these constants.

### 1.4 `reads/task_attempts.py`

| Constant | File:line | Tables | JOIN | TypeDecorators |
|---|---|---|---|---|
| `GET_ATTEMPT_QUERY` | task_attempts.py:52 | `task_attempts` | none | `JobNameType`, `WorkerIdType`, `TimestampMsType` |
| `LIST_FOR_TASK_QUERY` | task_attempts.py:57 | `task_attempts` | none | same |
| `GET_STATE_QUERY` | task_attempts.py:63 | `task_attempts` | none | — |
| `GET_WORKER_ID_QUERY` | task_attempts.py:68 | `task_attempts` | none | `WorkerIdType` |

All constants are used only by their wrappers.

### 1.5 `reads/workers.py`

| Constant | File:line | Tables | JOIN | TypeDecorators |
|---|---|---|---|---|
| `GET_ADDRESS_QUERY` | workers.py:99 | `workers` | none | `WorkerIdType` |
| `GET_DETAIL_QUERY` | workers.py:101 | `workers` | none | `WorkerIdType` |
| `LIST_DETAIL_BY_IDS_QUERY` | workers.py:103 | `workers` | none | `WorkerIdType` |
| `LIST_ADDRESSES_QUERY` | workers.py:107 | `workers` | none | `WorkerIdType` |
| `FILTER_EXISTING_QUERY` | workers.py:111 | `workers` | none | `WorkerIdType` |

All constants are used only by their wrappers.

### 1.6 `reads/reservations.py`

| Constant | File:line | Tables | JOIN | TypeDecorators |
|---|---|---|---|---|
| `LIST_CLAIMS_QUERY` | reservations.py:29 | `reservation_claims` | none | `WorkerIdType` |
| `GET_CLAIM_FOR_WORKER_QUERY` | reservations.py:35 | `reservation_claims` | none | `WorkerIdType` |
| `LIST_CLAIMS_FOR_JOB_QUERY` | reservations.py:39 | `reservation_claims` | none | `WorkerIdType` |
| `COUNT_CLAIMS_FOR_JOB_QUERY` | reservations.py:43 | `reservation_claims` | none | — |
| `GET_LAST_SUBMISSION_QUERY` | reservations.py:92 | `meta` | none | — |

All constants are used only by their wrappers.

### 1.7 `reads/dashboard.py`

| Constant | File:line | Notes |
|---|---|---|
| `PARENT_IDS_WITH_CHILDREN_QUERY` | dashboard.py:238 | Used by `parent_ids_with_children()` only; no external importers. |

The other dashboard queries (`list_jobs`, `task_summaries_for_jobs`) are fully
dynamic — no module-level `select(...)` constants.

### 1.8 `reads/budgets.py`

| Constant | File:line | Notes |
|---|---|---|
| `GET_USER_BUDGET_QUERY` | budgets.py:23 | Used by `get_user_budget()` only. |
| `LIST_USER_BUDGETS_QUERY` | budgets.py:31 | Used by `list_user_budgets()` and `get_all_user_budget_limits()`. |
| `GET_USER_ROLE_QUERY` | budgets.py:82 | Used by `get_user_role()` only. |

### 1.9 `controller.py` — private module-level constants (not in reads/)

`controller.py` defines 7 module-level `select(...)` constants that are not
inside `reads/` at all:

| Constant | Line | Tables | Single vs multi-site |
|---|---|---|---|
| `_JOB_SCHEDULING_QUERY` | 212 | `jobs`, `job_config` | 2 sites: `_jobs_by_id` (line 537) and nowhere else |
| `_TASK_ROW_QUERY` | 242 | `tasks` | 1 site: `_schedulable_tasks` (line 770) |
| `_TASK_DETAIL_QUERY` | 267 | `tasks` | defined but not used in controller.py — this is dead code (service.py has its own `_TASK_DETAIL_COLS`) |
| `_ATTEMPT_QUERY` | 291 | `task_attempts` | defined but not used in controller.py — dead code |
| `_BUILDING_COUNTS_QUERY` | 307 | `tasks`, `jobs` | 1 site: `_building_counts` (line 802) |
| `_RESERVATION_ENTRY_COUNT_QUERY` | 321 | `job_config` | 1 site: `_reservation_entry_count` (line 1711) |
| `_RUNNING_TASKS_WITH_BAND_QUERY` | 325 | `tasks`, `job_config` | 1 site: `_get_running_tasks_with_band_and_value` (line 569) |

`service.py` has its own parallel constants (not shared with `reads/`):
`_TASK_DETAIL_COLS` (line 337), `_ATTEMPT_COLS` (line 357), `_JOB_ROW_COLS` (line 725),
`_TASK_ROW_COLS` (local, inside a function at line 2565).

---

## Section 2 — Caller-Side API Requirements

### 2.1 Scheduler tick (`controller.py`)

The scheduler loop calls reads in three phases:

**Phase A — task collection** (`_schedulable_tasks`, lines 768-771):
- Executes `_TASK_ROW_QUERY` (private): `tasks` only, 13 columns, state=PENDING
- Columns used downstream: `task_id`, `job_id`, `state`, `current_attempt_id`,
  `failure_count`, `preemption_count`, `max_retries_failure`, `max_retries_preemption`,
  `submitted_at_ms`, `priority_band`, `priority_neg_depth`,
  `priority_root_submitted_ms`, `priority_insertion`
- This is NOT routed through `reads/tasks.py` — it remains a private query
  inside `controller.py`. The SA expression is already assembled inline.

**Phase B — job metadata** (`_jobs_by_id`, line 537):
- Executes `_JOB_SCHEDULING_QUERY` (private): `jobs + job_config`, 23-column join
- Columns consumed by `job_requirements_from_job()`: `res_*` (4), `constraints_json`,
  `has_coscheduling`, `coscheduling_group_by`
- Columns consumed by reservation logic: `has_reservation`, `job_id`, `state`,
  `is_reservation_holder`
- Remaining columns (`name`, `depth`, `started_at_ms`, etc.) consumed for
  `is_job_finished` check and `DemandEntry` construction

**Phase C — resource usage** (line 435, 1898, 2242):
- Calls `reads_scheduler.resource_usage_by_worker(snap)` → `WorkerResourceUsage`
- All 4 fields (`cpu_millicores`, `memory_bytes`, `gpu_count`, `tpu_count`)
  consumed by `worker_snapshot_from_row()`

**Phase D — reconcile** (line 2278):
- Calls `reads_scheduler.reconcile_rows_for_workers(snap, worker_ids)`
- All 6 `ReconcileRow` fields consumed

**Phase E — preemption** (lines 569-596):
- Executes `_RUNNING_TASKS_WITH_BAND_QUERY` (private): tasks + job_config,
  8 columns
- Columns consumed: `task_id`, `priority_band`, `worker_id`,
  `res_cpu_millicores`, `res_memory_bytes`, `res_disk_bytes`, `res_device_json`,
  `has_coscheduling`

**Phase F — timeout** (line 2211):
- Calls `reads_scheduler.timed_out_executing_tasks(tx, now)`
- Columns consumed: `task_id`, `worker_id` (both fields of `TimedOutTask`)

**Phase G — building counts** (line 802):
- Executes `_BUILDING_COUNTS_QUERY` (private): tasks + jobs, GROUP BY worker
- Columns consumed: `worker_id`, count

**Phase H — reservation claims** (lines 1711, 1755, 1763):
- `_reservation_entry_count`: executes `_RESERVATION_ENTRY_COUNT_QUERY` (private, single-site)
- `_claim_workers_for_reservations`: calls `_jobs_with_reservations()` →
  uses `job_id`, `reservation_json`

**Cross-call invariants**: All of Phases A–H that run within a single scheduling
tick open separate `read_snapshot()` calls. There is no single-snapshot
consistency guarantee across phases; the scheduler is designed to tolerate races.

### 2.2 Transitions (`transitions.py`)

`ControllerTransitions` methods and the module-level helpers that feed them
issue reads through the `reads/` layer. Grouped by transition path:

**`submit_job`** (line 903):
- `read_jobs.get_root_submitted_at_ms(cur, parent_id)` → `int | None`
  (only `root_submitted_at_ms` consumed)

**`cancel_job`** (line 1154):
- `read_jobs.list_subtree(cur, job_id)` → `list[JobName]`
- `read_tasks.list_active(cur, TaskScope(job_subtree=subtree), states=ACTIVE_TASK_STATES)`
  → fields consumed: `task_id`, `current_worker_id` (2 of 12 `ActiveTaskRow` fields)

**`_kill_non_terminal_tasks`** (line 446):
- `read_tasks.list_active(cur, TaskScope(job_id=job_id), states=NON_TERMINAL_TASK_STATES)`
  → fields: `task_id`, `current_attempt_id`, `current_worker_id`

**`_cascade_children`** (line 481):
- `read_jobs.list_descendants(cur, job_id, exclude_reservation_holders=...)` → `list[JobName]`

**`_resolve_preemption_policy`** (line 643):
- `read_jobs.get_preemption_info(cur, job_id)` → `(preemption_policy, num_tasks)`

**`_recompute_job_state`** (line 841):
- `read_jobs.get_recompute_basis(cur, job_id)` → `JobRecomputeBasis`
  (fields: `state`, `started_at_ms`, `max_task_failures`)
- `read_tasks.state_counts_for_job(cur, job_id)` → `dict[int, int]`
- `read_tasks.first_error_for_job(cur, job_id)` → `str | None`

**`run_request_template`** (line 800):
- `read_jobs.get_detail(snap, job_id)` → full 37-col Row; all fields consumed for
  proto assembly
- `read_jobs.get_workdir_files(snap, job_id)` → `dict[str, bytes]`

**`queue_assignments`** (line 1312):
- `read_tasks.get_detail(cur, assignment.task_id)` → 17-col Row; fields:
  `state`, `current_attempt_id` (via `task_row_can_be_scheduled`)
- `read_workers.active_healthy_address(cur, assignment.worker_id, ...)` →
  `str | None`
- `read_jobs.get_detail(cur, task.job_id)` (line 1344) → full Row for
  `_jobs_by_id` job-cache

**`apply_task_updates` / `apply_heartbeats_batch`** (lines 1610, 1630):
- `read_workers.filter_existing(cur, worker_ids)` → `set[str]`
- `read_tasks.bulk_get_detail(cur, task_ids)` → `dict[JobName, Row]`
  (fields: `state`, `current_attempt_id`, `job_id`, `failure_count`,
  `preemption_count`, `max_retries_failure`, `max_retries_preemption`)
- `read_attempts.bulk_get_for_updates(cur, attempt_keys)` → `dict[(JobName,int), Row]`
  (fields: `state`, `worker_id`, `started_at_ms`, `finished_at_ms`,
  `exit_code`, `error`)
- `read_jobs.get_config(cur, task.job_id)` → `dict` (fields: `has_coscheduling`)

**`_remove_failed_worker`** (line 1716):
- `read_tasks.list_active(cur, TaskScope(worker_id=wid), states=ACTIVE_TASK_STATES)`
  → all `ActiveTaskRow` fields: `state`, `is_reservation_holder`, `preemption_count`,
  `max_retries_preemption`, `task_id`, `current_attempt_id`

**`mark_task_unschedulable`** (line 1890):
- `read_tasks.get_job_id(cur, task_id)` → `JobName | None`

**`preempt_task`** (line 1909):
- `read_tasks.get_with_resources(cur, task_id)` → `ActiveTaskRow` (all fields:
  `state`, `preemption_count`, `max_retries_preemption`, `current_attempt_id`,
  `has_coscheduling`, `job_id`)
- `read_attempts.get_worker_id(cur, task_id, attempt_id)` → `WorkerId | None`

**`cancel_tasks_for_timeout`** (line 2002):
- `read_tasks.list_active(cur, TaskScope(task_ids=...), states=EXECUTING_TASK_STATES)`
  → fields: `task_id`, `job_id`, `current_worker_id`, `has_coscheduling`

**`register_or_refresh_worker`** (line 1193):
- No reads from `reads/` — inserts directly.

**`drain_for_direct_provider`** (line 2393):
- `read_tasks.list_assigned_null_worker_for_direct_provider(cur)` → `list[PendingDispatchRow]`
- `read_tasks.list_pending_for_direct_provider(cur, limit)` → `list[PendingDispatchRow]`
- `read_tasks.list_active(cur, TaskScope(null_worker=True), states=ACTIVE_TASK_STATES)`
  → fields: `task_id`, `current_attempt_id` only
- `read_jobs.get_workdir_files(cur, row.job_id)` → `dict[str, bytes]`

**`apply_direct_provider_updates`** (line 2499):
- `read_tasks.bulk_get_detail(cur, ids)` → same fields as heartbeat path
- `read_attempts.bulk_get_for_updates(cur, keys)` → same fields

**`get_running_tasks_for_poll`** (line 2228):
- `read_tasks.list_active(cur, TaskScope(worker_ids=...),  states=ACTIVE_TASK_STATES)`
  → fields: `task_id`, `current_attempt_id`, `current_worker_id`,
  `current_worker_address`, `container_id` — but `ActiveTaskRow` doesn't carry
  `container_id` or `current_worker_address`; this method actually reads the
  broader task detail Row, not `ActiveTaskRow`.
  (Checking: line 2250 calls `read_tasks.list_active` and accesses `task_id`,
  `current_attempt_id`; then line 2284 calls `reads_workers.list_active_by_ids`
  for per-worker addresses.)

### 2.3 Autoscaler

**`autoscaler/recovery.py`** (`load_autoscaler_checkpoint`):
- Issues raw `select(...)` calls directly against `scaling_groups_table`,
  `slices_table`, `workers_table` — does **not** go through `reads/` modules.
- Columns consumed from `workers_table`: `worker_id`, `slice_id`, `scale_group`, `address`
- These are autoscaler-specific columns not shared with the task reads surface.

**`autoscaler/scaling_group.py`**:
- Issues raw writes (`update`, `delete`, `insert`) against `scaling_groups_table`
  and `slices_table`.
- Does not call any `reads/` module.

**`controller.py::compute_demand_entries`** (line 361):
- Uses `_schedulable_tasks(queries)` and `_jobs_by_id(queries, ...)` (private
  helpers) plus `reads_scheduler.resource_usage_by_worker(snap)` (line 435).
- Passed a `ControllerDB` handle and calls `reads_workers.healthy_active_workers_with_attributes`
  at several sites (lines 945, 1755, 1897, 2411, 2522).

### 2.4 Service handlers (`service.py`)

`service.py` has two categories of reads:

**A — Delegated through `reads/`:**
- `reads_jobs.get_detail(tx, job_id)` via `_read_job()` wrapper (line 332)
- `reads_workers.get_detail(tx, worker_id)` in `_read_worker_detail()` (line 511)
- `reads_workers.healthy_active_workers_with_attributes(...)` (line 908)
- `reads_scheduler.running_tasks_by_worker(tx, ...)` (lines 1758, 1929)
- `reads_jobs.get_priority_bands(snap, ...)` (line 2584)

**B — Inline `select()` calls NOT routed through reads/:**
- `_read_task_with_attempts()` (lines 373, 376): uses local `_TASK_DETAIL_COLS`
  and `_ATTEMPT_COLS` — these **duplicate** the column lists in `reads/tasks.py`
  and `reads/task_attempts.py`.
- `_job_state()` (line 387): inline `select(jobs_table.c.state)`.
- `_worker_address()` (line 394): inline `select(workers_table.c.address)`.
- `_query_jobs()` (line 742): uses local `_JOB_ROW_COLS`, dynamic query builder.
- `_parent_ids_with_children()` (line 860): inline distinct select.
- `get_user_stats()` (lines 2565-2596): local `_TASK_ROW_COLS` inside function.

The `service.py` inline reads are a symptom of the same pattern — `service.py`
was written before `reads/` stabilised and accumulated its own parallel column
lists. Several of these duplicate what `reads/tasks.py`, `reads/task_attempts.py`,
and `reads/dashboard.py` already provide.

---

## Section 3 — Per-Canned-Query Disposition

### 3.1 `reads/scheduler.py` queries

| Query | Disposition | Replacement shape | Rationale |
|---|---|---|---|
| `JOBS_WITH_RESERVATIONS_QUERY` | **Inline** | Inline `select()` inside `jobs_with_reservations()` | Single wrapper, single caller, 2-col projection trivially readable inline |
| `HOLDER_JOBS_QUERY` | **Inline** | Inline inside `resource_usage_by_worker()` | Private two-step helper; constant buys nothing |
| `RESOURCE_USAGE_QUERY` | **Keep wrapper, inline constant** | Keep `resource_usage_by_worker()` with inline `select()`; rename return type to Protocol | 3 prod callers + 3 test callers; the post-processing Python aggregation is non-trivial, worth keeping in the wrapper. The `WorkerResourceUsage` dataclass is a stable cross-module type — **keep it**. |
| `RECONCILE_ROWS_QUERY` | **Keep wrapper, inline constant** | Keep `reconcile_rows_for_workers()` with inline `select()`; `ReconcileRow` dataclass is a stable type — **keep it**. | Python filter on `worker_ids` is non-trivial; single call site in prod but struct crosses into `controller.py` |
| `RUNNING_TASKS_QUERY` | **Inline** | Inline `select()` inside `running_tasks_by_worker()` | Simple 2-col projection, expanding IN list, 3 call sites but the wrapper is the right boundary |
| `TIMED_OUT_QUERY` | **Inline** | Inline `select()` inside `timed_out_executing_tasks()` | Python time comparison post-step; `TimedOutTask` dataclass is local to `reads/scheduler.py` — keep it local, inline the constant |

### 3.2 `reads/tasks.py` queries

| Query | Disposition | Replacement shape |
|---|---|---|
| `GET_TASK_DETAIL_QUERY` | **Inline** | `select(*_TASK_DETAIL_COLS).where(task_id == ...)` inside `get_detail()` |
| `BULK_TASK_DETAIL_QUERY` | **Inline** | `select(*_TASK_DETAIL_COLS).where(task_id.in_(...))` inside `bulk_get_detail()` |
| `GET_JOB_ID_QUERY` | **Inline** | Trivial 1-col select; inline inside `get_job_id()` |
| `GET_CURRENT_ATTEMPT_QUERY` | **Inline** | Trivial 1-col select; inline |
| `GET_PRIORITY_BAND_FOR_JOB_QUERY` | **Inline** | Trivial 1-col select with limit; inline |
| `STATE_COUNTS_FOR_JOB_QUERY` | **Inline** | GROUP BY select; inline inside `state_counts_for_job()` |
| `FIRST_ERROR_FOR_JOB_QUERY` | **Inline** | Ordered 1-col select; inline |
| `GET_WITH_RESOURCES_QUERY` | **Inline** | Uses shared `_ACTIVE_TASK_COLS`/`_ACTIVE_TASK_FROM` — inline as `select(*_ACTIVE_TASK_COLS).select_from(_ACTIVE_TASK_FROM).where(task_id == ...)` |
| `LIST_PENDING_DISPATCH_QUERY` | **Inline** | Uses shared `_DISPATCH_COLS`/`_DISPATCH_FROM` — inline inside `list_pending_for_direct_provider()` |
| `LIST_ASSIGNED_NULL_WORKER_DISPATCH_QUERY` | **Inline** | Same; inline inside `list_assigned_null_worker_for_direct_provider()` |

Note: `_ACTIVE_TASK_COLS`, `_ACTIVE_TASK_FROM`, `_DISPATCH_COLS`, `_DISPATCH_FROM` are
private column-list tuples (not `*_QUERY` constants) — they are **correct** to keep
as module-level helpers since they serve multiple functions within the module.

### 3.3 `reads/jobs.py` queries

| Query | Disposition | Replacement |
|---|---|---|
| `GET_STATE_QUERY` | **Inline** | 1-col select |
| `GET_ROOT_SUBMITTED_AT_QUERY` | **Inline** | 1-col select |
| `GET_PREEMPTION_INFO_QUERY` | **Inline** | 2-col join select |
| `GET_RECOMPUTE_BASIS_QUERY` | **Inline** | 3-col join select; `JobRecomputeBasis` dataclass kept — it crosses module boundary |
| `JOB_DETAIL_QUERY` | **Inline** | 37-col join select; single wrapper function, SA expression is long but self-documenting |
| `GET_CONFIG_QUERY` | **Inline** | Simple select with `.mappings()` |
| `FIND_PRUNABLE_QUERY` | **Inline** | Simple select with LIMIT 1 |
| `GET_WORKDIR_FILES_QUERY` | **Inline** | Simple select |

### 3.4 `reads/task_attempts.py` queries

| Query | Disposition | Replacement |
|---|---|---|
| `GET_ATTEMPT_QUERY` | **Inline** | `select(*_ATTEMPT_COLS).where(...)` |
| `LIST_FOR_TASK_QUERY` | **Inline** | Same with ORDER BY |
| `GET_STATE_QUERY` | **Inline** | 1-col select |
| `GET_WORKER_ID_QUERY` | **Inline** | 1-col select |

The `_ATTEMPT_COLS` tuple is a correct module-level helper — keep it.

### 3.5 `reads/workers.py` queries

| Query | Disposition | Replacement |
|---|---|---|
| `GET_ADDRESS_QUERY` | **Inline** | 1-col select |
| `GET_DETAIL_QUERY` | **Inline** | `select(*_WORKER_DETAIL_COLS).where(...)` |
| `LIST_DETAIL_BY_IDS_QUERY` | **Inline** | `select(*_WORKER_DETAIL_COLS).where(...in_(...))` |
| `LIST_ADDRESSES_QUERY` | **Inline** | 2-col select |
| `FILTER_EXISTING_QUERY` | **Inline** | 1-col select with IN |

The `_WORKER_DETAIL_COLS` tuple — keep it.

### 3.6 `reads/reservations.py` queries

All 5 constants: **Inline**. Each is used by exactly one wrapper function and
is ≤3 lines of SA expression.

### 3.7 `reads/dashboard.py`

`PARENT_IDS_WITH_CHILDREN_QUERY`: **Inline**. Simple distinct select with
expanding IN. Used at one site.

### 3.8 `reads/budgets.py`

All 3 constants: **Inline**. Each used once, trivial expressions.

### 3.9 `controller.py` private constants

| Constant | Disposition | Notes |
|---|---|---|
| `_JOB_SCHEDULING_QUERY` | **Inline** inside `_jobs_by_id()` | 2-site helper but both are inside `_jobs_by_id`; move select inline |
| `_TASK_ROW_QUERY` | **Inline** inside `_schedulable_tasks()` | Single site |
| `_TASK_DETAIL_QUERY` | **Delete** (dead code) | Not referenced anywhere in controller.py; `reads/tasks.py` covers this |
| `_ATTEMPT_QUERY` | **Delete** (dead code) | Same — service.py has its own and `reads/task_attempts.py` covers it |
| `_BUILDING_COUNTS_QUERY` | **Inline** inside `_building_counts()` | Single site |
| `_RESERVATION_ENTRY_COUNT_QUERY` | **Inline** inside `_reservation_entry_count()` | Single site |
| `_RUNNING_TASKS_WITH_BAND_QUERY` | **Inline** inside `_get_running_tasks_with_band_and_value()` | Single site |

### 3.10 `service.py` duplicate column lists

`_TASK_DETAIL_COLS`, `_ATTEMPT_COLS`, `_JOB_ROW_COLS` in `service.py` all
duplicate column sets from `reads/` modules. Disposition:

| Duplicate | Replace with |
|---|---|
| `_TASK_DETAIL_COLS` (service.py:337) | Call `read_tasks.get_detail(tx, task_id)` and `read_tasks.list_for_task(tx, task_id)` directly via `reads/` wrappers |
| `_ATTEMPT_COLS` (service.py:357) | Call `read_attempts.get(tx, task_id, attempt_id)` and `read_attempts.list_for_task(tx, task_id)` |
| `_JOB_ROW_COLS` + `_query_jobs()` (service.py:725+) | Delegate to `reads.dashboard.list_jobs()` — it already implements the same paged/sortable query with proper dynamic building |
| `_TASK_ROW_COLS` (service.py:2565, local) | Extract to `reads/tasks.py::pending_with_priority_fields()` helper |

---

## Section 4 — Risk + Ordering

### 4.1 Mechanical inline (zero blast radius)

The following are pure constant-to-inline substitutions — the `select()` moves
into the wrapper function body, the function signature is unchanged, and no
caller changes:

1. All `reads/*.py` module-level `*_QUERY` constants that are used **only** by
   a single wrapper in the same file.
   Files affected: `scheduler.py`, `tasks.py`, `jobs.py`, `task_attempts.py`,
   `workers.py`, `reservations.py`, `dashboard.py`, `budgets.py`.
   **~35 constants** total. Tests: run `tests/cluster/controller/` suite.

2. Dead-code deletion of `_TASK_DETAIL_QUERY` and `_ATTEMPT_QUERY` in
   `controller.py`.

3. Inline `_JOB_SCHEDULING_QUERY`, `_TASK_ROW_QUERY`, `_BUILDING_COUNTS_QUERY`,
   `_RESERVATION_ENTRY_COUNT_QUERY`, `_RUNNING_TASKS_WITH_BAND_QUERY`
   in `controller.py`.

### 4.2 Service.py deduplication (medium blast radius)

Replacing `service.py`'s duplicate column lists requires verifying that the
`reads/` wrappers return exactly the same columns. Callers in `service.py`
access `task_row.container_id`, `task_row.current_worker_address` which **are**
present in `reads/tasks.py::_TASK_DETAIL_COLS` — so the shapes match.

The `_query_jobs()` → `reads.dashboard.list_jobs()` migration requires
reconciling one difference: `service.py::_query_jobs` supports a
`job_id_prefix` filter (line 777) that `reads/dashboard.py::list_jobs()` does
not. The migration must either add `job_id_prefix` to `reads/dashboard.py`
first, or keep `_query_jobs` but source its column list from `reads/dashboard.py`.

Tests: `test_service.py`, `test_dashboard.py`.

### 4.3 Protocol extraction for `WorkerResourceUsage` and `ReconcileRow`

These dataclasses are correct as dataclasses (stable cross-module contract,
constructor does non-trivial aggregation). No change required. The owner can
optionally add Protocols if a different implementation needs to satisfy the
interface, but there is currently no second implementation.

### 4.4 `TaskScope` dataclass

`TaskScope` is a query-builder parameter, not a result row. It belongs in
`reads/tasks.py` rather than `rows.py` since it is only consumed by
`reads.tasks.list_active`. Move it closer to its user, but this is cosmetic.

### 4.5 `ActiveTaskRow` and `PendingDispatchRow` dataclasses

These carry non-trivial decoder logic (`resource_spec_from_scalars`). They cross
the `reads/` → `transitions.py` boundary at many sites. Keep them as dataclasses.

However, the `_decode_active_task_row` / `_ACTIVE_TASK_PROJECTION` residue in
`rows.py` is dead code (the SA Core version uses `_row_to_active_task` in
`reads/tasks.py`). The string-projection constants in `rows.py` (lines 81-86,
138-143) should be deleted.

---

## Section 5 — Final Ordered Task List

### Commit 1: Delete dead projection strings from `rows.py`

- **Files**: `rows.py`
- **Change**: Delete `_ACTIVE_TASK_PROJECTION`, `_DISPATCH_PROJECTION`,
  `_decode_active_task_row`, `_decode_dispatch_row` (lines 80-166).
  These are string-SQL remnants replaced by the SA Core `_row_to_active_task`
  and `_row_to_dispatch` in `reads/tasks.py`.
- **Acceptance**: `grep -n "_ACTIVE_TASK_PROJECTION\|_DISPATCH_PROJECTION" rows.py`
  returns nothing.

### Commit 2: Delete dead module-level constants from `controller.py`

- **Files**: `controller.py`
- **Change**: Delete `_TASK_DETAIL_QUERY` (line 267) and `_ATTEMPT_QUERY`
  (line 291) — neither is referenced anywhere.
- **Acceptance**: `grep -n "_TASK_DETAIL_QUERY\|_ATTEMPT_QUERY" controller.py`
  returns nothing.

### Commit 3: Inline all single-use `*_QUERY` constants in `reads/scheduler.py`

- **Files**: `reads/scheduler.py`
- **Change**: Inline `JOBS_WITH_RESERVATIONS_QUERY`, `HOLDER_JOBS_QUERY`,
  `RESOURCE_USAGE_QUERY`, `RECONCILE_ROWS_QUERY`, `RUNNING_TASKS_QUERY`,
  `TIMED_OUT_QUERY`. Each becomes the body of its wrapper function.
- **No signature changes.** No caller changes.
- **Acceptance**: No module-level `= select(` assignments remain in the file;
  test suite passes.

### Commit 4: Inline all single-use `*_QUERY` constants in `reads/tasks.py`

- **Files**: `reads/tasks.py`
- **Change**: Inline `GET_TASK_DETAIL_QUERY`, `BULK_TASK_DETAIL_QUERY`,
  `GET_JOB_ID_QUERY`, `GET_CURRENT_ATTEMPT_QUERY`, `GET_PRIORITY_BAND_FOR_JOB_QUERY`,
  `STATE_COUNTS_FOR_JOB_QUERY`, `FIRST_ERROR_FOR_JOB_QUERY`, `GET_WITH_RESOURCES_QUERY`,
  `LIST_PENDING_DISPATCH_QUERY`, `LIST_ASSIGNED_NULL_WORKER_DISPATCH_QUERY`.
  Keep `_ACTIVE_TASK_COLS`, `_ACTIVE_TASK_FROM`, `_DISPATCH_COLS`, `_DISPATCH_FROM`
  (used by multiple functions within the file).
- **No signature or caller changes.**
- **Acceptance**: No module-level public `*_QUERY = select(` in file.

### Commit 5: Inline all single-use `*_QUERY` constants in `reads/jobs.py`

- **Files**: `reads/jobs.py`
- **Change**: Inline all 8 `*_QUERY` constants listed in §3.3.
- **No signature or caller changes.**
- **Acceptance**: No module-level public `*_QUERY = select(` in file.

### Commit 6: Inline all single-use `*_QUERY` constants in `reads/task_attempts.py`

- **Files**: `reads/task_attempts.py`
- **Change**: Inline all 4 constants. Keep `_ATTEMPT_COLS` tuple.
- **Acceptance**: No module-level public `*_QUERY` in file.

### Commit 7: Inline all single-use `*_QUERY` constants in `reads/workers.py`, `reads/reservations.py`, `reads/dashboard.py`, `reads/budgets.py`

- **Files**: the 4 listed files
- **Change**: Inline all constants in each file.
  Workers: keep `_WORKER_DETAIL_COLS` tuple.
- **Acceptance**: `grep -rn "^[A-Z_]*_QUERY = " reads/` returns nothing.

### Commit 8: Inline remaining private constants in `controller.py`

- **Files**: `controller.py`
- **Change**: Inline `_JOB_SCHEDULING_QUERY` into `_jobs_by_id()`,
  `_TASK_ROW_QUERY` into `_schedulable_tasks()`,
  `_BUILDING_COUNTS_QUERY` into `_building_counts()`,
  `_RESERVATION_ENTRY_COUNT_QUERY` into `_reservation_entry_count()`,
  `_RUNNING_TASKS_WITH_BAND_QUERY` into `_get_running_tasks_with_band_and_value()`.
- **No caller changes.**
- **Acceptance**: No module-level `_*_QUERY = select(` in `controller.py`.

### Commit 9: Remove duplicate column lists from `service.py`

- **Files**: `service.py`
- **Change**:
  1. Replace `_TASK_DETAIL_COLS` + inline selects with calls to
     `read_tasks.get_detail()` / `read_tasks.list_for_task()` via a new
     import of `reads.tasks as read_tasks`.
  2. Replace `_ATTEMPT_COLS` + inline selects with calls to
     `read_attempts.get()` / `read_attempts.list_for_task()` via a new
     import of `reads.task_attempts as read_attempts`.
  3. Add `job_id_prefix` filter parameter to `reads/dashboard.py::list_jobs()`
     and delegate `_query_jobs()` to it.
  4. Delete the duplicated column-list constants.
- **Tests**: `test_service.py` + `test_dashboard.py`.
- **Acceptance**: `grep -n "_TASK_DETAIL_COLS\|_ATTEMPT_COLS\|_JOB_ROW_COLS" service.py`
  returns nothing; all service tests pass.

### Commit 10: Move `TaskScope` from `rows.py` to `reads/tasks.py`

- **Files**: `rows.py`, `reads/tasks.py`, `transitions.py`
- **Change**: Move `TaskScope` class and its import sites. Update
  `transitions.py` import.
- **Acceptance**: `from iris.cluster.controller.reads.tasks import TaskScope`
  works; `rows.py` no longer exports `TaskScope`.

---

## Key Findings Summary

1. **No external consumers** of the `*_QUERY` module-level constants. Every
   constant is used only by its own single wrapper function. The constants are
   genuinely legacy scaffolding with no architectural purpose today.

2. **Two groups of dead code** in `controller.py`: `_TASK_DETAIL_QUERY` and
   `_ATTEMPT_QUERY` are defined but never referenced.

3. **`service.py` has significant duplication** of `reads/` column lists.
   This is the highest-value cleanup in Commit 9 (reduces surface area for
   schema-change drift).

4. **`WorkerResourceUsage`, `ReconcileRow`, `ActiveTaskRow`, `PendingDispatchRow`,
   `JobRecomputeBasis` dataclasses are correctly typed** and should stay.
   They carry cross-module contracts with non-trivial constructors.

5. **`_ACTIVE_TASK_COLS`/`_ACTIVE_TASK_FROM` column-tuple helpers** (not
   `*_QUERY` constants) are the right pattern — keep them.

6. **Commits 1-8 are fully mechanical** (zero risk, zero callers to update).
   Commit 9 touches two test files. Commit 10 is cosmetic.

7. **Autoscaler** (`recovery.py`, `scaling_group.py`) does not use `reads/` and
   is out of scope for this cleanup.
