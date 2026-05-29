# Shared SchedulingSnapshot for Iris Scheduler + Autoscaler

Design proposal to unify the per-tick DB reads consumed by the Iris
scheduler and autoscaler under a single typed `SchedulingSnapshot`,
mirroring the `TransitionSnapshot` boundary already used by the
reconcile state machine. Authored 2026-05-27 as a follow-up to the
reconcile-path purification work in `20260527_iris_pure_transitions.md`.

## TL;DR

The scheduler and autoscaler today open five `read_snapshot()` blocks
per controller tick combined, and at least two of those re-read the
same data (pending tasks, healthy workers, per-worker resource usage).
Unify the reads behind one snapshot type, build it once per tick, and
let both components be pure functions over it. Net effect: same
total query count, fewer snapshot opens, no duplicated work, and the
autoscaler's in-memory dry-run no longer needs its own DB transactions.

Do **not** unify with `TransitionSnapshot`. The two snapshots are
keyed on different axes (caller focus vs. global state) and merging
would either bloat the reconcile snapshot or starve the scheduling
one. Keep them as sibling boundaries.

## Current state

### Scheduler

`Scheduler.find_assignments(context: SchedulingContext) -> SchedulingResult`
at `lib/iris/src/iris/cluster/controller/scheduler.py:622` is already
a pure function. The DB work happens in
`controller.py:1024 build_scheduling_context`, which issues **nine
queries inside one `read_snapshot()`**:

1. `_pending_tasks_with_jobs(snap)` — `tasks ⨝ jobs ⨝ job_config` filtered to `TASK_STATE_PENDING`.
2. `reads.healthy_active_workers_with_attributes(snap, health, worker_attrs)`.
3. `reads.resource_usage_by_worker(snap)` — `task_attempts ⨝ tasks ⨝ job_config` plus a reservation-holder scan.
4. `compute_user_spend(snap)` — `tasks ⨝ job_config` aggregated by user.
5. `reads.get_all_user_budget_limits(snap)`.
6. `reads.get_priority_bands(snap, pending_job_ids)`.
7. `SELECT job_id FROM jobs WHERE has_reservation = 1`.
8. `_reservation_entry_counts_for_pending(snap, pending)`.
9. `reads.building_counts(snap, worker_ids)`.

Worker rows are projected through `worker_snapshot_from_row`
(`scheduler.py:109`) and stuffed into `SchedulingContext`. The pure
passes (`apply_scheduling_gates`, `compute_scheduling_order`,
`_run_scheduler_pass`, `_apply_preemptions`) all operate on the
context with no further DB access — except `_apply_preemptions`
which mid-pass calls `_get_running_tasks_with_band_and_value`
(`controller.py:531`) for victim info that is **not** in
`SchedulingContext` today.

Output: `SchedulingResult.assignments`. Controller wraps each as
`Assignment(task_id, worker_id, priority_band)` and calls
`queue_assignments(cur, command)` to flip tasks to `ASSIGNED`.

### Autoscaler

`Autoscaler` at
`lib/iris/src/iris/cluster/controller/autoscaler/runtime.py:96`, driven
per-tick by `_run_autoscaler_once` in `controller.py:2556`. The tick is:

1. `_build_worker_status_map` (`controller.py:2581`) — one
   `read_snapshot()`, one query
   (`reads.running_tasks_by_worker(tx, healthy_active_worker_ids)`).
2. `Autoscaler.refresh(worker_status_map, ts)` — cloud-platform I/O
   (`handle.describe()`), no DB.
3. `Autoscaler.probe_health(ts)` — HTTP probes, no DB.
4. Another `read_snapshot()` for `healthy_active_workers_with_attributes`.
5. `compute_demand_entries(...)` (`controller.py:262`) — opens **two
   more** `read_snapshot()` blocks: one for `_pending_tasks_with_jobs`
   and one for `building_counts` + `resource_usage_by_worker`, then
   runs a dry-run `scheduler.find_assignments` in-memory against an
   unlimited-capacity context to filter out demand that idle workers
   already absorb. Also calls `_reserved_job_ids(queries)` to gate
   taint injection.
6. `_read_reservation_claims(self._db)` — another snapshot for
   `reads.list_claims(tx)`.
7. `Autoscaler.update(demand_entries, ts)` → `route_demand` →
   `build_scale_plan` → execute `SCALE_UP` decisions.

**Four separate `read_snapshot()` blocks**, several of which re-read
data the scheduler tick just consumed. Pending tasks, worker roster,
resource usage, and building counts are all read twice within the
controller's main loop.

### TransitionSnapshot is not the right fit

`reconcile_state.TransitionSnapshot` (`reconcile_state.py:109`) carries:
`now`, `tasks` (caller-supplied subset as `TaskDetailRow`), `attempts`,
`attempt_uid_index`, `job_configs`, `job_state_basis`,
`job_descendants`, `all_tasks_by_job`, `active_tasks_by_job`,
`active_workers` (membership set only).

Overlap with the scheduler is shallow: shared conceptual entities
(tasks, workers, job_configs), but the field sets are largely disjoint.
The reconcile snapshot is keyed by **caller focus** (specific task ids
and their ancestry — the workers/tasks the current reconcile call will
touch); the scheduling snapshot needs **global state** (all pending
tasks, all healthy workers, all reservation claims). Forcing one shape
would either bloat reconcile inputs or under-serve scheduling.

## Proposed shape

A new sibling: `SchedulingSnapshot`. Lives in `scheduling_state.py`
(name TBD; alternatives: `scheduling_snapshot.py`, `scheduler_state.py`).

```python
@dataclass(frozen=True)
class SchedulingSnapshot:
    # Temporal
    now: Timestamp

    # Pending work (both consumers)
    pending_tasks: tuple[PendingTask, ...]
    pending_by_job: Mapping[JobName, tuple[PendingTask, ...]]

    # Jobs / configs (both consumers, narrowly)
    job_requirements: Mapping[JobName, JobRequirements]
    job_constraints_json: Mapping[JobName, str | None]
    job_resource_specs: Mapping[JobName, ResourceSpecProto]
    reserved_job_ids: frozenset[JobName]
    has_reservation_for_job: Mapping[JobName, bool]
    has_direct_reservation_for_job: Mapping[JobName, bool]
    reservation_entry_counts: Mapping[JobName, int]  # scheduler-only

    # Worker roster (both)
    workers: tuple[SchedulableWorker, ...]
    worker_attrs: Mapping[WorkerId, dict[str, AttributeValue]]

    # Per-worker usage / counters (both)
    resource_usage_by_worker: Mapping[WorkerId, WorkerResourceUsage]
    building_counts: Mapping[WorkerId, int]
    running_task_ids_by_worker: Mapping[WorkerId, frozenset[JobName]]

    # Preemption inputs (scheduler-only)
    running_tasks_info: tuple[RunningTaskInfo, ...]

    # Budget / priority (scheduler-only)
    user_spend: Mapping[str, int]
    user_budget_limits: Mapping[str, int]
    user_budget_defaults: UserBudgetDefaults
    requested_bands: Mapping[JobName, int]

    # Reservation state (both)
    reservation_claims: Mapping[WorkerId, ReservationClaim]
```

### Per-field provenance

| Field | Source | Consumers |
|---|---|---|
| `now` | wall clock | both |
| `pending_tasks` | `_pending_tasks_with_jobs(tx)` | both |
| `pending_by_job` | derived | both |
| `job_requirements` | derived from `pending_tasks` | both |
| `job_constraints_json` / `job_resource_specs` | `pending_tasks` row | autoscaler |
| `workers` | `healthy_active_workers_with_attributes` | both |
| `worker_attrs` | derived | both |
| `resource_usage_by_worker` | `reads.resource_usage_by_worker` | both |
| `building_counts` | `reads.building_counts` | both |
| `running_task_ids_by_worker` | `reads.running_tasks_by_worker` | autoscaler |
| `running_tasks_info` | `_get_running_tasks_with_band_and_value` | scheduler (preempt) |
| `user_spend` | `compute_user_spend` | scheduler |
| `user_budget_limits` | `get_all_user_budget_limits` | scheduler |
| `requested_bands` | `get_priority_bands` | scheduler |
| `reserved_job_ids` | `SELECT … WHERE has_reservation=1` | both |
| `reservation_entry_counts` | `job_config.reservation_json` per reserved | scheduler |
| `reservation_claims` | `reads.list_claims` | both |

### Divergent-view tension

The only place scheduler and autoscaler want different *aggregations*
of the same data is per-worker capacity:

- Scheduler: per-worker free-resources vector (`WorkerCapacity.from_worker`).
- Autoscaler: per-pool aggregates live on `ScalingGroup`
  (in-memory autoscaler state, not DB).

The autoscaler's dry-run actually wants the same per-worker view the
scheduler uses — it literally calls `scheduler.find_assignments`. So
carry the source view (per-worker `SchedulableWorker` +
`WorkerResourceUsage`); consumers derive what they need. No per-pool
aggregate belongs in the snapshot.

### Cost

Today: ~14 queries across 5 snapshots per controller tick.
Proposed: ~12 queries in one `read_snapshot()`. Same query count,
fewer snapshot opens, no duplicated pending-tasks/workers/usage reads.
The dry-run inside the autoscaler stops needing its own snapshots.

## Migration shape

```python
def run_scheduling(snap: SchedulingSnapshot, *, config) -> SchedulingDecisions: ...

def run_autoscaler_evaluate(
    snap: SchedulingSnapshot,
    groups: dict[str, ScalingGroup],
    *,
    config,
) -> AutoscalerDecisions: ...
```

`SchedulingDecisions` carries `assignments`, `preemptions`,
`unschedulable_expired`. `AutoscalerDecisions` carries
`scaling_decisions`, the cached routing-decision proto, and the
pending-hints map. The controller persists the results through the
existing apply paths.

The autoscaler's `refresh` (slice describes) and `probe_health` (HTTP)
phases stay separate — they don't touch the DB and shouldn't share
the snapshot.

Per-tick sequence becomes: load `SchedulingSnapshot` once → run
scheduler → run autoscaler with the same snapshot. The autoscaler's
dry-run absorption pass calls into the scheduler's pure function with
the same snapshot plus an `unlimited_capacity=True` flag (or a tweaked
`SchedulingContext` view).

## Staging plan

### Stage S1 — Add `SchedulingSnapshot` + loader

Define the dataclass and a `load_scheduling_snapshot(cur, *, now,
health, worker_attrs)` function that returns the union of today's
reads, all inside one `read_snapshot()`. Includes
`running_task_ids_by_worker` (autoscaler input) and `running_tasks_info`
(scheduler preempt input) so the snapshot is fully self-contained.

### Stage S2 — Rebuild `SchedulingContext` as a thin view

`SchedulingContext` becomes a derivation over `SchedulingSnapshot`:
the per-cycle scratch fields (`capacities`, `assignment_counts`) stay
on the context, the input fields read from the snapshot. The pure
scheduler functions don't change. Replace `build_scheduling_context`
with the thin construction.

### Stage S3 — Eliminate duplicate fetches in `compute_demand_entries`

`compute_demand_entries(snap)` today opens two extra `read_snapshot()`
blocks for `_pending_tasks_with_jobs`, `building_counts`, and
`resource_usage_by_worker`. Rewrite to read from the passed-in
`SchedulingSnapshot`. **This is the immediate concrete win** and the
smallest change that proves the shape end-to-end.

### Stage S4 — Move preemption read into the loader

`_get_running_tasks_with_band_and_value` (`controller.py:531`) is
currently called mid-pass inside `_apply_preemptions`. Move it into
`load_scheduling_snapshot` as `running_tasks_info`, eliminating the
mid-pass DB hit.

### Stage S5 — Share one snapshot per controller tick

Make `_run_autoscaler_once` accept a pre-loaded snapshot from the
controller loop. The controller's main loop becomes: load
`SchedulingSnapshot` → run scheduler → run autoscaler. Alternative
(if the autoscaler runs on a longer interval): keep independent
ticks but share the loader code; the autoscaler still loads its own
snapshot, just via the same function.

### Stage S6 — Tests

State-machine tests against `SchedulingSnapshot` fixtures (analogous
to the work T5 outlines for `TransitionSnapshot`). Hand-built
snapshots driving scheduler + autoscaler pure functions; assertions
on `SchedulingDecisions` / `AutoscalerDecisions`.

## Open questions

- **Naming.** `SchedulingSnapshot` vs. `ScheduleSnapshot` vs.
  `ClusterSnapshot`. The first is consistent with `TransitionSnapshot`
  in naming the consumer (state machine) rather than the data.
- **Tick alignment.** Today the scheduler and autoscaler run on
  separate cadences (scheduler tighter than autoscaler). Sharing one
  snapshot per controller tick is cleaner but may change scaling
  latency. Concrete impact needs a benchmark.
- **Where the dry-run lives.** The autoscaler currently invokes the
  scheduler in-memory with unlimited capacity. After the migration,
  this could be (a) a method on the scheduler that takes a flag, or
  (b) a free function alongside the scheduler. (a) keeps the dry-run
  close to the real path; (b) keeps the scheduler's public surface
  smaller.
- **`SchedulingDecisions` vs. existing types.** Today's
  `SchedulingResult` only carries assignments; preemption is a side
  effect of the pass. Consolidating all outputs into one
  `SchedulingDecisions` is a clarity win but is independent of the
  snapshot work and could land separately.

## Non-goals

- Unifying with `TransitionSnapshot`. Different keying axis. Keep
  separate.
- Touching the autoscaler's cloud-side phases (`refresh`,
  `probe_health`). They don't read the DB and shouldn't share the
  snapshot.
- Changing `ScalingGroup` internal state or slice-lifecycle accounting.

## Key file references

- `lib/iris/src/iris/cluster/controller/scheduler.py` — scheduler pure functions, `SchedulingContext`, `WorkerSnapshot`
- `lib/iris/src/iris/cluster/controller/controller.py:262` — `compute_demand_entries`
- `lib/iris/src/iris/cluster/controller/controller.py:462` — `_pending_tasks_with_jobs`
- `lib/iris/src/iris/cluster/controller/controller.py:531` — `_get_running_tasks_with_band_and_value`
- `lib/iris/src/iris/cluster/controller/controller.py:1024` — `build_scheduling_context`
- `lib/iris/src/iris/cluster/controller/controller.py:1976` — `_run_scheduling`
- `lib/iris/src/iris/cluster/controller/controller.py:2556` — `_run_autoscaler_once`
- `lib/iris/src/iris/cluster/controller/controller.py:2581` — `_build_worker_status_map`
- `lib/iris/src/iris/cluster/controller/autoscaler/runtime.py:96` — `Autoscaler`
- `lib/iris/src/iris/cluster/controller/autoscaler/runtime.py:268` — `evaluate`
- `lib/iris/src/iris/cluster/controller/autoscaler/runtime.py:602` — `update`
- `lib/iris/src/iris/cluster/controller/autoscaler/routing.py:465` — `route_demand`
- `lib/iris/src/iris/cluster/controller/autoscaler/planning.py:129` — `build_scale_plan`
- `lib/iris/src/iris/cluster/controller/autoscaler/scaling_group.py:826` — `update_slice_activity`
- `lib/iris/src/iris/cluster/controller/reads.py:634` — `resource_usage_by_worker`
- `lib/iris/src/iris/cluster/controller/reads.py:722` — `building_counts`
- `lib/iris/src/iris/cluster/controller/reads.py:733` — `running_tasks_by_worker`
- `lib/iris/src/iris/cluster/controller/reads.py:1122` — `healthy_active_workers_with_attributes`
- `lib/iris/src/iris/cluster/controller/budget.py:66` — `compute_user_spend`
- `lib/iris/src/iris/cluster/controller/reconcile_state.py:109` — `TransitionSnapshot` (sibling boundary, kept separate)
