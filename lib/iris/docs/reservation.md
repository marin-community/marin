# Reservation System Design

**Status**: Proposal
**Author**: (planning document)
**Date**: 2026-02-27

## Problem Statement

Today, Iris scheduling is purely reactive: the autoscaler observes pending tasks,
computes demand, and creates slices. There is no way for a parent job to
declare ahead of time "my children will need N GPUs of type X" and have the
system pre-provision capacity before the children are submitted.

This creates two problems:

1. **Cold-start latency**: A parent job submits children that need v5p TPUs.
   The autoscaler sees demand, requests slices, waits for bootstrap (~5 min on
   GCP, longer on CoreWeave). The parent and children are blocked for the
   entire provisioning cycle.

2. **Scheduling fragmentation**: Children arrive one-by-one. Each independently
   competes for resources. With no advance knowledge of the total resource
   envelope, the scheduler cannot make globally optimal placement decisions
   (e.g., routing all children to the same region).

A **reservation** solves both by letting a job declare its resource envelope
ahead of time: "I will need 4× H100 workers and 2× v5p workers." The
autoscaler provisions capacity proactively, and the **reserving job itself is
held pending until the reservation is satisfied**. Once the job starts running,
it submits children which inherit its constraints (including region) via the
existing `merge_constraints` mechanism in `client.py` — so children naturally
land in the right region without any ongoing gate.

### Design Goals

- **Simple mechanism**: Easy to follow in code and in the dashboard. Operators
  should be able to look at a reservation and immediately understand what it
  claims and what's using it.
- **Best-effort convergence**: Reservations are scheduling hints, not SLAs.
  Preemptions can reduce a reservation below its target. The system converges
  back toward the target but does not provide hard guarantees.
- **One-time scheduling gate**: The **reserving job** is blocked from
  scheduling until its reservation is fulfilled. This is the region-locking
  mechanism: the reservation's constraints determine where capacity is
  provisioned, and the gate ensures the job doesn't start (and submit
  children) until that capacity exists. Children inherit the parent's region
  constraints and schedule normally — no ongoing gate needed.
- **Incremental adoption**: Jobs that don't use reservations behave exactly as
  today. The reservation system layers on top of existing scheduling and
  autoscaling without changing their core contracts.
- **Resource-generic**: A reservation is a list of resource requirements (not
  "slices"). Each entry is a `ResourceSpecProto` + constraints. To request
  multiple workers, duplicate the entry. This keeps the proto flat and the
  demand computation trivial.

### Non-Goals

- Time-based reservations ("give me X from T1 to T2")
- Hard capacity guarantees that survive preemption
- Multi-tenant quota management
- Reservation transfer between jobs

---

## Prior Art

### Borg Allocs (Google)

An **alloc** is a reserved set of resources on a machine. An **alloc set**
groups allocs across machines. Tasks are scheduled *into* existing allocs,
consuming their reserved resources. The alloc exists independently of any task;
it's "reserving a parking spot — whether or not you park a car there."

**Key insight**: Allocs decouple resource reservation from task execution.
The reservation is a first-class object with its own lifecycle.

**Strengths**: Cleanest claim-now-use-later primitive. Simple mental model.
**Weaknesses**: Full alloc semantics require per-machine reservation tracking.
At Iris's abstraction level (slices, not individual machines), we don't need
that granularity.

### Kubernetes Dynamic Resource Allocation (DRA)

A `ResourceClaim` is a request for specific devices. The scheduler finds nodes
with matching `ResourceSlice`s, allocates devices to the claim, then schedules
Pods to those nodes. Claims support prioritized fallback lists via
`firstAvailable`.

**Key insight**: The claim is a two-phase object — created first, then
fulfilled. Pods reference claims, not raw resource specs.

**Strengths**: Clean separation between "what I need" and "where to run."
**Weaknesses**: Operates at the node/device level, no time dimension.

### Kubernetes Kueue

Kueue sits above kube-scheduler as an admission controller. It decides *when*
a workload is admitted. Supports **partial admission** (reduced parallelism
when full quota is unavailable) and **flavor fungibility** (try preferred GPU,
fall back to alternative).

**Key insight**: Separate admission ("should this run?") from placement
("where does it run?"). Admission manages quotas; the scheduler handles
the rest.

**Strengths**: Clean layering. Partial admission is practical for training jobs.
**Weaknesses**: Requires a separate controller and CRDs.

### YARN ReservationSystem

Users submit resource "skylines" over time. A `PlanFollower` dynamically
creates scheduler queues to reflect reservations. Supports graceful degradation:
when capacity drops, the system replans, shrinking the least important
reservation.

**Key insight**: Explicit time dimension enables proactive capacity planning.
**Weaknesses**: Extremely complex. Overkill for our use case.

### Slurm Reservations

Admin creates a time-boxed, exclusive resource claim via `scontrol`. Jobs
associated with a reservation are scheduled before non-reservation jobs.
Backfill scheduling fills gaps around reservations.

**Key insight**: Reservations create scheduling priority, not just resource
holding. Reserved jobs are considered first.
**Weaknesses**: Admin-only creation. Fixed time windows.

### Summary of Patterns

| Pattern | Complexity | Fits Iris? | Why/Why Not |
|---------|-----------|-----------|-------------|
| Borg alloc (resource hold) | Low | Partially | Good model but we operate at slice level, not machine level |
| DRA claim (two-phase bind) | Medium | Yes | Clean "claim then fulfill" lifecycle maps well |
| Kueue admission control | Medium | Partially | We don't have a separate admission layer |
| YARN time skylines | High | No | We don't need time-based reservations |
| Slurm priority boost | Low | Yes | Simple and effective — reserved work gets priority |

---

## Proposal: Reservation as Demand Anchor

### Core Idea

A **reservation** is a persistent demand signal attached to a job. It tells
the autoscaler "provision these resources" and tells the scheduler "don't
start this job until the resources exist." The reserving job is held pending
until fulfilled; once running, it submits children that inherit its
constraints and schedule normally.

The reservation is **not** a separate first-class object with its own
lifecycle. It is a field on `LaunchJobRequest` that modifies two things:
1. **Demand routing**: reservation entries become `DemandEntry` objects every
   autoscaler cycle, keeping capacity alive.
2. **Scheduling**: the reserving job is gated until all entries are fulfilled.

### Why Not a First-Class Object?

We considered making reservations independent objects (like Borg allocs). The
advantages would be: explicit lifecycle management, reuse across jobs,
dashboard visibility as a top-level entity.

However, this adds significant complexity:

- New proto service with CRUD operations
- Separate lifecycle management (who cleans up abandoned reservations?)
- Reference counting between reservations and jobs
- A new concept for operators to learn

Instead, we tie the reservation to the job that creates it. The job's lifecycle
*is* the reservation's lifecycle. When the job terminates, the reservation
dissolves. This is simpler and covers the primary use case: a parent job
declaring resource needs for its children.

If we later need standalone reservations, we can extract the mechanism into a
first-class object. But we start simple.

### Mechanism

#### 1. Proto Extension

A reservation is a **flat list of resource entries**. Each entry describes a
single worker's worth of resources + constraints. To reserve 4 H100 workers,
include 4 identical entries. No `count` field — duplication is the counting
mechanism. This keeps the proto dead simple and the demand computation 1:1
(each entry becomes exactly one `DemandEntry`).

```protobuf
message ReservationEntry {
  // Resource spec for one worker. Same type used in LaunchJobRequest.resources.
  ResourceSpecProto resources = 1;

  // Constraints for routing (region, zone, preemptible, etc.).
  repeated Constraint constraints = 2;
}

message ReservationConfig {
  // Resource entries to reserve. Each entry = one worker. Duplicate entries
  // to reserve multiple workers of the same type.
  //
  // Example: reserve 4 H100 workers in us-east + 2 v5p workers:
  //   entries: [
  //     { resources: {device: {gpu: {variant: "H100", count: 8}}},
  //       constraints: [{key: "region", op: EQ, value: "us-east1"}] },
  //     { resources: ... },  // same — 4 total
  //     { resources: ... },
  //     { resources: ... },
  //     { resources: {device: {tpu: {variant: "v5litepod-16"}}} },
  //     { resources: ... },  // same — 2 total
  //   ]
  repeated ReservationEntry entries = 1;
}

message LaunchJobRequest {
  // ... existing fields ...

  // Resource reservation. When set, the autoscaler pre-provisions capacity
  // and the reserving job is held pending until the reservation is satisfied.
  ReservationConfig reservation = 30;
}
```

Each `ReservationEntry` carries its own `ResourceSpecProto` and `Constraint`
list. A single reservation can target multiple device types, regions, or scale
groups by including entries with different specs. The 1:1 mapping (entry =
worker = demand entry) eliminates counting/deduplication complexity.

#### 2. Demand Routing

Today, `compute_demand_entries()` creates `DemandEntry` objects only from
pending tasks. With reservations, we add **reservation demand** that persists
regardless of task state.

```python
def compute_demand_entries(state: ControllerState) -> list[DemandEntry]:
    entries = []
    entries.extend(_demand_from_pending_tasks(state))
    entries.extend(_demand_from_reservations(state))
    return entries
```

`_demand_from_reservations` iterates non-terminal jobs with a `reservation`
field. Each entry maps 1:1 to a `DemandEntry`:

```python
def _demand_from_reservations(state: ControllerState) -> list[DemandEntry]:
    entries = []
    for job in state.jobs_with_reservations():
        if job.is_terminal:
            continue
        for idx, res_entry in enumerate(job.request.reservation.entries):
            entries.append(DemandEntry(
                task_ids=[f"{job.job_id}:reservation:{idx}"],
                coschedule_group_id=None,
                device_type=extract_device_type(res_entry.resources),
                device_variant=extract_device_variant(res_entry.resources),
                constraints=list(res_entry.constraints),
                resources=res_entry.resources,
                reservation_job_id=job.job_id,
                reservation_entry_idx=idx,
            ))
    return entries
```

The 1:1 mapping means no counting or deduplication logic. Each entry always
produces exactly one demand entry. The demand entries are **idempotent** —
they exist every autoscaler cycle for the lifetime of the reserving job,
keeping the autoscaler from scaling down the reserved capacity.

**Interaction with task demand**: When children are running on reserved
workers, both the reservation demand *and* the task demand exist. This is
fine — `route_demand()` counts total demand, and the autoscaler won't
double-provision because the workers already exist. The reservation demand
keeps the workers alive; the task demand is redundant but harmless.

#### 3. Idle Scale-Down Suppression

Today, `ScalingGroup.scale_down_if_idle()` terminates slices when
`ready_slices > target_capacity`. The target is `max(current_demand, min_slices)`.

Since reservation demand flows through `route_demand()` as normal `DemandEntry`
objects, `current_demand` already includes reservation demand. **No separate
suppression mechanism is needed** — the demand entries themselves prevent
scale-down.

This is the elegance of the approach: reservations are just persistent demand.

#### 4. Scheduling Gate: Block Reserving Job Until Satisfied

This is the critical behavioral change. The **reserving job itself** is held
pending until its reservation is fulfilled. This is a one-time gate — once
satisfied, the job starts running and never re-checks.

**Why gate the job, not the children?**

Children inherit parent constraints via `merge_constraints` in `client.py`.
When the parent job has `constraints: [region=us-east1]`, its children
automatically get that constraint. So once the parent starts running (meaning
us-east1 capacity exists), children will naturally schedule to us-east1
through the normal constraint mechanism. No ongoing gate needed.

**Rule**: A job with a `reservation` field is not schedulable until every
reservation entry has a corresponding fulfilled worker.

"Satisfied" means: for each entry in the reservation, at least one worker
tagged with the reservation's identity exists for that entry index. Since
entries are flat (no count), the check is: `fulfilled_count >= len(entries)`.

Implementation in the scheduling loop (`_run_scheduling`):

```python
def _is_reservation_satisfied(self, job: ControllerJob) -> bool:
    """Check if a job's reservation is fully satisfied (one-time gate)."""
    if not job.has_reservation:
        return True

    fulfilled = self._count_reservation_workers(job.job_id)
    return fulfilled >= len(job.request.reservation.entries)
```

During the scheduling cycle, the reserving job's tasks are skipped while
unsatisfied. They get a `pending_reason` diagnostic:
"Waiting for reservation: 2/6 entries fulfilled."

Once the reservation is satisfied and the job starts running, the gate is
done. The job submits children which inherit its constraints and schedule
normally. If preemption later reduces reserved capacity, that's handled by:
1. The reservation demand entries keep the autoscaler re-provisioning.
2. Children that were preempted retry via the normal retry mechanism.
3. New children inherit parent constraints and schedule to the right region.

#### 5. Reservation State and Lifecycle

The reservation is **derived state** on `ControllerJob`, not a separate entity:

```python
@dataclass
class ControllerJob:
    # ... existing fields ...

    @property
    def has_reservation(self) -> bool:
        return self.request.HasField("reservation")

    @property
    def reservation_entries(self) -> list:
        if not self.has_reservation:
            return []
        return list(self.request.reservation.entries)
```

**Lifecycle**:
- **Created**: When the job is submitted with a `reservation` field.
- **Gating**: Job stays PENDING while reservation is unsatisfied. Reservation
  demand entries flow to the autoscaler every cycle.
- **Satisfied**: All entries fulfilled. Gate opens, job schedules normally.
  Reservation demand continues flowing (keeps capacity alive).
- **Dissolved**: When the job reaches a terminal state (SUCCEEDED, FAILED,
  KILLED). Demand entries disappear, normal scale-down reclaims capacity.

**Fulfillment tracking**: The controller counts workers tagged with
`reservation-job=<id>`. This count is maintained by the autoscaler when it
registers/unregisters workers (see Worker Attribute Tagging).

#### 6. Worker Attribute Tagging

When the autoscaler provisions a slice in response to reservation demand,
it tags the workers with:

```python
def _per_group_worker_config(self, group, reservation_job_id=None,
                              reservation_entry_idx=None):
    wc = ...  # existing config building
    if reservation_job_id:
        wc.worker_attributes["reservation-job"] = reservation_job_id
        wc.worker_attributes["reservation-entry"] = str(reservation_entry_idx)
    return wc
```

Tags are used for:
1. **Fulfillment counting**: Gate check counts workers with matching
   `reservation-job` tag.
2. **Dashboard display**: Show which workers belong to which reservation.

Tags do not prevent other jobs from using the worker — if the reservation's
children don't fill all capacity, other tasks schedule there normally.

#### 7. Dashboard Integration

Add a "Reservation" section to the job detail page showing:

- Total entries vs. fulfilled count
- Overall status: "Provisioning (2/6 entries fulfilled)" or "Satisfied"
- Per-entry resource spec and constraints (grouped by unique spec)

This is built from the `ReservationStatus` in the job status response.

#### 8. Scheduler Affinity (Phase 2)

*Not required for the initial implementation.* Once the gate works, children
schedule via inherited constraints. Phase 2 can add soft affinity to prefer
reserved workers:

1. Tag reserved workers with `reservation-job=<job_id>`.
2. When scheduling a child task whose parent has a reservation, try workers
   with matching tag first (fast path via posting list).
3. Fall through to normal scheduling if no reserved worker has capacity.

This is an optimization, not a correctness requirement. The gate + constraint
inheritance already ensures correct region placement.

---

## Alternative Considered: Virtual Scale Group per Reservation

An alternative approach: when a reservation is created, dynamically create a
temporary `ScalingGroup` with `min_slices = num_slices` and constraints that
pin the reserving job's children to it.

**Advantages**:
- Uses existing autoscaler machinery unchanged
- `min_slices` enforcement already handles scale-up and scale-down suppression
- Clean isolation between reserved and unreserved capacity

**Disadvantages**:
- Scale groups are currently static (defined in config, created at startup).
  Dynamic creation requires significant refactoring of `Autoscaler.__init__`,
  config validation, and the dashboard.
- Scale groups have heavyweight state (slice tracking, backoff, demand history).
  Creating/destroying them per-job adds lifecycle complexity.
- Hard isolation: other jobs cannot use idle reserved capacity, reducing
  utilization.

**Verdict**: Too much machinery for the benefit. The demand-anchor approach
achieves the same result with less code and better utilization (reserved
capacity is shared when idle).

## Alternative Considered: Standalone Reservation Object

Create a `Reservation` proto and service with CRUD lifecycle independent of
any job. Jobs reference reservations by ID.

**Advantages**:
- Clean separation of concerns
- Reservations can outlive individual jobs
- Multiple jobs can share a reservation
- Explicit lifecycle management with dashboard visibility

**Disadvantages**:
- Requires new proto service, new state management, new cleanup logic
- Who owns the reservation? Who is responsible for releasing it?
- Orphaned reservations waste capacity with no accountability
- Significantly more code and concepts to maintain

**Verdict**: Overkill for the primary use case (parent pre-provisions for
children). If we need cross-job reservations later, we can extract the
demand-anchor mechanism into a standalone object. Start simple.

---

## Comparison Matrix

| Property | Demand Anchor (proposed) | Virtual Scale Group | Standalone Object |
|----------|------------------------|--------------------|--------------------|
| New protos | 1 message, 1 field | 0 (reuse existing) | New service + messages |
| New state | 0 (derived from job) | Dynamic group lifecycle | Full CRUD lifecycle |
| Autoscaler changes | Demand computation only | None | Demand computation + lifecycle |
| Scheduler changes | One-time gate (Phase 1), affinity (Phase 2) | Constraint injection | 2-pass affinity |
| Utilization | High (shared capacity) | Low (isolated) | High (shared) |
| Dashboard | Section on job detail | New group appears | New top-level page |
| Cleanup | Automatic (job terminal) | Must delete group | Must handle orphans |
| Complexity | Low | Medium | High |
| Extensibility | Can extract to standalone later | Hard to generalize | Already general |

---

## Detailed Design: Demand Anchor

### Data Flow

```
Job submitted with reservation:
  entries: [
    { resources: H100×8, constraints: [region=us-east1] },  ─┐
    { resources: H100×8, constraints: [region=us-east1] },   │ 4 H100 entries
    { resources: H100×8, constraints: [region=us-east1] },   │
    { resources: H100×8, constraints: [region=us-east1] },  ─┘
    { resources: v5p },                                      ─┐ 2 v5p entries
    { resources: v5p },                                      ─┘
  ]
    │
    ▼
compute_demand_entries()
    ├── pending task demand (existing)
    └── reservation demand: 6 DemandEntry objects (1:1 with entries)
            │
            ▼
        route_demand()
            │  entries 0-3 → "h100-us-east" group
            │  entries 4-5 → "v5p-default" group
            ▼
        Autoscaler.evaluate()
            │  scale up both groups
            ▼
        4 H100 workers + 2 v5p workers created
        workers tagged reservation-job=<id>, reservation-entry=0..5
            │
            ▼
Scheduler checks gate on the reserving job:
    ├── _is_reservation_satisfied()? → 6/6 workers tagged ✓
    │     gate opens, job starts running
    ▼
Job submits children (inherit parent constraints via merge_constraints)
    │  children get constraints: [region=us-east1]
    ▼
Children schedule normally via existing constraint matching
    → land in us-east1 because of inherited region constraint
```

### Edge Cases

**Preemption reduces capacity after gate opened**. If a reserved worker is
preempted after the job started:
1. The `TrackedWorker` is unregistered.
2. Reservation demand entry for that slot persists → autoscaler reprovisions.
3. Children on the preempted worker retry via normal retry mechanism.
4. New children still inherit parent's region constraint → schedule correctly.
5. The gate does NOT re-close (it's one-time). This is fine because the
   constraint inheritance handles region correctness.

**More children than reserved workers**. If the reservation has 4 H100 entries
but 6 children need H100s, the extra 2 compete for resources normally. The
reservation is a *minimum*, not a cap.

**Reservation never satisfied**. If the autoscaler can't provision capacity
(quota, no matching group), the reserving job stays PENDING. The job's
`scheduling_timeout` (if set) will eventually mark it UNSCHEDULABLE, same
as any other unschedulable job.

### Proto Changes

```protobuf
// In cluster.proto

message ReservationEntry {
  ResourceSpecProto resources = 1;
  repeated Constraint constraints = 2;
}

message ReservationConfig {
  repeated ReservationEntry entries = 1;
}

message LaunchJobRequest {
  // ... existing fields ...
  ReservationConfig reservation = 30;
}

message ReservationStatus {
  int32 total_entries = 1;     // len(entries)
  int32 fulfilled = 2;         // workers alive with reservation tag
  bool satisfied = 3;          // fulfilled >= total_entries
}

message JobStatus {
  // ... existing fields ...
  ReservationStatus reservation = 25;
}
```

### Code Changes Summary

#### Phase 1 (core: demand anchor + gate)

| File | Change | Size |
|------|--------|------|
| `cluster.proto` | Add `ReservationEntry`, `ReservationConfig`, `ReservationStatus` | S |
| `controller/controller.py` | `_demand_from_reservations()` in `compute_demand_entries()` | S |
| `controller/controller.py` | `_is_reservation_satisfied()` gate in `_run_scheduling` | S |
| `controller/state.py` | `jobs_with_reservations()` query | S |
| `controller/autoscaler.py` | `reservation_job_id`, `reservation_entry_idx` on `DemandEntry` | S |
| `controller/autoscaler.py` | Pass reservation tags through `_per_group_worker_config` | S |
| Tests | Demand computation, gate blocked/unblocked, autoscaler integration | M |

Estimated Phase 1: ~300 lines of production code, ~400 lines of tests.

#### Phase 2 (optimization: affinity + dashboard)

| File | Change | Size |
|------|--------|------|
| `controller/scheduler.py` | `reservation_affinity` on `JobRequirements`, 2-pass scheduling | M |
| `controller/controller.py` | Wire affinity into `job_requirements_from_job` | S |
| `cluster/static/controller/job-detail.js` | Reservation status section | S |
| Tests | Affinity scheduling, E2E lifecycle | M |

### Testing Strategy

**Phase 1 tests:**

1. **Unit: demand emission** — Job with reservation produces correct
   `DemandEntry` objects (1:1 with entries, correct device/constraints).

2. **Unit: scheduling gate** — Reserving job held PENDING when workers < entries.
   Job schedules when workers >= entries. Gate is one-time (doesn't re-close).

3. **Unit: autoscaler integration** — Reservation demand prevents scale-down.
   Heterogeneous entries route to different scale groups.

4. **E2E: basic lifecycle** — Submit job with reservation, verify workers
   provisioned, job unblocked, children schedule with inherited constraints.
   Cancel job, verify capacity reclaimed.

**Phase 2 tests:**

5. **Unit: scheduler affinity** — Children of reserving job prefer reserved
   workers, fall back to unreserved.

6. **E2E: preemption** — Kill reserved worker, verify autoscaler reprovisions,
   children retry correctly.

---

## Implementation Plan

Following the spiral approach from AGENTS.md:

### Phase 1: Demand Anchor + Scheduling Gate

This is the core value: "block my job until capacity exists."

1. Add `ReservationEntry`, `ReservationConfig`, `ReservationStatus` to
   `cluster.proto`, regenerate protos.
2. Add `reservation_job_id`, `reservation_entry_idx` fields to `DemandEntry`.
3. Implement `_demand_from_reservations()` in `controller.py`: iterate
   non-terminal jobs with reservations, emit one `DemandEntry` per entry.
4. Wire worker attribute tagging (`reservation-job`, `reservation-entry`)
   through autoscaler `_per_group_worker_config`.
5. Implement `_is_reservation_satisfied()` in controller: count workers
   tagged with `reservation-job=<id>`, compare to `len(entries)`.
6. Gate the reserving job's tasks in `_run_scheduling`: skip if unsatisfied,
   set `pending_reason` diagnostic.
7. Add `ReservationStatus` to `GetJobStatusResponse`.
8. Unit + E2E tests.

**Testable outcome**: A job with a 6-entry reservation (4 H100 + 2 v5p)
causes the autoscaler to provision both types. The job stays PENDING until
all 6 workers exist, then starts and submits children. Children inherit
region constraints and schedule to the right place. Cancelling the job
releases all reserved capacity.

### Phase 2: Scheduler Affinity + Dashboard

Optimization: children prefer reserved workers over arbitrary workers.

1. Add `reservation_affinity` to `JobRequirements`.
2. Implement 2-pass scheduling in `try_schedule_task`.
3. Add reservation section to job detail dashboard page.
4. E2E test for affinity and preemption recovery.

**Testable outcome**: Children preferentially land on reserved workers.
Dashboard shows reservation fulfillment status.

---

## Open Questions

1. **Should reservations have a timeout?** If a job creates a reservation
   but capacity never materializes, it stays PENDING forever (absent a
   `scheduling_timeout`). For v1, the existing `scheduling_timeout` field
   provides a natural bound. We could add a separate `reservation_timeout`
   later if needed.

2. **Reservation priority vs. normal demand**: Should reservation demand be
   higher priority than normal pending-task demand in `route_demand`? Higher
   priority would ensure reservations fulfill first but could starve
   non-reservation jobs.

3. **Entry validation**: Should we validate at submission time that
   reservation entries can be satisfied (matching scale groups exist, entries
   don't exceed `max_slices`)? Probably yes — fail fast with a clear error.

4. **Convenience helpers**: The flat entry list is simple but verbose for
   "4 identical H100 entries." The client SDK could provide a helper like
   `ReservationConfig.replicate(entry, count=4)` that expands to a flat
   list. This is a client-side convenience, not a proto change.
