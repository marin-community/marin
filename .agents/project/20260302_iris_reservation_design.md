# Reservation System

A **reservation** lets a job declare its resource envelope ahead of time:
"I will need 4× H100 workers and 2× v5p workers." The autoscaler provisions
capacity proactively, and the **reserving job is held pending until the
reservation is satisfied**. Once the job starts, its children inherit
constraints (including region) via `merge_constraints` in `client.py`.

## Problem

Iris scheduling is purely reactive. A parent job submits children that need
capacity, the autoscaler sees demand, provisions slices, and everyone waits
5+ minutes. With no advance knowledge of the total resource envelope, the
scheduler can't make globally optimal placement decisions.

## Design Principles

- **Reservation = persistent demand**: No separate lifecycle object. The
  reservation is a field on `LaunchJobRequest` tied to the job's lifecycle.
- **Autoscaler is unmodified**: Reservation demand looks identical to task
  demand. The autoscaler just sees numbers and scales.
- **Scheduler is unmodified**: The controller injects taint attributes onto
  worker copies and taint constraints onto job requirements. The scheduler
  evaluates constraints as usual.
- **Best-effort convergence**: Reservations are hints, not SLAs. Preemptions
  reduce capacity; the system re-converges.
- **Reservations are a floor, not a ceiling**: Reserved jobs can use both
  claimed and unclaimed workers.
- **Incremental adoption**: Jobs without reservations behave exactly as before.

## Proto

```protobuf
message ReservationEntry {
  ResourceSpecProto resources = 1;
  repeated Constraint constraints = 2;
}

message ReservationConfig {
  // Each entry = one worker. Duplicate entries to reserve multiple workers
  // of the same type. 1:1 mapping: entry → DemandEntry.
  repeated ReservationEntry entries = 1;
}

message LaunchJobRequest {
  // ...
  ReservationConfig reservation = 30;
}

message ReservationStatus {
  int32 total_entries = 1;
  int32 fulfilled = 2;
  bool satisfied = 3;
}

message JobStatus {
  // ...
  ReservationStatus reservation = 25;
}
```

## Mechanism

### 1. Demand Pumping

`_demand_from_reservations()` generates plain `DemandEntry` objects — no
reservation metadata. All entries generate demand regardless of claim status,
preventing scale-down below reservation needs.

### 2. Demand Deduplication

Reservation entries and pending tasks can describe overlapping demand. Without
deduplication the autoscaler over-provisions. Deduplication is resource-aware:
an H100 reservation entry only absorbs H100 task demand.

| Reservation entries | Pending tasks | Total demand |
|---------------------|---------------|--------------|
| 2 H100              | 0             | 2            |
| 2 H100              | 2 H100        | 2            |
| 2 H100              | 5 H100        | 5            |
| 2 H100              | 2 A100        | 4            |

Per-job budget: `demand = max(reservation_count, task_count)` per
`(device_type, device_variant)`.

### 3. Worker Claiming

Each scheduling cycle, the controller scans unclaimed workers and assigns them
to unsatisfied reservation entries. Claims are tracked in
`_reservation_claims: dict[WorkerId, ReservationClaim]`.

### 4. Taint Injection

Before scheduling, the controller:
- Creates modified worker copies with `reservation-job=<id>` attribute on
  claimed workers (via `dataclasses.replace()` — avoids mutating shared state).
- Injects `NOT_EXISTS reservation-job` constraint on non-reservation jobs.
- Reservation jobs get no restrictive constraints — they can use claimed +
  unclaimed workers, with claimed workers ordered first for affinity.

### 5. Scheduling Gate

The reserving job stays PENDING until `fulfilled >= len(entries)`. Once
satisfied, the gate opens and the job schedules normally. The gate is one-time:
it doesn't re-close on preemption.

### 6. Cleanup

- Worker removed from state → claim removed from `_reservation_claims`.
- Job terminates → all claims for that job released; workers become available.

## Data Flow

```
Job submitted with reservation (2 entries)
    │
    ▼
_demand_from_reservations() → 2 plain DemandEntry objects
    │  autoscaler unaware these are for reservations
    ▼
Autoscaler scales up 2 generic workers
    ▼
_claim_workers_for_reservations()
    │  matches by device_type/variant + constraints
    │  sets _reservation_claims[worker_id] = ReservationClaim(job_id, entry_idx)
    ▼
_run_scheduling()
    │  injects claim attributes on worker copies
    │  injects NOT_EXISTS taint on non-reservation jobs
    ▼
Gate opens (2/2 claimed) → job schedules
    ▼
Children inherit parent constraints via merge_constraints
```

## Failure Scenarios

**Worker dies before job schedules**: Cleanup removes stale claim → demand
persists → autoscaler reprovisions → new worker claimed → gate re-opens.
Self-healing.

**Worker dies while job is running**: Task retry handles execution failures.
Reservation re-claims a replacement. If the job fails entirely, all claims
are released.

**Multiple reservations compete**: Each reservation's entries generate
independent demand. Autoscaler sees total demand and provisions enough for
all. Claims are first-come-first-served per cycle.

**Reservation never satisfied**: Job stays PENDING. Existing
`scheduling_timeout` provides the bound.

## Invariants

- Each worker maps to at most one claim. Each entry claimed by at most one worker.
- Claims per job never exceed number of reservation entries.
- Demand entries always emitted for every entry (prevents scale-down).
- Demand per reservation job per `(device_type, variant)` =
  `max(reservation_count, task_count)`.
- Cleanup → claiming → scheduling all run in the same thread — no races.

## Alternatives Considered

**Autoscaler-based approach**: Mixing reservation tags into `DemandEntry` and
having the autoscaler tag workers created cascading timing issues (1-minute
cooldown per group, demand routing competition, worker registration races).
Abandoned in favor of the taint-based approach.

**Virtual scale group per reservation**: Dynamic `ScalingGroup` with
`buffer_slices`. Too much machinery, hard isolation wastes capacity.

**Standalone reservation object**: Independent CRUD lifecycle. Overkill —
orphaned reservations, complex cleanup. Can extract later if needed.

## Soft Worker Preference for Reservation Jobs

### Problem

The current taint/constraint mechanism keeps non-reservation jobs off claimed
workers (via `NOT_EXISTS`), but does not steer reservation jobs *toward* their
claimed workers. The scheduler iterates `candidate_ids` — a `set` intersection
— so the "claimed workers first" ordering produced by
`_inject_reservation_taints` is lost. Reservation jobs can land on unclaimed
workers while their claimed workers sit idle and remain tainted (blocked for
regular jobs). This wastes reserved capacity and can trigger unnecessary
scale-up.

Reservations are a **floor**, not a ceiling — reservation jobs should be allowed
to use unclaimed workers too. So a hard constraint (`EQ reservation-job=<id>`)
is wrong. What we need is a soft preference: try claimed workers first, fall
back to any eligible worker.

### Proposed Fix: Two-Pass Scheduling in `_run_scheduling`

The claimed set per job is small (equal to the reservation entry count, typically
1–8). We can iterate it directly in the controller without touching the
scheduler's constraint engine.

**Approach**: Before calling `scheduler.find_assignments()`, do a first pass
over reservation-job tasks. For each such task, iterate the (small) set of
workers claimed for that job and check if any has capacity. If so, assign
directly and remove the task from the pending list. Remaining tasks fall through
to the normal scheduler path.

```python
def _run_scheduling(self) -> None:
    # ... existing setup: cleanup, claiming, gate checks ...

    # -- NEW: soft-preference pass for reservation jobs --
    # Iterate claimed workers for each reservation job. Since the claims set
    # is small (≤ reservation entry count), this is O(tasks × claims) which
    # is negligible.
    early_assignments: list[tuple[JobName, WorkerId]] = []
    remaining_tasks: list[JobName] = []

    # Build reverse index: job_id -> set of claimed worker IDs
    claimed_by_job: dict[str, set[WorkerId]] = defaultdict(set)
    for wid, claim in self._reservation_claims.items():
        claimed_by_job[claim.job_id].add(wid)

    for task_id in schedulable_task_ids:
        job_id = task_id.parent
        if job_id is None or job_id not in has_reservation:
            remaining_tasks.append(task_id)
            continue

        job_wire = job_id.to_wire()
        claimed_workers = claimed_by_job.get(job_wire, set())
        assigned = False
        for wid in claimed_workers:
            worker = ... # look up in available workers
            if worker is not None and _has_capacity(worker, jobs[job_id]):
                early_assignments.append((task_id, wid))
                assigned = True
                break
        if not assigned:
            remaining_tasks.append(task_id)

    # Buffer early assignments, then run normal scheduling on remaining tasks
    if early_assignments:
        self._buffer_assignments(early_assignments)

    # Pass remaining_tasks (not schedulable_task_ids) to the scheduler
    context = self._scheduler.create_scheduling_context(
        modified_workers,
        building_counts=building_counts,
        pending_tasks=remaining_tasks,
        jobs=jobs,
    )
    result = self._scheduler.find_assignments(context)
    # ...
```

**Key properties**:

- **Soft, not hard**: If no claimed worker has capacity, the task falls through
  to the normal scheduler and can land on any eligible worker.
- **No scheduler changes**: The scheduler remains a pure constraint engine.
  Preference logic stays in the controller where reservation state lives.
- **O(tasks × claims)**: Claims per job ≤ entry count (typically 1–8).
  The inner loop is bounded by the reservation size, not the cluster size.
- **Idempotent**: The preference pass uses the same capacity-checking logic as
  the scheduler. An early assignment commits resources identically to a
  scheduler assignment.

### Implementation Notes

- The capacity check needs to use the scheduler's `WorkerCapacity.can_fit()`
  and `deduct()` methods. The simplest path is to create a `SchedulingContext`
  first and use it for both the preference pass and the scheduler pass.
- The preference pass should respect building back-pressure (skip workers at
  the building limit).
- Non-reservation tasks must NOT go through the preference pass — they should
  only use the normal scheduler path with the `NOT_EXISTS` taint constraint.

## Phase 2 (Future)

- `reservation_timeout` separate from `scheduling_timeout`
- Entry validation at submission time
- Client-side `ReservationConfig.replicate(entry, count=N)` helper

