# Taint-Based Reservation System

## Context

The autoscaler-based approach to reservation fulfillment failed. Mixing reservation
logic into the autoscaler created cascading timing issues: 1-minute cooldown between
scale-ups per group, demand routing competition, and worker registration races made
it impossible for reservation gates to open reliably.

Reservations are now a **consumable resource** managed by the controller/scheduler.
The autoscaler remains **entirely unmodified** — it just sees demand numbers and
scales up/down. Reserved workers get a **taint** preventing normal jobs from using
them.

## Mechanism

1. **Demand pumping** (existing, simplified): `_demand_from_reservations()` generates
   plain `DemandEntry` objects — no reservation metadata on DemandEntry. The autoscaler
   sees demand, scales up generic workers. All reservation entries generate demand
   regardless of claim status (prevents scale-down of claimed workers).

2. **Worker claiming** (new): Each scheduling cycle, the controller scans unclaimed
   workers and assigns them to unsatisfied reservation entries. Claims are tracked in
   `_reservation_claims: dict[WorkerId, ReservationClaim]`.

3. **Taint injection** (new): Before scheduling, the controller:
   - Creates modified worker copies with `reservation-job=<id>` attribute on claimed
     workers.
   - Injects `NOT_EXISTS reservation-job` constraint on non-reservation jobs to prevent
     them from landing on claimed workers.
   - Reservation jobs do NOT get restrictive constraints — they can use both their
     claimed workers and any unclaimed workers. The reservation is a minimum guarantee,
     not a ceiling.
   - Reservation jobs _prioritize_ their claimed workers over unclaimed workers, e.g. by scanning through these first during scheduling or similar.
   - Uses `dataclasses.replace()` on workers to avoid mutating shared state (the
     heartbeat thread overwrites `worker.attributes` every cycle).

4. **Gate check** (existing, updated): `_is_reservation_satisfied()` uses the claims
   map instead of scanning worker attributes.

5. **Cleanup**:
   - Worker removed from state → controller removes from `_reservation_claims`.
   - Job terminates → controller removes all claims for that job; workers become
     available for other jobs or other reservations.

## Data Flow

```
Job submitted with reservation (2 entries)
    │
    ▼
_demand_from_reservations()
    │  2 plain DemandEntry objects (no reservation tag)
    │  autoscaler is unaware these are for reservations
    ▼
Autoscaler scales up 2 generic workers
    │
    ▼
Workers register (generic, no reservation attributes)
    │
    ▼
_claim_workers_for_reservations() [scheduling cycle]
    │  matches workers by device_type/variant + constraints
    │  sets _reservation_claims[worker_id] = ReservationClaim(job_id, entry_idx)
    ▼
_run_scheduling()
    │  creates worker copies with claim attributes injected
    │  injects NOT_EXISTS taint on non-reservation jobs
    │  reservation jobs can use claimed + unclaimed workers
    │  scheduler operates normally on modified inputs
    ▼
Gate opens (2/2 claimed) → job schedules on claimed workers
```

## Demand Deduplication

Reservation entries and pending tasks can describe overlapping demand. Without
deduplication the autoscaler sees double demand and over-provisions.

Deduplication is **resource-aware**: a reservation entry for an H100 only
absorbs task demand for H100s, not for A100s or CPUs. For each reservation
job, we build a budget from its reservation entries keyed by
`(device_type, device_variant)`. Task demand that matches a budget entry
consumes one unit; unmatched task demand passes through.

| Reservation entries  | Pending tasks        | Reservation demand | Task demand | Total |
|----------------------|----------------------|--------------------|-------------|-------|
| 2 H100               | 0                    | 2 H100             | 0           | 2     |
| 2 H100               | 2 H100               | 2 H100             | 0           | 2     |
| 2 H100               | 5 H100               | 2 H100             | 3 H100      | 5     |
| 2 H100               | 2 A100               | 2 H100             | 2 A100      | 4     |
| 2 H100, 1 A100       | 3 H100, 2 A100       | 2 H100, 1 A100     | 1 H100, 1 A100 | 5  |

Implementation: `compute_demand_entries` builds a per-job reservation budget
before emitting task demand.

```python
# Build budget from reservation entries
budget: Counter[tuple[DeviceType, str | None]] = Counter()
for res_entry in job.request.reservation.entries:
    key = (get_device_type_enum(res_entry.resources.device),
           get_device_variant(res_entry.resources.device))
    budget[key] += 1

# Emit task demand, absorbing matches against budget
for task in tasks:
    key = (device_type, device_variant)
    if budget[key] > 0:
        budget[key] -= 1
        continue  # reservation demand covers this task
    demand_entries.append(...)
```

## Invariants

- `_reservation_claims` is a `dict[WorkerId, ReservationClaim]`. Each worker
  maps to at most one claim. Each reservation entry is claimed by at most one
  worker.
- The number of claims for a job never exceeds the number of reservation entries.
- Demand entries are always emitted for every reservation entry (claimed or not).
  This prevents the autoscaler from scaling down below the reservation's needs.
- For each `(device_type, device_variant)`, demand per reservation job is
  `max(reservation_count, task_count)`, never the sum. Deduplication is
  resource-aware — an H100 reservation entry only absorbs H100 task demand.
- Cleanup runs before claiming. Claiming runs before scheduling. All in the same
  thread — no races.

## Why This Works

- **No autoscaler changes**: Reservation demand looks identical to task demand.
  The autoscaler scales up because demand > capacity. No reservation tags,
  no pending tracking, no cooldown issues.
- **No scheduler changes**: The scheduler evaluates constraints against worker
  attributes. The controller injects claim attributes onto worker copies and
  taint constraints onto job requirements. The scheduler doesn't know about
  reservations.
- **No timing issues**: Claiming happens in the scheduling loop, same thread
  as scheduling. No race between "tag worker" and "check gate."
- **Reservations are a floor, not a ceiling**: Reservation jobs prefer their
  claimed workers but can also use any unclaimed worker. This maximizes
  utilization.

## Prioritization

Reservation jobs should schedule onto their claimed workers before using
unclaimed ones. The controller achieves this by ordering the worker list:
claimed workers (for any reservation) are placed at the front. Since
non-reservation jobs have a `NOT_EXISTS reservation-job` constraint, they skip
past these workers and find unclaimed ones further in the list. Reservation jobs
have no such constraint, so they naturally pick from the front — their claimed
workers.

## Failure Scenarios

### Worker dies after being claimed, before job schedules

```
Worker W claimed for job J, entry 0
W disappears from controller state
    │
    ▼
_cleanup_stale_claims()          ← removes W from claims
    │  reservation now unsatisfied (1/2 claimed)
    │  gate stays closed, job J does not schedule
    ▼
_demand_from_reservations()      ← still emits 2 demand entries
    │  autoscaler sees demand > capacity, scales up
    ▼
New worker W' registers
    │
    ▼
_claim_workers_for_reservations() ← claims W' for entry 0
    │  reservation satisfied (2/2), gate opens
```

Self-healing. No special handling needed — the normal cleanup → claim → schedule
cycle covers it.

### Worker dies while job is running on it

```
Job J running on workers [W1, W2], both claimed
W1 crashes
    │
    ▼
_cleanup_stale_claims()          ← removes W1 from claims
    │  reservation now unsatisfied (1/2)
    ▼
Job J's tasks on W1 fail         ← normal task failure handling
    │  tasks return to pending (or job fails, depending on retry policy)
    ▼
Autoscaler sees demand, scales up replacement
New worker claimed, gate re-satisfied
    │
    ▼
Retried tasks schedule on new worker
```

Worker failure during execution is handled by the existing task retry mechanism.
The reservation system just needs to re-claim a replacement worker. If the job's
retry policy causes the job to fail entirely, `_cleanup_stale_claims()` releases
all claims for that job on the next cycle.

### Job cancelled before all workers claimed

```
Job J cancelled (2 entries, 1 claimed on W)
    │
    ▼
_cleanup_stale_claims()          ← sees job J is finished
    │  removes W from claims
    │  W is now available for other reservations or normal jobs
    ▼
_demand_from_reservations()      ← skips J (is_finished), no demand emitted
    │  autoscaler sees reduced demand, may scale down
```

Clean release. Workers freed immediately on the next scheduling cycle.

### Multiple reservations competing for the same worker type

```
Job A: reservation with 2 entries (gpu)
Job B: reservation with 3 entries (gpu)
    │  total demand: 5 gpu workers
    ▼
Autoscaler scales up 5 workers
    │
    ▼
_claim_workers_for_reservations()
    │  iterates jobs in order, claims first-come-first-served
    │  A gets 2 workers, B gets 3 workers
    │  each worker claimed by exactly one reservation
```

No starvation: all reservation entries generate demand regardless of which job
claims first. The autoscaler sees total demand = 5 and scales up enough for
everyone.

### Reservation job starts tasks that mirror the reservation

```
Job J: reservation with 2 H100 entries, submits 2 H100 tasks
    │
    ▼
compute_demand_entries()
    │  reservation demand: 2 H100 entries (anchor)
    │  task budget: {H100: 2}
    │  task demand: 2 H100 tasks absorbed by budget → 0 emitted
    │  total: 2 demand entries ← correct, not 4
    ▼
Autoscaler sees demand = 2, capacity = 2, no action
```

### Reservation job starts tasks with different resources

```
Job J: reservation with 2 H100 entries, submits 2 A100 tasks
    │
    ▼
compute_demand_entries()
    │  reservation demand: 2 H100 entries
    │  task budget: {H100: 2}
    │  task demand: 2 A100 tasks don't match budget → 2 emitted
    │  total: 4 demand entries (2 H100 + 2 A100)
    ▼
Autoscaler scales up 2 A100 workers (H100s already exist)
```

### Reservation job starts more tasks than entries

```
Job J: reservation with 2 H100 entries, submits 5 H100 tasks
    │
    ▼
compute_demand_entries()
    │  reservation demand: 2 H100 entries
    │  task budget: {H100: 2}
    │  task demand: first 2 H100 tasks absorbed, 3 excess emitted
    │  total: 5 demand entries
    ▼
Autoscaler scales up 3 more H100 workers for excess tasks
```

Excess tasks have no reservation — they schedule on unclaimed workers.

### Autoscaler scales down a claimed worker

Can this happen? Consider: 2 reservation entries, 2 workers claimed, no other
demand. Autoscaler sees demand = 2, capacity = 2 — no scale-down. The demand
entries for reservation entries are always emitted, so the autoscaler always
sees demand >= number of reservation entries.

The only way a claimed worker disappears is if the cloud provider terminates it
(preemption, hardware failure). This is equivalent to "worker dies after being
claimed" above — self-healing via re-claim.
