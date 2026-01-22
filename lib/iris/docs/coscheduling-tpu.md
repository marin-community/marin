# TPU Coscheduling Design

## Overview

Coscheduling enables multi-host TPU jobs to be scheduled atomically onto complete TPU slices. A coscheduled job specifies a `group_by` attribute (e.g., `tpu-name`); all tasks must land on workers sharing that attribute value.

This design supersedes the existing `gang_id` infrastructure.

## Design Principles

1. **Scheduler is stateless**: Pure function taking state as input, returning assignments. All persistent state lives in `ControllerState`.

2. **Single source of truth**: Worker capacity tracked exclusively in `ControllerWorker.committed_*` fields. Scheduler builds transient snapshots each cycle.

3. **Commit-then-dispatch**: Resources committed via `TaskAssignedEvent` before RPC. On failure, resources released via `TaskDispatchFailedEvent`.

4. **Parallel dispatch with timeouts**: RPCs dispatched in parallel via thread pool with 5s timeout to prevent slow workers from blocking the control plane.

## Data Model

### Worker Attributes

Workers report typed attributes (`string | int | float`) during registration:

- `tpu-name`: TPU slice identifier (e.g., `"my-tpu-v4"`)
- `tpu-worker-id`: Worker index within slice (0, 1, 2, ...)
- `tpu-topology`: Topology variant (e.g., `"v4-8"`)
- `taint:<name>`: Taint markers for maintenance, etc.

### Job Constraints

Jobs specify constraints to filter eligible workers:

| Operator | Meaning |
|----------|---------|
| `EQ` / `NE` | Equality / inequality |
| `EXISTS` / `NOT_EXISTS` | Attribute presence |
| `GT` / `GE` / `LT` / `LE` | Numeric comparisons (error on strings) |

All constraints must match for a worker to be eligible.

### Coscheduling Config

Jobs requesting coscheduling specify `group_by` attribute:

```
coscheduling.group_by = "tpu-name"
```

The scheduler finds a worker group where:
1. All workers share the same `group_by` value
2. Enough workers have capacity for all tasks
3. Tasks are assigned atomically (all-or-nothing)

## Scheduling Flow

```
1. Build WorkerCapacity snapshots from current state
2. For coscheduled jobs:
   - Group workers by `group_by` attribute
   - Find first group with enough capacity
   - Assign all tasks atomically, sorted by tpu-worker-id
3. For regular jobs:
   - First-fit matching with constraint filtering
4. Return SchedulingResult (assignments + timed-out tasks)
```

The scheduler never mutates persistent state—it returns proposed assignments that the controller commits via events.

## Dispatch Flow

```
1. Acquire scheduler lock
2. Get pending tasks and available workers
3. Run scheduler.find_assignments()
4. For each assignment:
   - Commit resources via TaskAssignedEvent
5. Dispatch RPCs in parallel (5s timeout)
6. On RPC failure: release via TaskDispatchFailedEvent
7. Release lock
```

Key insight: Worker's `run_task` RPC returns immediately (execution is async), so RPC latency is just network round-trip.

## Failure Handling

### Coscheduled Group Failure

When one task in a coscheduled job fails terminally:
1. All running sibling tasks are killed
2. Resources are released on their workers
3. Job transitions to FAILED state

This prevents wasted compute from partially-running distributed jobs where collective ops would timeout anyway.

### Dispatch Failure Recovery

If RPC dispatch fails after resource commitment:
1. `TaskDispatchFailedEvent` fired
2. Task reverted to PENDING state
3. Worker resources released
4. Task will be rescheduled next cycle

## Validation

Coscheduled jobs are validated at submission:
- Must specify TPU device
- Replica count must match TPU topology's VM count

## Out of Scope

**Preemption**: This design uses first-fit without preemption. Large coscheduled jobs may wait if the cluster is fragmented. Mitigations:
- Dedicated TPU pools via constraints
- Starvation monitoring/alerting

Preemption support is deferred to future work.
