# Issue #5470: TPU Placement Collision — Root Cause Analysis

## Status: Root cause identified — NOT a scheduler bug, it's a worker cleanup race

## What happened (from controller logs)

Incident B (v5p-256, 32 VMs/slice) — the most documented case:

```
01:38:36  lr0.5 parent submitted
01:38:38  lr0.67 parent submitted
01:47:09  Both train_lm children submitted (32 tasks each)

--- Initial placement: DIFFERENT slices (correct) ---
09:14:31  lr0.5  assigned → slice 389585fe (32 workers)
09:14:34  lr0.67 assigned → slice 2e06c8f1 (32 workers)

--- Both run on separate slices for ~9.5 hours ---

--- GCP preemption cycle begins ---
12:58:19  lr0.67 reassigned → slice c4535e6f (different slice, still separate)

--- THE COLLISION ---
18:43:32  lr0.5  reassigned → slice 8996b868
18:43:58  lr0.67 reassigned → slice 8996b868  ← SAME SLICE, 26s later
18:43:57  lr0.5  train_lm terminated (25s after its assignment!)
18:44:07  lr0.67 train_lm terminated
```

The survivor (lr0.33) was preempted and reassigned **4 times** across different
slices, always landing alone. The cluster had multiple v5p-256 slices cycling
through GCP preemption.

## Key finding

**The collision is NOT at initial placement.** Both jobs were correctly placed
on separate slices at 09:14. The collision happened ~9.5 hours later during a
GCP preemption-induced reassignment wave.

The 26-second gap between the two reassignments to slice `8996b868` means they
were in **different scheduling ticks** (scheduler min interval = 1s). After
lr0.5 was committed to `8996b868`, `committed_tpu` should have been 4 on each
worker — blocking lr0.67 in the next tick.

**But lr0.5 was terminated 25 seconds after assignment (18:43:32 → 18:43:57).**
This means lr0.5's tasks failed during BUILDING, triggering
`_requeue_coscheduled_siblings` which decommitted `committed_tpu` back to 0.
lr0.67 was then assigned to the now-free slice at 18:43:58 — 1 second after
lr0.5's decommit. Then lr0.5's retry would also target the same slice.

## Confirmed facts from production data

- `has_reservation = 0` on ALL affected jobs. **Reservations are not involved.**
- `res_device_json = {"tpu": {"variant": "v5p-256", "count": 4}}` — tpu.count
  is correctly set in the DB.
- `has_coscheduling = 1`, `coscheduling_group_by = "tpu-name"` on all train_lm
  children.
- Parent jobs are plain CPU executor jobs (`res_device_json = "{}"`).
- The cluster had multiple v5p-256 slices, all preemptible, cycling through GCP
  preemption events.

## What we tested and ruled out

| Hypothesis | Tests | Result |
|---|---|---|
| tpu.count=0 in production | DB query | **Ruled out**: count=4 in all jobs |
| Reservations involved | DB query | **Ruled out**: has_reservation=0 |
| Same-tick double-booking (tpu.count=4) | 14 tests | **Ruled out**: in-memory deduction works |
| Cross-tick double-booking (tpu.count=4) | 6 tests | **Ruled out**: committed_tpu blocks correctly |
| Reservation resource release | 21 tests | **Ruled out**: lifecycle is correct |
| Committed resource double-decommit | 17 tests | **Ruled out**: accounting is correct |
| Preemption requeue accounting | 12 tests | **Ruled out** (with tpu.count=4) |
| Taint injection bypass | 8 tests | **Ruled out**: no taint gap with tpu.count=4 |
| Full controller pipeline | 10 tests | **Ruled out**: matches scheduler-only behavior |
| **Exact production sequence** | 4 tests | **Ruled out**: preempt both slices → new slice → race → no collision |

Total: **92 tests** with tpu.count=4, all passing. The scheduler is correct
when tested through the state machine (ControllerTransitions).

## The exact production reproduction test

`test_5470_preemption_reassignment.py` simulates the exact incident:
1. Parent CPU jobs + child coscheduled TPU gangs (matching DB schema)
2. Initial placement on separate slices (2 slices, 2 gangs)
3. GCP preemption of both slices (via `fail_worker` → `_requeue_coscheduled_siblings`)
4. New slice appears
5. Tick 1: Job A assigned to new slice, committed_tpu=4
6. Job A fails during BUILDING (25s gap from logs) → decommit → committed_tpu=0
7. Tick 2: Job B assigned to same slice (correct — resources were freed)
8. Tick 3: Job A retried — **BLOCKED by Job B's committed_tpu** ✓

With `tpu.count=4`, the scheduler correctly prevents the collision at every
step. The in-memory deduction works within ticks, and committed_tpu in the DB
works across ticks.

## What we still don't understand

The production collision DID happen (controller logs prove it). But we cannot
reproduce it through the test state machine. The gap must be in something our
tests don't exercise:

1. **Concurrency between scheduling and heartbeat threads.** The production
   controller has separate threads for scheduling and heartbeat processing.
   The test exercises both paths but sequentially (single-threaded). A real
   race between the two threads could create a window where committed_tpu is
   transiently incorrect.

2. **Worker registration refresh during preemption.** When a GCP preemptible
   TPU slice is reclaimed and a new one spins up, the new workers register with
   the controller. The `upsert` ON CONFLICT clause preserves `committed_*`
   counters — but the new workers have worker IDs that have NEVER had committed
   resources. They start at committed_tpu=0. If the old workers (with
   committed resources) are pruned before the scheduling tick reads them, the
   new workers appear free.

3. **Worker pruning during the race window.** The `find_prunable` method
   removes workers that are unhealthy and haven't heartbeated recently. If
   slice-1's workers are pruned (removing their committed_tpu entries), and
   slice-3's NEW workers register with committed_tpu=0, the scheduler sees a
   clean slate for the new slice — regardless of what was committed on the old
   slice.

   BUT — the workers on the new slice-3 (`8996b868`) are genuinely new workers
   with no prior committed resources. Job A gets assigned to them in tick N,
   committed_tpu becomes 4. Job B reads committed_tpu=4 in tick N+1 and should
   be blocked. The only way Job B gets through is if committed_tpu was reset to
   0 between ticks — which happens if Job A's tasks fail and trigger a requeue.

4. **The 25-second window.** lr0.5 was assigned at 18:43:32 and terminated at
   18:43:57. lr0.67 was assigned at 18:43:58. The sequence is:
   - 18:43:32: lr0.5 assigned, committed_tpu=4
   - 18:43:57: lr0.5 terminated → decommit → committed_tpu=0
   - 18:43:58: lr0.67 assigned, committed_tpu=4

   This means lr0.67 was assigned AFTER lr0.5 was decommitted. This is
   correct behavior! The scheduler gave the slice to lr0.67 because lr0.5
   no longer held it. But then lr0.5 retried and ALSO got the same slice —
   creating the collision.

   **The collision is between lr0.67 (newly assigned) and lr0.5 (retrying)
   in a SUBSEQUENT tick.** Our test covers this exact path and it doesn't
   reproduce — the retry is blocked by lr0.67's committed_tpu.

## Root cause: worker cleanup race, NOT scheduler double-booking

Detailed controller logs for slice `8996b868` show the **exact sequence**:

```
18:38:44     Autoscaler creates slice 8996b868 (v5p-256)
18:43:20-32  32 workers register over ~12 seconds
18:43:32     lr0.5 assigned to all 32 workers (committed_tpu=4 per worker)
18:43:57     lr0.5 train_lm TERMINATED (event=job_terminated, 25s after assignment)
             → resources decommitted, committed_tpu=0 on all workers
18:43:58     lr0.67 assigned to same 32 workers (1 second after lr0.5 freed them)
18:44:07     lr0.67 train_lm TERMINATED
```

**The scheduler did NOT double-book.** It assigned lr0.5, lr0.5 failed and was
fully terminated (not retried — `event=job_terminated`), resources were freed,
and lr0.67 was legitimately assigned to the now-free workers. This is correct
scheduler behavior — it gave the slice to the next pending gang after the
first one released it.

The collision manifests as a **JAX coordinator port conflict** because:
1. lr0.5's tasks started on the workers and began binding port 8476
2. lr0.5 failed during BUILDING (25 seconds later) — possibly because its
   JAX init conflicted with lr0.33 which was already running on a different
   slice? Or a container build failure.
3. The controller terminated lr0.5 and sent kill RPCs to the workers
4. 1 second later, lr0.67's tasks were dispatched to the same workers
5. lr0.67's tasks arrived on workers where lr0.5's processes were still
   being cleaned up — the kill hadn't fully completed
6. Both sets of processes now compete for port 8476 on the same hosts

This is a **worker-side process cleanup race**, not a scheduler placement bug.
The fix should ensure:
- Kill RPCs are awaited before reassigning workers, OR
- Tasks use job-scoped ports (not a fixed port 8476), OR
- The JAX coordinator init is made idempotent/conflict-tolerant

## Test files created during investigation

- `test_tpu_placement_collision.py` — 19 tests (initial same-tick/cross-tick probes)
- `test_reservation_resource_release.py` — 21 tests (reservation lifecycle)
- `test_preemption_coscheduled_collision.py` — 16 tests (preemption paths)
- `test_taint_coscheduled_bypass.py` — 10 tests (taint injection)
- `test_committed_resource_accounting.py` — 17 tests (DB accounting)
- `test_full_pipeline_collision.py` — 12 tests (full Controller pipeline)
- `test_5470_preemption_reassignment.py` — 4 tests (exact production sequence)

Controller logs saved to `.agents/ops/5470_controller_logs.txt`.
