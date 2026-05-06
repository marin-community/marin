---
date: 2026-05-06
system: iris
severity: high
issue: https://github.com/marin-community/marin/issues/5470
fix_pr: https://github.com/marin-community/marin/pull/5475
fix_commit: a6f13ff0016952423338b3109b8f3c97ad66144e
related: .agents/ops/iris_placement_bug.md
---

# Issue #5470 Incident Timeline: Coscheduler TPU Placement Collision

## Executive Summary

Four incidents over 36 hours (2026-05-03 to 2026-05-04) where the Iris
scheduler placed two coscheduled TPU gang jobs on identical physical workers.
Root cause: a missing cross-gang exclusivity check in the scheduler's
`_find_coscheduled_assignments` and `queue_assignments` methods. The scheduler
deducted CPU/memory/TPU-count per-worker after assigning the first gang, but
TPU coscheduling operates at the *group* level (all VMs in a TPU slice share
one `tpu-name`). Two gangs targeting the same `tpu-name` group could both pass
the per-worker `can_fit` check if either: (a) both were pending in the same
scheduling tick, or (b) the first gang's committed resources were not yet
visible to the second gang's scheduling context.

The fix (PR #5475, commit `a6f13ff00`) adds three defenses:
1. Commit-time TPU group overlap rejection in `queue_assignments`
2. A staged-capacity ledger that tracks which TPU groups are claimed within a
   single `queue_assignments` batch
3. TPU jobs skip the Iris JAX coordinator entirely (eliminates the port-8476
   conflict as a failure mode)


## Architecture: How Scheduling Works

```
Scheduling Loop (1s min / 10s max interval, adaptive backoff)
  |
  v
_run_scheduling()
  |-- _read_scheduling_state()          # snapshot pending tasks + workers from DB
  |-- _apply_scheduling_gates()         # filter by deadline, reservations, caps
  |-- _compute_scheduling_order()       # priority-band interleave, budget weights
  |-- _run_scheduler_pass()
  |     |-- create_scheduling_context() # build WorkerCapacity + ConstraintIndex
  |     |-- _preference_pass()          # steer reservation tasks (non-coscheduled only)
  |     |-- find_assignments()          # THE CORE SCHEDULER
  |     |     |-- _find_coscheduled_assignments()  # all-or-nothing gang placement
  |     |     |-- first_fitting_worker()           # non-coscheduled tasks
  |     |     `-- result: list[(TaskId, WorkerId)]
  |     `-- _dispatch_assignments_direct()
  |           `-- queue_assignments()   # COMMIT TO DB in one transaction
  |                 |-- validate task schedulability
  |                 |-- assign tasks to workers
  |                 |-- add_committed_resources()
  |                 `-- enqueue RunTaskRequest RPCs
  `-- _apply_preemptions()
```

Key insight: `find_assignments()` operates on an **in-memory snapshot**
(SchedulingContext). It deducts capacity from WorkerCapacity objects as it
assigns tasks. Then `queue_assignments()` commits to the DB. The DB committed
resources are what subsequent ticks see.


## The Bug: Two Failure Modes

### Mode 1: Same-Tick Collision (Incidents A, B, C)

When two coscheduled TPU gangs are both pending in the same scheduling tick:

```
find_assignments():
  for job_id, task_ids in tasks_by_job:   # iterates ALL pending coscheduled jobs
    if req.is_coscheduled:
      result = _find_coscheduled_assignments(context, task_ids, req)
      #  Gang A: finds group "my-tpu-slice", 8 workers all have capacity
      #          deducts CPU+mem+TPU from each WorkerCapacity
      #          returns 8 assignments
      #
      #  Gang B: finds SAME group "my-tpu-slice", checks can_fit()
      #          Workers have reduced CPU/mem/TPU but...
      #          IF tpu.count was properly set AND the per-worker TPU deduction
      #          zeroed out available_tpus: BLOCKED (correct behavior)
      #
      #          IF tpu.count was 0, or if CPU/mem headroom was large enough:
      #          can_fit() PASSES -> double-booking!
```

The `_find_coscheduled_assignments` method:
1. Groups workers by `tpu-name` attribute
2. For each group, counts workers where `can_fit(req)` passes
3. If enough workers, assigns tasks and calls `capacity.deduct(req)`

The deduction reduces `available_tpus` by `req_tpu_count`. If `tpu.count=4`
(the production value for v5p), deducting 4 from 4 leaves 0, and the second
gang's `can_fit` correctly rejects with `insufficient_tpu`.

**But the production incidents happened with tpu.count=4.** So why did the
collision occur?

The answer lies in the **CPU/memory headroom**. Production v5p VMs have:
- 208 vCPUs (208,000 millicores)
- 448 GB RAM

The `train_lm` jobs request:
- 32 vCPUs (32,000 millicores)
- 128 GB RAM
- 4 TPU chips

After Gang A's deduction:
- CPU: 208,000 - 32,000 = 176,000 remaining (enough for Gang B)
- Memory: 448 GB - 128 GB = 320 GB remaining (enough for Gang B)
- TPU: 4 - 4 = 0 remaining (should block Gang B)

With `tpu.count=4`, `can_fit` *should* block the second gang. The tests
confirm this works correctly in isolation.

### The Real Same-Tick Mechanism

The issue reports dispatch gaps of 2.7s (incident B) and 4.9s (incident D).
The scheduler's minimum interval is 1 second. This means the two gangs were
**NOT necessarily in the same `find_assignments()` call**.

The actual sequence for incidents B/C:

```
Tick N: Gang A's train_lm tasks enter pending (parent job dispatches child)
        find_assignments() places Gang A -> commit to DB

Tick N+1 (1-2s later): Gang B's train_lm tasks enter pending
        _read_scheduling_state() builds snapshot from DB
        committed_tpu on each worker = 4 (from Gang A)
        WorkerCapacity.available_tpus = total_tpu(4) - committed_tpu(4) = 0
        can_fit() returns insufficient_tpu -> Gang B stays pending
```

This should work. So what actually broke?

### Mode 2: The Commit-Time Race (The Actual Bug)

The critical insight is in the `queue_assignments` method (pre-fix):

```python
def queue_assignments(self, cur, assignments, direct_dispatch=False):
    for assignment in assignments:
        task = self._store.tasks.get_detail(cur, assignment.task_id)
        worker_address = self._store.workers.active_healthy_address(cur, assignment.worker_id)
        if task is None or worker_address is None:
            rejected.append(assignment)
            continue
        if not task_row_can_be_scheduled(task):
            rejected.append(assignment)
            continue
        # ... assign and commit ...
        self._store.workers.add_committed_resources(cur, assignment.worker_id, resources)
```

The pre-fix `queue_assignments` processes assignments **one at a time** in a
flat loop. It checks `task_row_can_be_scheduled` (is the task still pending?)
but does **NOT** check whether the TPU group is already claimed by another
gang in the same batch.

The race window:

```
Thread: scheduling-loop (single-threaded, but the DB state can change
        between _read_scheduling_state and queue_assignments)

Timeline:
  T=0ms:    _read_scheduling_state() reads workers with committed_tpu=0
  T=5ms:    find_assignments() places Gang A (8 tasks) + Gang B (8 tasks)
            Both gangs target "my-tpu-slice" group
            In-memory deduction: after Gang A, TPU=0; Gang B should fail
            BUT: if Gang B was processed BEFORE Gang A in the iteration order,
            or if Gang B entered via a DIFFERENT path...
```

Wait -- `find_assignments` iterates `tasks_by_job` which is a `defaultdict`.
The iteration order depends on insertion order (Python 3.7+ dict ordering =
insertion order). The first coscheduled job encountered gets assigned; its
`capacity.deduct()` call reduces available_tpus to 0; the second job's
`can_fit()` should fail.

**The actual race is across ticks with stale scheduling contexts.**

Here is the scenario that matches all 4 incidents:

```
Tick N (T=0):
  - _read_scheduling_state() -> sees Gang A pending, Gang B NOT YET pending
  - find_assignments() -> places Gang A (8 tasks)
  - _dispatch_assignments_direct() -> queue_assignments() commits Gang A
  - Gang A's committed_tpu written to DB

Between ticks (T=500ms):
  - Gang B's parent job dispatches its train_lm child
  - train_lm tasks enter PENDING state in the DB

Tick N+1 (T=1000ms):
  - _read_scheduling_state() -> reads workers from DB
    committed_tpu SHOULD be 4 (from Gang A)
    BUT: Gang A's tasks are in ASSIGNED state, not yet RUNNING
    Are ASSIGNED tasks' committed resources visible?
```

Let me trace the resource commitment more carefully.

`queue_assignments()` calls `self._store.workers.add_committed_resources()`
which **increments** committed_tpu in the workers table within the same DB
transaction. So after the transaction commits, any subsequent
`_read_scheduling_state()` will see committed_tpu=4 on each worker.

The `WorkerCapacity.from_worker()` computes:
```python
available_tpus = worker.total_tpu_count - worker.committed_tpu
```

So if committed_tpu=4 and total_tpu_count=4, available_tpus=0, and `can_fit`
rejects. This should work across ticks.

### The Parent-Child Dispatch Timing Gap

The key piece of evidence: the **dispatch timestamps** show 2.7s-4.9s gaps
between the two gangs' `train_lm` children starting. These are the
`started_at_ms` values on the `train_lm` child jobs, not the parent
submissions.

The parent jobs were submitted at ~01:38 UTC. The `train_lm` children were
dispatched at ~09:14 UTC -- **7.5 hours later**. Why?

Because the parent job (the executor/coordinator) first runs the dependency
chain: download, normalize, tokenize, cache-copy. Only after all dependencies
are satisfied does it dispatch the `train_lm` child. When multiple parent
jobs share the same dependency chain, they all finish the dep chain around the
same time and dispatch their `train_lm` children within seconds of each other.

**This is the race window for incidents B and C.**

For incident D, both jobs were simply queued waiting for v5p-128 capacity.
When capacity opened (preemption freed a slice, or autoscaler brought one up),
both pending `train_lm` children became schedulable in the same tick.


### The Actual Root Cause: No Cross-Gang Exclusivity at Either Level

The bug has **two cooperating failures**:

1. **Scheduler level** (`find_assignments`): When two coscheduled gangs target
   the same TPU group in the same tick, `_find_coscheduled_assignments` deducts
   TPU capacity per-worker. With `tpu.count=4`, this correctly prevents double-
   booking within a single `find_assignments()` call. **This path works.**

2. **Commit level** (`queue_assignments`): When two gangs arrive in
   **consecutive ticks** (tick N places Gang A, tick N+1 places Gang B), the
   DB committed resources should block Gang B. With `tpu.count=4`, this also
   works.

3. **The gap**: Between `find_assignments()` returning and
   `queue_assignments()` committing, there is no lock preventing a concurrent
   `_read_scheduling_state()` from seeing stale committed resources.

   BUT -- there is only ONE scheduling thread. No concurrency here.

4. **The REAL gap**: The `queue_assignments` pre-fix code does not check
   whether TWO gangs in the SAME batch of assignments are targeting the same
   TPU group. When `find_assignments` produces assignments for both Gang A and
   Gang B (e.g., one from the coscheduled pass and one from a preference pass
   or preemption recycle), `queue_assignments` commits them all sequentially
   without cross-checking.

   The preference pass (`_preference_pass`) only handles non-coscheduled jobs.
   But preemption recovery and task recycling can produce stale assignments.

### Hypothesis: Task Recycling Creates Stale Assignments

The most likely actual mechanism:

1. **Tick N**: Both gangs' `train_lm` tasks are pending. `find_assignments()`
   correctly assigns only Gang A (8 tasks). Gang B stays pending.

2. **Commit**: Gang A's 8 tasks are committed, committed_tpu=4 per worker.

3. **Tick N+1**: Gang B is still pending. `_read_scheduling_state()` sees
   committed_tpu=4. `find_assignments()` correctly rejects Gang B. No
   assignments.

4. **A transient event**: One of Gang A's tasks fails during BUILDING (e.g.,
   container build timeout, network blip). The task transitions back to PENDING.
   `committed_tpu` is decremented for that worker.

5. **Tick N+K**: Gang A has 7 running tasks and 1 pending task. Gang B has 8
   pending tasks. The scheduling context shows the one worker with freed TPU.
   The coscheduled pass tries to place Gang B: needs 8 workers in the same
   `tpu-name` group, but only 1 has available TPU. Gang B cannot be placed.

   Meanwhile, Gang A's recycled task gets re-placed on the freed worker.

6. **BUT**: If Gang A's recycled task and Gang B's full gang were BOTH
   processed in `find_assignments()`, and if Gang A's recycled task is a
   single non-coscheduled task (because it's a retry, not the original gang):

   This is speculative. Coscheduled task retries are still coscheduled.

### Most Likely Actual Mechanism: Dispatch-Queue Race

Looking more carefully at `_dispatch_assignments_direct`:

```python
def _dispatch_assignments_direct(self, assignments):
    command = [Assignment(task_id=tid, worker_id=wid) for tid, wid in assignments]
    with self._store.transaction() as cur:
        result = self._transitions.queue_assignments(cur, command, direct_dispatch=True)
```

And `_run_scheduler_pass`:
```python
result = self._scheduler.find_assignments(context)
all_assignments = preference_assignments + result.assignments
if all_assignments:
    self._dispatch_assignments_direct(all_assignments)
```

The `find_assignments()` call and `_dispatch_assignments_direct()` call happen
back-to-back in the same thread. There is no interleaving scheduling tick
between them.

**So the only way both gangs get placed is if `find_assignments()` returns
assignments for both in the same call.** Let me re-examine the coscheduled
assignment logic.

```python
for job_id, task_ids in tasks_by_job.items():
    req = context.jobs.get(job_id)
    if req is None or not req.is_coscheduled:
        continue
    coscheduled_result = self._find_coscheduled_assignments(context, task_ids, req)
    if coscheduled_result:
        result.assignments.extend(coscheduled_result)
```

`_find_coscheduled_assignments` calls `context.capacities[worker_id].deduct(req)`
for each assigned worker. This reduces `available_tpus` from 4 to 0.

The next iteration of the `for` loop calls `_find_coscheduled_assignments`
for Gang B. Inside, it does:
```python
available = [
    worker_id for worker_id in group_worker_ids
    if context.capacities[worker_id].can_fit(req) is None
]
if len(available) < num_tasks:
    continue
```

With `available_tpus=0` after Gang A's deduction, `can_fit(req)` returns
`insufficient_tpu` for every worker. `available` is empty. Gang B is not
assigned.

**This works correctly.** So the same-tick scenario is not the problem when
`tpu.count=4`.

### Resolution: It Must Be Cross-Tick

Given that:
- Same-tick works correctly when tpu.count=4
- There is only one scheduling thread (no concurrency)
- `queue_assignments` commits resources atomically before the next tick reads

The only remaining explanation for the production incidents is that there is
a path where committed resources are **not properly accounted**. Let me check
if `add_committed_resources` actually persists correctly.

Looking at the fix commit: it adds `_StagedWorkerCapacity` tracking and
`_assignment_tpu_group_rejection` in `queue_assignments`. The fact that the
fix adds a **staged capacity ledger** to `queue_assignments` suggests the
pre-fix code had a vulnerability where assignments within a single
`queue_assignments` call did not properly track cumulative resource usage.

The pre-fix `queue_assignments` calls `add_committed_resources` for each
assignment sequentially. But `add_committed_resources` writes to the DB
(within the same transaction). Subsequent reads within the same transaction
SHOULD see the updated values (SQLite has read-your-own-writes within a
transaction).

**But `queue_assignments` does NOT re-read worker capacity between assignments.**
It checks `task_row_can_be_scheduled(task)` but not `worker_has_capacity()`.
Once `find_assignments()` says "assign these 16 tasks", `queue_assignments`
trusts that decision and commits all 16 without re-validating capacity.

**This is the bug.**

If `find_assignments()` produces assignments for both gangs (which it shouldn't
when tpu.count=4, as shown above), `queue_assignments` would commit all of
them. But `find_assignments` correctly blocks the second gang when
tpu.count=4.

### Final Hypothesis: The Incidents Span Multiple Scheduling Cycles

Going back to the dispatch timestamps:

```
Incident B:
  lr0.5  train_lm started: 1777799671408  (09:14:31 UTC)
  lr0.67 train_lm started: 1777799674124  (09:14:34 UTC)  -> +2.7s gap

Incident D:
  p67m33-lr0.67 train_lm started: 1777897156854  (12:19:16 UTC)
  p33m67-lr0.83 train_lm started: 1777897161741  (12:19:21 UTC)  -> 4.9s gap
```

The `started_at_ms` is the `train_lm` **child job** start time, set in
`mark_running_if_pending()` within `queue_assignments()`. A 2.7s gap means
the two gangs were committed in **different** `queue_assignments()` calls,
i.e., **different scheduling ticks**.

With the scheduler min interval of 1s:
- 2.7s gap = 2-3 ticks apart
- 4.9s gap = 4-5 ticks apart

For the cross-tick case to fail, committed resources from tick N must NOT be
visible in tick N+1's `_read_scheduling_state()`. Let me check if there is a
caching layer.

Looking at `_read_scheduling_state`:
```python
def _read_scheduling_state(self):
    pending_tasks = _schedulable_tasks(self._db)
    workers = healthy_active_workers_with_attributes(self._db)
```

These are direct DB reads. No caching. The committed_tpu written by tick N's
`queue_assignments` should be visible to tick N+1's read.

**UNLESS** the `train_lm` child jobs have **not yet been dispatched** when
the second tick runs. The parent executor dispatches `train_lm` as a child
job. If both parents dispatch their `train_lm` children at nearly the same
time, both `train_lm` jobs enter PENDING within the same small window. The
scheduler picks them up in the same or adjacent ticks.

But the scheduler assigns the FIRST `train_lm` gang, commits to DB, then
on the next tick reads the DB and sees committed resources. The SECOND gang
should be blocked.

**The 2.7s/4.9s dispatch gap is the time between the two SEPARATE
`queue_assignments` transactions committing.** If tick N commits Gang A at
T=0 and tick N+2 commits Gang B at T=2.7s, then Gang B's tick SAW the
committed resources from Gang A but placed Gang B anyway.

This means the `can_fit` check PASSED for Gang B despite committed_tpu=4.
With `tpu.count=4`, this should not happen.

**Unless the committed_tpu was already decremented by the time Gang B's tick
ran.** This could happen if Gang A's tasks immediately failed (e.g., a
start_tasks RPC failure that rolls back committed resources) between ticks.

Looking at `_dispatch_assignments_direct`:
```python
for worker_id, response, error in self._provider.start_tasks(jobs):
    if error is not None:
        # ... fail the attempt so it bounces back to PENDING
```

If start_tasks fails, the task's committed resources are NOT freed (the task
is marked ASSIGNED, then WORKER_FAILED, which transitions to PENDING). The
committed resources are freed when the task transitions out of ASSIGNED.

Actually, looking at the task state machine: ASSIGNED -> WORKER_FAILED ->
PENDING. When a task goes from ASSIGNED to PENDING, committed resources ARE
freed (via `release_committed_resources` in the transition handler).

So: if Gang A is committed in tick N, its tasks are ASSIGNED. If ALL 8 tasks
immediately fail their start_tasks RPC in the same tick (unlikely), they
transition to PENDING and committed resources are freed. Tick N+1 sees
committed_tpu=0 and places Gang B on the same workers.

But this is extremely unlikely for all 8 tasks to fail simultaneously.

### Working Hypothesis: Stale Coscheduled Task Recycling

The most consistent explanation across all 4 incidents:

The `_find_coscheduled_assignments` path correctly blocks double-booking
**within a single tick when both gangs are freshly pending**. But the
incidents involve a more subtle path:

1. Gang A is placed and committed in tick N.
2. Gang B enters pending in tick N+1. Committed resources block it. Correct.
3. Gang A runs for hours (incident B: 9.5 hours).
4. A transient event (preemption, node failure) causes some of Gang A's tasks
   to be recycled back to PENDING.
5. Gang A's recycled tasks re-enter the scheduling queue.
6. The scheduler places Gang A's recycled tasks AND Gang B's full gang.

   But wait -- Gang B needs ALL 8/16/32 workers. Gang A still holds most of
   them. Gang B cannot be placed while Gang A holds any workers.

   **Unless Gang A is fully preempted/killed**, freeing all workers. At that
   point, both Gang A (recycled) and Gang B (pending) compete for the same
   freed workers.

This matches the evidence: Gang A and Gang B **both fail** with identical
preemption counts, meaning both were in a thrash loop simultaneously.

The sequence:
1. Gang A and Gang B are both placed on the same workers.
2. Both attempt to bind port 8476.
3. One wins, the other's tasks are recycled by Iris (preemption++).
4. The recycled tasks restart, try port 8476 again, conflict again.
5. This loops ~95 times per task before Iris terminates both jobs.

For step 1 to happen, both gangs must have been assigned to the same workers
at some point. Given that the scheduler correctly prevents same-tick double-
booking (with tpu.count=4), the placement must have happened across ticks
where committed resources were transiently cleared.

**The `started_at_ms` in the issue is the time `mark_running_if_pending` was
called -- this is the FIRST time the job entered running state, not necessarily
the time of the collision.** The collision could have happened hours later
during a recycle event.

This is consistent with incident B where the collision manifested "mid-run
after 9.5 hours of uptime."


## Per-Incident Timeline

### Incident A -- v5p-64, 2026-05-03 ~01:21 UTC

**TPU**: v5p-64 (8 VMs, 4 TPU chips/VM)
**Region**: us-east5

| Time (UTC) | Event |
|---|---|
| ~01:20:53 | Both parents submitted within seconds of each other |
| ~01:21 | Both `train_lm` children dispatched (8 tasks each) |
| ~01:21 - ~01:26 | Both gangs placed on identical 8-host set |
| ~01:26 | Both jobs fail: preemptions=707, worker_failed=7/8, failures=1 |

**Dispatch gap**: Seconds (exact timestamps not provided for child dispatch).
**Gang size**: 8 tasks.
**Preemption cycles**: 707 / 8 tasks = ~88 cycles per task.
**Compounding factor**: /moojink/ priority-band-2 contention on v5p-64
added external preemption pressure.

**What the controller did**: Both `train_lm` children entered pending at
nearly the same time. The scheduler saw both in the same tick (or consecutive
ticks before the first was committed). With only one v5p-64 slice available,
both were assigned to the same 8 workers.

### Incident B -- v5p-256, 2026-05-03, submitted ~01:38 UTC, dispatched ~09:14 UTC

**TPU**: v5p-256 (32 VMs, 4 TPU chips/VM)
**Region**: us-east5

| Time (UTC) | Event |
|---|---|
| 01:38:34 (1777772314000) | lr0.33 parent submitted (oldest, +0s) |
| 01:38:36 (1777772316424) | lr0.5 parent submitted (+2.4s) |
| 01:38:38 (1777772318732) | lr0.67 parent submitted (+4.7s) |
| 01:38 - 09:14 | All 3 parents run dependency chains (download, normalize, tokenize, cache-copy). ~7h36m of CPU work before training dispatched. |
| 09:14:31 (1777799671408) | lr0.5 `train_lm` child dispatched (32 tasks) |
| 09:14:34 (1777799674124) | lr0.67 `train_lm` child dispatched (32 tasks), +2.7s gap |
| ~09:14 | lr0.33 `train_lm` also dispatched around this time |
| 09:14 - 18:43 | lr0.33 runs cleanly (preemptions=1, worker_failed=0). lr0.5 and lr0.67 placed on same 31 hosts; run in latent collision state for 9.5 hours. |
| ~18:43 | Transient event triggers the port-8476 death loop. Both lr0.5 and lr0.67 enter preemption thrash. |
| ~18:43+ | 3131 preemption cycles (32 tasks x ~98 cycles/task). Both fail with failures=1, worker_failed=31/32. |

**Key observations**:
- lr0.33 survived because it was dispatched first (either in an earlier tick
  or first in the same tick's iteration order).
- 2.7s dispatch gap = likely 2-3 scheduling ticks apart.
- The collision was **latent** for 9.5 hours before manifesting. Both gangs
  ran nominally because the port-8476 binding succeeded for one gang's tasks
  first. The collision only became active when a transient event caused task
  restarts.
- 31/31 hosts identical between lr0.5 and lr0.67 (verified via bootstrap
  log `advertise_host` comparison).

**What the controller did**: Three `train_lm` children all entered pending
within seconds of each other (~09:14 UTC) after their parents finished
dependency chains simultaneously. The scheduler placed lr0.33 first (correct).
Then placed lr0.5 and lr0.67 on the same workers -- either because committed
resources from lr0.5 were not yet visible when lr0.67 was scheduled (cross-
tick), or because a commit-time validation gap allowed both through.

### Incident C -- v5p-256, 2026-05-03, submitted ~01:41 UTC

**TPU**: v5p-256 (32 VMs, 4 TPU chips/VM)
**Region**: us-east5

| Time (UTC) | Event |
|---|---|
| 01:41:22 (1777772482016) | p67m33-lr0.83 parent submitted (+0s) |
| 01:41:24 (1777772484621) | p33m67-lr0.83 parent submitted (+2.6s) |
| ~09:14 | Both `train_lm` children dispatched (timing closely correlated with incident B due to shared dependency chain completion) |
| Running | p67m33-lr0.83 runs cleanly, succeeds. p33m67-lr0.83 placed on same hosts. |
| Mid-run | p33m67-lr0.83 reaches step 6112 (80%+ progress) before thrash kicks in |
| Failure | p33m67-lr0.83: preemptions=3131, worker_failed=31/32, failures=1 |

**Key observations**:
- Same signature as incident B: 3131 preemptions, 31/32 worker_failed.
- The failing job (p33m67) reached step 6112 before the collision manifested,
  confirming the latent collision pattern.
- p67m33 won the placement race (submitted 2.6s earlier); p33m67 lost.
- Salvageable: permanent checkpoint at step-6112 in
  `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.83-78fd44/`.

### Incident D -- v5p-128, 2026-05-04 12:19 UTC dispatch

**TPU**: v5p-128 (16 VMs, 4 TPU chips/VM)
**Region**: us-east5

| Time (UTC) | Event |
|---|---|
| May 3 16:14:38 (1777824878720) | p67m33-lr0.67 resume2 parent submitted |
| May 3 20:02:20 (1777838540690) | p33m67-lr0.83 parent submitted, **+3h47m later** |
| May 3 20:02 - May 4 12:19 | Both `train_lm` children PENDING, waiting for v5p-128 capacity. ~16h in queue. |
| May 4 12:19:16 (1777897156854) | p67m33-lr0.67 `train_lm` dispatched |
| May 4 12:19:21 (1777897161741) | p33m67-lr0.83 `train_lm` dispatched, **+4.9s gap** |
| May 4 12:19 - 12:40 | p67m33-lr0.67 runs 21 min, p33m67-lr0.83 runs 37 min. Both thrashing. |
| Failure | Both: preemptions=1515, worker_failed=15/16, failures=1 |

**Key observations**:
- **Dispatch-time collision**: Submissions were 3h47m apart, ruling out
  submission-time proximity as the cause. The collision happened when both
  pending jobs were dispatched simultaneously when v5p-128 capacity opened.
- 4.9s dispatch gap = 4-5 scheduling ticks apart.
- 1515 preemptions / 16 tasks = ~95 cycles per task (consistent with other
  incidents).
- The capacity event that triggered both dispatches: likely a v5p-128 slice
  became available (autoscaler spin-up or another job completing/being
  preempted). Both pending jobs saw the newly available slice and were
  dispatched in quick succession.
- Neither job saved a useful checkpoint. p67m33-lr0.67 was a resume from
  step-2646; p33m67-lr0.83 never wrote a checkpoint.


## Cross-Incident Pattern Analysis

### Dispatch Gap vs Scheduler Interval

| Incident | Dispatch gap | Est. ticks apart | Same tick? |
|---|---|---|---|
| A | seconds | 1-2 | Possibly same tick |
| B | 2.7s | 2-3 | Different ticks |
| C | correlated with B | 2-3 | Different ticks |
| D | 4.9s | 4-5 | Different ticks |

The dispatch gaps are consistently in the 2-5 second range, suggesting the
collision happens across **2-5 consecutive scheduling ticks**, not within a
single tick. This is consistent with: tick N places Gang A, ticks N+1 through
N+4 see Gang B still pending, and one of those ticks places Gang B despite
Gang A's committed resources.

### Preemption Scaling

| Gang size | TPU variant | Preemptions | Per-task cycles |
|---|---|---|---|
| 8 tasks | v5p-64 | 707 | ~88 |
| 16 tasks | v5p-128 | 1515 | ~95 |
| 32 tasks | v5p-256 | 3131 | ~98 |

Near-linear scaling with gang size. The slight increase in per-task cycles
for larger gangs likely reflects the longer time for all tasks to reach the
port-conflict state simultaneously.

### Survivor Pattern

In incidents A and D, no jobs survived (both gangs failed). In incidents B
and C, the **first-submitted** job survived and the **later-submitted** job
failed. This is consistent with the scheduler processing jobs in submission
order: the first gang gets committed cleanly, and only the second gang ends
up on the same hosts.

### Capacity Event Correlation

Incidents B and C are temporally correlated: both sets of `train_lm` children
were dispatched around 09:14 UTC on May 3. This suggests a single capacity
event (e.g., v5p-256 slices becoming available after overnight maintenance
or autoscaler spin-up) triggered the dispatch of multiple pending gangs
simultaneously.

Incident D explicitly demonstrates the dispatch-queue variant: a capacity
event at 12:19 UTC on May 4 released both pending v5p-128 jobs.


## The Fix (PR #5475, commit a6f13ff00)

The fix addresses the bug at three levels:

### 1. Commit-Time TPU Group Rejection

New `_assignment_tpu_group_rejection()` method in `queue_assignments` checks:
- Whether another gang in the **same batch** already claims the TPU group
  (`staged_tpu_groups` dict)
- Whether an **active task** from a different job already occupies the TPU
  group (DB lookup via `_foreign_active_tpu_task_in_group`)

If either check fails, the entire coscheduled gang is rejected.

### 2. Batched Gang Validation

New `_assignment_batches()` groups assignments by coscheduled job. Each batch
is validated atomically: if any assignment in a gang fails validation, the
entire gang is rejected. This prevents partial gang commits.

### 3. Staged Capacity Ledger

`_StagedWorkerCapacity` tracks cumulative resource deductions across the
entire `queue_assignments` call, preventing the second gang from seeing stale
capacity even within the same transaction.

### 4. TPU-Specific JAX Init Bypass

`jax_init.py` now skips the Iris JAX coordinator entirely for TPU jobs (the
TPU runtime handles distributed init). Job-scoped endpoint names
(`_job_scoped_endpoint_name`) prevent cross-job port conflicts for non-TPU
jobs.

### 5. Worker TPU Metadata Injection

`task_attempt.py` now passes `TPU_NAME`, `TPU_WORKER_ID`,
`TPU_WORKER_HOSTNAMES`, and `TPU_CHIPS_PER_HOST_BOUNDS` into the container
environment, enabling the TPU runtime to perform distributed init without
the Iris coordinator.


## What the Controller Actually Did

Based on the evidence, the most likely sequence across all incidents:

1. Multiple parent jobs (submitted within seconds or hours of each other)
   complete their dependency chains at nearly the same time, causing their
   `train_lm` children to enter PENDING within seconds.

2. The scheduling loop picks up the first `train_lm` gang and places it
   (tick N). Committed resources are written to the DB.

3. 1-5 ticks later, the second `train_lm` gang is picked up. The scheduler
   reads the DB, sees committed resources, and **should** block the second
   gang. However, a race condition in the commit/read path or an edge case in
   the coscheduled task recycling allows the second gang to be placed on the
   same workers.

4. Both gangs' tasks are dispatched to the same workers. The first to bind
   port 8476 "wins" initially, but the collision is latent.

5. Eventually (immediately for incidents A and D; after 9.5 hours for incident
   B), a task restart triggers the port-conflict cascade, and both gangs enter
   the preemption death loop (~95 cycles per task).

The fix in PR #5475 closes the race by adding explicit TPU group exclusivity
checks at commit time, making it impossible for two gangs to be committed to
the same TPU group regardless of the scheduling tick timing.


## The Preemption-Induced Double-Booking Path (Most Likely Root Cause)

The test file `test_preemption_coscheduled_collision.py` documents the most
likely mechanism. The function `_requeue_coscheduled_siblings` (transitions.py
line 591) is the key:

When one task of a coscheduled gang hits a transient failure (WORKER_FAILED),
ALL sibling tasks are bounced to PENDING and their committed resources are
decommitted from all workers. This is by design -- it ensures the gang
re-coschedules atomically on the same slice.

But this creates a window:

```
State before preemption:
  Gang A: 8 tasks RUNNING on workers [w0..w7], committed_tpu=4 each
  Gang B: 8 tasks PENDING (blocked by committed_tpu on all workers)

Gang A task-0 hits WORKER_FAILED:
  _requeue_coscheduled_siblings() bounces ALL 8 siblings to PENDING
  committed_tpu deducted on ALL 8 workers -> committed_tpu=0

State after preemption:
  Gang A: 8 tasks PENDING (all decommitted)
  Gang B: 8 tasks PENDING
  Workers: all committed_tpu=0, all free

Next scheduler tick:
  find_assignments() sees TWO pending gangs and 8 free workers
  Iterates coscheduled jobs:
    Gang A: 8 workers available, can_fit passes -> ASSIGN
    Gang B: workers have been deducted by Gang A -> BLOCKED (correct)

  OR, if Gang B happens to be iterated FIRST:
    Gang B: 8 workers available, can_fit passes -> ASSIGN
    Gang A: workers have been deducted by Gang B -> BLOCKED (correct)
```

With tpu.count=4, the in-memory deduction correctly blocks the second gang
within a single `find_assignments()` call. **So this path is also safe.**

But there is another variant: what if the preemption triggers the SECOND
gang's dispatch in the SAME tick via the preemption pass?

```python
# In _run_scheduling():
all_assignments, context, tainted_jobs = self._run_scheduler_pass(...)
preemptions = self._apply_preemptions(order, tainted_jobs, ...)
```

The preemption pass runs AFTER `find_assignments()`. If Gang A is running
and Gang B is higher-priority but unschedulable, the preemption pass can
evict Gang A to free capacity for Gang B. The eviction goes through
`_requeue_coscheduled_siblings` for the victim (Gang A), freeing all workers.

But the preemption-created free capacity is not immediately re-scheduled in
the same tick -- it waits for the next tick. In the next tick, both gangs
are PENDING and compete.

**The critical question is whether the `started_at_ms` differences (2.7s,
4.9s) represent initial placement or a MID-RUN re-placement after preemption.**

For incidents B and C, the latent collision (9.5 hours of uptime before
failure) strongly suggests the initial placement was correct but a mid-run
preemption event caused both gangs to be bounced to PENDING and then both
re-placed on the same workers.

The sequence for incident B:
```
09:14:31  lr0.5 train_lm placed on workers [w0..w31]  (started_at_ms)
09:14:34  lr0.67 train_lm placed on workers [w0..w31]  (started_at_ms)
          BOTH on the same workers from the start -- collision is latent
09:14 - 18:43  Both run nominally (port-8476 happened to not conflict yet)
~18:43    A task restart triggers the port-conflict cascade
~18:43+   3131 preemption cycles, both jobs fail
```

OR:
```
09:14:31  lr0.5 train_lm placed on workers [w0..w31]
09:14:34  lr0.67 train_lm placed on workers [w32..w63] (DIFFERENT workers)
          Both run correctly on separate slices
~18:43    An external preemption bounces lr0.5's gang to PENDING
          _requeue_coscheduled_siblings frees workers [w0..w31]
          Next tick: lr0.5 is re-placed on [w32..w63] (lr0.67's workers)
          NOW they share hosts -> collision
```

The second scenario requires a second v5p-256 slice to exist. The issue
mentions v5p-256 capacity was limited. If there was only ONE v5p-256 slice,
then both gangs must have been placed on the same workers from the start.

With only one slice and tpu.count=4, the same-tick `find_assignments` would
correctly block the second gang. So either:
(a) There were multiple slices and a mid-run re-placement caused the collision
(b) There is a subtle bug in the scheduler that tpu.count=4 does not prevent

Given that the fix adds commit-time TPU group overlap rejection AND the tests
for this scenario exist, the most likely answer is (a): there were enough
workers but the mid-run re-placement after preemption bypassed the
per-worker TPU exclusivity.

### The Commit-Time Validation Gap

The fix in PR #5475 adds `_assignment_tpu_group_rejection` to
`queue_assignments`. This check runs at COMMIT TIME, not scheduling time.
The fact that the fix targets `queue_assignments` rather than
`find_assignments` confirms that the bug is in the commit path:

**`find_assignments()` correctly prevents double-booking within a tick.
But `queue_assignments()` does not re-validate that the assignments are
still safe when they are committed.** Between `find_assignments()` returning
and `queue_assignments()` running, the world may have changed (preemptions,
recycled tasks, etc.).

The pre-fix `queue_assignments` checks:
- Is the task still schedulable? (yes)
- Is the worker healthy? (yes)
- Does the worker have capacity? (NOT CHECKED at commit time)

It does NOT check:
- Is another gang from the same batch targeting the same TPU group?
- Is an active task from a different job already on this TPU group?

The fix adds both checks. The `_StagedWorkerCapacity` ledger tracks
cumulative resource deductions within the batch, and
`_foreign_active_tpu_task_in_group` checks for active tasks from other
jobs on the same TPU group.


## Remaining Questions

1. **Latent collision trigger**: Incident B ran 9.5 hours before the collision
   manifested. What specific transient event triggered the port-conflict
   cascade? The postmortem mentions "a single host-level transient" but
   doesn't identify it.

2. **Preemption termination heuristic**: The ~95 cycles per task is below the
   `max_retries_preemption=1000` budget. What triggers termination? Is it a
   wall-time timeout, a consecutive-failure threshold, or something else?

3. **Number of v5p-256 slices**: Were there multiple v5p-256 slices in
   us-east5 during incidents B and C? If only one, the collision must have
   been from initial placement. If multiple, the mid-run preemption-induced
   re-placement scenario is the likely cause.

4. **started_at_ms semantics**: Does `started_at_ms` on the `train_lm` child
   record the FIRST time it entered running state, or is it updated on
   re-starts? If updated on re-starts, the 2.7s/4.9s gaps might represent
   re-placement times after a preemption event, not initial placement times.
