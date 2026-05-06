# Fix Proposal: Issue #5470 — TPU Placement Collision

## Root Cause (confirmed from controller logs)

The collision is NOT a scheduler double-booking bug. The scheduler places gangs
sequentially and correctly. The problem is a **worker cleanup race**:

```
18:43:32  lr0.5 assigned to slice 8996b868 (32 workers)
18:43:53  lr0.5 tasks start building (syncing deps, installing pip)
18:43:57  lr0.5 task 28 fails (502 Bad Gateway downloading Python from GitHub)
          → _requeue_coscheduled_siblings: all 32 tasks → PENDING
          → decommit_resources: committed_tpu=0 on all 32 workers
          → transaction commits
          → kill RPCs queued for 31 still-building tasks (ASYNC)
18:43:58  lr0.67 assigned to same 32 workers (scheduler sees committed_tpu=0)
          → start RPCs sent to all 32 workers
          → workers receive start for lr0.67 BEFORE kill for lr0.5 completes
          → both task sets now running on same workers → port 8476 conflict
```

The decommit and the kill are not atomic from the workers' perspective. The DB
transaction frees the capacity instantly, but the worker-side process cleanup
is asynchronous.

## Fix: Deferred decommit for coscheduled TPU gangs

### Approach

When `_requeue_coscheduled_siblings` bounces a coscheduled gang, do NOT decommit
the resources immediately. Instead, leave committed_tpu held on the workers and
mark the tasks with a "pending_decommit" flag. The decommit happens only when
the poll cycle confirms the workers have stopped the old tasks.

This keeps the workers "reserved" from the scheduler's perspective until they
are actually clean, preventing the next gang from being assigned to a dirty
slice.

### Where to change

**`_requeue_coscheduled_siblings` (transitions.py ~line 591)**

Currently passes `resources=resources` to `_terminate_task` for each sibling,
which triggers `decommit_resources`. Change to pass `resources=None` so the
decommit is skipped. Instead, track which workers need deferred decommit.

```python
# BEFORE (current code):
_terminate_task(
    ...,
    worker_id=worker_id_str,
    resources=resources if sib.current_worker_id is not None else None,
)

# AFTER:
_terminate_task(
    ...,
    worker_id=worker_id_str,
    resources=None,  # defer decommit — workers still have stale processes
)
# Track that this worker needs decommit after kill confirmation
if sib.current_worker_id is not None:
    tasks.mark_pending_decommit(cur, sib.task_id, resources)
```

**New: `mark_pending_decommit` in TaskStore**

Add a column (or use an existing mechanism) to track tasks that are PENDING
but whose worker resources haven't been decommitted yet. When the poll cycle
or a heartbeat confirms the task is no longer running on the worker, trigger
the decommit.

**Poll reconciliation (controller.py ~line 2403)**

After PollTasks confirms a worker is no longer running a stale task, process
any pending decommits for that worker.

### Simpler alternative: hold committed_tpu for N seconds

Instead of a full pending_decommit tracking system, keep a
`decommit_cooldown` dict in the controller that holds (worker_id, timestamp)
entries. When a coscheduled gang is requeued, record the decommit intent with
a timestamp. The scheduler's `_read_scheduling_state` inflates committed_tpu
by the cooldown entries that haven't expired. After the cooldown (e.g., 30s —
enough for kill RPCs to complete), the cooldown entry expires and committed_tpu
naturally returns to its DB value.

This requires no schema change and no new RPC. The downside is that reassignment
is delayed by the cooldown period even if the kills complete faster.

### Simplest fix: await kill RPCs before releasing transaction

The most minimal fix: in `_run_task_updater_loop`, when a coscheduled requeue
produces kills, send the StopTasks RPCs BEFORE committing the decommit
transaction.

Currently (controller.py ~line 2434):
```python
with self._store.transaction() as cur:
    results = self._transitions.apply_heartbeats_batch(cur, requests)
# Transaction commits → decommit is visible to scheduler
# THEN kills are sent:
if all_tasks_to_kill:
    self._stop_tasks_direct(all_tasks_to_kill, all_task_kill_workers)
```

Change to: split the transaction so kills happen before the decommit commits.
Or: send kills inside the transaction (before commit), then commit. The DB
write lock would be held longer, but the kills would land before the scheduler
sees the freed capacity.

This is fragile — StopTasks RPCs can take seconds, and holding the DB write
lock that long would stall all other DB operations.

### Recommended fix: coscheduled gang cooldown

The best balance of simplicity and correctness:

1. Add a `_gang_cooldown: dict[str, float]` to Controller (keyed by tpu-name
   group, value = monotonic timestamp when requeue happened).

2. In `_requeue_coscheduled_siblings`, after the transaction commits, record
   the TPU group in the cooldown dict with a 30-second expiry.

3. In `_find_coscheduled_assignments` (or in `_build_context`), check the
   cooldown dict. If a TPU group has an active cooldown, skip it — treat all
   workers in that group as if they still have committed TPU resources.

4. Kill RPCs proceed async as before. After 30s (or when poll confirms), the
   cooldown expires and the group is available for reassignment.

This approach:
- Requires no DB schema change
- Requires no new RPC
- Adds ~10 lines of code
- Delays reassignment by at most 30s after a gang failure
- Is safe: the cooldown is conservative (over-reserves, never under-reserves)

## Long-term fix: UpdateTasks RPC

Replace the separate Start/Stop RPCs with a single `UpdateTasks(to_start,
to_stop)` RPC that the worker processes atomically. The worker kills the old
tasks, waits for cleanup, then starts the new ones. This eliminates the race
entirely but requires a worker rollout.
