# Stale Worker Resource State: Investigation & Fix

## Symptom

A coscheduled `train_lm` job (8 replicas on TPU v3) is stuck pending with:

> No worker has sufficient resources (need cpu=32, memory=137438953472)

The dashboard shows workers with **phantom `Running Tasks`** (1, 3, etc.) even though **no jobs are actually running on them**. The previous job (`161655`) failed, but its resources were never released from the worker state.

## Root Cause: `_cascade_coscheduled_failure` leaks committed resources

When a coscheduled task fails terminally, `_cascade_coscheduled_failure` kills all running siblings. However, it **never calls `_cleanup_task_resources`** for those siblings, leaving their committed CPU/memory/TPU on the worker and their task IDs in `worker.running_tasks`.

The subsequent `_mark_remaining_tasks_killed` (called by `_finalize_job_state`) **skips these siblings** because they are already `is_finished()` (the cascade intentionally sets `preemption_count = max_retries_preemption` to make them terminal).

## Control Flow Diagram

```
Worker A fails (heartbeat timeout or task crash)
        │
        v
_on_worker_failed(worker_A)
        │
        ├── Cascades TASK_STATE_WORKER_FAILED to task_0 on worker_A
        │       │
        │       v
        │   _on_task_state_changed(task_0, WORKER_FAILED)
        │       │
        │       ├── task_0.handle_attempt_result(WORKER_FAILED)
        │       │       → result = EXCEEDED_RETRY_LIMIT (or SHOULD_RETRY)
        │       │
        │       ├── _cleanup_task_resources(task_0)     ← resources freed for task_0 ✓
        │       │
        │       ├── task_0.is_finished() == True, job.is_coscheduled == True
        │       │
        │       v
        │   _cascade_coscheduled_failure(task_0, job)
        │       │
        │       ├── For each sibling (task_1..task_7) on workers B..H:
        │       │       │
        │       │       ├── sibling.preemption_count = max_retries_preemption
        │       │       ├── sibling.handle_attempt_result(WORKER_FAILED)
        │       │       │       → sibling is now terminal (is_finished() == True)
        │       │       ├── job.on_task_transition(old, new)
        │       │       ├── txn.tasks_to_kill.add(sibling_id)
        │       │       │
        │       │       └── *** _cleanup_task_resources(sibling) NEVER CALLED ***  ← BUG
        │       │               Workers B..H still have:
        │       │                 - sibling in running_tasks
        │       │                 - committed_cpu += 32
        │       │                 - committed_mem += 128GB
        │       │
        │       v
        │   job.on_task_transition(old, task_0.state)
        │       → new_job_state = JOB_STATE_FAILED (too many failures)
        │       │
        │       v
        │   _finalize_job_state(job, FAILED)
        │       │
        │       v
        │   _mark_remaining_tasks_killed(job_id, "Job exceeded max_task_failures")
        │       │
        │       ├── For each task in job:
        │       │       if task.is_finished(): continue   ← SKIPS SIBLINGS (already terminal)
        │       │                                            Their resources are NEVER freed
        │       │
        │       └── (no-op for siblings)
        │
        ├── del self._workers[worker_A]      ← worker_A is pruned (resources gone with it)
        │
        v
    Worker A eventually re-registers → fresh ControllerWorker (clean)
    Workers B..H → still have phantom committed resources and running_tasks
```

## Impact

After the cascade:
- Workers B through H each show 1+ "Running Tasks" in the dashboard
- Each phantom task consumes 32 CPUs and ~128GB committed memory
- With only 112 CPUs and 188.7GB per worker, a single phantom task blocks scheduling any new 32-CPU/128GB task
- The new coscheduled job needs all 8 workers in the same `tpu-name` group to have capacity, but 7 of 8 are blocked
- Result: permanent scheduling deadlock until the controller is restarted

## Fix

In `_cascade_coscheduled_failure`, add `_cleanup_task_resources` for each killed sibling:

```python
# state.py, _cascade_coscheduled_failure
def _cascade_coscheduled_failure(self, trigger_task, job, txn):
    for sibling_id in self._tasks_by_job.get(job.job_id, []):
        if sibling_id == trigger_task.task_id:
            continue
        sibling = self._tasks[sibling_id]
        if sibling.state not in (TASK_STATE_RUNNING, TASK_STATE_ASSIGNED):
            continue

        sibling_old = sibling.state
        sibling.preemption_count = sibling.max_retries_preemption
        sibling.handle_attempt_result(
            cluster_pb2.TASK_STATE_WORKER_FAILED,
            error=f"Coscheduled sibling {trigger_task.task_id} failed",
        )
        job.on_task_transition(sibling_old, sibling.state)
        self._cleanup_task_resources(sibling, job, txn)   # ← ADD THIS LINE
        txn.tasks_to_kill.add(sibling_id)
        txn.log(
            "coscheduled_sibling_killed",
            sibling_id,
            trigger_task=str(trigger_task.task_id),
        )
```

### Why not route through `_on_task_state_changed`?

`_on_worker_failed` routes each affected task through `_on_task_state_changed`, which
handles cleanup. However, `_on_task_state_changed` itself calls
`_cascade_coscheduled_failure` when a coscheduled task fails terminally. Routing
siblings through `_on_task_state_changed` would cause re-entrancy: sibling_1's
cascade would process still-RUNNING sibling_2, then the outer loop would process
sibling_2 again, causing double resource release. Adding the direct
`_cleanup_task_resources` call is the minimal, correct fix.

### Why doesn't the heartbeat path recover?

When the worker eventually reports the killed task as completed, `complete_heartbeat`
(line 1940) checks `if task and not task.is_finished()` — but the cascade already
set `is_finished() == True`. The heartbeat report is silently dropped, and resources
are never freed.

## Secondary Issue: Worker re-registration without prior failure

In `_on_worker_registered`, when a worker re-registers while still in `_workers` (e.g., worker process restarted before the heartbeat timeout fires), the code only updates metadata and heartbeat timestamp. It does **not** reset `running_tasks`, `committed_cpu`, `committed_mem`, `committed_gpu`, or `committed_tpu`.

This is a separate (less likely) leak path. The coscheduled cascade bug above is the primary cause of the observed behavior.

## Immediate Workaround

Restart the controller to clear all in-memory state. Workers will re-register with fresh `ControllerWorker` objects.

## Verification

Use the RPC interface to inspect committed resources:

```bash
# List workers and their running tasks
uv run iris --verbose --config lib/iris/examples/eu-west4.yaml rpc controller ListWorkers

# Check the failed job's tasks and their worker assignments
uv run iris --verbose --config lib/iris/examples/eu-west4.yaml rpc controller GetJobStatus \
  --job_id /iris-run-power-reference_hyperparameter_sweep-20260213-161655
```

Workers showing `running_job_ids` for the failed job confirm the leak.
