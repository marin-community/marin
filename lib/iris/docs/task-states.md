# Task States Reference

This document describes the `TaskState` enum, the state machine governing task
lifecycle, retry semantics, and how states appear in the dashboard.

A **task** is the unit of execution in Iris. Each job expands into one or more
tasks (controlled by `replicas`). Tasks are independently scheduled, retried,
and tracked. Job state is derived from task state counts -- there is no
independent job state machine.

## State Diagram

```
                          +-----------+
                          |  PENDING  |<-----------------+
                          +-----+-----+                  |
                                |                        |
                          dispatch to worker              |
                                |                        |
                                v                        |
                          +-----------+                  |
                          | ASSIGNED  |                  |
                          +-----+-----+                  |
                                |                        |
                        worker starts task               |
                                |                        |
                                v                        |
                          +-----------+                  |
                          | BUILDING  |                  |
                          +-----+-----+                  |
                                |                        |
                        build completes                  |
                                |                        |
                                v                        |
                          +-----------+                  |
                          |  RUNNING  |                  |
                          +-----+-----+                  |
                                |                        |
            +-------------------+-------------------+    |
            |                   |                   |    |
            v                   v                   v    |
      +-----------+       +-----------+     +------------+
      | SUCCEEDED |       |  FAILED   |---->| retry      |
      +-----------+       +-----------+     +------------+
                                |                  ^
                                | exhausted        |
                                v                  |
                          (terminal)               |
                                                   |
                          +-----------+            |
                          |WORKER_FAIL|------------+
                          +-----------+
                                |
                                | exhausted
                                v
                          (terminal)

      Other terminal states: KILLED, UNSCHEDULABLE (never retried)
```

## State Table

| State | Proto Value | Terminal | Retriable | Set By | Dashboard Display |
|---|---|---|---|---|---|
| `UNSPECIFIED` | 0 | -- | -- | Default zero value; never used in practice | `unspecified` (grey) |
| `PENDING` | 1 | No | -- | Job submission (`_on_job_submitted`), retry requeue (`_requeue_task`) | `pending` (amber) |
| `ASSIGNED` | 9 | No | -- | Scheduler dispatch (`_on_task_assigned` / `create_attempt`) | `assigned` (orange) |
| `BUILDING` | 2 | No | -- | Worker heartbeat report; worker sets this during bundle download and dependency sync | `building` (purple) |
| `RUNNING` | 3 | No | -- | Worker heartbeat report; worker sets this when user command starts | `running` (blue) |
| `SUCCEEDED` | 4 | Yes | No | Worker heartbeat report; task exited with code 0 | `succeeded` (green) |
| `FAILED` | 5 | Yes | Yes | Worker heartbeat report; task exited with non-zero code | `failed` (red) |
| `KILLED` | 6 | Yes | No | Controller: job cancellation (`_on_job_cancelled`), job failure cascade (`_mark_remaining_tasks_killed`), per-task timeout | `killed` (grey) |
| `WORKER_FAILED` | 7 | Yes | Yes | Controller: worker death cascade (`_on_worker_failed`), coscheduled sibling kill | `worker_failed` (purple) |
| `UNSCHEDULABLE` | 8 | Yes | No | Controller: scheduling timeout expired (`_mark_task_unschedulable`) | `unschedulable` (red) |


## State Transitions in Detail

### PENDING

The initial state for every task. Set in two contexts:

1. **Job submission**: `_on_job_submitted` calls `expand_job_to_tasks`, which
   creates `ControllerTask` objects with `state=TASK_STATE_PENDING`. Tasks are
   enqueued into the priority-sorted scheduling queue.

2. **Retry requeue**: `_requeue_task` resets `task.state` to `TASK_STATE_PENDING`
   and re-inserts the task into the scheduling queue. This happens after a
   retriable `FAILED` or `WORKER_FAILED` when retry budget remains.

### ASSIGNED

Set by `_on_task_assigned` after the scheduler selects a worker and commits
resources. `create_attempt` creates a new `ControllerTaskAttempt` in
`TASK_STATE_ASSIGNED` state. The task is now bound to a specific worker and
consuming its resources.

The worker has not yet acknowledged the task -- it will receive the dispatch
in the next heartbeat cycle.

### BUILDING

Reported by the worker via heartbeat. The worker transitions internally:

- `PENDING -> BUILDING` when bundle download starts (`task_attempt.py:433`)
- Later, `BUILDING` again when dependency sync starts (`task_attempt.py:549`)

The controller processes this transition in `complete_heartbeat`. Note: if the
worker reports `PENDING`, the controller ignores it to prevent regressing an
`ASSIGNED` task and confusing the building-count backpressure window.

### RUNNING

Reported by the worker via heartbeat after the user command starts executing
(`task_attempt.py:570`). The controller records `started_at` on the attempt.

### SUCCEEDED

Reported by the worker via heartbeat when the task process exits with code 0.
The controller sets `exit_code=0`, `finished_at`, and marks the task terminal.
No retry logic applies.

### FAILED

Reported by the worker via heartbeat when the task process exits with a non-zero
code. Triggers retry evaluation:

1. `handle_attempt_result` calls `_handle_failure`, which increments
   `failure_count` and compares against `max_retries_failure`.
2. If `failure_count <= max_retries_failure`: returns `SHOULD_RETRY`. The caller
   (`_on_task_state_changed`) calls `_requeue_task`, which resets state to
   `PENDING` and re-enqueues the task. Resources are released from the current
   worker.
3. If `failure_count > max_retries_failure`: returns `EXCEEDED_RETRY_LIMIT`. The
   task remains in `FAILED` state and is terminal. `error` and `exit_code` are
   recorded.

### KILLED

Set by the controller in three scenarios:

1. **User cancellation**: `_on_job_cancelled` iterates non-terminal tasks and
   transitions each to `KILLED`. Tasks with workers assigned are queued for
   kill RPCs.

2. **Job failure cascade**: When a job exceeds `max_task_failures`,
   `_finalize_job_state` calls `_mark_remaining_tasks_killed` to terminate all
   surviving tasks.

3. **Parent job termination**: `_cancel_child_jobs` recursively cancels child
   jobs when a parent reaches a terminal state (except `SUCCEEDED`).

`KILLED` is always terminal and never retried.

### WORKER_FAILED

Set by the controller when a worker dies. `_on_worker_failed` iterates all
tasks on the dead worker and emits `TaskStateChangedEvent` with
`TASK_STATE_WORKER_FAILED` for each non-terminal task.

Retry evaluation uses the preemption budget:

1. `_handle_failure` increments `preemption_count` and compares against
   `max_retries_preemption` (default: 100).
2. If budget remains: `SHOULD_RETRY` -- task is requeued to `PENDING`.
3. If exhausted: `EXCEEDED_RETRY_LIMIT` -- task stays in `WORKER_FAILED`
   and is terminal.

**Coscheduled jobs**: When a task in a coscheduled (gang-scheduled) job fails
terminally, `_cascade_coscheduled_failure` exhausts the preemption budget of
all running siblings and transitions them to `WORKER_FAILED` (terminal). This
prevents other hosts from hanging on collective operations.

### UNSCHEDULABLE

Set by the controller's scheduling loop when a task's scheduling deadline
expires (`_mark_task_unschedulable` in `controller.py`). The deadline is
derived from the job's `scheduling_timeout` field.

`UNSCHEDULABLE` is always terminal. If any task becomes unschedulable, the
entire job transitions to `JOB_STATE_UNSCHEDULABLE` and all remaining tasks
are killed.

## Retry Semantics

Iris maintains two independent retry budgets per task:

| Budget | Counter | Limit Field | Default | Trigger State |
|---|---|---|---|---|
| Failure | `failure_count` | `max_retries_failure` | 0 (no retries) | `FAILED` |
| Preemption | `preemption_count` | `max_retries_preemption` | 100 | `WORKER_FAILED` |

### Retry flow

1. Worker reports terminal state via heartbeat.
2. `handle_attempt_result` delegates to `_handle_failure`.
3. The appropriate counter is incremented.
4. If `counter <= limit`: `TaskTransitionResult.SHOULD_RETRY`.
   - `_on_task_state_changed` calls `_requeue_task`.
   - Task state is reset to `PENDING`. A new attempt will be created when the
     scheduler re-dispatches.
   - Worker resources are released via `_cleanup_task_resources`.
5. If `counter > limit`: `TaskTransitionResult.EXCEEDED_RETRY_LIMIT`.
   - Task remains in its failure state and is terminal.
   - `is_finished()` returns `True`.
   - The job's `_compute_job_state` may trigger a job-level state change
     (e.g., `JOB_STATE_FAILED` if `max_task_failures` is exceeded).

### What counts toward job failure

Only `TASK_STATE_FAILED` counts toward the job's `max_task_failures` threshold.
Worker failures (preemptions) do not count. This means a job can survive
unlimited preemptions as long as the per-task preemption budget is not
exhausted.

### States that are never retried

- `SUCCEEDED`: task completed successfully
- `KILLED`: explicit termination by user or cascade
- `UNSCHEDULABLE`: scheduling timeout expired

## Terminal State Summary

A task is considered finished (`is_finished() == True`) when:

| State | Condition |
|---|---|
| `SUCCEEDED` | Always finished |
| `KILLED` | Always finished |
| `UNSCHEDULABLE` | Always finished |
| `FAILED` | Finished when `failure_count > max_retries_failure` |
| `WORKER_FAILED` | Finished when `preemption_count > max_retries_preemption` |

The distinction matters: a task in `FAILED` state with retry budget remaining
is in a terminal state at the attempt level but is not finished at the task
level. `can_be_scheduled()` returns `True` for such tasks.

## Dashboard Display

The dashboard uses `stateToName()` from `shared/utils.js` to convert proto enum
strings (e.g., `TASK_STATE_RUNNING`) to lowercase display names by stripping the
`TASK_STATE_` prefix. Each name maps to a CSS class `status-{name}`:

| Display Name | CSS Class | Color |
|---|---|---|
| `pending` | `.status-pending` | Amber (#9a6700) |
| `assigned` | `.status-assigned` | Orange (#bc4c00) |
| `building` | `.status-building` | Purple (#8250df) |
| `running` | `.status-running` | Blue (#0969da) |
| `succeeded` | `.status-succeeded` | Green (#1a7f37) |
| `failed` | `.status-failed` | Red (#cf222e) |
| `killed` | `.status-killed` | Grey (#57606a) |
| `worker_failed` | `.status-worker_failed` | Purple (#8250df) |
| `unschedulable` | `.status-unschedulable` | Red (#cf222e) |

The job detail page shows per-task attempt history. Each attempt has its own
state badge, and worker failures are annotated with "(worker failure)" in the
attempt rows.

Pending tasks display a `pending_reason` diagnostic below the state badge when
the controller can identify why the task cannot be scheduled (e.g., no workers
match constraints).

## Job State Derivation

Job state is computed from task state counts in `_compute_job_state()`:

1. **SUCCEEDED**: All tasks are in `TASK_STATE_SUCCEEDED`.
2. **FAILED**: Count of `TASK_STATE_FAILED` tasks exceeds `max_task_failures`.
3. **UNSCHEDULABLE**: Any task is `TASK_STATE_UNSCHEDULABLE`.
4. **KILLED**: Any task is `TASK_STATE_KILLED` (and job is not already terminal).
5. **RUNNING**: Any task is `ASSIGNED`, `BUILDING`, or `RUNNING`.
6. **PENDING**: Default (no tasks have started).

The ordering matters -- earlier rules take priority. A job with one succeeded
task and one failed task (beyond tolerance) is `FAILED`, not `RUNNING`.
