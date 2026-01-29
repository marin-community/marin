# Task Dispatch Improvements

**Issue:** https://github.com/marin-community/marin/issues/2540

## Problem

The task dispatch lifecycle has a confusing intermediate state. When the scheduler assigns a task to a worker and the controller fires `TaskAssignedEvent`, the task gets an attempt created in `TASK_STATE_PENDING` state. This means:

- A task shows as PENDING on the dashboard even though it has been assigned to a worker and resources committed
- The RPC to the worker may be in-flight, but the task looks identical to one that hasn't been scheduled at all
- `revert_attempt()` exists but is dead code — dispatch failures are handled via `WorkerFailedEvent` instead

## Design

Add `TASK_STATE_ASSIGNED` as an explicit state between PENDING and RUNNING.

### State Transitions (new)

```
PENDING → ASSIGNED → RUNNING → SUCCEEDED/FAILED/KILLED/WORKER_FAILED
                                    ↓ (retry)
                                  PENDING
```

- **PENDING**: Task created, not yet scheduled
- **ASSIGNED**: Scheduler matched task to worker, attempt created, resources committed, RPC may be in-flight
- **RUNNING**: Worker confirmed execution started
- **SUCCEEDED/FAILED/etc.**: Terminal states (unchanged)

### Changes Required

#### 1. Proto: Add TASK_STATE_ASSIGNED

**File:** `src/iris/rpc/cluster.proto`

Add `TASK_STATE_ASSIGNED = 9;` to the `TaskState` enum. We use 9 (not 2/BUILDING) to avoid breaking the existing enum values.

#### 2. State: Use ASSIGNED in create_attempt

**File:** `src/iris/cluster/controller/state.py`

- `ControllerTask.create_attempt()`: Change default `initial_state` from `TASK_STATE_PENDING` to `TASK_STATE_ASSIGNED`
- `ControllerTask.revert_attempt()`: Already resets to `TASK_STATE_PENDING` — correct behavior
- `ControllerTask.can_be_scheduled()`: No change needed — checks `is_terminal()` on attempt, ASSIGNED is not terminal
- `ControllerTask.is_terminal()`: No change needed — ASSIGNED is not in terminal set
- `_on_task_assigned()` handler: Update to pass `TASK_STATE_ASSIGNED` explicitly

#### 3. State: ASSIGNED is schedulable only when fresh

The `can_be_scheduled()` method checks:
- No attempts → schedulable (fresh task)
- Latest attempt is terminal AND task not finished → schedulable (retry)

ASSIGNED is not terminal, so a task in ASSIGNED state with an active attempt won't be re-scheduled. This is correct.

#### 4. Scheduler: ASSIGNED tasks are not pending

**File:** `src/iris/cluster/controller/scheduler.py`

Check that the scheduler only considers PENDING tasks. Currently it receives `pending_tasks` from the controller — need to verify the controller filters correctly.

#### 5. Dashboard: Show ASSIGNED state

**Files:** `src/iris/cluster/static/worker/app.js`, `src/iris/cluster/static/controller/job-detail.js`

- Worker dashboard: Add ASSIGNED to task counts
- Controller dashboard: ASSIGNED tasks should appear as a distinct state
- CSS: Add `status-assigned` class (yellow/orange to distinguish from pending=gray and running=green)

#### 6. Remove dead code

- Remove `revert_attempt()` — it's never called. Dispatch failures use `WorkerFailedEvent`.

#### 7. Regenerate protos

Run `scripts/generate-protos.py` after proto changes.

#### 8. Update tests

**File:** `tests/cluster/controller/test_state.py`

- Update tests that check task state after assignment to expect ASSIGNED instead of PENDING
- Add test: task in ASSIGNED state is not considered schedulable
- Add test: ASSIGNED → RUNNING transition works

### Files to modify

1. `src/iris/rpc/cluster.proto` — add enum value
2. `scripts/generate-protos.py` — run to regenerate
3. `src/iris/cluster/controller/state.py` — use ASSIGNED in create_attempt, remove revert_attempt
4. `src/iris/cluster/controller/controller.py` — verify pending task filtering
5. `src/iris/cluster/static/worker/app.js` — add ASSIGNED to counts
6. `src/iris/cluster/static/controller/job-detail.js` — handle ASSIGNED state
7. `src/iris/cluster/static/shared/styles.css` — add status-assigned style (if CSS file exists)
8. `tests/cluster/controller/test_state.py` — update expectations
