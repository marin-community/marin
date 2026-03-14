# Design: JobPreemptionPolicy and ExistingJobPolicy

Reference: [marin-community/marin#2967](https://github.com/marin-community/marin/issues/2967)

## Problem

When a worker dies and the controller removes it via `record_heartbeat_failure`
(`lib/iris/src/iris/cluster/controller/transitions.py:1088`), each task on the dead
worker is transitioned to `TASK_STATE_WORKER_FAILED` or `TASK_STATE_PENDING` (if retries
remain). The method calls `_recompute_job_state` (line 1184) to derive the new job
state, but **never calls `_cascade_terminal_job`** when the job reaches a terminal state.

Compare with `apply_task_updates` (line 1056-1072), which does cascade:

```python
new_job_state = self._recompute_job_state(cur, task.job_id)
if new_job_state in (JOB_STATE_FAILED, JOB_STATE_KILLED, ...):
    cascade_kills = _cascade_terminal_job(cur, task.job_id, now_ms, reason)
    tasks_to_kill.update(cascade_kills)
```

This means:

1. **Permanent failure path** (task reports failure via heartbeat → `apply_task_updates`):
   children are correctly cascaded via `_cascade_terminal_job` (line 1070).
2. **Worker death path** (`record_heartbeat_failure`): children are **orphaned**.
   The parent job may enter `JOB_STATE_WORKER_FAILED` but its child jobs keep running.

Even when the parent task is retried (goes back to PENDING because
`preemption_count <= max_retries_preemption` at line 1145), the old child jobs are
still running. When the retried parent creates children with the same names, the
controller raises `ALREADY_EXISTS` because `fail_if_exists=False` only replaces
**finished** jobs (`service.py:594`), not running ones (`service.py:607`).

The practical impact (from issue #2967): Zephyr coordinator and worker actors are
orphaned when the parent host is preempted, causing duplicate pipelines.

## Goals and Non-Goals

### Goals

1. Add `JobPreemptionPolicy` to control whether child jobs are killed when a parent
   task is retried after worker loss:
   - `TERMINATE_CHILDREN` (default for single-task jobs): kill children on preemption retry.
   - `PRESERVE_CHILDREN` (default for multi-task jobs): leave children running.

2. Add `ExistingJobPolicy` enum with values `{ ERROR, KEEP, RECREATE }`:
   - `ERROR`: always error if job exists (replaces `fail_if_exists=true`).
   - `KEEP`: if a job with the same name is running, return a handle to it; replace finished.
   - `RECREATE`: terminate existing job and replace it.
   - Default (`UNSPECIFIED`): current behavior — replace finished jobs, error on running.

3. Fix the missing cascade in `record_heartbeat_failure`.

### Non-Goals

- Changing retry budgets or preemption semantics.
- Modifying reservation holder behavior (holders already requeue on worker death).
- Fray/Ray actor lifecycle management (separate layer).
- Dashboard changes.

## Proposed Solution

### 1. Proto changes

**`lib/iris/src/iris/rpc/cluster.proto`** — add two enums and wire them into
`LaunchJobRequest`:

```protobuf
enum JobPreemptionPolicy {
  JOB_PREEMPTION_POLICY_UNSPECIFIED = 0;  // Controller picks default based on replicas
  JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN = 1;
  JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN = 2;
}

enum ExistingJobPolicy {
  EXISTING_JOB_POLICY_UNSPECIFIED = 0;  // Default: replace finished, error on running
  EXISTING_JOB_POLICY_ERROR = 1;        // Always error if job exists (any state)
  EXISTING_JOB_POLICY_KEEP = 2;         // Return handle to existing running job
  EXISTING_JOB_POLICY_RECREATE = 3;     // Terminate existing job, submit new one
}
```

In `Controller.LaunchJobRequest`, add new fields (keep `fail_if_exists` for wire compat):

```protobuf
message LaunchJobRequest {
    // ... existing fields 1-30 ...

    JobPreemptionPolicy preemption_policy = 31;
    ExistingJobPolicy existing_job_policy = 32;
}
```

The `fail_if_exists` bool (field 22) remains for backward compatibility.
`existing_job_policy` takes precedence when non-`UNSPECIFIED`.

### 2. Fix missing cascade in `record_heartbeat_failure`

**`lib/iris/src/iris/cluster/controller/transitions.py`**

The bug: line 1184 calls `_recompute_job_state` but discards the result. When
the job enters a terminal state (e.g., preemption budget exhausted), child jobs
are not cascaded.

Extract child-only cascade from `_cascade_terminal_job` (line 254):

```python
def _cascade_children(cur, job_id: JobName, now_ms: int, reason: str) -> set[JobName]:
    """Kill descendant jobs of job_id (not job_id's own tasks)."""
    proto_cache: dict[str, cluster_pb2.Controller.LaunchJobRequest] = {}
    tasks_to_kill: set[JobName] = set()
    descendants = cur.execute(
        "WITH RECURSIVE subtree(job_id) AS ("
        "  SELECT job_id FROM jobs WHERE parent_job_id = ? "
        "  UNION ALL "
        "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
        ") SELECT job_id FROM subtree",
        (job_id.to_wire(),),
    ).fetchall()
    for child_row in descendants:
        child_job_id = str(child_row["job_id"])
        tasks_to_kill.update(_kill_non_terminal_tasks(cur, child_job_id, reason, now_ms, proto_cache))
        cur.execute(
            "UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
            "WHERE job_id = ? AND state NOT IN (?, ?, ?, ?)",
            (cluster_pb2.JOB_STATE_KILLED, reason, now_ms, child_job_id,
             *TERMINAL_JOB_STATES),
        )
    return tasks_to_kill
```

Then refactor `_cascade_terminal_job` to use it:
```python
def _cascade_terminal_job(cur, job_id, now_ms, reason):
    proto_cache = {}
    tasks_to_kill = _kill_non_terminal_tasks(cur, job_id.to_wire(), reason, now_ms, proto_cache)
    tasks_to_kill.update(_cascade_children(cur, job_id, now_ms, "Parent job terminated"))
    return tasks_to_kill
```

### 3. Preemption policy enforcement

Add to `record_heartbeat_failure` after line 1184:

```python
new_job_state = self._recompute_job_state(cur, parent_job_id)
# NEW: cascade children if job reached terminal state
if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
    tasks_to_kill.update(
        _cascade_terminal_job(cur, parent_job_id, now_ms, f"Worker {worker_id} failed")
    )
# NEW: cascade children on retry if TERMINATE_CHILDREN policy
elif new_task_state == cluster_pb2.TASK_STATE_PENDING:
    policy = _resolve_preemption_policy(cur, parent_job_id)
    if policy == cluster_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
        tasks_to_kill.update(
            _cascade_children(cur, parent_job_id, now_ms, "Parent task preempted")
        )
```

Similarly in `apply_task_updates` (line 1055), gate the existing cascade on policy
for non-SUCCESS terminal states. SUCCESS always cascades (children should be cleaned
up when parent finishes).

Policy resolution helper:
```python
def _resolve_preemption_policy(cur, job_id: JobName) -> int:
    row = cur.execute(
        "SELECT request_proto FROM jobs WHERE job_id = ?", (job_id.to_wire(),)
    ).fetchone()
    if row is None:
        return cluster_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    req = cluster_pb2.Controller.LaunchJobRequest()
    req.ParseFromString(row["request_proto"])
    if req.preemption_policy != cluster_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED:
        return req.preemption_policy
    # Default: single-task → TERMINATE, multi-task → PRESERVE
    if req.replicas <= 1:
        return cluster_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    return cluster_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN
```

### 4. ExistingJobPolicy in controller service

**`lib/iris/src/iris/cluster/controller/service.py`** — replace the `fail_if_exists`
logic at line 591-607:

```python
existing_job = _read_job(self._db, job_id)
if existing_job:
    policy = _effective_existing_job_policy(request)
    if policy == cluster_pb2.EXISTING_JOB_POLICY_KEEP and not existing_job.is_finished():
        # Return handle to existing running job
        return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
    elif policy == cluster_pb2.EXISTING_JOB_POLICY_RECREATE:
        if not existing_job.is_finished():
            self._transitions.cancel_job(job_id, "Replaced by new submission")
        self._transitions.remove_finished_job(job_id)
    elif existing_job.is_finished():
        self._transitions.remove_finished_job(job_id)
    else:
        raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")
```

Where `_effective_existing_job_policy` maps `UNSPECIFIED` to current default behavior
and respects `fail_if_exists` for backward compat:

```python
def _effective_existing_job_policy(request) -> int:
    if request.existing_job_policy != cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED:
        return request.existing_job_policy
    if request.fail_if_exists:
        return cluster_pb2.EXISTING_JOB_POLICY_ERROR
    # Default: replace finished, error on running (current behavior)
    return cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED
```

### 5. Client API changes

**`lib/iris/src/iris/client/client.py:595`** — add to `IrisClient.submit()`:

```python
def submit(
    self,
    ...,
    preemption_policy: int = 0,      # JOB_PREEMPTION_POLICY_UNSPECIFIED
    existing_job_policy: int = 0,     # EXISTING_JOB_POLICY_UNSPECIFIED
) -> Job:
```

**`lib/iris/src/iris/cluster/client/remote_client.py:85`** — set on the proto:

```python
request = cluster_pb2.Controller.LaunchJobRequest(
    ...,
    preemption_policy=preemption_policy,
    existing_job_policy=existing_job_policy,
)
```

### 6. Storage

No DB schema change needed. Both policies are stored in `request_proto` blob
(the `jobs` table stores the serialized `LaunchJobRequest`). The policy is read
back from `request_proto` when needed during preemption handling.

## Implementation Outline

Each step is independently testable:

### Step 1: Proto + codegen
- **File**: `lib/iris/src/iris/rpc/cluster.proto`
- Add `JobPreemptionPolicy`, `ExistingJobPolicy` enums
- Add fields 31, 32 to `LaunchJobRequest`
- Regenerate proto
- **Test**: proto compiles, fields round-trip

### Step 2: Refactor `_cascade_terminal_job`
- **File**: `lib/iris/src/iris/cluster/controller/transitions.py`
- Extract `_cascade_children` (pure refactor, no behavior change)
- **Test**: existing tests pass unchanged

### Step 3: Fix worker-death cascade bug + preemption policy
- **File**: `lib/iris/src/iris/cluster/controller/transitions.py`
- After `_recompute_job_state` in `record_heartbeat_failure` (line 1184):
  cascade children on terminal state, apply policy on retry
- Same for `apply_task_updates` (line 1070): gate non-SUCCESS cascade on policy
- **Test**: add to `lib/iris/tests/cluster/controller/`:
  - Single-task parent preempted → child jobs killed (TERMINATE_CHILDREN)
  - Multi-task parent preempted → children preserved (PRESERVE_CHILDREN)
  - Parent exhausts preemption budget → children always killed

### Step 4: ExistingJobPolicy in controller service
- **File**: `lib/iris/src/iris/cluster/controller/service.py`
- Replace `fail_if_exists` logic with policy dispatch
- **Test**: extend `lib/iris/tests/cluster/controller/test_service.py`:
  - KEEP on running job → returns existing handle
  - RECREATE on running job → terminates and replaces
  - ERROR on finished job → errors
  - UNSPECIFIED preserves current behavior

### Step 5: Client API propagation
- **Files**: `lib/iris/src/iris/client/client.py`,
  `lib/iris/src/iris/cluster/client/remote_client.py`
- Add parameters, wire through to proto
- **Test**: unit test that proto fields are set correctly

## Risks and Open Questions

1. **Cascade timing with retries**: When a parent has retries remaining, its task
   goes back to PENDING (not terminal). With TERMINATE_CHILDREN, children are killed
   immediately on retry — before the parent re-runs. This is intentional: the retried
   parent will recreate children. But if child creation is expensive (large downloads),
   PRESERVE_CHILDREN + KEEP policy may be preferred. This is a caller decision.

2. **KEEP policy semantics**: With KEEP, `LaunchJob` returns a response for the
   existing running job without actually creating anything. The caller gets back a
   `Job` handle pointing at the existing job. The caller must be prepared for the job
   to be in any state (building, running, etc.).

3. **RECREATE atomicity**: `cancel_job` → `remove_finished_job` → resubmit is not
   atomic from the caller's perspective. A crash between cancel and resubmit leaves
   the child terminated with no replacement. Acceptable for v1.

4. **`fail_if_exists` deprecation path**: Keeping field 22 avoids wire breakage.
   New `existing_job_policy` takes precedence. If both are set, policy wins. Worth
   logging a deprecation warning if `fail_if_exists` is used.

5. **apply_task_updates cascade change**: Currently `apply_task_updates` cascades
   children on ALL terminal states including `JOB_STATE_SUCCEEDED`. SUCCESS cascade
   must remain unconditional (clean up children when parent finishes). Only
   WORKER_FAILED → terminal cascade should consult the policy.
