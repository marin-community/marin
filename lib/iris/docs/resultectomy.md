# Proposal: Remove `_result.pkl` / Callable Job Result Return

**Status:** Proposal
**Date:** 2026-02-23

## Problem

Iris supports "callable" tasks via `Entrypoint.from_callable(fn, *args, **kwargs)`.
When the callable completes, the generated runner script (`CALLABLE_RUNNER` in
`cluster/types.py`) pickles the return value to `_result.pkl` in the task's working
directory. The worker's `TaskAttempt._monitor()` then reads this file into
`self.result`.

With `KubernetesRuntime` on CoreWeave, this is broken by design. The task Pod runs
in an isolated Pod with an `emptyDir`-backed `/app`. The worker runs in a *different*
Pod. When the task writes `_result.pkl` inside its Pod, the worker cannot read it --
the file never leaves the task Pod's filesystem. Callable tasks silently return `None`.

Beyond the K8s incompatibility, the feature is also broken end-to-end even on Docker:

1. `TaskAttempt` reads `_result.pkl` into `self.result` (bytes) at line 633 of
   `task_attempt.py`.
2. `TaskInfo` protocol exposes `result -> bytes | None` (line 67 of `worker_types.py`).
3. `WorkerServiceImpl.get_task_status()` has a dead code path (lines 59-62 of
   `service.py`) with a comment: *"TaskStatus doesn't have serialized_result field,
   but we could add it."*
4. The `CompletedTaskEntry` heartbeat message has no result field, so task results
   never reach the controller.
5. `JobStatus.serialized_result` (proto field 13) exists in `cluster.proto` but is
   never populated by the controller.

The result: even with Docker runtime, `self.result` is a dead store. It is read from
disk and held in memory, but never transmitted back to the client. No production code
path consumes it.

The one place that *does* return callable results to callers is `WorkerPool`, but it
does so via the actor RPC protocol (`ActorServiceClientSync.call()`), bypassing
`_result.pkl` entirely.

## Why not fix it?

Copying `_result.pkl` back from a K8s task Pod to the worker Pod would require:

- `kubectl cp` or a `read_file` RPC on the task Pod before it terminates.
- Shared PVCs between worker and task Pods (adding infrastructure coupling).
- A new "result relay" sidecar or init container.

All of these add complexity and tight coupling between the worker and the runtime.
They also introduce failure modes (result file too large, Pod evicted before copy,
network partition during transfer) that must be handled. The feature has zero known
consumers -- `WorkerPool` uses actor RPC, and the only tests exercising `from_callable`
never inspect the return value through Iris infrastructure.

The simpler path is to remove the dead plumbing.

## Scope of removal

### Files to change

**`lib/iris/src/iris/cluster/types.py`**

- `CALLABLE_RUNNER` (lines 512-537): Remove the `_result.pkl` write. The runner
  should call the function and exit. Change:
  ```python
  result = fn(*args, **kwargs)
  with open(os.path.join(workdir, "_result.pkl"), "wb") as f:
      f.write(cloudpickle.dumps(result))
  ```
  to:
  ```python
  fn(*args, **kwargs)
  ```

**`lib/iris/src/iris/cluster/worker/task_attempt.py`**

- `self.result` field (line 277): Remove the `result: bytes | None = None` attribute.
- `_monitor()` result reading (lines 628-635): Remove the block that reads
  `_result.pkl` after container exit:
  ```python
  if status.exit_code == 0 and self.workdir:
      result_path = self.workdir / "_result.pkl"
      if result_path.exists():
          try:
              self.result = result_path.read_bytes()
          except Exception as e:
              ...
  ```

**`lib/iris/src/iris/cluster/worker/worker_types.py`**

- `TaskInfo.result` protocol property (lines 66-68): Remove the `result` property
  from the `TaskInfo` protocol.

**`lib/iris/src/iris/cluster/worker/service.py`**

- `get_task_status()` dead code (lines 59-62): Remove the `include_result` handling
  that currently does nothing.

**`lib/iris/src/iris/rpc/cluster.proto`**

- `JobStatus.serialized_result` (line 183): Remove field 13. Mark it `reserved` to
  prevent accidental reuse.
- `Controller.GetJobStatusRequest.include_result` (line 446): Remove field 2. Mark
  it `reserved`.
- `Worker.GetTaskStatusRequest.include_result` (line 703): Remove field 2. Mark it
  `reserved`.

Regenerate proto files via `scripts/generate_protos.py` after proto changes.

### Files that need no changes (but are worth noting)

- **`lib/iris/src/iris/cluster/controller/scheduler.py`**: No result handling exists
  here. The scheduler is pure assignment logic.
- **`lib/iris/src/iris/cluster/controller/controller.py`**: Never reads
  `serialized_result`. `CompletedTaskEntry` has no result field.
- **`lib/iris/src/iris/cluster/client/remote_client.py`**: Never reads
  `serialized_result` from `JobStatus` or `TaskStatus`.
- **`lib/iris/src/iris/client/worker_pool.py`**: Uses actor RPC for results
  (`ActorClient.execute()`), not `_result.pkl`. No changes needed.
- **`lib/iris/src/iris/actor/`**: Actor framework uses its own cloudpickle-over-RPC
  for results. Completely independent of `_result.pkl`.

### Tests to update

- **`lib/iris/tests/cluster/test_types.py`**: `test_entrypoint_from_callable_resolve_roundtrip`
  (line 27) tests `Entrypoint.from_callable` serialization, which is unaffected.
  `CALLABLE_RUNNER` changes may require updating tests that inspect the runner script
  content, if any exist.
- **`lib/iris/tests/client/test_worker_pool.py`**: Tests callable return values via
  `WorkerPool` actor RPC. No changes needed.
- **E2E tests using `from_callable`**: The many tests using `Entrypoint.from_callable`
  (e.g., in `test_smoke.py`, `test_scheduling.py`, `test_docker.py`, `conftest.py`)
  submit callables but never read `_result.pkl` or `serialized_result`. They check
  job status (SUCCEEDED/FAILED), not return values. No changes needed.

## Migration path

Users who need to return values from callable tasks should:

1. **Use `WorkerPool`**: The `WorkerPool` + `ActorPool` pattern already provides
   callable dispatch with result return via actor RPC. This is the recommended
   approach for interactive result retrieval.

2. **Write results to object storage**: For batch jobs, write results to GCS/S3
   from within the task. This is more robust than pickle files and works across
   all runtimes.

3. **Use task logs**: For small outputs (status codes, metrics), log them from the
   task and read via `fetch_task_logs()`.

## Alternatives considered

### Add `kubectl cp` / shared volume for K8s result transfer

Rejected. This would:

- Couple the worker to Kubernetes internals (kubectl exec, PVC provisioning).
- Require the worker to poll for `_result.pkl` existence after task Pod completion.
- Add a new failure mode: task Pod evicted before result file is copied.
- Add latency between task completion and result availability.
- Solve a problem that has no known consumers.

### Add a result field to `CompletedTaskEntry` and propagate through heartbeats

Rejected. This would:

- Add arbitrary-sized binary payloads to the heartbeat protocol, which is designed
  for lightweight state synchronization.
- Require buffering large results in the controller's memory.
- Still not solve the K8s runtime problem (worker can't read the file).

## Implementation plan

1. **Remove `_result.pkl` write from `CALLABLE_RUNNER`** in `cluster/types.py`.
   The callable's return value is simply discarded.

2. **Remove `self.result` and `_result.pkl` reading** from `task_attempt.py`.
   Delete the attribute and the file-reading block in `_monitor()`.

3. **Remove `TaskInfo.result`** from `worker_types.py`. Remove the dead
   `include_result` handling from `service.py`.

4. **Clean up proto fields**: Reserve `serialized_result`, `include_result` fields
   in `cluster.proto`. Regenerate proto files.

5. **Run tests**: `uv run pytest lib/iris/tests/ -x -o "addopts="` to verify
   nothing breaks.
