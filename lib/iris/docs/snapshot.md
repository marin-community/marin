# Controller Snapshot/Checkpoint Design

## Problem

The Iris controller holds all state in memory. A restart -- whether due to a bug fix, image upgrade, or crash -- wipes job queues, worker registrations, autoscaler slice ownership, and reservation claims. Workers re-register on the next heartbeat timeout, but their running tasks are orphaned: the controller has no record of them, so it sends kill requests for "unknown" tasks. The autoscaler loses its slice inventory, so it cannot scale down orphaned VMs. The result is resource leaks and user-visible job failures on every controller restart.

This document designs a checkpoint/restore mechanism that lets the controller serialize its essential state to remote storage before stopping and reload it on startup, preserving the continuity of the task scheduler and autoscaler across restarts.

## Goals

1. **Lossless restart**: Running and pending jobs survive a controller restart. Tasks already dispatched to workers continue executing; pending tasks resume scheduling.
2. **Autoscaler continuity**: Slice ownership and scaling group state (backoff timers, demand counters) are preserved so the autoscaler does not orphan VMs or re-provision already-running slices.
3. **Simple implementation**: A single JSON blob on GCS/S3/local FS, written atomically. No external database. Versioned schema for forward compatibility.
4. **Testable in isolation**: Checkpoint and restore are pure functions over a snapshot dataclass. E2E tests exercise the full cycle using the existing `LocalController` + `FakePlatform` infrastructure.

## Non-Goals

- **External state store** (Redis, etcd, SQLite): Out of scope. The checkpoint file approach is sufficient for the current single-controller architecture.
- **Hot standby / HA failover**: Not addressed. This is cold restart only.
- **Incremental / streaming checkpoints**: Full snapshot on each checkpoint. The state is small (tens of MB for large clusters).
- **Worker-side state persistence**: Workers already handle restart via container cleanup + re-registration. No changes needed on the worker side. The controller serializes its view of workers so it can resume heartbeating immediately.

## State Inventory

The controller's mutable state lives in four major locations. Each needs explicit checkpoint coverage.

### 1. ControllerState (`cluster/controller/state.py`)

The thread-safe container holding all job/task/worker/endpoint data.

| Field | Type | Checkpoint? | Notes |
|-------|------|-------------|-------|
| `_jobs` | `dict[JobName, ControllerJob]` | **Yes** | Core job metadata, request protos, timestamps, state counters |
| `_tasks` | `dict[JobName, ControllerTask]` | **Yes** | Task state, retry counters, attempt history |
| `_tasks_by_job` | `dict[JobName, list[JobName]]` | Derived | Rebuilt from `_tasks` on restore |
| `_workers` | `dict[WorkerId, ControllerWorker]` | **Yes** | Serialized so the controller can resume heartbeats immediately. Health state is reset to fresh on restore. |
| `_task_queue` | `list[QueueEntry]` | Derived | Rebuilt from `_tasks` (enqueue all `can_be_scheduled()` tasks) |
| `_endpoints` | `dict[str, ControllerEndpoint]` | **Yes** | Service discovery endpoints for actors |
| `_endpoints_by_task` | `dict[JobName, set[str]]` | Derived | Rebuilt from `_endpoints` |
| `_transactions` | `deque[TransactionLog]` | **No** | Debug log, not needed for correctness |
| `_pending_dispatch` | `dict[WorkerId, PendingDispatch]` | **No** | Stale after restart; drained before checkpoint |

**Key decision: workers ARE checkpointed.** The controller uses a push-model heartbeat: it initiates heartbeat RPCs _to_ workers at the worker's known address. Workers passively wait for these heartbeats and reset + re-register only when heartbeats stop arriving (default timeout ~60s). By serializing worker identity (`worker_id`, `address`, `metadata`, `attributes`), the restored controller can resume heartbeating immediately on startup. Workers never notice the restart as long as it completes within the heartbeat timeout.

On restore, each `ControllerWorker` is reconstructed with:
- Identity fields (`worker_id`, `address`, `metadata`, `attributes`) restored from checkpoint
- Health fields reset to fresh: `healthy=True`, `consecutive_failures=0`, `last_heartbeat=now()`
- `running_tasks` and committed resources rebuilt from restored task state (tasks in ASSIGNED/BUILDING/RUNNING whose `worker_id` matches)
- `resource_snapshot` and `resource_history` set to empty (live metrics repopulate on first heartbeat)

If a worker died during the restart window, the first heartbeat RPC fails. The normal failure cascade (consecutive failures → WORKER_FAILED → task retry) handles this without any special reconciliation logic.

### 2. Autoscaler (`cluster/controller/autoscaler.py`)

| Field | Type | Checkpoint? | Notes |
|-------|------|-------------|-------|
| `_workers` (TrackedWorker) | `dict[str, TrackedWorker]` | **Yes** | Maps worker_id to slice_id/scale_group. Essential for slice ownership. |
| `_action_log` | `deque[AutoscalerAction]` | **No** | Debug log |
| `_last_routing_decision` | `RoutingDecision` | **No** | Recomputed each cycle |
| `_last_evaluation` | `Timestamp` | **No** | Recomputed each cycle |

### 3. ScalingGroup (`cluster/controller/scaling_group.py`)

Per-group state that must survive restart:

| Field | Type | Checkpoint? | Notes |
|-------|------|-------------|-------|
| `_slices` | `dict[str, SliceState]` | **Yes** | Core slice inventory: slice_id, lifecycle state, VM addresses |
| `_pending_scale_ups` | `int` | **No** | Reset to 0; in-flight scale-ups are lost. The next autoscaler cycle will re-evaluate and re-issue if needed. |
| `_current_demand` | `int` | **No** | Recomputed each cycle |
| `_peak_demand` | `int` | **No** | Diagnostic only |
| `_backoff_until` | `Deadline` | **Yes** | Prevents thrashing after restart if we were in backoff |
| `_consecutive_failures` | `int` | **Yes** | Needed to compute correct backoff duration |
| `_last_scale_up` | `Timestamp` | **Yes** | Cooldown gate |
| `_last_scale_down` | `Timestamp` | **Yes** | Cooldown gate |
| `_quota_exceeded_until` | `Deadline` | **Yes** | Prevents quota-exceeded thrashing |
| `_quota_reason` | `str` | **Yes** | Diagnostic context |
| `_idle_threshold`, cooldown durations | `Duration` | **No** | Immutable config, not state |

**SliceState details**: For each slice we checkpoint:
- `slice_id`, `scale_group` (from SliceHandle)
- `lifecycle` state (BOOTING, READY, FAILED, etc.)
- `vm_addresses` (list of worker IP addresses in this slice)
- `created_at` timestamp
- `last_active` timestamp

The `SliceHandle` itself is a platform object that cannot be directly serialized. On restore, we use `platform.list_slices()` to re-discover slice handles from the cloud and match them to checkpointed slice state by `slice_id`.

### 4. Controller (`cluster/controller/controller.py`)

| Field | Type | Checkpoint? | Notes |
|-------|------|-------------|-------|
| `_reservation_claims` | `dict[WorkerId, ReservationClaim]` | **Yes** | Prevents reservation re-claiming from scratch |
| `_heartbeat_iteration` | `int` | **No** | Cosmetic counter |

## Snapshot Schema

All checkpointed state is collected into a single `ControllerSnapshot` dataclass, serialized to JSON via a `snapshot_pb2.ControllerSnapshot` protobuf message. Using protobuf for the wire format gives us forward-compatible schema evolution and leverages the existing proto infrastructure.

```protobuf
// snapshot.proto
edition = "2023";
package iris.snapshot;

import "time.proto";
import "cluster.proto";
import "config.proto";

message ControllerSnapshot {
  // Schema version for forward compatibility. The restore path
  // rejects snapshots with a major version it does not understand.
  int32 schema_version = 1;
  iris.time.Timestamp created_at = 2;

  // --- ControllerState ---
  repeated JobSnapshot jobs = 10;
  repeated EndpointSnapshot endpoints = 11;
  repeated WorkerSnapshot workers = 12;

  // --- Autoscaler ---
  repeated ScalingGroupSnapshot scaling_groups = 20;
  repeated TrackedWorkerSnapshot tracked_workers = 21;

  // --- Controller ---
  repeated ReservationClaimSnapshot reservation_claims = 30;
}

message JobSnapshot {
  string job_id = 1;
  iris.cluster.Controller.LaunchJobRequest request = 2;
  int32 state = 3;  // JobState enum value

  iris.time.Timestamp submitted_at = 4;
  iris.time.Timestamp root_submitted_at = 5;
  iris.time.Timestamp started_at = 6;
  iris.time.Timestamp finished_at = 7;

  string error = 8;
  int32 exit_code = 9;
  int32 num_tasks = 10;

  // Scheduling deadline: if set, the absolute epoch_ms when the job
  // becomes unschedulable. Deadlines are monotonic-clock based in
  // memory but stored as wall-clock for portability across restarts.
  int64 scheduling_deadline_epoch_ms = 11;

  repeated TaskSnapshot tasks = 20;
}

message TaskSnapshot {
  string task_id = 1;
  string job_id = 2;
  int32 state = 3;

  string error = 4;
  int32 exit_code = 5;
  iris.time.Timestamp started_at = 6;
  iris.time.Timestamp finished_at = 7;
  iris.time.Timestamp submitted_at = 8;

  int32 max_retries_failure = 10;
  int32 max_retries_preemption = 11;
  int32 failure_count = 12;
  int32 preemption_count = 13;

  repeated TaskAttemptSnapshot attempts = 20;
}

message TaskAttemptSnapshot {
  int32 attempt_id = 1;
  string worker_id = 2;
  int32 state = 3;
  string log_directory = 4;

  iris.time.Timestamp created_at = 5;
  iris.time.Timestamp started_at = 6;
  iris.time.Timestamp finished_at = 7;

  int32 exit_code = 8;
  string error = 9;
}

message EndpointSnapshot {
  string endpoint_id = 1;
  string name = 2;
  string address = 3;
  string job_id = 4;
  map<string, string> metadata = 5;
  iris.time.Timestamp registered_at = 6;
}

message WorkerSnapshot {
  string worker_id = 1;
  string address = 2;
  iris.cluster.WorkerMetadata metadata = 3;
  map<string, iris.cluster.AttributeValue> attributes = 4;
}

message ScalingGroupSnapshot {
  string name = 1;

  repeated SliceSnapshot slices = 10;

  int32 consecutive_failures = 20;
  iris.time.Timestamp backoff_until = 21;
  iris.time.Timestamp last_scale_up = 22;
  iris.time.Timestamp last_scale_down = 23;
  iris.time.Timestamp quota_exceeded_until = 24;
  string quota_reason = 25;
}

message SliceSnapshot {
  string slice_id = 1;
  string scale_group = 2;
  string lifecycle = 3;  // SliceLifecycleState string value
  repeated string vm_addresses = 4;
  iris.time.Timestamp created_at = 5;
  iris.time.Timestamp last_active = 6;
  string error_message = 7;
}

message TrackedWorkerSnapshot {
  string worker_id = 1;
  string slice_id = 2;
  string scale_group = 3;
  string internal_address = 4;
}

message ReservationClaimSnapshot {
  string worker_id = 1;
  string job_id = 2;
  int32 entry_idx = 3;
}
```

## Serialization Format and Storage

**Format**: Protocol Buffers serialized to JSON (via `google.protobuf.json_format`). JSON is human-readable for debugging and inspectable with standard tools. The proto schema handles versioning.

**Storage**: Written to the configured `bundle_prefix` location (GCS, S3, or local FS) using `fsspec`, which the controller already depends on for bundle storage. The snapshot path is:

```
{bundle_prefix}/../controller-snapshots/snapshot-{timestamp_ms}.json
```

The latest snapshot is also written to a well-known path:

```
{bundle_prefix}/../controller-snapshots/latest.json
```

Writing uses atomic rename: write to a `.tmp` file, then rename. On GCS this is `copy + delete` which is eventually consistent but sufficient for our use case (single writer).

## Checkpoint Protocol

### BeginCheckpoint RPC

```protobuf
// Added to ControllerService in cluster.proto
rpc BeginCheckpoint(BeginCheckpointRequest) returns (BeginCheckpointResponse);

message BeginCheckpointRequest {}
message BeginCheckpointResponse {
  string snapshot_path = 1;  // Where the snapshot was written
  iris.time.Timestamp created_at = 2;
}
```

The checkpoint sequence:

1. **Pause scheduling**: Set a `_checkpoint_in_progress` flag that the scheduling loop checks. The scheduling loop skips `_run_scheduling()` while the flag is set. This prevents new task assignments during checkpoint.

2. **Pause autoscaler**: The autoscaler loop similarly checks the flag and skips `_run_autoscaler_once()`.

3. **Wait for in-flight heartbeats to drain**: The heartbeat loop completes its current round. We wait for the dispatch executor to be idle (no in-flight heartbeat RPCs). This ensures all pending task state changes from heartbeat responses have been applied.

4. **Snapshot state under lock**: Acquire `ControllerState._lock` and serialize all checkpointed fields into a `ControllerSnapshot` proto. Also snapshot autoscaler and scaling group state (these have their own locks).

5. **Write to storage**: Write the serialized snapshot to remote storage via fsspec.

6. **Resume or stop**: If the checkpoint was for a graceful restart, the controller can now stop. If it was a periodic checkpoint, clear the flag and resume.

```python
def begin_checkpoint(self) -> str:
    """Pause, snapshot, and write controller state to remote storage.

    Returns the path where the snapshot was written.
    """
    self._checkpoint_in_progress = True
    try:
        # Wait for in-flight dispatches to settle. The heartbeat loop
        # will finish its current round and then idle because the
        # scheduling loop is paused (no new dispatches buffered).
        self._drain_inflight_heartbeats()

        # Snapshot under locks
        snapshot = self._create_snapshot()

        # Write to storage
        path = self._write_snapshot(snapshot)
        return path
    finally:
        self._checkpoint_in_progress = False
```

### Periodic Checkpointing

In addition to the explicit `BeginCheckpoint` RPC, the controller writes periodic checkpoints on a configurable interval (default: 60 seconds). This provides crash recovery even without an explicit checkpoint request. The periodic checkpoint runs in the autoscaler loop thread (which already has the appropriate cadence) and does NOT pause scheduling -- it takes a best-effort snapshot that may include partially-committed state. For crash recovery this is acceptable: the worst case is that a few recently-assigned tasks are re-assigned after restore.

### Handling In-Flight Operations

**In-flight heartbeats**: The `_drain_inflight_heartbeats()` call waits up to 10 seconds for the dispatch executor to become idle. If heartbeats are still in flight after the timeout, the checkpoint proceeds anyway -- the state is still consistent because heartbeat responses are applied under the state lock, and any responses that arrive after the snapshot is taken simply update state that will be lost (the controller is about to stop).

**In-flight scale-ups**: Scale-up threads run asynchronously and may be creating VMs while the checkpoint runs. These threads call `complete_scale_up()` or `cancel_scale_up()` on ScalingGroup. `_pending_scale_ups` is NOT checkpointed (reset to 0 on restore). The restore path reconciles via `platform.list_slices()`:

- **GCP (TPU and VM slices)**: `create_slice()` is synchronous — the cloud resource exists (and is visible to `list_slices()`) before `create_slice()` returns. The gap between `create_slice()` return and `complete_scale_up()` call is small (typically sub-second). If a checkpoint happens in this gap, the slice exists in the cloud but not in the ScalingGroup. On restore, `list_slices()` discovers it and it is adopted as BOOTING. GCP VM slices use startup-script metadata for bootstrap, so no SSH is needed and the VM self-bootstraps on boot. GCP TPU slices use SSH-based bootstrap in a background thread; the TPU resource itself is visible to `list_slices()` immediately regardless of bootstrap state.

- **CoreWeave**: `create_slice()` returns immediately with an in-memory handle, then submits Pod creation to a background thread. There is a real gap between handle creation and `kubectl apply` where the Pod does not yet exist in K8s. If a checkpoint happens during this gap, the pending counter is lost and `list_slices()` won't find the Pod. This is safe: the next autoscaler cycle re-evaluates demand and re-issues the scale-up. The worst case is a brief delay in provisioning.

- **Manual/Local**: Slices are tracked in-memory immediately upon `create_slice()` return. No cloud resources are involved. On restore, `list_slices()` returns from the in-memory dict (which is empty after restart), so all slices must come from the checkpoint.

In all cases, the autoscaler re-evaluates demand on its first cycle after restore, so any lost in-flight scale-ups are re-issued promptly.

## Restore Protocol

### On Startup

The controller checks for a snapshot file at the well-known path. If found:

1. **Deserialize snapshot**: Read and parse the JSON proto.

2. **Version check**: Reject snapshots with incompatible schema versions.

3. **Restore ControllerState**: For each checkpointed job, task, and worker:
   - Reconstruct `ControllerJob` and `ControllerTask` objects from the snapshot.
   - Rebuild derived indexes (`_tasks_by_job`, `_task_queue`, `_endpoints_by_task`).
   - Reconstruct `ControllerWorker` objects from `WorkerSnapshot` entries: restore `worker_id`, `address`, `metadata`, and `attributes`. Set health fields to fresh defaults (`healthy=True`, `consecutive_failures=0`, `last_heartbeat=Timestamp.now()`). Rebuild `running_tasks` and committed resource counters from restored tasks (tasks in ASSIGNED/BUILDING/RUNNING state whose latest attempt references this worker).

4. **Restore autoscaler state**: For each scaling group:
   - Re-discover slice handles via `platform.list_slices()` and match to checkpointed `SliceSnapshot` by `slice_id`.
   - Restore lifecycle state, timing state (backoff, cooldowns), and VM addresses.
   - Slices in the checkpoint that no longer exist in the cloud are discarded (they were terminated while the controller was down).
   - Slices in the cloud that are NOT in the checkpoint are adopted as BOOTING (they were created by an in-flight scale-up that completed after the checkpoint).

5. **Restore reservation claims**: Restored as-is. Stale claims (for workers that have died) are cleaned up by the existing `_cleanup_stale_claims()` on the first scheduling cycle.

6. **Resume normal operation**: Start scheduling, heartbeat, and autoscaler loops.

### Worker Resumption via Serialized State

Because worker state is checkpointed, the controller resumes heartbeating to workers immediately on startup. Workers use a push-model heartbeat: they passively wait for the controller to send heartbeat RPCs and only reset + re-register when heartbeats stop arriving. As long as the controller restarts within the worker's heartbeat timeout (~60s default), workers never notice the restart.

The existing heartbeat reconciliation protocol handles all consistency cases naturally:

**Worker is alive and has the expected tasks**: The first heartbeat succeeds. The worker reports its running tasks, which match the restored task records. Normal operation continues.

**Worker reports tasks the controller does NOT know about**: This can happen if a task was assigned between the checkpoint and the controller stop. The controller sends a kill request for the unknown task on the next heartbeat. The worker terminates the container. This is the existing behavior.

**Controller expects tasks the worker does NOT report**: If a restored task was ASSIGNED/BUILDING/RUNNING but the worker does not report it (task finished or worker restarted during the restart window), the heartbeat response indicates the task is missing. The controller marks it as WORKER_FAILED and retries it.

**Worker died during the restart window**: The first heartbeat RPC to this worker fails. The controller increments `consecutive_failures`. After `HEARTBEAT_FAILURE_THRESHOLD` (10) consecutive failures, the worker is pruned from state and its tasks are marked WORKER_FAILED. The autoscaler discovers the slice is gone via `platform.list_slices()` and removes it. No special reconciliation logic is needed — this is the existing failure cascade.

**Controller restart takes longer than heartbeat timeout**: If the restart exceeds the worker's heartbeat timeout, workers will reset their state (kill containers) and attempt to re-register. The restored controller accepts the re-registration via the existing `_on_worker_registered` path, which updates the worker record in-place. Tasks that were running on the worker are now gone (containers killed), so the heartbeat reconciliation marks them as WORKER_FAILED and retries them. This is a degraded but safe path — it matches the behavior of the original "no worker checkpoint" design.

### Scheduling Deadline Adjustment

`ControllerJob.scheduling_deadline` uses `Deadline` which is based on monotonic time internally. On restore, we convert the checkpointed wall-clock deadline back to a monotonic `Deadline` by computing the remaining time:

```python
remaining_ms = snapshot.scheduling_deadline_epoch_ms - snapshot.created_at.epoch_ms
if remaining_ms > 0:
    job.scheduling_deadline = Deadline.from_now(Duration.from_ms(remaining_ms))
else:
    # Deadline already expired at checkpoint time
    job.scheduling_deadline = Deadline.from_now(Duration.from_ms(0))
```

Similarly, `_backoff_until` and `_quota_exceeded_until` in ScalingGroup use `Deadline` and need the same wall-clock-to-monotonic conversion on restore.

## Edge Cases

### Autoscaler groups mid-boot

A scaling group may have slices in BOOTING or INITIALIZING state at checkpoint time. On restore:

- The slice handle is re-discovered via `platform.list_slices()`.
- The lifecycle state is restored from the checkpoint (e.g., BOOTING).
- The autoscaler's `refresh()` method polls the slice via `handle.describe()` and will transition it to READY or FAILED as usual.
- If the slice completed booting while the controller was down, `describe()` returns READY and the autoscaler handles the transition.

### Tasks in flight during checkpoint

Tasks in ASSIGNED/BUILDING/RUNNING state are checkpointed as-is. On restore:

- The task record is restored with its full attempt history.
- The worker is also restored (from `WorkerSnapshot`), so the worker_id reference is valid.
- The controller resumes heartbeating to the worker. The first successful heartbeat reconciles task state: the worker reports which tasks are actually running.
- If the worker died during the restart window, heartbeat failures trigger the normal failure cascade.

### Workers that disappeared during restart

Workers that were healthy at checkpoint time but disappeared during the restart window:

- Both the worker and its tasks are restored from the checkpoint.
- The controller attempts to heartbeat the restored worker. The RPC fails.
- After `HEARTBEAT_FAILURE_THRESHOLD` (10) consecutive failures, the worker is pruned from state and its tasks are marked WORKER_FAILED and retried.
- The autoscaler discovers the slice is gone via `platform.list_slices()` and removes it.

No special handling is needed because the existing heartbeat failure cascade covers this case. The only difference from a normal worker failure is that the controller starts with the worker in a "healthy" state and discovers it is dead via heartbeat failures rather than never seeing a registration.

### Timer/deadline state that depends on wall clock

Three types of time-dependent state:

1. **Scheduling deadlines**: Converted from wall-clock to monotonic on restore (see above).
2. **Backoff/quota deadlines**: Same conversion. If the backoff should have expired during the restart, it is set to already-expired.
3. **Heartbeat timestamps** (`last_heartbeat` on ControllerWorker): Set to `Timestamp.now()` on restore, effectively resetting the heartbeat clock. The first heartbeat RPC to each worker will update it. This prevents the restored controller from immediately declaring all workers dead due to stale timestamps.
4. **Slice `last_active` timestamps**: Checkpointed as wall-clock epoch_ms. On restore, preserved as-is. If a slice was idle before the restart and remains idle after, it will correctly be eligible for scale-down. The idle timer effectively "pauses" during the restart window, which is conservative (it delays scale-down rather than causing premature scale-down).

### Endpoint survival

Endpoints registered by actor servers are checkpointed. On restore, the endpoints are available for resolution immediately. If the actor server has actually died during the restart, the next RPC to that endpoint will fail, and the client retries or the job fails naturally. The endpoint will be cleaned up when its parent job transitions to a terminal state.

### Job hierarchy (parent/child jobs)

Jobs are checkpointed individually with their `job_id` which encodes the hierarchy. On restore, the `root_submitted_at` field is preserved, so depth-first scheduling priority is maintained. Child jobs whose parent was cancelled at checkpoint time are already in a terminal state and are restored as-is.

## New Module: `cluster/controller/snapshot.py`

All snapshot logic lives in a single module with pure functions:

```python
# cluster/controller/snapshot.py

"""Controller checkpoint and restore.

Provides pure functions for creating and restoring controller snapshots.
The snapshot captures all non-transient state needed to resume controller
operation after a restart.
"""

from dataclasses import dataclass
from iris.rpc import snapshot_pb2

@dataclass(frozen=True)
class SnapshotResult:
    """Result of a snapshot operation."""
    proto: snapshot_pb2.ControllerSnapshot
    job_count: int
    task_count: int
    worker_count: int
    slice_count: int

def create_snapshot(
    state: ControllerState,
    autoscaler: Autoscaler | None,
    reservation_claims: dict[WorkerId, ReservationClaim],
) -> SnapshotResult:
    """Create a snapshot from current controller state.

    Pure function: reads state under locks, returns a proto.
    Captures jobs, tasks, workers, endpoints, autoscaler state,
    and reservation claims. Does not write to storage.
    """
    ...

def restore_snapshot(
    snapshot: snapshot_pb2.ControllerSnapshot,
    state: ControllerState,
    autoscaler: Autoscaler | None,
    platform: Platform,
) -> RestoreResult:
    """Restore controller state from a snapshot.

    Populates the (empty) state and autoscaler with checkpointed data:
    - Jobs and tasks with full state and attempt history
    - Workers with fresh health state (healthy=True, last_heartbeat=now())
    - Running task assignments rebuilt from task state
    - Slice handles re-discovered via platform.list_slices() and matched
      to checkpointed slice state

    Must be called before starting the scheduling/heartbeat loops.
    """
    ...

def write_snapshot(
    snapshot: snapshot_pb2.ControllerSnapshot,
    storage_prefix: str,
) -> str:
    """Write snapshot to remote storage. Returns the path."""
    ...

def read_latest_snapshot(
    storage_prefix: str,
) -> snapshot_pb2.ControllerSnapshot | None:
    """Read the latest snapshot from remote storage, or None if not found."""
    ...
```

## RPC Endpoints

Two new RPCs on the Controller service:

```protobuf
// In cluster.proto, added to the Controller service
rpc BeginCheckpoint(Controller.BeginCheckpointRequest)
    returns (Controller.BeginCheckpointResponse);
rpc LoadCheckpoint(Controller.LoadCheckpointRequest)
    returns (Controller.LoadCheckpointResponse);

message Controller {
  // ... existing messages ...

  message BeginCheckpointRequest {}
  message BeginCheckpointResponse {
    string snapshot_path = 1;
    iris.time.Timestamp created_at = 2;
    int32 job_count = 3;
    int32 task_count = 4;
    int32 worker_count = 5;
    int32 slice_count = 6;
  }

  message LoadCheckpointRequest {
    // If empty, loads the latest snapshot from the well-known path.
    string snapshot_path = 1;
  }
  message LoadCheckpointResponse {
    int32 jobs_restored = 1;
    int32 tasks_restored = 2;
    int32 workers_restored = 3;
    int32 slices_restored = 4;
    int32 slices_reconciled = 5;  // Slices found in cloud but not in checkpoint
  }
}
```

A CLI command wraps the RPC:

```bash
# Checkpoint and stop
iris --config=... cluster controller checkpoint

# Checkpoint without stopping (periodic backup)
iris --config=... cluster controller checkpoint --no-stop

# Restore from latest checkpoint on startup (automatic)
# The controller checks for a snapshot on startup and restores if found.

# Restore from a specific snapshot
iris --config=... cluster controller restore --snapshot-path gs://bucket/snapshots/snapshot-123.json
```

## Testing Strategy

### Test Infrastructure: Platform Mocking for Reconciliation

The `restore_snapshot()` function calls `platform.list_slices()` to reconcile the checkpointed slice inventory against live cloud state. Testing this reconciliation requires a platform fake that lets tests control exactly which slices are "visible in the cloud" independently of which slices were checkpointed.

The existing `FakePlatform` (in `tests/cluster/platform/fakes.py`) already supports `list_slices()` with label filtering and returns `FakeSliceHandle` instances. For snapshot reconciliation tests, we extend it with a method to inject pre-existing slices (simulating slices that exist in the cloud but were not created during this test run) and to remove slices (simulating termination during a restart window).

```python
# Extension to FakePlatform for snapshot reconciliation tests

class FakePlatform:
    # ... existing implementation ...

    def inject_slice(self, handle: FakeSliceHandle) -> None:
        """Add a slice to the platform's inventory without going through create_slice().

        Simulates a slice that exists in the cloud but was not created by the
        current controller instance (e.g., an in-flight scale-up that completed
        after the checkpoint was taken).
        """
        with self._lock:
            self._slices[handle.slice_id] = handle

    def remove_slice(self, slice_id: str) -> None:
        """Remove a slice from the platform's inventory without calling terminate().

        Simulates a slice that was terminated while the controller was down.
        After this call, list_slices() will not return this slice.
        """
        with self._lock:
            self._slices.pop(slice_id, None)
```

For GCP-specific reconciliation tests (testing the `list_slices()` behavior itself, not the snapshot restore logic), we use the existing `FakeGcloud` which intercepts `subprocess.run` calls and maintains in-memory TPU/VM state. Tests can manipulate `FakeGcloud._tpus` and `FakeGcloud._vms` directly to simulate cloud-side state changes.

For CoreWeave-specific tests, we mock `kubectl` responses at the subprocess level (similar to `FakeGcloud`) to control what Pods are visible.

### Unit Tests: `tests/cluster/test_snapshot.py`

Test the pure snapshot/restore functions in isolation:

```python
def test_snapshot_roundtrip_preserves_jobs():
    """Create state with jobs/tasks, snapshot, restore to fresh state, verify."""
    state = ControllerState()
    # Add jobs, tasks, advance some to RUNNING
    ...
    snapshot = create_snapshot(state, autoscaler=None, reservation_claims={})
    new_state = ControllerState()
    restore_snapshot(snapshot.proto, new_state, autoscaler=None, platform=fake_platform)
    # Assert jobs, tasks, states match
    ...

def test_snapshot_roundtrip_preserves_scaling_group_state():
    """Backoff timers, cooldowns, and slice inventory survive roundtrip."""
    ...

def test_snapshot_roundtrip_preserves_workers():
    """Worker identity and metadata survive roundtrip. Health fields are reset to fresh."""
    ...

def test_restore_rebuilds_running_tasks_from_task_state():
    """Worker.running_tasks and committed resources are rebuilt from restored tasks."""
    ...

def test_restore_converts_wall_clock_deadlines():
    """Scheduling deadlines are converted from wall-clock to monotonic."""
    ...

def test_snapshot_write_read_roundtrip():
    """Write to local FS, read back, verify proto equality."""
    ...

def test_schema_version_mismatch_raises():
    """Reject snapshots with incompatible schema versions."""
    ...
```

### Reconciliation Tests: `tests/cluster/test_snapshot_reconciliation.py`

These tests exercise the `platform.list_slices()` reconciliation that happens during `restore_snapshot()`. This is the most critical testing area: bugs here cause orphaned VMs (resource leaks) or lost slice inventory (capacity gaps).

All tests in this file use `FakePlatform` with its `inject_slice()` / `remove_slice()` methods to set up the cloud state independently of the checkpoint state.

#### Fixture: `reconciliation_env`

```python
@dataclass
class ReconciliationEnv:
    """Test environment for snapshot reconciliation tests.

    Provides a FakePlatform, a pre-built ScalingGroup, and helpers
    for creating checkpoint state and invoking restore.
    """
    platform: FakePlatform
    config: config_pb2.ScaleGroupConfig
    label_prefix: str

    def make_slice_snapshot(
        self,
        slice_id: str,
        lifecycle: str = "ready",
        vm_addresses: list[str] | None = None,
        created_at_ms: int | None = None,
    ) -> SliceSnapshot:
        """Build a SliceSnapshot proto for use in checkpoint state."""
        ...

    def make_fake_slice(
        self,
        slice_id: str,
        state: CloudSliceState = CloudSliceState.READY,
        vm_addresses: list[str] | None = None,
    ) -> FakeSliceHandle:
        """Build a FakeSliceHandle for injection into the platform."""
        ...

@pytest.fixture
def reconciliation_env() -> ReconciliationEnv:
    config = make_scale_group_config(
        name="tpu-group",
        min_slices=0,
        max_slices=10,
    )
    platform = FakePlatform(FakePlatformConfig(config=config))
    return ReconciliationEnv(
        platform=platform,
        config=config,
        label_prefix="test",
    )
```

#### Group 1: Slices present in both checkpoint and cloud

These test the normal case and variants where the slice's cloud state has changed during the restart window.

```python
def test_restore_slice_in_checkpoint_and_cloud_preserves_lifecycle(reconciliation_env):
    """A READY slice that exists in both checkpoint and cloud keeps its READY lifecycle.

    The most common case: the controller checkpointed a healthy slice, restarted,
    and the slice is still running. The restored ScalingGroup should have the same
    slice with the same lifecycle state and the SliceHandle from list_slices().
    """
    env = reconciliation_env
    # Checkpoint has slice-1 as READY
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="ready", vm_addresses=["10.0.0.1"])
    # Cloud has slice-1 alive
    cloud_handle = env.make_fake_slice("slice-1", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert len(result.slices) == 1
    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.READY
    # The handle should be the one from list_slices(), not a stale reference
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_booting_slice_that_became_ready_transitions_on_refresh(reconciliation_env):
    """A slice checkpointed as BOOTING that finished booting during the restart window.

    The checkpoint records lifecycle=BOOTING. The cloud shows READY (boot completed
    while controller was down). On restore, we set lifecycle=BOOTING from the checkpoint.
    The autoscaler's next refresh() cycle will call describe() on the handle, see READY,
    and transition the slice. This test verifies the restore sets up the state correctly
    so that the normal refresh path handles the transition.
    """
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="booting")
    cloud_handle = env.make_fake_slice("slice-1", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    # Restore preserves the checkpoint lifecycle; the autoscaler refresh will fix it
    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_initializing_slice_with_cloud_ready(reconciliation_env):
    """A slice checkpointed as INITIALIZING where workers finished initializing.

    Similar to the BOOTING case: checkpoint says INITIALIZING, cloud says READY.
    The autoscaler refresh will transition it. Restore should preserve the
    checkpoint lifecycle and associate the live handle.
    """
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="initializing")
    cloud_handle = env.make_fake_slice("slice-1", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.INITIALIZING
    assert result.slices["slice-1"].handle is cloud_handle
```

#### Group 2: Slices in checkpoint but missing from cloud

These test the case where slices were terminated or disappeared while the controller was down. Bugs here would cause the autoscaler to reference stale handles and fail on describe()/terminate().

```python
def test_restore_discards_slice_missing_from_cloud(reconciliation_env):
    """A slice in the checkpoint that no longer exists in the cloud is discarded.

    This happens when a preemptible slice is terminated by the cloud provider
    during the restart window, or when an operator manually deletes a TPU.
    The restored ScalingGroup must not contain a stale entry.
    """
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-gone", lifecycle="ready", vm_addresses=["10.0.0.99"])
    # Do NOT inject into platform -- slice is gone from cloud

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-gone" not in result.slices
    assert result.discarded_count == 1


def test_restore_discards_failed_slice_missing_from_cloud(reconciliation_env):
    """A FAILED slice that disappeared from cloud is discarded cleanly.

    A slice that failed and was then cleaned up by cloud GC should not
    produce an error during restore. It should be silently dropped.
    """
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-failed", lifecycle="failed")

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-failed" not in result.slices


def test_restore_multiple_slices_some_missing(reconciliation_env):
    """Mixed case: some checkpoint slices exist in cloud, some don't.

    Verifies that present slices are kept and missing slices are discarded,
    without the missing ones corrupting the present ones.
    """
    env = reconciliation_env
    snap_alive = env.make_slice_snapshot("slice-alive", lifecycle="ready")
    snap_gone = env.make_slice_snapshot("slice-gone", lifecycle="ready")

    cloud_alive = env.make_fake_slice("slice-alive", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud_alive)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(
            name="tpu-group",
            slices=[snap_alive, snap_gone],
        ),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-alive" in result.slices
    assert "slice-gone" not in result.slices
    assert result.slices["slice-alive"].handle is cloud_alive
```

#### Group 3: Slices in cloud but NOT in checkpoint

These test the case where in-flight scale-ups completed after the checkpoint was taken. Bugs here cause orphaned VMs that the autoscaler cannot track or scale down.

```python
def test_restore_adopts_unknown_cloud_slice_as_booting(reconciliation_env):
    """A slice visible in list_slices() but absent from the checkpoint is adopted.

    This happens when a scale-up was in flight at checkpoint time: create_slice()
    returned (GCP: cloud resource exists) but complete_scale_up() had not yet
    been called, so the slice was not in the ScalingGroup's inventory.

    The restored ScalingGroup must adopt it as BOOTING so the autoscaler can
    track it and eventually transition it to READY or scale it down.
    """
    env = reconciliation_env
    orphan = env.make_fake_slice("slice-orphan", state=CloudSliceState.READY)
    env.platform.inject_slice(orphan)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-orphan" in result.slices
    assert result.slices["slice-orphan"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-orphan"].handle is orphan
    assert result.adopted_count == 1


def test_restore_adopts_creating_cloud_slice(reconciliation_env):
    """A cloud slice in CREATING state (not yet READY) is adopted as BOOTING.

    On GCP, a TPU in CREATING state has been requested but has not finished
    provisioning. It should still be adopted so we don't leak it.
    """
    env = reconciliation_env
    creating = env.make_fake_slice("slice-creating", state=CloudSliceState.CREATING)
    env.platform.inject_slice(creating)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-creating" in result.slices
    assert result.slices["slice-creating"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_mixed_known_and_unknown_slices(reconciliation_env):
    """Checkpoint has slice-A; cloud has slice-A and slice-B.

    slice-A is reconciled normally. slice-B (unknown to checkpoint) is adopted
    as BOOTING. This is the common case for a checkpoint taken mid-scale-up.
    """
    env = reconciliation_env
    snap_a = env.make_slice_snapshot("slice-a", lifecycle="ready")

    cloud_a = env.make_fake_slice("slice-a", state=CloudSliceState.READY)
    cloud_b = env.make_fake_slice("slice-b", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud_a)
    env.platform.inject_slice(cloud_b)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[snap_a]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.slices["slice-a"].lifecycle == SliceLifecycleState.READY
    assert result.slices["slice-b"].lifecycle == SliceLifecycleState.BOOTING
    assert result.adopted_count == 1
```

#### Group 4: Multiple scaling groups with mixed states

```python
def test_restore_multiple_groups_independent_reconciliation():
    """Each scaling group reconciles independently against the same platform.

    Group-A has 2 slices (1 alive, 1 gone). Group-B has 1 slice plus
    1 orphan from an in-flight scale-up. Verifies that cross-group
    slices don't interfere with each other.
    """
    config_a = make_scale_group_config(name="group-a", ...)
    config_b = make_scale_group_config(name="group-b", ...)
    platform = FakePlatform(...)

    # Group A: slice-a1 alive, slice-a2 gone
    platform.inject_slice(make_fake_slice("slice-a1", scale_group="group-a"))
    # slice-a2 not injected (terminated during restart)

    # Group B: slice-b1 alive, slice-b-orphan appeared during restart
    platform.inject_slice(make_fake_slice("slice-b1", scale_group="group-b"))
    platform.inject_slice(make_fake_slice("slice-b-orphan", scale_group="group-b"))

    result_a = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(
            name="group-a",
            slices=[make_slice_snapshot("slice-a1"), make_slice_snapshot("slice-a2")],
        ),
        platform=platform, config=config_a, label_prefix="test",
    )

    result_b = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(
            name="group-b",
            slices=[make_slice_snapshot("slice-b1")],
        ),
        platform=platform, config=config_b, label_prefix="test",
    )

    assert set(result_a.slices.keys()) == {"slice-a1"}
    assert set(result_b.slices.keys()) == {"slice-b1", "slice-b-orphan"}
    assert result_b.slices["slice-b-orphan"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_empty_checkpoint_with_cloud_slices(reconciliation_env):
    """Fresh controller start (no checkpoint) with existing cloud slices.

    This covers the case where a controller crashes before its first checkpoint.
    A new controller starts, finds no snapshot, but list_slices() discovers
    slices from the previous controller instance. All should be adopted.
    """
    env = reconciliation_env
    env.platform.inject_slice(env.make_fake_slice("slice-1"))
    env.platform.inject_slice(env.make_fake_slice("slice-2"))

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert len(result.slices) == 2
    assert all(s.lifecycle == SliceLifecycleState.BOOTING for s in result.slices.values())


def test_restore_empty_checkpoint_empty_cloud(reconciliation_env):
    """Fresh start with no checkpoint and no cloud slices. Clean slate."""
    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[]),
        platform=reconciliation_env.platform,
        config=reconciliation_env.config,
        label_prefix=reconciliation_env.label_prefix,
    )

    assert len(result.slices) == 0
```

#### Group 5: Lifecycle state mismatches between checkpoint and cloud

```python
def test_restore_ready_slice_now_failed_in_cloud(reconciliation_env):
    """Checkpoint says READY, but the cloud reports the slice as FAILED.

    The slice hardware failed during the restart window. On restore, we
    use the checkpoint lifecycle (READY). The autoscaler refresh will call
    describe(), see FAILED, and handle the transition. This test verifies
    we don't crash or lose the slice handle during restore.
    """
    env = reconciliation_env
    snap = env.make_slice_snapshot("slice-1", lifecycle="ready")
    cloud = env.make_fake_slice("slice-1", state=CloudSliceState.FAILED)
    env.platform.inject_slice(cloud)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    # Checkpoint lifecycle is preserved; autoscaler refresh will detect FAILED
    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.READY
    assert result.slices["slice-1"].handle is cloud


def test_restore_preserves_vm_addresses_from_checkpoint(reconciliation_env):
    """vm_addresses from the checkpoint are restored for immediate heartbeating.

    The controller needs worker IP addresses to resume heartbeating immediately
    on startup. These come from the checkpoint, not from describe() (which may
    involve cloud API latency).
    """
    env = reconciliation_env
    snap = env.make_slice_snapshot(
        "slice-1", lifecycle="ready", vm_addresses=["10.0.0.1", "10.0.0.2"]
    )
    cloud = env.make_fake_slice("slice-1", state=CloudSliceState.READY)
    env.platform.inject_slice(cloud)

    result = restore_scaling_group(
        group_snapshot=ScalingGroupSnapshot(name="tpu-group", slices=[snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.slices["slice-1"].vm_addresses == ["10.0.0.1", "10.0.0.2"]
```

#### Group 6: GCP platform `list_slices()` behavior during restore

These tests exercise the GCP-specific `list_slices()` implementation via `FakeGcloud` to verify that the cloud discovery step itself works correctly for restore scenarios. These are _not_ tests of the snapshot restore logic, but of the platform layer that restore depends on.

```python
def test_gcp_list_slices_discovers_tpu_created_during_restart(fake_gcloud):
    """A TPU created by a concurrent scale-up thread is visible to list_slices().

    GCP create_slice() for TPUs is synchronous: the TPU exists in the cloud
    before the call returns. If a checkpoint happened between create_slice()
    returning and complete_scale_up() running, the TPU is visible to
    list_slices() but not in the checkpoint. This test verifies that
    list_slices() actually finds it.
    """
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test", zones=["zone-a"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    labels = Labels("iris")

    # Simulate a TPU that was created by the previous controller
    fake_gcloud._tpus[("iris-tpu-group-123", "zone-a")] = {
        "name": "iris-tpu-group-123",
        "state": "READY",
        "acceleratorType": "v5litepod-16",
        "labels": {
            labels.iris_managed: "true",
            labels.iris_scale_group: "tpu-group",
        },
        "networkEndpoints": [{"ipAddress": "10.0.0.50"}],
        "createTime": "2024-01-15T10:30:00.000Z",
    }

    slices = platform.list_slices(zones=["zone-a"], labels={labels.iris_managed: "true"})
    assert any(s.slice_id == "iris-tpu-group-123" for s in slices)


def test_gcp_list_slices_discovers_vm_slice_created_during_restart(fake_gcloud):
    """A GCE VM slice created during the restart window is visible to list_slices().

    Similar to the TPU case, but for GCE VM-backed slices. The VM is tagged
    with the iris-slice-id label at creation time, making it discoverable.
    """
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test", zones=["zone-a"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    labels = Labels("iris")

    fake_gcloud._vms[("iris-cpu-vm-123", "zone-a")] = {
        "name": "iris-cpu-vm-123",
        "status": "RUNNING",
        "networkInterfaces": [{"networkIP": "10.0.0.60"}],
        "labels": {
            labels.iris_managed: "true",
            labels.iris_scale_group: "cpu-group",
            labels.iris_slice_id: "cpu-slice-123",
        },
    }

    slices = platform.list_slices(zones=["zone-a"], labels={labels.iris_managed: "true"})
    assert any(s.slice_id == "cpu-slice-123" for s in slices)


def test_gcp_list_slices_excludes_terminated_tpu_during_restart(fake_gcloud):
    """A TPU that went to DELETING state during the restart window is excluded.

    The cloud may have preempted or deleted a TPU while the controller was
    down. GcpPlatform.list_slices() filters TPUs not in (READY, CREATING)
    state, so the reconciliation step will correctly see it as "missing."
    """
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test", zones=["zone-a"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    labels = Labels("iris")

    fake_gcloud._tpus[("iris-tpu-dead", "zone-a")] = {
        "name": "iris-tpu-dead",
        "state": "DELETING",
        "acceleratorType": "v5litepod-16",
        "labels": {labels.iris_managed: "true"},
        "networkEndpoints": [],
        "createTime": "2024-01-15T10:30:00.000Z",
    }

    slices = platform.list_slices(zones=["zone-a"], labels={labels.iris_managed: "true"})
    assert not any(s.slice_id == "iris-tpu-dead" for s in slices)


def test_gcp_list_slices_finds_creating_tpu_during_restart(fake_gcloud):
    """A TPU still in CREATING state (provisioning) is visible to list_slices().

    If the controller checkpointed while a create_slice() was in progress
    and the TPU has not yet finished provisioning, it will be in CREATING
    state. list_slices() includes CREATING TPUs so the restore can adopt them.
    """
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test", zones=["zone-a"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    labels = Labels("iris")

    fake_gcloud._tpus[("iris-tpu-creating", "zone-a")] = {
        "name": "iris-tpu-creating",
        "state": "CREATING",
        "acceleratorType": "v5litepod-16",
        "labels": {labels.iris_managed: "true"},
        "networkEndpoints": [],
        "createTime": "2024-01-15T10:30:00.000Z",
    }

    slices = platform.list_slices(zones=["zone-a"], labels={labels.iris_managed: "true"})
    assert any(s.slice_id == "iris-tpu-creating" for s in slices)
```

#### Group 7: CoreWeave platform `list_slices()` behavior during restore

CoreWeave `list_slices()` queries Kubernetes Pods via kubectl. These tests verify the behavior under restore-relevant scenarios.

```python
def test_coreweave_list_slices_pod_not_yet_created():
    """A scale-up was initiated (kubectl apply submitted) but the Pod doesn't exist yet.

    CoreWeave create_slice() returns immediately with an in-memory handle, then
    submits Pod creation to a background thread. If the controller checkpoints
    during this gap, the pending counter is lost and list_slices() returns
    nothing for this slice. This is safe: the next autoscaler cycle re-issues.

    This test verifies that list_slices() returns empty when no Pods exist,
    confirming the "lost pending scale-up" behavior described in the design.
    """
    # Mock kubectl to return empty pod list
    ...
    slices = platform.list_slices(zones=[], labels={labels.iris_managed: "true"})
    assert len(slices) == 0


def test_coreweave_list_slices_pod_created_during_restart():
    """A Pod was created (kubectl apply completed) during the restart window.

    The background thread completed kubectl apply, so the Pod now exists in K8s.
    list_slices() should discover it. The reconciliation step will adopt it.
    """
    # Mock kubectl to return a Pod in Pending state
    ...
    slices = platform.list_slices(zones=[], labels={labels.iris_managed: "true"})
    assert len(slices) == 1
    assert slices[0].describe().state == CloudSliceState.BOOTSTRAPPING


def test_coreweave_list_slices_pod_ready_during_restart():
    """A Pod that was Pending at checkpoint time became Ready during restart.

    Similar to the GCP BOOTING->READY case. The Pod transitions from Pending
    to Running with Ready=True while the controller is down.
    """
    # Mock kubectl to return a Pod with Ready=True
    ...
    slices = platform.list_slices(zones=[], labels={labels.iris_managed: "true"})
    assert len(slices) == 1
    assert slices[0].describe().state == CloudSliceState.READY


def test_coreweave_list_slices_pod_deleted_during_restart():
    """A Pod that existed at checkpoint time was deleted during the restart window.

    The Pod was evicted or manually deleted. list_slices() should not return it.
    The reconciliation step will see it as "missing from cloud" and discard
    the checkpoint entry.
    """
    # Mock kubectl to return empty pod list
    ...
    slices = platform.list_slices(zones=[], labels={labels.iris_managed: "true"})
    assert len(slices) == 0
```

#### Group 8: Scaling group timing state preservation

```python
def test_restore_preserves_backoff_state(reconciliation_env):
    """Backoff timers survive checkpoint/restore.

    If a scaling group was in backoff (recent creation failure), the restored
    group should remain in backoff so it doesn't immediately retry and
    hit the same quota/capacity issue.
    """
    env = reconciliation_env
    # Snapshot with backoff active (5 minutes remaining)
    snapshot = ScalingGroupSnapshot(
        name="tpu-group",
        consecutive_failures=3,
        backoff_until=Timestamp.from_ms(Timestamp.now().epoch_ms() + 300_000).to_proto(),
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.consecutive_failures == 3
    assert result.backoff_active


def test_restore_expired_backoff_is_inactive(reconciliation_env):
    """Backoff that expired during the restart window is correctly inactive."""
    env = reconciliation_env
    # Snapshot with backoff that expired 1 minute ago
    snapshot = ScalingGroupSnapshot(
        name="tpu-group",
        consecutive_failures=2,
        backoff_until=Timestamp.from_ms(Timestamp.now().epoch_ms() - 60_000).to_proto(),
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.consecutive_failures == 2
    assert not result.backoff_active


def test_restore_preserves_quota_exceeded_state(reconciliation_env):
    """Quota exceeded state and reason survive restore."""
    env = reconciliation_env
    snapshot = ScalingGroupSnapshot(
        name="tpu-group",
        quota_exceeded_until=Timestamp.from_ms(Timestamp.now().epoch_ms() + 300_000).to_proto(),
        quota_reason="RESOURCE_EXHAUSTED: out of v5 TPUs in us-central2",
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.quota_exceeded_active
    assert "v5 TPUs" in result.quota_reason
```

### E2E Tests: `tests/e2e/test_snapshot.py`

Test the full checkpoint-restore cycle with a running local cluster:

```python
@pytest.mark.e2e
def test_checkpoint_restore_running_job(cluster):
    """Submit a long-running job, checkpoint, restart controller, verify job resumes."""
    client = cluster.client
    job = client.submit(name="long-job", entrypoint=..., resources=...)

    # Wait for job to start running
    wait_for_condition(lambda: client.get_job_status("long-job").state == RUNNING)

    # Checkpoint
    resp = cluster.rpc.begin_checkpoint(BeginCheckpointRequest())
    assert resp.snapshot_path

    # Restart controller (LocalController.restart() with snapshot loading)
    cluster.restart_with_snapshot(resp.snapshot_path)

    # Verify job is still tracked and running (workers are restored, no re-registration needed)
    status = client.get_job_status("long-job")
    assert status.state == RUNNING

@pytest.mark.e2e
def test_checkpoint_restore_preserves_autoscaler_slices(cluster):
    """Verify slice inventory survives checkpoint/restore."""
    ...

@pytest.mark.e2e
def test_checkpoint_restore_with_worker_death(cluster):
    """Worker dies during restart window; tasks are retried after restore."""
    ...
```

## Implementation Plan

Following the spiral approach from AGENTS.md, each step is independently testable.

### Step 1: Snapshot proto and pure snapshot/restore functions

- Add `snapshot.proto` with the schema defined above (including `WorkerSnapshot`).
- Implement `create_snapshot()` for ControllerState (jobs, tasks, workers, endpoints).
- Implement `restore_snapshot()` for ControllerState: restore jobs, tasks, endpoints; restore workers with fresh health state; rebuild `running_tasks` and committed resources from task state.
- Add `write_snapshot()` / `read_latest_snapshot()` using fsspec.
- Unit tests for roundtrip, schema versioning, deadline conversion, worker health reset.
- **Test**: `test_snapshot_roundtrip_preserves_jobs`, `test_snapshot_roundtrip_preserves_workers`, `test_restore_rebuilds_running_tasks_from_task_state`, `test_snapshot_write_read_roundtrip`.

### Step 2: Autoscaler and ScalingGroup snapshot support

- Extend `create_snapshot()` to capture ScalingGroup state (slices, backoff, cooldowns).
- Extend `create_snapshot()` to capture TrackedWorker entries from the autoscaler.
- Extend `restore_snapshot()` to restore ScalingGroup state, including `platform.list_slices()` reconciliation.
- Unit tests for scaling group roundtrip, cloud reconciliation.
- **Test**: `test_snapshot_roundtrip_preserves_scaling_group_state`, `test_restore_discards_slices_missing_from_cloud`.

### Step 3: Controller integration (checkpoint flag, drain, RPC)

- Add `_checkpoint_in_progress` flag to Controller.
- Implement `begin_checkpoint()` method on Controller with pause/drain/snapshot/write.
- Add `BeginCheckpoint` RPC endpoint to ControllerServiceImpl.
- Add snapshot loading on Controller startup (check for latest.json).
- Add reservation claim checkpointing.
- **Test**: E2E test with LocalController: submit job, checkpoint, restore, verify job.

### Step 4: CLI commands and periodic checkpointing

- Add `iris cluster controller checkpoint` CLI command.
- Add periodic checkpoint to the autoscaler loop (configurable interval).
- Add `--snapshot-path` flag to controller startup for explicit restore.
- **Test**: E2E test for periodic checkpoint, CLI integration test.

### Step 5: Worker failure and edge case hardening

- Test that workers which die during the restart window are correctly detected via heartbeat failures and their tasks are retried.
- Test the degraded path where restart exceeds the worker heartbeat timeout (workers reset + re-register).
- Test edge cases: worker reports different attempt_id, worker reports tasks from before checkpoint, etc.
- **Test**: E2E test with worker death during restart window, E2E test with slow restart exceeding heartbeat timeout.
