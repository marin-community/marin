# Sub-doc: The Reconcile Protocol

Companion to `spec.md` §4.2-4.3. Full proto, semantics, error codes, idempotency, compat translation. Draft 3.

## Changes from Draft 2

- **Single direction: controller → worker.** Dropped `ControllerService.Reconcile`. Dropped the symmetric `ReconcileMessage` shape; replaced with conventional `ReconcileRequest` / `ReconcileResponse`. Workers never initiate state pushes; the controller learns of state changes only on its next tick.
- **No Nudge / worker-initiated push.** Removed. Latency for terminal-state propagation is now bounded by the tick interval (1 s, see `spec.md` §4.4-§5.5).
- **No rate-limiting section.** Without worker-initiated RPCs there is nothing to rate-limit; the controller alone drives the cadence.

## Changes from Draft 1 (still in effect)

- **No etag-based spec caching.** Replaced with "observed implies cached" rule (§3).
- **No `controller_epoch`.** Not needed without etags.
- **`attempt_uid` is the sole routing key.** Compat fields `task_id`+`attempt_id` retained on the wire through Phase D, deprecated thereafter.
- **No `desired_generation`.** Spec mutation handled by re-sending the spec inline.

## Full proto

In `lib/iris/src/iris/rpc/worker.proto`:

```proto
edition = "2023";
package iris.cluster;

import "job.proto";
import "time.proto";

message Worker {
  // ============================================================
  // RECONCILE RPC (controller -> worker, unary)
  // ============================================================

  message ReconcileRequest {
    string worker_id = 1;
    repeated DesiredAttempt desired = 2;        // complete expected set
  }

  message ReconcileResponse {
    string worker_id = 1;
    repeated AttemptObservation observed = 2;   // complete observed set
    WorkerHealth health = 3;
  }

  message DesiredAttempt {
    // Routing key. 16 hex chars (128-bit random). Set in Phase C+;
    // empty in Phase B (recipient routes by task_id+attempt_id then).
    string attempt_uid = 1;

    oneof intent {
      AttemptSpec run = 10;
      StopReason stop = 11;
    }

    // Phase-B compat: legacy composite key.
    string task_id = 100;
    int32 attempt_id = 101;
  }

  message AttemptSpec {
    // Populated only when the DB attempt state is ASSIGNED (the one
    // dispatch tick that hands the worker the spec). Omitted on every
    // subsequent tick — the worker is expected to have it cached. See
    // spec.md §4.3.
    iris.job.RunTaskRequest request = 1;
  }

  message AttemptObservation {
    string attempt_uid = 1;

    iris.job.TaskState state = 2;
    int32 exit_code = 3;
    string error = 4;
    string container_id = 5;
    iris.time.Timestamp finished_at = 6;
    iris.job.ResourceUsage resource_usage = 7;

    // Phase-B compat: legacy composite key.
    string task_id = 100;
    int32 attempt_id = 101;
  }

  enum StopReason {
    STOP_REASON_UNSPECIFIED = 0;
    STOP_REASON_CANCELLED = 1;
    STOP_REASON_PREEMPTED = 2;
    STOP_REASON_SUPERSEDED = 3;
    STOP_REASON_JOB_TERMINATED = 4;
    STOP_REASON_TASK_TIMEOUT = 5;
    STOP_REASON_WORKER_DRAIN = 6;
  }

  message WorkerHealth {
    bool healthy = 1;
    string health_error = 2;
    iris.job.WorkerResourceSnapshot resources = 3;
  }
}

service WorkerService {
  rpc Reconcile(Worker.ReconcileRequest) returns (Worker.ReconcileResponse);
  // legacy RPCs (StartTasks / StopTasks / PollTasks / Ping) kept through Phase D.
}
```

`controller.proto` augments `Register` with capability bits (unchanged from v2):

```proto
message Controller {
  message RegisterResponse {
    bool accepted = 1;
    string worker_id = 2;
    WorkerCapabilities capabilities = 10;
  }

  message WorkerCapabilities {
    bool reconcile_rpc = 1;
  }
}
```

## Semantics

### `ReconcileRequest`

| Field | Meaning |
|---|---|
| `worker_id` | Destination worker (controller stamps; worker verifies match) |
| `desired` | Complete expected set. Anything the worker has that's not here and not terminal gets stopped. |

### `ReconcileResponse`

| Field | Meaning |
|---|---|
| `worker_id` | Echo from request (defense check) |
| `observed` | Complete observed set: every attempt the worker has any record of, including freshly-terminal ones. The signal "this attempt is finished" rides here. |
| `health` | Worker health summary. Replaces today's `Ping` response. |

### `DesiredAttempt.intent`

`run` ⇒ "start or keep running."
- Worker has uid running → no-op (report current state).
- Worker has uid terminal → no-op (terminal sticks; report it).
- Worker doesn't have uid AND `AttemptSpec.request` is set → build + start.
- Worker doesn't have uid AND `AttemptSpec.request` is unset → report `AttemptObservation(state=MISSING)`. Controller treats this as `worker_lost_spec` and fails the attempt.

`stop` ⇒ "stop." Idempotent.

### Spec inclusion rule

Recap of `spec.md` §4.3:

> Controller sends `AttemptSpec(request=...)` inline exactly when the DB attempt state is `ASSIGNED`. Every other state (`BUILDING`, `RUNNING`, etc.) sends `AttemptSpec()` (empty submessage); the worker is expected to have it cached.
>
> If the worker has lost the cache (cold restart, never received), it reports `AttemptObservation(state=MISSING)`. The controller's apply layer fails the attempt forward as `worker_lost_spec` and the scheduler places a fresh attempt under a new uid on the next tick.

No `observed_uids_by_worker` bookkeeping on the controller. The dispatch rule is a pure function of the DB row's state.

### No sequence number

Considered and rejected. Defenses a `seq` field could provide:

- **Out-of-order delivery detection.** Unary Connect RPC over HTTP/2; each call is independent at the wire level. Out-of-order doesn't exist.
- **Overlapping ticks per worker.** `_run_polling_loop` runs ticks serially; tick N+1 cannot start until tick N's `asyncio.gather` over all per-worker RPCs completes. No overlap possible.
- **Replay protection.** Not a concern at this layer.
- **Controller restart detection.** Handled by `Register`.
- **Request/response correlation.** Unary RPC already correlates.

Adding a counter is dead weight.

## Error codes

| Code | When | Controller action |
|---|---|---|
| `UNAUTHENTICATED` | Bad auth token | Log; alert; don't retry |
| `PERMISSION_DENIED` | Wrong scope | Same |
| `UNAVAILABLE`, `DEADLINE_EXCEEDED` | Transport / worker overload | No retry inline; next tick reconverges |
| `INTERNAL` | Bug in worker | Log; next tick reconverges |
| (worker returns malformed observed) | Worker bug | Log at ERROR; apply what we can; next tick |

Critical: the controller **never inline-retries**. The next tick (1 s away) is the retry. This is the level-triggered guarantee — every tick is independent, every failure is recovered on the next.

## Idempotency

The RPC is idempotent at the protocol level:
- `desired` is a set. Sending twice has the same observable effect.
- `observed` likewise.

The non-idempotent bits (DB state transitions) live in `apply_reconcile_response` and must check current DB state before writing. Specifically: don't overwrite a terminal `finished_at` on duplicate terminal observation; don't transition a CANCELLED row back to RUNNING because a stale observation carried RUNNING.

## Compat translation (Phase B legacy shim)

Workers without `reconcile_rpc` capability get the old wire. The `legacy_translator` helper in `controller/reconcile.py` translates a `WorkerReconcilePlan` (output of `reconcile_worker`) into legacy RPCs:

| Pure-compute output | Phase B+ wire | Legacy wire |
|---|---|---|
| `DesiredAttempt(uid=X, intent=run, AttemptSpec(request=...))` | `Reconcile.desired=[X with spec]` | `StartTasks(tasks=[RunTaskRequest])` then `PollTasks(expected=[X])` |
| `DesiredAttempt(uid=X, intent=run, AttemptSpec())` (cached) | `Reconcile.desired=[X without spec]` | `PollTasks(expected=[X])` |
| `DesiredAttempt(uid=X, intent=stop)` | `Reconcile.desired` omits X + stop directive | `StopTasks(task_ids=[X])` |
| `AttemptObservation` in response | `ReconcileResponse.observed` | Extracted from `PollTasksResponse.tasks` |
| `WorkerHealth` in response | `ReconcileResponse.health` | Separate `Ping` RPC return |

The shim is one-pass: pure function emits a plan; translator picks the wire.

## Capability negotiation

`Register` response carries `WorkerCapabilities`. Controller sets `reconcile_rpc=true` for workers it knows speak the new RPC (based on `Register.metadata.client_revision_date` ≥ Phase B release date).

Worker stores capabilities; if `reconcile_rpc=true`, it installs the `Reconcile` handler (legacy handlers also remain, so a controller in mixed-deployment can still call old RPCs).

There is no negotiated parameter for "throttle" — without worker-initiated RPCs, the worker doesn't generate traffic to throttle.

## Wire examples

### Steady state

```
POST /iris.cluster.WorkerService/Reconcile

ReconcileRequest {
  worker_id: "w-tpu-001"
  desired: [
    DesiredAttempt { attempt_uid: "a1b2c3d4e5f6a7b8", intent: run { AttemptSpec {} } }
    DesiredAttempt { attempt_uid: "f9e8d7c6b5a4f3e2", intent: run { AttemptSpec {} } }
  ]
}

⇒ ReconcileResponse {
  worker_id: "w-tpu-001"
  observed: [
    AttemptObservation { attempt_uid: "a1b2c3d4e5f6a7b8", state: RUNNING }
    AttemptObservation { attempt_uid: "f9e8d7c6b5a4f3e2", state: RUNNING }
  ]
  health: { healthy: true, resources: { ... } }
}
```

Both specs omitted — worker has them cached. ~250 bytes / RTT.

### Fresh assignment

```
ReconcileRequest {
  worker_id: "w-tpu-001"
  desired: [
    DesiredAttempt {
      attempt_uid: "newone1234567890"
      intent: run { AttemptSpec { request: RunTaskRequest { ... } } }   // spec inline
    }
  ]
}

⇒ ReconcileResponse {
  observed: [
    AttemptObservation { attempt_uid: "newone1234567890", state: BUILDING }
  ]
  health: { healthy: true, ... }
}
```

Worker now has `newone1234567890` in its echoed observed. Next tick: spec omitted.

### Terminal-state propagation (controller→worker, no worker push)

This is the case the v2 worker-initiated Reconcile path handled. v3 simplification: the worker just waits.

Tick N (worker has running attempts):
```
ReconcileRequest {
  desired: [
    DesiredAttempt { attempt_uid: "a1b2c3", intent: run { AttemptSpec {} } }
    DesiredAttempt { attempt_uid: "d4e5f6", intent: run { AttemptSpec {} } }
  ]
}
⇒ ReconcileResponse {
  observed: [
    AttemptObservation { attempt_uid: "a1b2c3", state: RUNNING }
    AttemptObservation { attempt_uid: "d4e5f6", state: RUNNING }
  ]
}
```

Container `a1b2c3` exits at T+2 s. Worker records the terminal state locally. No RPC fires.

Tick N+1 (1 s after N):
```
ReconcileRequest {
  desired: [
    DesiredAttempt { attempt_uid: "a1b2c3", intent: run { AttemptSpec {} } }   // controller doesn't know yet
    DesiredAttempt { attempt_uid: "d4e5f6", intent: run { AttemptSpec {} } }
  ]
}
⇒ ReconcileResponse {
  observed: [
    AttemptObservation { attempt_uid: "a1b2c3", state: SUCCEEDED, exit_code: 0, finished_at: {...} }
    AttemptObservation { attempt_uid: "d4e5f6", state: RUNNING }
  ]
}
```

Controller's apply layer transitions `a1b2c3` to SUCCEEDED in DB. Cross-worker cascades (coscheduling, job-state recompute) fire in the same transaction.

Tick N+2 (1 s after N+1):
```
ReconcileRequest {
  desired: [
    DesiredAttempt { attempt_uid: "d4e5f6", intent: run { AttemptSpec {} } }   // a1b2c3 omitted (terminal acked)
    // possibly DesiredAttempt for a newly-placed attempt, etc.
  ]
}
```

Terminal-state propagation latency: 1 s (the tick interval, in the worst case). Total convergence to "all parties agree" across a coscheduled-sibling cascade: 2 s (one tick to learn, one tick to propagate cascades).

### Cold worker after restart (spec cache lost mid-attempt)

DB still has the attempt in `RUNNING`. Controller dispatches without spec (DB isn't ASSIGNED). Worker has no record:

```
ReconcileRequest {
  worker_id: "w-tpu-001"
  desired: [
    DesiredAttempt { attempt_uid: "a1b2c3d4e5f6a7b8", intent: run { AttemptSpec {} } }   // no spec
  ]
}

⇒ ReconcileResponse {
  observed: [
    AttemptObservation { attempt_uid: "a1b2c3d4e5f6a7b8", state: MISSING }
  ]
  health: { healthy: true, ... }
}
```

Controller's apply layer transitions `a1b2c3d4e5f6a7b8` to `FAILED("worker_lost_spec")` and fires cross-worker cascades (coscheduled-sibling failure, job-state recompute). The scheduler places a fresh attempt with a new uid on the next tick; that attempt is in `ASSIGNED`, so the next dispatch carries spec inline.

`MISSING` is the only new task state in the observation enum. It is non-terminal in the protocol sense but the controller always converts it to a terminal DB state immediately — workers never report MISSING twice for the same uid because the controller stops sending it.

## What this does NOT touch

- **Log streaming** (finelog): orthogonal side channel. Log keys migrate to `attempt_uid` in Phase C.
- **Profile capture**: `ProfileTask` RPC unchanged; gains optional `attempt_uid` parameter in Phase C.
- **Exec in container**: `ExecInContainer` unchanged; gains optional `attempt_uid` in Phase C.
- **Worker registration**: `Register` unchanged; extends response with capabilities.
- **Bundle distribution**: `BundleStore` unchanged.

Reconcile is the state-convergence protocol. Other RPCs serve orthogonal concerns and don't change.

## Spec cache locality

The worker-side spec cache (`dict[AttemptUid, RunTaskRequest]`) lives in `iris/cluster/worker/reconcile.py:SpecCache`. It is in-memory only; if it is lost, the worker reports `MISSING` and the attempt fails forward. Persistence is a fail-open follow-up (`spec.md` §5.9), not a protocol concern.
