# Design: Long-Running Operations for Actor RPC

## Problem

Actor RPC calls block an HTTP connection for the entire duration of the method.
When `ZephyrCoordinator.run_pipeline` (`lib/zephyr/src/zephyr/coordinator.py`)
takes hours, the httpx read timeout fires and `call_with_retry`
(`lib/iris/src/iris/rpc/errors.py`) retries the call — creating duplicate
pipeline executions on the same coordinator (#3459).

The timeout default was changed to `None` as a stopgap, but the fundamental
issue remains: a long-lived blocking RPC is fragile. The caller can't
distinguish "still running" from "server died", can't cancel, and
`.result(timeout=30)` only abandons the Python future while the RPC thread
and server keep running.

**Current flow** (`_IrisActorMethod.remote()` in `lib/fray/src/fray/iris_backend.py`):
```
.remote()  →  ThreadPoolExecutor.submit(blocking_rpc)  →  Future
.result()  →  Future.result()  (waits on thread)
```

The thread holds an open HTTP connection for the entire call. The timeout on
`.result()` is disconnected from the RPC timeout, and there's no way to
observe progress or cancel server-side work.

## Goals

- Decouple call submission from result retrieval — no long-lived HTTP connections
- Make `.result(timeout=X)` actually stop server-side work when it expires
- Keep the `ActorFuture` protocol (`lib/fray/src/fray/actor.py`) unchanged, so any
  new future type drops into existing call sites
- Support the existing synchronous `__call__` path for short RPCs

**Non-goals**: Persistence across server restarts, distributed tracing, streaming progress

## Proposed Solution

Add three RPCs to `ActorService`: `StartOperation` (submit work, get ID back),
`GetOperation` (poll for result), `CancelOperation` (stop work). The server
runs the method in its existing thread pool and stores the result keyed by
operation ID.

### Proto additions (`actor.proto`)

```protobuf
message Operation {
  string operation_id = 1;
  enum State { PENDING = 0; RUNNING = 1; SUCCEEDED = 2; FAILED = 3; CANCELLED = 4; }
  State state = 2;
  bytes serialized_result = 3;   // set on SUCCEEDED
  ActorError error = 4;          // set on FAILED
}

message OperationId { string operation_id = 1; }

service ActorService {
  // ... existing RPCs ...
  rpc StartOperation(ActorCall) returns (Operation);
  rpc GetOperation(OperationId) returns (Operation);
  rpc CancelOperation(OperationId) returns (Operation);
}
```

`StartOperation` reuses the existing `ActorCall` message — same serialized
args/kwargs. It returns immediately with an `Operation` in `RUNNING` state.

### Server (`lib/iris/src/iris/actor/server.py`)

The server gets an `_operations` dict. `StartOperation` submits work to the
existing executor and returns the ID. `GetOperation` returns current state.
`CancelOperation` sets a flag; the actor can check it via a context object.

The worker records its own outcome on the `OperationState`, so no `Future` has
to be retained and `state` is derived rather than stored:

```python
@dataclass
class OperationState:
    operation_id: str
    cancelled: threading.Event
    serialized_result: bytes | None   # set on success
    error: ActorError | None          # set on failure
    completed_at: float | None        # stamped by _run_operation's finally block

    @property
    def state(self) -> int:
        ...  # RUNNING until completed_at, then CANCELLED / FAILED / SUCCEEDED

async def start_operation(self, request, ctx):
    op = OperationState(operation_id=uuid.uuid4().hex)
    self._operations[op.operation_id] = op
    self._executor.submit(self._run_operation, op, method, args, kwargs)
    return op.to_proto()

async def get_operation(self, request, ctx):
    op = self._operations[request.operation_id]
    return op.to_proto()
```

### Client (`lib/iris/src/iris/actor/client.py`)

`ActorClient` gets three methods that call the new RPCs directly (short calls,
normal retry/timeout): `start_operation()`, `poll_operation_status()` for a
single-shot state read, and `cancel_operation()`. A `get_operation()` helper
wraps `poll_operation_status()` in a backoff loop for callers that just want to
block until the operation reaches a terminal state.

### Fray integration (`lib/fray/src/fray/iris_backend.py`)

The long-running path lands on a new `_IrisActorMethod.submit()`: a fast
`start_operation` RPC returns an operation ID, and the returned
`OperationFuture` polls for the result. `remote()` keeps its existing shape — a
daemon thread running a direct `Call` RPC — because one RPC beats polling for
short methods. So `ActorMethod` grows a `submit()` alongside `remote()`, while
`ActorFuture` itself is unchanged.

```python
class OperationFuture:
    def __init__(self, client, operation_id, poll_interval=1.0):
        self._client = client
        self._op_id = operation_id
        self._poll_interval = poll_interval

    def result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            op = self._client.poll_operation_status(self._op_id)
            if op.state == SUCCEEDED:
                return cloudpickle.loads(op.serialized_result)
            if op.state in (FAILED, CANCELLED):
                raise ...
            if deadline is not None and time.monotonic() >= deadline:
                self._client.cancel_operation(self._op_id)
                raise TimeoutError(...)
            time.sleep(self._poll_interval)
```

This satisfies the `ActorFuture` protocol — `result(timeout=...)` works as
users expect: `None` waits forever, a value actually cancels after the deadline.

### Synchronous path

`ActorMethod.__call__()` (blocking) keeps using the existing `Call` RPC rather
than `start_operation` + immediate poll: for a short method the extra round
trips buy nothing. The `Call` RPC stays in the proto — no breaking change.

## Implementation Outline

1. Proto — add `Operation`, `OperationId` messages and three RPCs to `ActorService`; regenerate
2. Server — add `_operations` dict, implement `start_operation`/`get_operation`/`cancel_operation` handlers, record results from the worker thread
3. Client — add `start_operation()`/`poll_operation_status()`/`get_operation()`/`cancel_operation()` on `ActorClient`
4. Fray — add `_IrisActorMethod.submit()` returning an `OperationFuture`; keep `remote()` and `__call__` on the existing `Call` RPC
5. Cleanup — drop a completed operation once a poll has carried its result back to the caller
6. Test — e2e test: start long operation, poll, cancel, verify timeout behavior

## Notes

- **Backwards compatible** — existing `Call` RPC stays for synchronous use and the local backend
- **No persistence** — operations are in-memory; server restart loses them. Acceptable since Iris already handles job-level restarts.
- **Cancellation is cooperative** — `CancelOperation` sets an event; the actor method must check it. For methods that don't check, cancellation just marks the operation as cancelled and discards the result.
- **Polling overhead** — one short RPC per `poll_interval` seconds. At 1s interval, negligible for hour-long pipelines.
- **Operation cleanup** — `get_operation` evicts an operation as soon as it reports a terminal state, so results are delivered exactly once and memory does not grow without bound. A caller that never polls leaks one entry; a background sweep would be the fix if that ever matters.

## Future Work

- Server-side streaming (`WaitOperation` with long-poll) to reduce polling
- Operation persistence for crash recovery
- Progress reporting (percentage, stage name) in the `Operation` message
- Adaptive poll interval (back off when operation is long-running)
