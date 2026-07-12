# Design: Long-Running Operations for Actor RPC

## Problem

Actor RPC calls block an HTTP connection for the entire duration of the method.
When `run_pipeline` takes hours (`execution.py:1151`), the httpx read timeout
fires and `call_with_retry` (`errors.py:187`) retries the call — creating
duplicate pipeline executions on the same coordinator (#3459).

The timeout default was changed to `None` as a stopgap, but the fundamental
issue remains: a long-lived blocking RPC is fragile. The caller can't
distinguish "still running" from "server died", can't cancel, and
`.result(timeout=30)` only abandons the Python future while the RPC thread
and server keep running.

**Current flow** (`iris_backend.py:276-280`):
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
- Keep the `ActorFuture` protocol (`actor.py:65-70`) unchanged — callers don't change
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

### Server (`server.py`)

The server gets an `_operations` dict. `StartOperation` submits work to the
existing executor and returns the ID. `GetOperation` returns current state.
`CancelOperation` sets a flag; the actor can check it via a context object.

```python
@dataclass
class OperationState:
    id: str
    state: Operation.State
    future: Future            # from executor.submit()
    result: bytes | None      # serialized on completion
    error: ActorError | None
    cancelled: threading.Event

async def start_operation(self, request, ctx):
    op_id = uuid.uuid4().hex
    cancelled = threading.Event()
    future = self._executor.submit(self._run_op, op_id, method, args, kwargs, cancelled)
    self._operations[op_id] = OperationState(id=op_id, state=RUNNING, future=future, ...)
    return Operation(operation_id=op_id, state=RUNNING)

async def get_operation(self, request, ctx):
    op = self._operations[request.operation_id]
    if op.future.done() and op.state == RUNNING:
        self._finalize(op)  # move to SUCCEEDED/FAILED, store result
    return op.to_proto()
```

### Client (`client.py`)

`ActorClient` gets `start_operation()` and `get_operation()` methods that
call the new RPCs directly (short calls, normal retry/timeout).

### Fray integration (`iris_backend.py`)

`_IrisActorMethod.remote()` calls `start_operation` (fast RPC, returns
immediately) and returns an `OperationFuture`. No thread pool needed.

```python
class OperationFuture:
    def __init__(self, client, operation_id, poll_interval=1.0):
        self._client = client
        self._op_id = operation_id
        self._poll_interval = poll_interval

    def result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            op = self._client.get_operation(self._op_id)
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

`ActorMethod.__call__()` (blocking) can either use `start_operation` +
immediate poll, or keep using the existing `Call` RPC for simplicity. The
existing `Call` RPC stays in the proto — no breaking change.

## Implementation Outline

1. Proto — add `Operation`, `OperationId` messages and three RPCs to `ActorService`; regenerate
2. Server — add `_operations` dict, implement `start_operation`/`get_operation`/`cancel_operation` handlers, finalize results on poll
3. Client — add `start_operation()`/`get_operation()`/`cancel_operation()` on `ActorClient`
4. Fray — replace thread-pool dispatch in `_IrisActorMethod.remote()` with `OperationFuture`; keep `__call__` using existing `Call` RPC
5. Cleanup — garbage-collect completed operations after a TTL (e.g. 1 hour)
6. Test — e2e test: start long operation, poll, cancel, verify timeout behavior

## Notes

- **Backwards compatible** — existing `Call` RPC stays for synchronous use and the local backend
- **No persistence** — operations are in-memory; server restart loses them. Acceptable since Iris already handles job-level restarts.
- **Cancellation is cooperative** — `CancelOperation` sets an event; the actor method must check it. For methods that don't check, cancellation just marks the operation as cancelled and discards the result.
- **Polling overhead** — one short RPC per `poll_interval` seconds. At 1s interval, negligible for hour-long pipelines.
- **Operation cleanup** — server should evict completed operations after a TTL to avoid unbounded memory growth. A background sweep or lazy eviction on `get_operation` works.

## Future Work

- Server-side streaming (`WaitOperation` with long-poll) to reduce polling
- Operation persistence for crash recovery
- Progress reporting (percentage, stage name) in the `Operation` message
- Adaptive poll interval (back off when operation is long-running)
