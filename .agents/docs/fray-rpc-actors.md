# Fray RPC Actor Support - Design Document

## Status

**Phase**: Design
**Created**: 2026-01-08
**Related**: `wv-2db6` - Add actor support for Fray

---

## Executive Summary

This document describes the design and implementation plan for adding **actor support** to the Fray RPC backend. Actors are stateful services that persist across multiple method calls and provide the foundation for RL workloads (e.g., inference servers, rollout workers).

**Current state**: Fray RPC supports stateless task execution via `FrayController` and `FrayWorker`. Actors are NOT YET IMPLEMENTED (raises `NotImplementedError` in `FrayContext.create_actor()`).

**Goal**: Extend Fray RPC to support full actor lifecycle (create, method calls, restart on failure, deletion) matching the `JobContext` protocol.

---

## Background

### Why Actors?

Marin's RL pipeline requires **stateful services** that:
1. Load expensive models once and serve many requests (inference servers)
2. Maintain rollout state across episodes (rollout workers)
3. Survive worker preemption with automatic restart (fault tolerance)

Ray provides actors via `ray.remote(ActorClass)`, and Fray must match this interface to enable Ray → Fray migration.

### Current Fray RPC Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│ FrayContext  │────────▶│ FrayController   │────────▶│ FrayWorker   │
│  (Client)    │   RPC   │  (Coordinator)   │   RPC   │  (Executor)  │
└──────────────┘         └──────────────────┘         └──────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Controller State │
                         │ - Task Queue     │
                         │ - Task Results   │
                         │ - Worker Registry│
                         └──────────────────┘
```

**What exists**:
- Task submission (`submit_task`)
- Task execution on workers (`get_next_task`, `report_task_complete`)
- Worker registration and health checks
- Storage-first execution (Phase 1)

**What's missing**: Actor lifecycle management

---

## Design

### Actor Lifecycle

Actors follow a **create → call → restart → destroy** lifecycle:

```
Client                Controller              Worker
  │                       │                      │
  │  1. Create Actor      │                      │
  ├──────────────────────▶│                      │
  │  ActorCreate(cls)     │                      │
  │                       │  2. Select Worker    │
  │                       │     (least loaded)   │
  │                       │                      │
  │                       │  3. Instantiate      │
  │                       ├──────────────────────▶
  │                       │  ActorInstantiate    │
  │                       │  (actor_id, spec)    │
  │                       │                      │
  │                       │                      │  4. Create Instance
  │                       │                      │     actor = cls()
  │                       │                      │
  │                       │  5. Actor Ready      │
  │                       │◀──────────────────────┤
  │                       │                      │
  │  6. Actor Handle      │                      │
  │◀──────────────────────┤                      │
  │  ActorHandle          │                      │
  │                       │                      │
  │  7. Call Method       │                      │
  ├──────────────────────▶│                      │
  │  ActorCall(id, meth)  │                      │
  │                       │  8. Route to Worker  │
  │                       ├──────────────────────▶
  │                       │  ActorExecuteMethod  │
  │                       │                      │
  │                       │                      │  9. Execute
  │                       │                      │     actor.method()
  │                       │                      │
  │                       │  10. Return Result   │
  │                       │◀──────────────────────┤
  │                       │                      │
  │  11. Method Result    │                      │
  │◀──────────────────────┤                      │
```

### Core Principles

1. **Controller-owned placement**: Controller assigns actors to workers (like tasks)
2. **Worker-hosted state**: Workers maintain actor instances in memory
3. **Restart on failure**: Controller detects worker failure and restarts actors on new workers
4. **Storage-first model**: Actor initialization args are stored, state is NOT persisted (Phase 1)
5. **Named actors**: Support `name` parameter for singleton actors (e.g., "global_inference_server")

### Controller State

The controller must track:

```python
# Actor management
actor_registry: dict[str, ActorInfo]      # actor_id -> info
named_actors: dict[str, str]              # name -> actor_id
actor_specs: dict[str, bytes]             # actor_id -> serialized spec (for restart)

@dataclass
class ActorInfo:
    actor_id: str
    worker_id: str                        # Current hosting worker
    name: str | None
    created_at: timestamp
    last_used: timestamp
    status: ActorStatus                   # CREATING, READY, RESTARTING, FAILED
```

### Worker State

Workers must maintain:

```python
# Actor instances
actor_instances: dict[str, Any]           # actor_id -> Python object

# Example:
# actor_instances["actor-123"] = InferenceServer(model_path="...")
```

---

## Protocol Extensions

### Proto Definitions (fray.proto)

```protobuf
// Actor lifecycle
enum ActorStatus {
  ACTOR_STATUS_UNSPECIFIED = 0;
  ACTOR_STATUS_CREATING = 1;
  ACTOR_STATUS_READY = 2;
  ACTOR_STATUS_RESTARTING = 3;
  ACTOR_STATUS_FAILED = 4;
}

message ActorSpec {
  string actor_id = 1;
  bytes serialized_actor = 2;           // cloudpickle(cls, args, kwargs)
  string name = 3;                      // Optional named actor
  bool get_if_exists = 4;               // Return existing if name matches
}

message ActorHandle {
  string actor_id = 1;
  string worker_id = 2;
  string name = 3;
  ActorStatus status = 4;
}

message ActorCall {
  string actor_id = 1;
  bytes serialized_call = 2;            // cloudpickle(method_name, args, kwargs)
}

message ActorCallResult {
  string actor_id = 1;
  string task_id = 2;                   // Actor calls return task handles
}

message ActorDeleteRequest {
  string actor_id = 1;
}

// Extend FrayController service
service FrayController {
  // ... existing task methods ...

  // Actor RPCs
  rpc CreateActor(ActorSpec) returns (ActorHandle);
  rpc CallActor(ActorCall) returns (TaskHandle);  // Returns task handle for async
  rpc GetActorStatus(ActorHandle) returns (ActorHandle);
  rpc DeleteActor(ActorDeleteRequest) returns (Empty);
}

// Extend FrayWorker service
service FrayWorker {
  // ... existing health check methods ...

  // Actor RPCs
  rpc InstantiateActor(ActorSpec) returns (ActorHandle);
  rpc ExecuteActorMethod(ActorCall) returns (TaskResult);
  rpc DestroyActor(ActorDeleteRequest) returns (Empty);
  rpc ListActors(Empty) returns (ActorList);
}

message ActorList {
  repeated ActorHandle actors = 1;
}
```

### API Changes

#### FrayContext (Client)

```python
class FrayContext:
    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> ActorHandle:
        """Create actor and return handle."""
        payload = {"cls": actor_class, "args": args, "kwargs": kwargs}
        serialized_actor = cloudpickle.dumps(payload)

        spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name=name or "",
            get_if_exists=get_if_exists,
        )
        handle = self._client.create_actor(spec)

        return _FrayActorHandle(handle.actor_id, self._client)


class _FrayActorHandle:
    """Actor handle for remote method calls."""

    def __init__(self, actor_id: str, client: FrayControllerClientSync):
        self._actor_id = actor_id
        self._client = client

    def __getattr__(self, method_name: str):
        return _FrayActorMethod(self._actor_id, method_name, self._client)


class _FrayActorMethod:
    """Actor method wrapper."""

    def __init__(self, actor_id: str, method_name: str, client: FrayControllerClientSync):
        self._actor_id = actor_id
        self._method_name = method_name
        self._client = client

    def remote(self, *args, **kwargs) -> _FrayFuture:
        """Call actor method, returns future."""
        payload = {"method": self._method_name, "args": args, "kwargs": kwargs}
        serialized_call = cloudpickle.dumps(payload)

        call = fray_pb2.ActorCall(
            actor_id=self._actor_id,
            serialized_call=serialized_call,
        )
        task_handle = self._client.call_actor(call)

        return _FrayFuture(task_handle.task_id, self._client)
```

#### FrayControllerServicer (Controller)

```python
class FrayControllerServicer(FrayController):
    def __init__(self):
        # ... existing task state ...
        self._actors: dict[str, ActorInfo] = {}
        self._named_actors: dict[str, str] = {}
        self._actor_specs: dict[str, bytes] = {}

    async def create_actor(self, request: fray_pb2.ActorSpec, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """Create actor and assign to worker."""
        # Check for existing named actor
        if request.name and request.name in self._named_actors:
            if request.get_if_exists:
                actor_id = self._named_actors[request.name]
                actor = self._actors[actor_id]
                return fray_pb2.ActorHandle(
                    actor_id=actor_id,
                    worker_id=actor.worker_id,
                    name=request.name,
                    status=fray_pb2.ACTOR_STATUS_READY,
                )
            else:
                raise ConnectError(Code.ALREADY_EXISTS, f"Actor {request.name} already exists")

        # Generate actor ID
        actor_id = str(uuid.uuid4())

        with self._lock:
            # Select worker with fewest actors
            worker_id = min(
                self._workers.keys(),
                key=lambda w: sum(1 for a in self._actors.values() if a.worker_id == w)
            )

            # Store spec for restart
            self._actor_specs[actor_id] = request.serialized_actor

            # Create ActorInfo
            actor_info = ActorInfo(
                actor_id=actor_id,
                worker_id=worker_id,
                name=request.name or None,
                status=fray_pb2.ACTOR_STATUS_CREATING,
                created_at=time.time(),
            )
            self._actors[actor_id] = actor_info

            if request.name:
                self._named_actors[request.name] = actor_id

        # Instantiate on worker (async)
        worker_stub = self._get_worker_stub(worker_id)
        actor_spec = fray_pb2.ActorSpec(
            actor_id=actor_id,
            serialized_actor=request.serialized_actor,
            name=request.name,
        )

        try:
            await worker_stub.instantiate_actor(actor_spec)

            with self._lock:
                self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_READY

        except Exception as e:
            with self._lock:
                self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_FAILED
            raise ConnectError(Code.INTERNAL, f"Failed to create actor: {e}")

        return fray_pb2.ActorHandle(
            actor_id=actor_id,
            worker_id=worker_id,
            name=request.name,
            status=fray_pb2.ACTOR_STATUS_READY,
        )

    async def call_actor(self, request: fray_pb2.ActorCall, ctx: RequestContext) -> fray_pb2.TaskHandle:
        """Route actor method call to hosting worker as a task."""
        with self._lock:
            if request.actor_id not in self._actors:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found")

            actor_info = self._actors[request.actor_id]

            # Check if worker is alive
            if actor_info.worker_id not in self._workers:
                # Worker died - restart actor
                await self._restart_actor(request.actor_id)
                raise ConnectError(Code.UNAVAILABLE, f"Actor {request.actor_id} restarting")

        # Create task for method call
        task_id = str(uuid.uuid4())

        # Wrap as special actor task
        actor_task_payload = {
            "actor_id": request.actor_id,
            "serialized_call": request.serialized_call,
        }
        serialized_fn = cloudpickle.dumps(actor_task_payload)

        task = Task(
            task_id=task_id,
            serialized_fn=serialized_fn,
            status=fray_pb2.TASK_STATUS_PENDING,
            actor_id=request.actor_id,  # Mark as actor task
        )

        with self._lock:
            self._tasks[task_id] = task
            # Route directly to actor's worker
            self._pending_queue.append(task_id)
            self._condition.notify_all()

        return fray_pb2.TaskHandle(
            task_id=task_id,
            status=fray_pb2.TASK_STATUS_PENDING,
        )

    async def _restart_actor(self, actor_id: str):
        """Restart actor on new worker after failure."""
        if actor_id not in self._actor_specs:
            # Can't restart without spec
            del self._actors[actor_id]
            return

        actor_info = self._actors[actor_id]
        old_worker_id = actor_info.worker_id
        actor_spec = self._actor_specs[actor_id]

        # Select new worker (exclude failed worker)
        available_workers = [
            w_id for w_id in self._workers.keys()
            if w_id != old_worker_id
        ]

        if not available_workers:
            actor_info.status = fray_pb2.ACTOR_STATUS_FAILED
            return

        new_worker_id = min(
            available_workers,
            key=lambda w_id: sum(1 for a in self._actors.values() if a.worker_id == w_id)
        )

        # Update status
        actor_info.status = fray_pb2.ACTOR_STATUS_RESTARTING

        # Instantiate on new worker
        worker_stub = self._get_worker_stub(new_worker_id)
        spec = fray_pb2.ActorSpec(
            actor_id=actor_id,
            serialized_actor=actor_spec,
            name=actor_info.name or "",
        )

        try:
            await worker_stub.instantiate_actor(spec)
            actor_info.worker_id = new_worker_id
            actor_info.status = fray_pb2.ACTOR_STATUS_READY
        except Exception:
            actor_info.status = fray_pb2.ACTOR_STATUS_FAILED
```

#### FrayWorkerServicer (Worker)

```python
class FrayWorkerServicer(FrayWorker):
    def __init__(self, worker_id: str):
        # ... existing state ...
        self._actor_instances: dict[str, Any] = {}

    async def instantiate_actor(self, request: fray_pb2.ActorSpec, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """Instantiate actor from spec."""
        # Deserialize actor spec
        actor_data = cloudpickle.loads(request.serialized_actor)
        cls = actor_data["cls"]
        args = actor_data["args"]
        kwargs = actor_data["kwargs"]

        # Create instance
        try:
            instance = cls(*args, **kwargs)

            with self._lock:
                self._actor_instances[request.actor_id] = instance

            return fray_pb2.ActorHandle(
                actor_id=request.actor_id,
                worker_id=self.worker_id,
                name=request.name,
                status=fray_pb2.ACTOR_STATUS_READY,
            )

        except Exception as e:
            raise ConnectError(Code.INTERNAL, f"Failed to instantiate actor: {e}")

    async def execute_actor_method(self, request: fray_pb2.ActorCall, ctx: RequestContext) -> fray_pb2.TaskResult:
        """Execute method on actor instance."""
        with self._lock:
            if request.actor_id not in self._actor_instances:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found on this worker")

            instance = self._actor_instances[request.actor_id]

        # Deserialize method call
        call_data = cloudpickle.loads(request.serialized_call)
        method_name = call_data["method"]
        args = call_data["args"]
        kwargs = call_data["kwargs"]

        # Execute method
        try:
            method = getattr(instance, method_name)
            result = method(*args, **kwargs)

            serialized_result = cloudpickle.dumps(result)

            return fray_pb2.TaskResult(
                task_id="",  # Not used for direct calls
                serialized_result=serialized_result,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
            serialized_error = cloudpickle.dumps(e)

            return fray_pb2.TaskResult(
                task_id="",
                error=error_msg,
                serialized_error=serialized_error,
            )

    async def destroy_actor(self, request: fray_pb2.ActorDeleteRequest, ctx: RequestContext) -> fray_pb2.Empty:
        """Destroy actor instance."""
        with self._lock:
            self._actor_instances.pop(request.actor_id, None)

        return fray_pb2.Empty()
```

---

## Failure Scenarios & Recovery

### Scenario 1: Worker Crash Hosting Actors

**What happens**:
1. Worker hosting actors `a-1, a-2` crashes
2. Controller detects failure via health check (30s timeout)
3. Controller marks worker as DEAD
4. Controller restarts actors on new workers:
   - Deserializes original actor specs from `actor_specs`
   - Instantiates on least-loaded workers
   - Updates actor registry with new `worker_id`
5. **Actor state is lost** (Phase 1 limitation)

**Recovery time**: <35s (30s detection + 5s restart)

**Implications**:
- RL inference servers: Must reload models on restart (acceptable, models are in GCS)
- RL rollout workers: In-flight episodes are lost (acceptable, environment resets)

### Scenario 2: Actor Method Call During Restart

**What happens**:
1. Client calls `actor.method.remote()` while actor is restarting
2. Controller returns error: `UNAVAILABLE: Actor {actor_id} restarting`
3. Client retries after backoff (exponential)
4. Actor becomes READY, call succeeds

**Mitigation**: Client library should auto-retry on `UNAVAILABLE` with exponential backoff.

### Scenario 3: Named Actor Collision

**What happens**:
1. Client creates actor with `name="server"`
2. Worker hosting actor crashes
3. Controller restarts actor on new worker (same name)
4. Different client creates actor with `name="server", get_if_exists=True`
5. Returns handle to existing (restarted) actor

**Expected behavior**: Named actors are singletons, `get_if_exists=True` returns existing.

---

## Implementation Phases

### Phase 1: Basic Actor Support (MVP)

**Goal**: Enable actor creation and method calls on Fray RPC.

**Tasks**:
1. Extend protobuf definitions (add Actor messages)
2. Implement controller actor registry and placement logic
3. Implement worker actor instantiation and method execution
4. Update `FrayContext.create_actor()` to create actors via RPC
5. Add actor handle and method wrappers (`_FrayActorHandle`, `_FrayActorMethod`)
6. Basic tests (create actor, call methods, verify state persistence)

**Exit criteria**:
- ✅ `ctx.create_actor(Counter)` creates actor on worker
- ✅ `actor.increment.remote()` executes method and returns future
- ✅ Actor state persists across multiple method calls
- ✅ Named actors work (`get_if_exists=True`)

### Phase 2: Fault Tolerance

**Goal**: Restart actors on worker failure.

**Tasks**:
1. Store actor specs in controller for restart
2. Detect actor hosting worker failure (via health checks)
3. Restart actors on new workers (automatic)
4. Update actor registry with new worker_id
5. Tests for actor restart (kill worker, verify actor comes back)

**Exit criteria**:
- ✅ Worker crash triggers actor restart on new worker
- ✅ Actor method calls succeed after restart
- ✅ Actor state is reset (expected behavior for Phase 1)

### Phase 3: Actor Lifecycle Management

**Goal**: Complete actor lifecycle (delete, status queries).

**Tasks**:
1. Implement `delete_actor()` RPC
2. Implement `get_actor_status()` for monitoring
3. Add garbage collection for unused actors (optional)
4. Add actor metrics (method call count, latency)

**Exit criteria**:
- ✅ `ctx.delete_actor(actor_handle)` removes actor
- ✅ Status queries return correct actor state

### Phase 4: Performance Optimization (Future)

**Goal**: Reduce actor method call latency.

**Tasks**:
1. Direct worker-to-worker actor calls (bypass controller routing)
2. Actor method call batching
3. Locality-aware actor placement (co-locate with data)
4. Actor checkpointing for faster restart

---

## Testing Strategy

### Unit Tests

```python
# test_fray_actors.py

def test_create_actor():
    """Test basic actor creation."""
    ctx = FrayContext("http://localhost:50051")

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1
            return self.count

    actor = ctx.create_actor(Counter)

    result1 = ctx.get(actor.increment.remote())
    assert result1 == 1

    result2 = ctx.get(actor.increment.remote())
    assert result2 == 2  # State persists


def test_named_actor():
    """Test named actor singleton behavior."""
    ctx = FrayContext("http://localhost:50051")

    actor1 = ctx.create_actor(Counter, name="singleton")
    actor2 = ctx.create_actor(Counter, name="singleton", get_if_exists=True)

    ctx.get(actor1.increment.remote())  # count = 1
    result = ctx.get(actor2.increment.remote())  # count = 2

    assert result == 2  # Same actor instance


def test_actor_restart():
    """Test actor restart on worker failure."""
    ctx = FrayContext("http://localhost:50051")

    actor = ctx.create_actor(Counter)
    ctx.get(actor.increment.remote())  # count = 1

    # Kill worker hosting actor
    worker_id = get_actor_worker(actor)
    kill_worker(worker_id)

    # Wait for restart
    time.sleep(5)

    # Method calls should succeed (but state is reset)
    result = ctx.get(actor.increment.remote())
    assert result == 1  # State reset on restart
```

### Integration Tests

```python
def test_rl_inference_server():
    """Test RL inference server actor pattern."""
    ctx = FrayContext("http://localhost:50051")

    class InferenceServer:
        def __init__(self, model_path: str):
            self.model = load_model(model_path)  # Expensive

        def predict(self, obs):
            return self.model(obs)

    server = ctx.create_actor(InferenceServer, model_path="gs://models/policy.pkl")

    # Multiple clients can call
    futures = [server.predict.remote(obs) for obs in observations]
    results = ctx.get(futures)

    assert len(results) == len(observations)
```

---

## Performance Characteristics

### Phase 1 Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Actor creation latency | <100ms | Includes worker selection + instantiation |
| Actor method call latency | <10ms p95 | Controller routing + worker execution |
| Actor restart time | <5s | Includes worker selection + re-instantiation |
| Actors per worker | 100+ | Memory-limited, not network-limited |

### Bottlenecks

**Phase 1**:
- Actor method calls go through controller (extra hop)
- Actor state is NOT persisted (lost on restart)
- cloudpickle serialization for every method call

**Future optimizations**:
- Direct worker-to-worker method calls (Phase 4)
- Actor checkpointing to GCS (Phase 4)
- Method call batching (Phase 4)

---

## Comparison with Ray

### What Fray Matches

✅ **Basic actor creation**: `ctx.create_actor(cls, *args, **kwargs)`
✅ **Named actors**: `name="server", get_if_exists=True`
✅ **Method calls**: `actor.method.remote(*args) → Future`
✅ **Restart on failure**: Automatic (though state is lost)

### What Fray Does NOT Match (Phase 1)

❌ **Actor state persistence**: Ray can checkpoint, Fray cannot (yet)
❌ **Actor lifetime**: Fray only supports `non_detached` (dies with controller)
❌ **Actor resources**: No CPU/GPU resource constraints (yet)
❌ **Actor placement groups**: No affinity control (yet)

### Migration Path

For RL workloads:
1. **Inference servers**: Work immediately (models reload on restart)
2. **Rollout workers**: Work immediately (episodes reset on restart)
3. **Parameter servers**: NOT YET (requires state persistence)

---

## Dependencies

### Proto Changes

- Add `ActorSpec`, `ActorHandle`, `ActorCall` messages
- Add `ActorStatus` enum
- Extend `FrayController` service with actor RPCs
- Extend `FrayWorker` service with actor RPCs

### Code Changes

- `lib/fray/src/fray/job/rpc/proto/fray.proto` (proto definitions)
- `lib/fray/src/fray/job/rpc/controller.py` (controller logic)
- `lib/fray/src/fray/job/rpc/worker.py` (worker logic)
- `lib/fray/src/fray/job/rpc/context.py` (client API)

### Testing

- `lib/fray/tests/test_fray_actors.py` (new file)
- `lib/fray/tests/test_job_context.py` (extend with actor tests)

---

## Open Questions

1. **Actor task routing**: Should actor method calls bypass the task queue and route directly to the hosting worker?
   - **Proposal**: Yes, but only in Phase 4 (for now, treat as special tasks)

2. **Actor resource limits**: Should actors count toward worker CPU/memory limits?
   - **Proposal**: Not in Phase 1, track in Phase 3

3. **Actor garbage collection**: When should unused actors be destroyed?
   - **Proposal**: Manual deletion only in Phase 1, add TTL in Phase 3

4. **Actor method call semantics**: Should calls be async (task-based) or sync (blocking RPC)?
   - **Proposal**: Async (return futures), matches Ray semantics

5. **Actor preemption**: Should actors survive controller restarts?
   - **Proposal**: No in Phase 1 (in-memory state), yes in Phase 2 (persist to Redis)

---

## Summary

This design extends Fray RPC to support **stateful actors** with automatic restart on worker failure. The implementation follows a phased approach:

1. **Phase 1 (MVP)**: Basic actor creation and method calls
2. **Phase 2**: Fault tolerance (restart on worker failure)
3. **Phase 3**: Lifecycle management (delete, status, metrics)
4. **Phase 4**: Performance optimization (direct calls, checkpointing)

The design matches Ray's actor API closely enough to enable RL workloads while keeping Phase 1 simple (no state persistence). Future phases will add advanced features like checkpointing and resource constraints.

**Next steps**: File weaver tasks for each implementation phase.
