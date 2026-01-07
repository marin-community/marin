# Fray Architecture: Complete Execution Flows

## Status

**Phase 1: Storage-First Execution** ✅ COMPLETE

The storage-first chunk execution model described here has been implemented. See:
- `lib/zephyr/src/zephyr/storage.py` - InlineRef, StorageRef, ChunkWriter, StorageManager
- `lib/zephyr/src/zephyr/backends.py` - Storage-aware Backend
- `lib/zephyr/src/zephyr/plan.py` - run_stage returns ChunkRef directly

**Phase 2: Distributed FrayController/FrayWorker** - Future work documented below.

---

## Overview

This document defines the complete end-to-end execution flows for tasks and actors in Fray, including worker registration, task dispatch, failure handling, and actor lifecycle.

**Note**: This document describes the logical operations and interfaces without assuming a specific RPC protocol (gRPC, HTTP, etc.). The implementation plan will specify the concrete protocol choice.

---

## Component Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│ FrayContext  │────────▶│ FrayController   │────────▶│ FrayWorker   │
│  (Client)    │   RPC   │  (Coordinator)   │   RPC   │  (Executor)  │
└──────────────┘         └──────────────────┘         └──────────────┘
                                  │                           │
                                  │ Polls health              │
                                  │◀──────────────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Controller State │
                         │ - Task Queue     │
                         │ - Results        │
                         │ - Worker Registry│
                         │ - Actor Registry │
                         └──────────────────┘
```

**Components:**
1. **FrayContext** - Client library for submitting tasks/actors
2. **FrayController** - Central coordinator that assigns work and monitors health
3. **FrayWorker** - Executor that runs tasks and hosts actors
4. **Controller State** - In-memory state (Phase 1) or external store (Phase 2+)

**Key Architectural Principles:**
- **Push-based dispatch**: Controller assigns tasks to workers (workers don't poll)
- **Controller-driven health**: Controller polls workers for health status
- **Controller-owned retry**: Controller detects failures and retries tasks
- **Stateless workers**: Workers execute what they're told, maintain minimal state

---

## Flow 1: Task Execution (End-to-End)

### Complete Flow Diagram

```
Client                Controller              Worker
  │                       │                      │
  │  1. Submit Task       │                      │
  ├──────────────────────▶│                      │
  │  TaskCreate(spec)     │                      │
  │                       │                      │
  │  2. Task Handle       │                      │
  │◀──────────────────────┤                      │
  │  {task_id: "t-123"}   │                      │
  │                       │                      │
  │                       │  3. Assign Task      │
  │                       ├──────────────────────▶
  │                       │  ExecuteTask(t-123,  │
  │                       │              spec)   │
  │                       │                      │
  │                       │                      │  4. Execute
  │                       │                      │     fn(*args)
  │                       │                      │
  │                       │  5. Report Result    │
  │                       │◀──────────────────────┤
  │                       │  TaskComplete(t-123, │
  │                       │              result) │
  │                       │                      │
  │  6. Poll Status       │                      │
  ├──────────────────────▶│                      │
  │  TaskStatus(t-123)    │                      │
  │                       │                      │
  │  7. Get Result        │                      │
  ├──────────────────────▶│                      │
  │  TaskResult(t-123)    │                      │
  │                       │                      │
  │  8. Result Data       │                      │
  │◀──────────────────────┤                      │
  │  {result: 42}         │                      │
```

### Operations

#### Client → Controller

**Operation: TaskCreate**
```
Input:
  serialized_task: bytes     # cloudpickle(fn, args, kwargs)
  resources: dict            # CPU/memory/GPU requirements
  max_retries: int           # Retry limit

Output:
  task_id: str
  status: TaskStatus
```

**Operation: TaskStatus**
```
Input:
  task_id: str

Output:
  task_id: str
  status: TaskStatus         # PENDING, IN_PROGRESS, FINISHED, FAILED
  worker_id: str | None
  created_at: timestamp
  started_at: timestamp | None
  finished_at: timestamp | None
  error: str | None
```

**Operation: TaskResult**
```
Input:
  task_id: str

Output:
  result_data: bytes         # cloudpickle(result)

Error:
  NOT_FOUND if task doesn't exist
  NOT_READY if task not finished
```

#### Controller → Worker

**Operation: ExecuteTask**
```
Input:
  task_id: str
  serialized_task: bytes

Output:
  accepted: bool             # Worker can execute now

Worker behavior:
  - Deserialize task
  - Execute function
  - Report result via TaskComplete or TaskFail
  - Return immediately (async execution)
```

#### Worker → Controller

**Operation: TaskComplete**
```
Input:
  task_id: str
  result_data: bytes         # cloudpickle(result)

Output:
  acknowledged: bool
```

**Operation: TaskFail**
```
Input:
  task_id: str
  error: str                 # Exception traceback

Output:
  acknowledged: bool
  retry: bool                # Should worker retry or controller will?
```

### Controller Logic

**Task Assignment Algorithm:**
```python
def assign_task(task_id: str, task_spec: bytes):
    """Assign task to least-loaded worker."""

    # 1. Select worker with lowest load
    worker_id = min(
        worker_registry.keys(),
        key=lambda w: worker_registry[w].active_tasks
    )

    # 2. Mark task as assigned
    task_registry[task_id].status = IN_PROGRESS
    task_registry[task_id].worker_id = worker_id
    task_registry[task_id].started_at = now()

    # 3. Push task to worker
    try:
        worker_registry[worker_id].execute_task(task_id, task_spec)
        worker_registry[worker_id].active_tasks += 1
    except Exception:
        # Worker unreachable - mark for retry
        task_registry[task_id].status = PENDING
        task_queue.append(task_id)
```

**Retry Logic:**
```python
def handle_task_failure(task_id: str, error: str):
    """Handle task failure with retry logic."""

    task_info = task_registry[task_id]

    # Increment retry count
    task_info.retries += 1

    if task_info.retries < task_info.max_retries:
        # Retry: reset to pending and requeue
        task_info.status = PENDING
        task_info.worker_id = None
        task_queue.append(task_id)
    else:
        # Max retries exceeded - mark as failed
        task_info.status = FAILED
        task_info.error = error
        task_info.finished_at = now()
```

---

## Flow 2: Worker Registration & Health Monitoring

### Flow Diagram

```
Worker                Controller
  │                       │
  │  1. Register          │
  ├──────────────────────▶│
  │  WorkerRegister(addr) │  - Add to registry
  │                       │  - Start health checks
  │                       │
  │  2. Registration OK   │
  │◀──────────────────────┤
  │                       │
  │                       │
  │                       │  3. Health Check
  │                       │     (every 10s)
  │  4. Health Response   │
  │◀──────────────────────┤
  │  WorkerHealth()       │
  │                       │
  │  5. Health Status     │
  ├──────────────────────▶│
  │  {active_tasks: 2}    │  - Update last_seen
  │                       │
  │   ... time passes ... │
  │                       │
  │   (no response)       │  6. Detect Failure
  │                       │     (timeout >30s)
  │                       │     - Mark worker dead
  │                       │     - Requeue tasks
  │                       │     - Restart actors
```

### Operations

#### Worker → Controller

**Operation: WorkerRegister**
```
Input:
  worker_id: str
  address: str               # RPC address for callbacks
  hostname: str
  pid: int
  resources: ResourceInfo    # CPU/memory/GPU available

Output:
  worker_id: str             # Confirmed or assigned
  heartbeat_interval: int    # Seconds between health checks
```

**Operation: WorkerUnregister**
```
Input:
  worker_id: str

Output:
  acknowledged: bool
```

#### Controller → Worker

**Operation: WorkerHealth**
```
Input:
  (none)

Output:
  active_tasks: int          # Current task count
  resources_available: ResourceInfo
  uptime: int                # Seconds since start

Worker behavior:
  - Return current load metrics
  - Used by controller to track health
```

### Controller Health Monitoring

**Health Check Loop:**
```python
async def monitor_workers():
    """Periodically check worker health."""

    while True:
        await sleep(10.0)  # Check every 10s

        now_ts = now()
        dead_workers = []

        for worker_id, worker in worker_registry.items():
            # Poll worker health
            try:
                health = await worker.health_check()
                worker.last_seen = now_ts
                worker.active_tasks = health.active_tasks

            except Exception:
                # Worker unreachable
                if now_ts - worker.last_seen > 30.0:
                    # Dead if no response for >30s
                    dead_workers.append(worker_id)

        # Handle failures
        for worker_id in dead_workers:
            await handle_worker_failure(worker_id)
```

**Worker Failure Handling:**
```python
async def handle_worker_failure(worker_id: str):
    """Handle worker failure: requeue tasks, restart actors."""

    # 1. Find tasks assigned to this worker
    failed_tasks = [
        task_id for task_id, info in task_registry.items()
        if info.worker_id == worker_id and info.status == IN_PROGRESS
    ]

    # 2. Requeue tasks (automatic retry)
    for task_id in failed_tasks:
        task_registry[task_id].status = PENDING
        task_registry[task_id].worker_id = None
        task_queue.append(task_id)

    # 3. Find actors hosted on this worker
    failed_actors = [
        actor_id for actor_id, actor_info in actor_registry.items()
        if actor_info.worker_id == worker_id
    ]

    # 4. Restart actors on new workers
    for actor_id in failed_actors:
        await restart_actor(actor_id)

    # 5. Remove worker from registry
    del worker_registry[worker_id]
```

---

## Flow 3: Actor Execution (End-to-End)

### Complete Flow Diagram

```
Client                Controller              Worker
  │                       │                      │
  │  1. Create Actor      │                      │
  ├──────────────────────▶│                      │
  │  ActorCreate(cls)     │                      │
  │                       │                      │
  │                       │  2. Select Worker    │
  │                       │     (least loaded)   │
  │                       │                      │
  │                       │  3. Create Actor     │
  │                       ├──────────────────────▶
  │                       │  ActorInstantiate    │
  │                       │  (actor_id, spec)    │
  │                       │                      │
  │                       │                      │  4. Create Instance
  │                       │                      │     actor = cls()
  │                       │                      │
  │                       │  5. Actor Ready      │
  │                       │◀──────────────────────┤
  │                       │  {actor_id: "a-123"} │
  │                       │                      │
  │  6. Actor Handle      │                      │
  │◀──────────────────────┤                      │
  │  {actor_id: "a-123"}  │                      │
  │                       │                      │
  │  7. Call Method       │                      │
  ├──────────────────────▶│                      │
  │  ActorCall(a-123,     │                      │
  │            method)    │                      │
  │                       │                      │
  │                       │  8. Route to Worker  │
  │                       ├──────────────────────▶
  │                       │  ActorExecuteMethod  │
  │                       │  (a-123, method)     │
  │                       │                      │
  │                       │                      │  9. Execute
  │                       │                      │     actor.method()
  │                       │                      │
  │                       │  10. Return Result   │
  │                       │◀──────────────────────┤
  │                       │  {result_data}       │
  │                       │                      │
  │  11. Method Result    │                      │
  │◀──────────────────────┤                      │
```

### Operations

#### Client → Controller

**Operation: ActorCreate**
```
Input:
  serialized_actor: bytes    # cloudpickle(cls, args, kwargs)
  name: str | None           # Optional named actor

Output:
  actor_id: str
  worker_id: str             # Where actor is hosted
  name: str | None
```

**Operation: ActorCall**
```
Input:
  actor_id: str
  serialized_call: bytes     # cloudpickle(method_name, args, kwargs)

Output:
  task_id: str               # Actor calls return task handles

Note: Actor method calls are treated as tasks internally
```

**Operation: ActorDelete**
```
Input:
  actor_id: str

Output:
  acknowledged: bool
```

#### Controller → Worker

**Operation: ActorInstantiate**
```
Input:
  actor_id: str
  serialized_actor: bytes

Output:
  actor_id: str
  created: bool

Worker behavior:
  - Deserialize actor spec
  - Instantiate: actor = cls(*args, **kwargs)
  - Store in actor registry
  - Return immediately
```

**Operation: ActorExecuteMethod**
```
Input:
  actor_id: str
  serialized_call: bytes     # cloudpickle(method, args, kwargs)

Output:
  result_data: bytes         # cloudpickle(result)

Worker behavior:
  - Look up actor instance
  - Deserialize method call
  - Execute: result = actor.method(*args, **kwargs)
  - Serialize and return result

Error:
  ACTOR_NOT_FOUND if actor not on this worker
```

**Operation: ActorDestroy**
```
Input:
  actor_id: str

Output:
  acknowledged: bool

Worker behavior:
  - Delete actor instance
  - Free resources
```

### Controller Logic

**Actor Assignment:**
```python
def create_actor(actor_spec: bytes, name: str | None) -> str:
    """Create actor and assign to worker."""

    # Check for named actor
    if name and name in named_actors:
        return named_actors[name]  # Return existing

    # Generate ID
    actor_id = f"actor-{uuid.uuid4()}"

    # Select worker with fewest actors
    worker_id = min(
        worker_registry.keys(),
        key=lambda w: sum(
            1 for a in actor_registry.values()
            if a.worker_id == w
        )
    )

    # Instantiate on worker
    worker = worker_registry[worker_id]
    worker.instantiate_actor(actor_id, actor_spec)

    # Register
    actor_registry[actor_id] = ActorInfo(
        actor_id=actor_id,
        worker_id=worker_id,
        name=name,
        created_at=now()
    )

    if name:
        named_actors[name] = actor_id

    # Store spec for restart
    actor_specs[actor_id] = actor_spec

    return actor_id
```

**Actor Method Routing:**
```python
def call_actor_method(actor_id: str, call_spec: bytes) -> bytes:
    """Route actor method call to hosting worker."""

    if actor_id not in actor_registry:
        raise ActorNotFound(actor_id)

    actor_info = actor_registry[actor_id]
    worker_id = actor_info.worker_id

    # Check worker health
    if worker_id not in worker_registry:
        # Worker died - restart actor
        restart_actor(actor_id)
        raise ActorRestarting(actor_id)

    # Forward to worker
    worker = worker_registry[worker_id]
    result = worker.execute_actor_method(actor_id, call_spec)

    # Update last used
    actor_info.last_used = now()

    return result
```

**Actor Restart:**
```python
def restart_actor(actor_id: str):
    """Restart actor on new worker after failure."""

    if actor_id not in actor_registry:
        return

    actor_info = actor_registry[actor_id]
    old_worker_id = actor_info.worker_id

    # Get original spec
    if actor_id not in actor_specs:
        # Can't restart without spec
        del actor_registry[actor_id]
        return

    actor_spec = actor_specs[actor_id]

    # Select new worker (exclude failed worker)
    available_workers = [
        w_id for w_id in worker_registry.keys()
        if w_id != old_worker_id
    ]

    if not available_workers:
        # No workers available
        return

    new_worker_id = min(
        available_workers,
        key=lambda w_id: sum(
            1 for a in actor_registry.values()
            if a.worker_id == w_id
        )
    )

    # Instantiate on new worker
    worker = worker_registry[new_worker_id]
    worker.instantiate_actor(actor_id, actor_spec)

    # Update registry
    actor_info.worker_id = new_worker_id
```

---

## Worker Interface Specification

Workers must implement the following operations to be compatible with the controller:

### Required Operations

**1. ExecuteTask(task_id, serialized_task) → accepted**
- Execute task asynchronously
- Report result via TaskComplete or TaskFail
- Return immediately (non-blocking)

**2. WorkerHealth() → health_status**
- Return current load metrics
- Called by controller every 10s
- Must respond within 5s or considered unhealthy

**3. ActorInstantiate(actor_id, serialized_actor) → created**
- Instantiate actor from cloudpickle spec
- Store in local actor registry
- Return immediately

**4. ActorExecuteMethod(actor_id, serialized_call) → result**
- Execute method on existing actor
- Return serialized result
- Block until method completes

**5. ActorDestroy(actor_id) → acknowledged**
- Delete actor instance
- Free resources

### Worker Responsibilities

Workers are responsible for:
- ✅ Executing tasks when assigned by controller
- ✅ Hosting actor instances
- ✅ Responding to health checks
- ✅ Reporting task results
- ✅ Graceful shutdown (finish in-progress tasks)

Workers are NOT responsible for:
- ❌ Task scheduling/queueing
- ❌ Retry logic
- ❌ Worker failure detection
- ❌ Actor placement decisions

---

## Controller State Model

### Core Data Structures

```python
# Task management
task_queue: deque[str]                    # Pending task IDs
task_registry: dict[str, TaskInfo]        # task_id -> info
task_specs: dict[str, bytes]              # task_id -> serialized spec
task_results: dict[str, bytes]            # task_id -> serialized result

# Worker management
worker_registry: dict[str, WorkerInfo]    # worker_id -> info

# Actor management
actor_registry: dict[str, ActorInfo]      # actor_id -> info
named_actors: dict[str, str]              # name -> actor_id
actor_specs: dict[str, bytes]             # actor_id -> spec (for restart)
```

### State Transitions

**Task States:**
```
PENDING ──(assign)──▶ IN_PROGRESS ──(complete)──▶ FINISHED
                           │
                           └──(fail)──▶ PENDING (if retries left)
                                    └─▶ FAILED (max retries)
```

**Worker States:**
```
(register) ──▶ HEALTHY ──(timeout)──▶ DEAD ──(cleanup)──▶ (removed)
                  │
                  └──(health OK)──▶ HEALTHY
```

**Actor States:**
```
(create) ──▶ ACTIVE ──(worker failure)──▶ RESTARTING ──▶ ACTIVE
                │                                            │
                └──(delete)──────────────────────────────────▶ DESTROYED
```

---

## Failure Scenarios & Recovery

### Scenario 1: Worker Crash During Task Execution

**What happens:**
1. Worker executing task `t-123` crashes
2. Controller health check fails (no response for >30s)
3. Controller marks worker as DEAD
4. Controller requeues task `t-123` (status → PENDING)
5. Controller assigns `t-123` to different worker
6. Task executes successfully on new worker

**Recovery time:** <30s (health check timeout)

### Scenario 2: Worker Crash Hosting Actors

**What happens:**
1. Worker hosting actors `a-1, a-2` crashes
2. Controller detects failure via health check
3. Controller restarts actors on new workers:
   - Deserialize original actor specs
   - Instantiate on least-loaded workers
   - Update actor registry with new worker_ids
4. Actor state is lost (Phase 1 limitation)

**Recovery time:** <35s (30s detection + 5s restart)

**Phase 2 improvement:** Checkpoint actor state to GCS

### Scenario 3: Controller Crash (Phase 1)

**What happens:**
1. Controller crashes - all state lost (in-memory)
2. In-progress tasks fail
3. Workers continue running but can't report results
4. Clients timeout waiting for results

**Recovery:** Requires full restart
- Tasks resubmitted by clients
- Actors recreated by clients

**Phase 2 improvement:** Persist state to external store (Redis/PostgreSQL)

### Scenario 4: Network Partition

**What happens:**
1. Worker isolated from controller
2. Controller health checks fail → worker marked DEAD
3. Controller requeues tasks, restarts actors
4. Worker continues executing (doesn't know it's dead)
5. When network recovers:
   - Worker reports stale results (controller ignores)
   - Worker sees actors already restarted elsewhere

**Mitigation:** Worker should gracefully shutdown if can't reach controller for >60s

---

## Performance Characteristics

### Phase 1 Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Task dispatch latency | <10ms p95 | Controller → Worker RPC |
| Task execution overhead | <50ms | Serialization + dispatch |
| Actor method call latency | <5ms p95 | Controller routing |
| Worker failure detection | 30s | Health check timeout |
| Actor restart time | <5s | Includes worker selection + instantiation |
| Controller task throughput | 1000+ tasks/sec | Limited by serialization, not network |

### Bottlenecks & Optimizations

**Phase 1 Bottlenecks:**
- In-memory state (single controller, no HA)
- cloudpickle serialization (slow for large objects)
- Linear worker scan for assignment

**Phase 2 Optimizations:**
- External state store (Redis) for persistence
- Multiple controller replicas (active-passive)
- Efficient serialization (msgpack for simple types)
- Locality-aware scheduling (prefer workers with warm data)

**Phase 3 Optimizations:**
- Active-active controller cluster (Raft consensus)
- Work-stealing between workers
- Resource-based scheduling (CPU/GPU matching)
- Predictive worker scaling

---

## Usage Examples

### Example 1: Simple Task

```python
# Client code (protocol-agnostic)
ctx = FrayContext("controller-address")

# Submit task
future = ctx.run(lambda x: x * 2, 21)

# Wait for result
result = future.result()  # Blocks until complete
print(f"Result: {result}")  # 42
```

**Behind the scenes:**
1. Client serializes lambda → sends TaskCreate to controller
2. Controller queues task, returns task_id
3. Controller assigns to Worker1 (least loaded)
4. Worker1 executes, returns result to controller
5. Client polls TaskStatus, then fetches result

### Example 2: Stateful Actor

```python
# Define actor
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

# Client code
ctx = FrayContext("controller-address")

# Create actor
counter = ctx.create_actor(Counter, name="global_counter")

# Call methods
result1 = counter.increment.remote()
print(result1.result())  # 1

result2 = counter.increment.remote()
print(result2.result())  # 2
```

**Behind the scenes:**
1. Client serializes Counter class → ActorCreate
2. Controller assigns to Worker2, forwards ActorInstantiate
3. Worker2 creates instance: `counter_obj = Counter()`
4. Client calls `increment.remote()` → ActorCall to controller
5. Controller routes to Worker2 → ActorExecuteMethod
6. Worker2 executes `counter_obj.increment()` → returns result

### Example 3: Failure Recovery

```python
# Submit long-running task
future = ctx.run(expensive_computation, data)

# Worker crashes mid-execution
# ... 30s passes ...

# Controller detects failure, retries task on Worker2
# Client is unaware of retry

result = future.result()  # Eventually completes
```

---

## Summary of Design Decisions

### 1. Push-Based Task Dispatch
**Decision:** Controller assigns tasks to workers (not pull-based polling)

**Rationale:**
- Controller has global view of worker load
- More efficient (no polling overhead)
- Better load balancing (least-loaded assignment)

### 2. Controller-Driven Health Monitoring
**Decision:** Controller polls workers for health (not worker heartbeats)

**Rationale:**
- Controller owns failure detection logic
- Simpler worker implementation
- Controller can adjust health check frequency

### 3. Controller-Owned Retry Logic
**Decision:** Controller handles task retry (not workers)

**Rationale:**
- Centralized retry policy (max_retries, backoff)
- Workers are stateless executors
- Easier to debug (single source of truth)

### 4. Stateless Workers (Phase 1)
**Decision:** Workers maintain minimal state, rely on controller

**Rationale:**
- Simplifies worker implementation
- Makes workers replaceable
- Phase 2 will add actor checkpointing

### 5. In-Memory Controller State (Phase 1)
**Decision:** Controller state is in-memory (not persisted)

**Rationale:**
- Simplest implementation for Phase 1
- Good enough for development/testing
- Phase 2 will add Redis/PostgreSQL backend
