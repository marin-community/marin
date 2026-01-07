# Implementation Plan: Fray Migration - Ray Replacement

## Background

### Source Documents

- **Design Specification**: [fray_design_spec.md](fray_design_spec.md)
- **Architecture Flows**: [fray_architecture_flows.md](fray_architecture_flows.md)

**Goal**: Replace Ray with Fray's storage-first execution model, eliminating dependency on Ray's object store while maintaining compatibility for training, RL, and data processing workloads.

**Scope**: This plan covers Phase 1 (Storage-First Execution) and Phase 2 (Distributed Tasks) of the Fray migration.

---

## Phase 1: Storage-First Execution ✅ COMPLETE

### Summary

Phase 1 has been completed. Zephyr now uses a storage-first execution model where:
- Chunks are represented by `ChunkRef` types (`InlineRef` for small data, `StorageRef` for spilled data)
- Workers make spill decisions based on configurable size thresholds
- The generator-based streaming protocol has been replaced with direct return values
- Data flows through storage rather than object references

### Completed Work

#### Stage 0: Storage-First Chunk References ✅

**Created `lib/zephyr/src/zephyr/storage.py`:**
- `InlineRef` - Data kept inline in memory (small chunks)
- `StorageRef` - Reference to data stored in GCS/S3/local filesystem
- `ChunkWriter` - Writes chunk data, choosing inline vs storage based on size threshold
- `StorageManager` - Manages storage paths and cleanup for a job execution

#### Stage 1: Storage-First Backend Option ✅

**Updated `lib/zephyr/src/zephyr/backends.py`:**
- `Backend.__init__` accepts `StorageManager`
- `Backend.execute()` has `storage_path` and `spill_threshold_bytes` parameters
- `Shard.iter_chunks()` handles both `InlineRef` and `StorageRef` transparently
- Workers return `list[tuple[ChunkHeader, ChunkRef]]` instead of generators

**Updated `lib/zephyr/src/zephyr/plan.py`:**
- `run_stage()` returns `list[tuple[ChunkHeader, ChunkRef]]` directly
- `_collect_chunks()` uses `ChunkWriter` with spill decisions
- `StageContext` includes `storage_path` and `spill_threshold_bytes` for worker decisions

#### Stage 1.5: Context Module Refactor ✅

**Split `lib/fray/src/fray/job/context.py` into isolated modules:**
- `sync_ctx.py` - `SyncContext` with `_SyncFuture`, `_SyncActorHandle`, `_SyncActorMethod`
- `threadpool_ctx.py` - `ThreadContext` with `_GeneratorFuture`, `_ThreadActorHandle`, `_ThreadActorMethod`
- `ray_ctx.py` - `RayContext` for Ray distributed execution
- `context.py` - `JobContext` protocol, `ContextConfig`, factory functions only

Each context is fully isolated with its own internal classes - no cross-imports between contexts.

### Architecture After Phase 1

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Backend     │────▶│ StorageRef  │────▶│ GCS/S3/     │
│ (Controller)│     │ or InlineRef│     │ Local FS    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        ▲
      │ context.run(run_stage)                 │
      ▼                                        │
┌─────────────┐                                │
│ Worker      │ ─────(spill if large)──────────┘
│ (run_stage) │
└─────────────┘
```

---

## Phase 2: Distributed Task Execution ✅ COMPLETE

Phase 2 adds a new RPC-based `JobContext` implementation for distributed execution using Connect RPC.

### Implementation Note: Connect RPC vs gRPC

Phase 2 was implemented using **Connect RPC** instead of standard gRPC. Connect RPC provides:
- HTTP/1.1 and HTTP/2 support (vs gRPC's HTTP/2-only requirement)
- Browser compatibility via Connect-Web
- Simpler deployment (works with standard HTTP servers like uvicorn)
- Better compatibility with Python async/await patterns
- JSON and binary protobuf wire formats

The implementation uses:
- `connectrpc` Python package for RPC framework
- `uvicorn` for serving the ASGI application
- Standard Protocol Buffers for message definitions
- `betterproto` for Python code generation (cleaner dataclasses than grpcio)

### Directory Structure

```
lib/fray/src/fray/job/
├── context.py          # JobContext protocol + factory (existing)
├── sync_ctx.py         # SyncContext (existing)
├── threadpool_ctx.py   # ThreadContext (existing)
├── ray_ctx.py          # RayContext (existing)
└── rpc/                # NEW: RPC-based distributed context
    ├── __init__.py
    ├── context.py      # FrayContext - JobContext implementation ✅
    ├── controller.py   # FrayController - task queue, worker registry ✅
    ├── worker.py       # FrayWorker - task executor ✅
    └── proto/          # Protocol definitions ✅
        ├── fray.proto
        └── (generated files)
```

### Stage 2: Protocol Buffer Definitions ✅

#### Objective

Define Connect RPC service API for controller-worker communication.

#### Changes

- [x] Create `lib/fray/src/fray/job/rpc/proto/fray.proto`:

```protobuf
syntax = "proto3";

package fray;

enum TaskStatus {
  TASK_STATUS_UNSPECIFIED = 0;
  TASK_STATUS_PENDING = 1;
  TASK_STATUS_IN_PROGRESS = 2;
  TASK_STATUS_FINISHED = 3;
  TASK_STATUS_FAILED = 4;
}

message TaskSpec {
  bytes serialized_data = 1;  // cloudpickle(fn, args, kwargs)
  map<string, int32> resources = 2;
  int32 max_retries = 3;
}

message TaskHandle {
  string task_id = 1;
  TaskStatus status = 2;
  string worker_id = 3;
  string error = 4;
}

message TaskResult {
  bytes result_data = 1;
}

message WorkerInfo {
  string worker_id = 1;
  string address = 2;
  int32 cpu_cores = 3;
  int64 memory_bytes = 4;
}

message Empty {}

service FrayController {
  rpc SubmitTask(TaskSpec) returns (TaskHandle);
  rpc GetTaskStatus(TaskHandle) returns (TaskHandle);
  rpc GetTaskResult(TaskHandle) returns (TaskResult);
  rpc RegisterWorker(WorkerInfo) returns (Empty);
  rpc ReportTaskComplete(TaskHandle) returns (Empty);
  rpc ReportTaskFailed(TaskHandle) returns (Empty);
}

service FrayWorker {
  rpc ExecuteTask(TaskSpec) returns (TaskResult);
  rpc HealthCheck(Empty) returns (Empty);
}
```

- [x] Add connectrpc dependencies to `lib/fray/pyproject.toml`
- [x] Create `lib/fray/scripts/generate_proto.sh`

#### Validation

- [x] Proto files compile without errors
- [x] Generated Python files importable

---

### Stage 3: FrayContext Implementation ✅

#### Objective

Implement `FrayContext` as a `JobContext` that talks to `FrayController` via Connect RPC.

#### Changes

- [x] Create `lib/fray/src/fray/job/rpc/__init__.py`
- [x] Create `lib/fray/src/fray/job/rpc/context.py`:

```python
"""FrayContext - RPC-based distributed JobContext implementation."""

import cloudpickle
from collections.abc import Callable
from typing import Any

from fray.job.rpc.proto.fray_connect import FrayControllerClientSync

class _FrayFuture:
    """Future for RPC-based task execution."""

    def __init__(self, task_id: str, client: FrayControllerClientSync):
        self._task_id = task_id
        self._client = client
        self._result = None
        self._done = False

    def result(self, timeout: float | None = None) -> Any:
        """Poll for task completion and return result."""
        if not self._done:
            # Poll task status until complete
            ...
        return self._result

    def done(self) -> bool:
        """Check if task is complete."""
        ...


class FrayContext:
    """JobContext implementation using Connect RPC controller/worker."""

    def __init__(self, controller_address: str):
        if not controller_address.startswith("http"):
            controller_address = f"http://{controller_address}"
        self._client = FrayControllerClientSync(controller_address)

    def put(self, obj: Any) -> Any:
        return obj  # Storage-first: identity

    def get(self, ref: Any) -> Any:
        if isinstance(ref, _FrayFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _FrayFuture:
        payload = {"fn": fn, "args": args}
        serialized_fn = cloudpickle.dumps(payload)
        task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)
        handle = self._client.submit_task(task_spec)
        return _FrayFuture(handle.task_id, self._client)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        # Poll futures until num_returns are done
        ...

    def create_actor(self, actor_class: type, *args, **kwargs) -> Any:
        raise NotImplementedError("Actors not yet supported")
```

- [x] Update `lib/fray/src/fray/job/context.py` to add `"fray"` context type:

```python
def create_job_ctx(
    context_type: Literal["ray", "threadpool", "sync", "fray", "auto"] = "auto",
    max_workers: int = 1,
    controller_address: str | None = None,
    **ray_options,
) -> JobContext:
    ...
    elif context_type == "fray":
        if controller_address is None:
            raise ValueError("controller_address required for fray context")
        from fray.job.rpc.context import FrayContext
        return FrayContext(controller_address)
```

#### Validation

- [x] `FrayContext` implements `JobContext` protocol
- [x] Can submit tasks to controller
- [x] `wait()` polls and returns when tasks complete

---

### Stage 4: Controller Implementation ✅

#### Objective

Implement `FrayController` Connect RPC service with task queue and worker registry.

#### Changes

- [x] Create `lib/fray/src/fray/job/rpc/controller.py`:

```python
"""FrayController - Central coordinator for distributed task execution."""

import threading
import uuid
from collections import deque
from dataclasses import dataclass

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayController, FrayControllerASGIApplication

@dataclass
class Task:
    task_id: str
    serialized_fn: bytes
    status: int  # fray_pb2.TaskStatus enum value
    result: bytes | None = None
    error: str | None = None
    worker_id: str | None = None


class FrayControllerServicer(FrayController):
    """Connect RPC servicer implementing the FrayController service."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._pending_queue: deque[str] = deque()
        self._workers: dict[str, fray_pb2.WorkerInfo] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    async def submit_task(self, request, ctx):
        # Create task, add to pending queue
        ...

    async def get_task_status(self, request, ctx):
        ...

    async def get_task_result(self, request, ctx):
        ...


class FrayControllerServer:
    """Wraps the FrayControllerServicer in an ASGI server using uvicorn."""

    def __init__(self, port: int = 0):
        self._port = port
        self._servicer = FrayControllerServicer()
        self._app = FrayControllerASGIApplication(self._servicer)

    def start(self) -> int:
        # Start uvicorn server in background thread
        ...

    def stop(self, grace: float = 5.0):
        ...
```

#### Validation

- [x] Controller starts and accepts Connect RPC connections
- [x] Tasks queue and assign to workers
- [x] Task status updates correctly

---

### Stage 5: Worker Implementation ✅

#### Objective

Implement `FrayWorker` that executes assigned tasks.

#### Changes

- [x] Create `lib/fray/src/fray/job/rpc/worker.py`:

```python
"""FrayWorker - Task executor for distributed execution."""

import cloudpickle
from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClientSync

class FrayWorker:
    """Worker that registers with controller and executes tasks."""

    def __init__(self, controller_address: str, worker_id: str | None = None, port: int = 0):
        self.worker_id = worker_id or str(uuid.uuid4())
        self._port = port
        self._running = False

        if not controller_address.startswith("http"):
            controller_address = f"http://{controller_address}"
        self.controller_address = controller_address

        self._controller_client = FrayControllerClientSync(controller_address)

    def register(self):
        """Register with controller."""
        worker_info = fray_pb2.WorkerInfo(
            worker_id=self.worker_id,
            address=self.address,
            num_cpus=os.cpu_count() or 1,
            memory_bytes=0,
        )
        self._controller_client.register_worker(worker_info)

    def _execute_task(self, task: fray_pb2.TaskSpec) -> fray_pb2.TaskResult:
        """Execute a task and return serialized result."""
        task_data = cloudpickle.loads(task.serialized_fn)
        fn = task_data["fn"]
        args = task_data["args"]
        result = fn(*args)
        serialized_result = cloudpickle.dumps(result)
        return fray_pb2.TaskResult(
            task_id=task.task_id,
            serialized_result=serialized_result,
        )

    def run(self):
        """Main worker loop - poll for tasks, execute, report results."""
        self.register()
        while self._running:
            # Get task from controller
            # Execute
            # Report result
            ...
```

#### Validation

- [x] Worker registers with controller
- [x] Worker executes assigned tasks
- [x] Results reported back to controller
- [x] End-to-end test: submit task via `FrayContext`, execute on worker

---

### Stage 6: Integration Testing ✅

#### Validation

- [x] End-to-end tests pass
- [x] Can create FrayContext via `create_job_ctx("fray", controller_address="...")`
- [x] Tasks execute successfully across controller and worker
- [x] Error handling works correctly (failed tasks)

---

### Stage 7: Documentation ✅

#### Changes

- [x] Updated implementation plan to mark Phase 2 complete
- [x] Documented Connect RPC migration rationale
- [x] Added comprehensive module docstrings
- [x] Integrated FrayContext into `create_job_ctx()` factory

---

## Stage Dependencies

```
Phase 1 ✅ COMPLETE
─────────────────────────────────────────────────────────────
Stage 0 (Storage Refs) ✅ ──► Stage 1 (Storage Backend) ✅
                                        │
Stage 1.5 (Context Refactor) ✅ ────────┘

Phase 2 (RPC Backend) ✅ COMPLETE
─────────────────────────────────────────────────────────────
Stage 2 (Proto) ✅ ──► Stage 3 (FrayContext) ✅ ──► Stage 4 (Controller) ✅
                                                            │
                                                            ▼
                                                    Stage 5 (Worker) ✅
                                                            │
                                                            ▼
                                                  Stage 6 (Integration) ✅
                                                            │
                                                            ▼
                                                  Stage 7 (Documentation) ✅
```

---

## Migration Path

| Component | Phase 1 ✅ | Phase 2 (Future) |
|-----------|---------|---------|
| **Training** | ThreadContext ✅ | FrayContext (optional) |
| **RL** | Existing contexts ✅ | FrayContext with actors |
| **Zephyr** | Storage-first Backend ✅ | Storage + FrayContext |
| **Ray Object Store** | Not used (ChunkRef) ✅ | Removed |

---

## Usage After Phase 2 ✅

```python
# Local testing (existing)
ctx = create_job_ctx("sync")
ctx = create_job_ctx("threadpool", max_workers=4)

# Ray (existing)
ctx = create_job_ctx("ray")

# Fray distributed (Phase 2) - using Connect RPC
ctx = create_job_ctx("fray", controller_address="http://localhost:50051")

# Example: Submit and execute tasks
def add_numbers(a, b):
    return a + b

future = ctx.run(add_numbers, 5, 3)
result = ctx.get(future)  # Returns 8
```

---

## See Also

- [Fray Design Specification](fray_design_spec.md)
- [Fray Architecture Flows](fray_architecture_flows.md)
- `AGENTS.md` - Project coding guidelines
