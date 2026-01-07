# Fray: Replacing Ray in Marin - Design Specification & Analysis

## Status

**Phase 1: Storage-First Execution** ✅ COMPLETE

Zephyr has been migrated to a storage-first execution model. See [Implementation Plan](fray_implementation_plan.md) for details.

---

## Background

**Related documents:**
- Ray Infrastructure Challenges
- Marin - Ray Migration
- Marin Infrastructure 2026
- Marin RL Pipeline

**Context:**
Marin's code base currently relies on Ray for distribution. We've documented challenges with Ray extensively, including weak TPU support, slow scheduling, and poor isolation. We have abstracted Ray usage behind `lib/zephyr` (for data processing) and `lib/fray` (for task dispatch via `JobContext`).

## Workload Analysis

Marin uses Ray for three distinct types of computation:

| Workload | Ray Usage | Data Movement |
|Base Workload|Current Ray Usage|Proposed Fray Approach|
|---|---|---|
| **Data processing (Zephyr)** | Task spawn, Object Store | Ray moves data between workers. **Change:** Decouple execution from movement. Stages write tables to storage (GCS/S3). |
| **Training** | Task spawn only | Independent in-task communication (JAX/TPU). **Change:** Task retry is sufficient. |
| **RL** | Actors | Out-of-band (Apache Arrow). **Change:** Actor restart-on-failure. |

**Key Insight:** Preemption handling is trivial for Training and RL (stateless retry or actor restart). Zephyr's complexity can be reduced by observing that stage inputs/outputs are always tables (potentially sharded). This allows decoupling execution from data movement.

## Design

We propose replacing Ray with **Fray** in three phases.

### Phase 1: Storage-First Execution (The "Vortex" Approach)

The simplest correct implementation. Fray provides task and actor execution with trivial retry semantics. Zephyr stages become independent jobs that communicate exclusively via persistent storage.

**Fray Responsibilities:**
- Task dispatch with automatic retry on preemption.
- Actor lifecycle management with restart-on-failure.
- **NO** object store, **NO** distributed reference counting.

**Zephyr Changes:**
- **Refactor `zephyr.backends.Backend`**:
    - Instead of `context.put()` returning an object ref to the next stage, Map phases must write complete output tables (Vortex/Parquet) to GCS/S3.
    - Writers must output a manifest or deterministic path structure.
- Reduce phases read exclusively from materialized tables using `fsspec`.
- Map and Reduce phases run as separate task groups ("Bundle of Tasks") with explicit synchronization at the Scheduler level.

**Architecture:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Mapper  │────▶│ GCS/S3  │────▶│ Reducer │
└─────────┘     └─────────┘     └─────────┘
     │               ▲               │
     │               │               │
     └───────────────┘               ▼
       write complete            read from
         table                   storage
```

**Fault Tolerance:** Trivial.
- Mapper fails: Retry. Overwrites output shard.
- Reducer fails: Retry. Inputs are immutable in storage.

**Tradeoffs:**
- Higher latency (materialization).
- Higher storage costs/IOPS.
- **Benefit:** Extremes simplicity and debuggability.

### Phase 2: Write-Through Caching

Retain storage-first semantics but add a write-through cache layer.

**Changes:**
- Mappers stream writes to both local memory (RAM/tmpfs) and remote storage.
- Reducers prefer reading from Mapper memory via RPC (lower latency).
- On Mapper preemption/failure, Reducers fallback seamlessly to storage.

**Tradeoffs:**
- Lower latency in happy path.
- Same storage reliability.
- Increased complexity (Coordination of memory locations).

### Phase 3: In-Memory Tables

Tables exist primarily in worker memory, spilling only under pressure. Requires `TableConfig` and explicit lifecycle management.

## Integration with Existing Codebase

### Existing JobContext Protocol

Fray already provides a `JobContext` protocol (`lib/fray/src/fray/job/context.py`) that abstracts execution backends:

```python
class JobContext(Protocol):
    def put(self, obj: Any) -> Any           # Store object, return reference
    def get(self, ref: Any) -> Any           # Retrieve from reference
    def run(self, fn: Callable, *args) -> Any  # Execute, return future
    def wait(self, futures, num_returns) -> tuple[list, list]
    def create_actor(self, actor_class, *args, **kwargs) -> Any
```

**Three implementations exist:**
- `SyncContext`: Single-threaded, identity put/get
- `ThreadContext`: ThreadPoolExecutor-based parallelism
- `RayContext`: Ray distributed execution (to be replaced)

### Design Principle

FrayContext should be a **4th JobContext implementation** that:
- Implements the existing protocol (drop-in replacement for RayContext)
- Talks to FrayController via gRPC
- Returns FrayFuture from `run()`
- Uses storage paths instead of object refs for `put()`/`get()` (Phase 1)

This preserves Zephyr's abstraction and allows gradual migration.

## API Reference

### Core Types

```python
class FrayController:
    """Central coordinator for distributed task execution."""
    address: str

    @staticmethod
    def create(hostport: str) -> FrayController

class TableConfig:
    """Configuration for table storage (Phase 2+)."""
    serialization_mode: Literal["IN_MEMORY", "MATERIALIZED", "INLINE"]
    spill_target: str
    spill_threshold_bytes: int
    shards: int
    replicas: int = 1

TaskStatus = Literal["PENDING", "IN_PROGRESS", "FINISHED", "FAILED"]

class FrayFuture[T]:
    """Future representing async task execution."""
    def status(self) -> TaskStatus
    def wait(self) -> TaskStatus
    def result(self) -> T
```

### FrayContext (JobContext Implementation)

```python
class FrayContext:
    """Distributed JobContext implementation using Fray controller."""

    def __init__(self, controller_address: str): ...

    # JobContext protocol
    def put(self, obj: Any) -> Any
    def get(self, ref: Any) -> Any
    def run(self, fn: Callable[..., T], *args, **kwargs) -> FrayFuture[T]
    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]
    def create_actor(self, actor_class: type, *args, **kwargs) -> ActorHandle

    # Zephyr-specific table operations (Phase 2+)
    def create_table(self, f: TableFunction, config: TableConfig) -> ShardedTable
```

### RPC Services

The system uses gRPC for controller-worker communication:

```protobuf
service FrayController {
    rpc TaskCreate(TaskSpec) returns (TaskHandle);
    rpc TaskStatus(TaskHandle) returns (TaskHandle);
    rpc TaskResult(TaskHandle) returns (TaskResult);
    rpc WorkerRegister(WorkerRegisterRequest) returns (WorkerHandle);
}

service FrayWorker {
    rpc ExecuteTask(ExecuteTaskRequest) returns (ExecuteTaskResponse);
    rpc WorkerHealth(HealthRequest) returns (HealthResponse);
}
```

## Migration Path

| Component | Phase 1 ✅ | Phase 2 | Phase 3 |
|---|---|---|---|
| **Training** | ✓ Complete (via Fray JobContext) | No changes | No changes |
| **RL** | ✓ Complete (via Fray Actors) | No changes | No changes |
| **Zephyr** | ✅ **Storage-first** (InlineRef/StorageRef) | Write-through | In-memory |
| **Ray Obj Store** | ✅ **Not used** | Removed | Removed |

---

## Engineer's Analysis & Reflections

### 1. Codebase Reality Check

The current codebase is well-positioned for this migration:

- **JobContext Protocol**: Fray already provides a `JobContext` protocol with three implementations (`SyncContext`, `ThreadContext`, `RayContext`). FrayContext becomes a 4th implementation—this is the cleanest migration path.

- **Zephyr Backend**: `zephyr.backends.Backend` already abstracts execution logic via `JobContext`. It relies on `Chunk` objects wrapping `context.put()` references. The primary refactor will be changing `Chunk` to wrap **storage paths** (e.g., `gs://bucket/job/stage/shard_0.vortex`) instead of object references during Phase 1.

- **Vortex Support**: `zephyr.writers.write_vortex_file` and `zephyr.readers.load_vortex` exist and support `fsspec`. This is a critical enabler for Phase 1.

- **Cluster vs. JobContext**: Important distinction:
  - `Cluster` (`LocalCluster`, `RayCluster`) handles **job submission** (entire processes)
  - `JobContext` handles **in-job execution** (task dispatch within a running job)
  - FrayController is part of the JobContext layer, not the Cluster layer

### 2. Crucial Refactor: `Zephyr.Backend`

Current Zephyr execution flows:
```
Dataset → compute_plan() → PhysicalPlan → Backend._execute_plan → _execute_stage
                                                     ↓
                                        context.run(run_stage, ctx, ops)
                                                     ↓
                                        Worker yields ChunkHeader + data
                                                     ↓
                                        Results reassembled into Shards
```

**Key insight**: The refactor is simpler than it appears. `Shard.iter_chunks()` calls `context.get(chunk.data)`. In Phase 1:
- `chunk.data` becomes a `StorageRef` (path + count)
- `StorageRef.load()` reads from GCS/S3
- No changes to `run_stage()` or the generator protocol

**Risk:** Writing to GCS is slow compared to `context.put()`.
**Mitigation:** Ensure operations are coarse-grained (default chunk_size=100,000 items).

### 3. Deployment Model

**Phase 1 (Storage-First)**:
- No FrayController needed
- Use existing `ThreadContext` or `SyncContext`
- Each shard can run as independent process
- Coordination via storage paths (deterministic naming)

**Phase 2+ (Distributed)**:
- FrayController runs as a standalone service (K8s deployment or head node)
- Workers register at startup, receive task assignments
- Controller handles retry, health monitoring

### 4. Interactive Latency

Phase 1 will be significantly slower for interactive exploration due to GCS round-trips.

**Recommendation:** For development/debugging:
- Use `ThreadContext` with in-memory chunks (default behavior)
- Enable storage mode only for production pipelines
- Consider local disk cache for repeated reads

### 5. Conclusion

The "Storage-First" approach (Phase 1) is a robust, low-risk starting point:
1. It leverages existing GCS/S3 infrastructure
2. It preserves the JobContext abstraction
3. It enables trivial fault tolerance (retry reads from storage)
4. It can be implemented without building the full distributed system

**Recommendation:** Proceed with Phase 1:
1. Add `StorageRef` type to represent storage-backed chunks
2. Update `Shard.iter_chunks()` to handle `StorageRef`
3. Add `storage_path` option to `Backend.execute()`
4. Existing code continues to work unchanged
