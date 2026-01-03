# Fluster: A Modern Cluster Abstraction for Marin

**Status**: Design Proposal (Revised after Principal Engineer Review)
**Authors**: Marin Team
**Last Updated**: 2026-01-02

## Background

### Current Architecture

Marin's distributed computing infrastructure currently relies on three main components:

1. **Ray**: Used for job scheduling, remote task execution, and actor-based coordination
2. **Zephyr**: Our data processing abstraction layer (already Ray-independent for data pipelines)
3. **Fray**: Our emerging abstraction for job and task scheduling

Our codebase uses Ray in several distinct ways:

| Use Case | Files | Pattern |
|----------|-------|---------|
| Data Processing | `processing/classification/inference.py` | `@ray.remote` + manual `ray.wait()` loops |
| Job Execution | `execution/executor.py` | Ray job submission via Fray cluster |
| RL Training | `rl/rl_job.py` | Multiple coordinated TPU jobs via Fray |
| Actor Pools | `processing/classification/autoscaler.py` | Stateful GPU/TPU actors |
| Inference | `generation/inference.py` | vLLM actor pools |

### Existing Fray Implementation

Fray already provides:

- **Cluster Interface** (`fray.cluster.base`): Abstract `Cluster` class with `launch()`, `poll()`, `monitor()`, `wait()`, and `terminate()` methods
- **Job Context** (`fray.job.context`): `JobContext` protocol with `put()`, `get()`, `run()`, `wait()`, and `create_actor()` methods
- **Multiple Backends**: `LocalCluster` (subprocess-based), `RayCluster` (Ray job submission), and an emerging Rust RPC backend
- **Resource Configuration**: `ResourceConfig`, `TpuConfig`, `GpuConfig` for specifying job requirements
- **Environment Configuration**: `EnvironmentConfig` for workspace/docker-based execution

The Rust RPC backend (`lib/fray/src/fray/job/rpc/`) implements a gRPC-based coordinator server with:
- Object store for put/get operations
- Task scheduler for remote execution
- Actor registry and host for stateful services
- Worker pool management

### Zephyr's Execution Model (Critical Context)

Zephyr (`lib/zephyr/`) is our data processing framework that already abstracts away Ray. Understanding its execution model is critical for this design:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Zephyr Pipeline                               │
│   Dataset.from_files(...).flat_map(...).map(...).write_jsonl()  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ compute_plan()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PhysicalPlan                                  │
│   source_items: [SourceItem(shard_idx=0, data=...), ...]        │
│   stages: [Stage(ops=[Map, ForkChunks, Write]), ...]            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Backend.execute()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend._run_tasks()                          │
│   For each shard:                                                │
│     future = context.run(run_stage, ctx, operations)            │
│     # Streaming: iterate generator results                       │
│     while True:                                                  │
│       header = context.get(next(future))  # ChunkHeader         │
│       data_ref = next(future)             # Object reference     │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Zephyr uses **streaming generators** for worker-to-controller data flow. Workers yield `(ChunkHeader, data)` pairs as they produce results, enabling:
1. Backpressure (controller doesn't fetch faster than it can process)
2. Early result availability (don't wait for full shard completion)
3. Memory efficiency (chunks flow through, not buffered)

## Motivation

### Problems with Current Ray Integration

1. **Cluster Management Complexity**: Ray's cluster setup requires significant orchestration, and Ray's scheduler doesn't provide gang scheduling natively - we've built workarounds for TPU slice allocation.

2. **Job Isolation**: Ray clusters share state across jobs. When we want per-user or per-experiment isolation, we must spin up separate Ray clusters, adding operational complexity.

3. **HTTP Interface Limitations**: Ray's dashboard and job submission APIs don't expose a clean HTTP interface for programmatic job management, requiring tunneling or proxying for remote access.

4. **Reference Counting Semantics**: Ray's complex reference counting and ownership semantics create subtle bugs. However, Ray's **generator streaming** pattern (simpler than refcounting) is actually useful.

5. **Dependency on Ray Internals**: Our codebase contains workarounds for Ray quirks scattered throughout (`ray_run`, `ray_deps`, special TPU actor handling).

### Goals

1. **Optionality**: Enable migration away from Ray to alternatives like Monarch for internal execution
2. **Simplicity**: Provide a minimal API surface that covers our actual use cases
3. **Job Isolation**: Support per-user, per-experiment isolation without cluster overhead
4. **Cross-cluster Scheduling**: Enable scheduling across multiple physical clusters/regions
5. **Better Observability**: HTTP-native APIs for job status, logs, and control
6. **Preserve Streaming**: Support Zephyr's generator-based streaming pattern

### Non-Goals

1. We are not building a general-purpose distributed computing framework
2. We are not implementing Ray's complex reference counting semantics
3. We are not replacing Ray for local development - Ray remains useful for single-machine parallelism

## Challenges

### Technical Challenges

1. **Gang Scheduling**: TPU training requires all workers in a slice to start simultaneously. The scheduler must coordinate multi-VM allocation atomically.

2. **Actor State Management**: RL training uses actors for curriculum management and weight transfer coordination. These actors must survive individual worker failures.

3. **Incremental Migration**: We cannot stop development for a rewrite. The new system must coexist with Ray and allow gradual migration.

4. **Performance**: Data transfer between workers (tables, objects) must be efficient enough not to bottleneck data processing pipelines.

5. **Streaming Results**: Zephyr workers stream results via generators. The RPC backend must support this pattern efficiently.

### Organizational Challenges

1. **Codebase Complexity**: Our codebase has evolved organically with Ray assumptions baked in. Configuration inheritance patterns make refactoring difficult.

2. **Testing**: We need to maintain test coverage during migration without doubling test maintenance burden.

## Design

The design introduces three core abstractions: **Fluster** (cluster management), **Fray** (job execution context), and **Tables** (distributed data exchange for shuffle operations).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│   (experiments, training scripts, data processing)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Executor Framework                          │
│   (DAG of ExecutorSteps, versioning, output path management)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼─────┐  ┌───────▼───────┐  ┌─────▼─────┐
│   Fluster     │  │     Fray      │  │   Tables  │
│   (Cluster)   │  │    (Job)      │  │  (Data)   │
│               │  │               │  │           │
│ - create_job  │  │ - put/get     │  │ - write   │
│ - poll/wait   │  │ - run         │  │ - read    │
│ - terminate   │  │ - wait        │  │ - scan    │
│ - list        │  │ - actors      │  │           │
└───────┬───────┘  └───────┬───────┘  └─────┬─────┘
        │                  │                │
┌───────▼──────────────────▼────────────────▼─────────────────────┐
│                    Backend Layer                                 │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│   │ Local       │  │ RPC         │  │ Ray         │             │
│   │ (subprocess)│  │ (Rust/gRPC) │  │ (compat)    │             │
│   └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Fluster: Cluster Management API

Fluster manages job lifecycle across physical clusters. It allocates resources, launches jobs, and monitors their status.

```python
class Fluster(Protocol):
    """Cluster-level job management."""

    def connect(self, namespace: str) -> FlusterConnection:
        """Connect to a cluster with a specific namespace for isolation."""
        ...

    def create_job(
        self,
        request: JobRequest,
    ) -> JobId:
        """Create and launch a job.

        Args:
            request: Job specification including:
                - name: Human-readable job name
                - entrypoint: Command or callable to execute
                - resources: ResourceConfig (CPU/GPU/TPU, RAM, disk)
                - environment: EnvironmentConfig (workspace, docker, pip packages)
        """
        ...

    def poll(self, job_id: JobId) -> JobInfo:
        """Get current job status without blocking."""
        ...

    def wait(self, job_id: JobId | list[JobId]) -> JobInfo | list[JobInfo]:
        """Block until job(s) complete."""
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        ...

    def list_jobs(self, namespace: str | None = None) -> list[JobInfo]:
        """List jobs, optionally filtered by namespace."""
        ...
```

**Execution Flow**:

```
ray run exp123.py
  └─> cluster.run(coordinator)
      └─> boots up initial task on non-preemptible machine
          └─> that task runs w/ CPU & RAM = small
              └─> executor spawns sub-jobs as ExecutorSteps complete
```

The coordinator address comes from the executor process. Fluster sets up the Fray environment (cookie, user, workspace, namespace, coordinator) before launching jobs.

### Fray: Job Execution Context

Fray provides the runtime context within a job for task execution, object storage, and actor management.

```python
class JobContext(Protocol):
    """Job-level execution context."""

    def put(self, obj: Any) -> ObjectRef:
        """Store an object, return a reference."""
        ...

    def get(self, ref: ObjectRef | StreamingRef) -> Any:
        """Retrieve an object or next streaming value from its reference."""
        ...

    def run(self, fn: Callable, *args) -> StreamingRef:
        """Execute a function remotely, return a streaming reference.

        The returned StreamingRef can be:
        1. Passed to wait() for completion checking
        2. Iterated via get(next(ref)) for streaming results
        3. Used with get() to retrieve final result (for non-generators)

        Generator functions yield results incrementally, each accessible
        via successive get(next(ref)) calls.
        """
        ...

    def wait(self, futures: list[StreamingRef], num_returns: int = 1) -> tuple[list, list]:
        """Wait for streaming refs to have available results.

        Returns (ready, pending) where ready refs have at least one
        result available for get(next(ref)).
        """
        ...

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["job", "detached"] = "job",
        preemptible: bool = True,
        **kwargs,
    ) -> ActorHandle:
        """Create a stateful actor.

        Named actors enable discovery across workers:

            # Worker 1: Create
            curriculum = ctx.create_actor(Curriculum, config, name="curriculum")

            # Worker 2: Get same instance
            curriculum = ctx.create_actor(Curriculum, config, name="curriculum",
                                          get_if_exists=True)
        """
        ...
```

**Key Design Decisions**:

1. **No implicit `@ray.remote`**: Unlike Ray, functions are not decorated. Remote execution is explicit via `ctx.run()`.

2. **Streaming-first `run()`**: The return type is `StreamingRef`, not just `Future`. This supports both completion-waiting AND incremental result streaming.

3. **Shared namespace for actors**: Jobs within the same namespace can discover named actors, enabling coordination across workers.

4. **Gang scheduling for TPU**: TPU jobs are gang-scheduled automatically - all VMs in a slice start together.

### Streaming Protocol (Critical for Zephyr)

The streaming protocol enables workers to yield results incrementally:

```python
# Worker (generator function)
def run_stage(ctx: StageContext, ops: list[PhysicalOp]) -> Iterator[ChunkHeader | list]:
    stream = iter(ctx.shard)
    for op in ops:
        stream = apply_op(op, stream)

    # Yield chunks as they're produced
    chunk = []
    for item in stream:
        chunk.append(item)
        if len(chunk) >= ctx.chunk_size:
            yield ChunkHeader(shard_idx=ctx.shard_idx, count=len(chunk))
            yield chunk
            chunk = []
    if chunk:
        yield ChunkHeader(shard_idx=ctx.shard_idx, count=len(chunk))
        yield chunk

# Controller (backend)
def _run_tasks(self, contexts: list[StageContext], ops: list[PhysicalOp]):
    active_gens = []
    for ctx in contexts:
        active_gens.append(self.context.run(run_stage, ctx, ops))

    while active_gens:
        ready, _ = self.context.wait(active_gens, num_returns=1)
        for gen in ready:
            try:
                header = self.context.get(next(gen))   # ChunkHeader
                data_ref = next(gen)                    # Object reference to chunk
                yield header, data_ref
            except StopIteration:
                active_gens.remove(gen)
```

**RPC Implementation**:

```protobuf
service Coordinator {
    // Streaming task execution
    rpc SubmitStreamingTask(Task) returns (StreamHandle);
    rpc StreamNext(StreamHandle) returns (StreamItem);  // Returns header or data

    // Object store
    rpc Put(PyObject) returns (ObjectRef);
    rpc Get(ObjectRef) returns (PyObject);

    // Completion waiting
    rpc WaitStreams(WaitStreamsRequest) returns (WaitStreamsResponse);

    // Actor management
    rpc CreateActor(ActorSpec) returns (ActorRef);
    rpc GetActorByName(GetActorByNameRequest) returns (ActorRef);
    rpc CallActorMethod(ActorMethodCall) returns (TaskRef);
}

message StreamItem {
    oneof item {
        bytes data = 1;        // Pickled object
        bool exhausted = 2;    // Stream complete
    }
}
```

### Tables: Distributed Data Exchange

Tables provide a [Piccolo](https://www.usenix.org/conference/osdi10/piccolo-building-fast-distributed-programs-partitioned-tables)-inspired abstraction for **shuffle operations** (scatter/gather). They are NOT a replacement for the object store - they complement it for specific use cases.

**When to use Tables vs Object Store**:

| Pattern | Abstraction | Example |
|---------|-------------|---------|
| Pass data to worker | Object Store (`put/get`) | Zephyr shard data |
| Stream results back | Streaming `run()` | Zephyr chunk results |
| Shuffle (many-to-many) | Tables | GroupBy, Scatter |
| Intermediate shuffle data | Tables with GCS backend | Large dataset shuffle |

```python
@dataclass
class TableConfig:
    """Configuration for a distributed table."""
    name: str
    partition_fn: Callable[[Any], int]  # e.g., xxhash
    num_partitions: int
    storage: Literal["memory", "gcs", "local"]

class Table(Protocol):
    """Distributed key-value table with partitioned writes."""

    def write(self, partition: int, key: Any, value: Any) -> None:
        """Write a key-value pair to a specific partition.

        The caller determines the partition (typically via hash).
        """
        ...

    def write_batch(self, partition: int, items: list[tuple[Any, Any]]) -> None:
        """Write multiple items to a partition (more efficient)."""
        ...

    def read(self, partition: int) -> Iterator[tuple[Any, Any]]:
        """Read all entries in a partition."""
        ...

    def flush(self) -> None:
        """Ensure all writes are persisted."""
        ...
```

**Replay Semantics**:

Each table entry is tagged with `(source_task_id, sequence_number)`. On task replay:
1. The table discards entries from the failed task's `source_task_id`
2. The replayed task writes fresh entries
3. Downstream tasks re-read the partition

```python
# Internal entry format
@dataclass
class TableEntry:
    source_task_id: TaskId
    sequence_number: int
    key: Any
    value: Any
```

### Zephyr Backend Walkthrough

Here's how Zephyr's backend works with the new system:

```python
# lib/zephyr/src/zephyr/backends.py (updated for Fluster)

class Backend:
    def __init__(self, context: JobContext, config: BackendConfig):
        self.context = context
        self.config = config

    def _shards_from_source_items(self, source_items: list[SourceItem]) -> list[Shard]:
        """Create Shards from SourceItems.

        Each SourceItem's data is stored via put() and wrapped in a Chunk.
        """
        items_by_shard: dict[int, list[SourceItem]] = defaultdict(list)
        for item in source_items:
            items_by_shard[item.shard_idx].append(item)

        shards = []
        for shard_idx in sorted(items_by_shard.keys()):
            items = items_by_shard[shard_idx]
            chunks = []
            for item in items:
                # Store item data, get reference back
                ref = self.context.put([item.data])
                chunks.append(Chunk(count=1, data=ref))
            shards.append(Shard(idx=shard_idx, chunks=chunks, context=self.context))
        return shards

    def _run_tasks(
        self,
        contexts: list[StageContext],
        operations: list[PhysicalOp],
    ) -> dict[int, list[tuple[ChunkHeader, Any]]]:
        """Run stage tasks, streaming results back.

        This is where the streaming protocol is critical:
        1. Submit all tasks via context.run() - returns StreamingRefs
        2. Wait for any ref to have results available
        3. Pull header/data pairs via next() iteration
        4. Repeat until all generators exhausted
        """
        results_by_shard: dict[int, list[tuple[ChunkHeader, Any]]] = defaultdict(list)

        if not contexts:
            return results_by_shard

        # Launch all tasks - each returns a StreamingRef (generator)
        active_gens: list[tuple[StreamingRef, StageContext]] = []
        for ctx in contexts:
            gen = self.context.run(run_stage, ctx, operations)
            active_gens.append((gen, ctx))

        # Process streaming results
        while active_gens:
            gen_refs = [g for g, _ in active_gens]
            ready, _ = self.context.wait(gen_refs, num_returns=1)

            for ready_gen in ready:
                # Find the matching context
                for g, ctx in active_gens:
                    if g is ready_gen:
                        try:
                            # Pull next header/data pair from generator
                            header = self.context.get(next(ready_gen))
                            data_ref = next(ready_gen)  # Don't get() yet - keep as ref
                            results_by_shard[header.shard_idx].append((header, data_ref))
                        except StopIteration:
                            # Generator exhausted
                            active_gens.remove((g, ctx))
                        break

        return results_by_shard

    def _execute_stage(
        self,
        stage: Stage,
        shards: list[Shard],
        hints: ExecutionHint,
    ) -> list[Shard]:
        """Execute a single stage on shards."""
        # Reshard: just redistribute refs (no worker execution)
        if stage.stage_type == StageType.RESHARD:
            return reshard_refs(shards, stage.output_shards or len(shards))

        # Build StageContext for each shard
        contexts = [
            StageContext(
                shard=shard,
                shard_idx=shard.idx,
                total_shards=len(shards),
                chunk_size=hints.chunk_size,
                aux_shards=self._get_aux_shards(stage, shard.idx),
                execution_context=self.context,
            )
            for shard in shards
        ]

        # Run tasks with streaming
        results = self._run_tasks(contexts, stage.operations)

        # Assemble output shards from streamed chunks
        output_shards = []
        for idx in range(len(shards)):
            if idx in results:
                chunks = [Chunk(h.count, ref) for h, ref in results[idx]]
                output_shards.append(Shard(idx=idx, chunks=chunks, context=self.context))
            else:
                output_shards.append(Shard(idx=idx, chunks=[], context=self.context))

        return output_shards
```

**Key Points**:

1. **put/get for shard data**: Source items are stored via `put()`, workers retrieve via `get()` inside `StageContext.shard`.

2. **Streaming for results**: Workers yield `(header, data)` pairs. The backend pulls these incrementally via `next()` on the `StreamingRef`.

3. **Data stays as references**: The backend stores `data_ref` (not `get(data_ref)`) in results. Materialization happens only when iterating final results.

4. **Scatter uses inline yielding**: Zephyr's `Scatter` operation yields chunks to different shard indices directly in the generator - no Table needed for this case.

### When Tables Are Needed

Tables are specifically for **shuffle persistence** scenarios:

```python
# Scenario: Large dataset shuffle that doesn't fit in memory
def scatter_to_table(
    ctx: JobContext,
    table: Table,
    input_shard: Shard,
    key_fn: Callable,
    num_output_shards: int,
):
    """Write items to table partitions based on key hash."""
    batch_by_partition: dict[int, list] = defaultdict(list)

    for item in input_shard:
        partition = hash(key_fn(item)) % num_output_shards
        batch_by_partition[partition].append((key_fn(item), item))

        # Flush batches periodically
        if len(batch_by_partition[partition]) >= 10000:
            table.write_batch(partition, batch_by_partition[partition])
            batch_by_partition[partition] = []

    # Flush remaining
    for partition, items in batch_by_partition.items():
        if items:
            table.write_batch(partition, items)

    table.flush()

def gather_from_table(
    ctx: JobContext,
    table: Table,
    partition: int,
) -> Iterator:
    """Read items from a table partition."""
    for key, value in table.read(partition):
        yield value
```

### RPC Layer

All Fray operations are accessible via HTTP/2 (gRPC) APIs:

```protobuf
service Coordinator {
    // Object store
    rpc Put(PyObject) returns (ObjectRef);
    rpc Get(ObjectRef) returns (PyObject);

    // Streaming task execution (replaces simple SubmitTask)
    rpc SubmitStreamingTask(StreamingTask) returns (StreamHandle);
    rpc StreamNext(StreamNextRequest) returns (StreamItem);
    rpc WaitStreams(WaitStreamsRequest) returns (WaitStreamsResponse);

    // Table operations
    rpc CreateTable(CreateTableRequest) returns (TableRef);
    rpc TableWriteBatch(TableWriteBatchRequest) returns (TableWriteBatchResponse);
    rpc TableRead(TableReadRequest) returns (stream TableEntry);
    rpc TableFlush(TableFlushRequest) returns (TableFlushResponse);

    // Actor management
    rpc CreateActor(ActorSpec) returns (ActorRef);
    rpc GetActorByName(GetActorByNameRequest) returns (ActorRef);
    rpc CallActorMethod(ActorMethodCall) returns (TaskRef);

    // Worker management
    rpc RegisterWorker(RegisterWorkerRequest) returns (RegisterWorkerResponse);
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
}
```

### Migration Path

The migration proceeds in phases:

**Phase 1: Resource Configuration** (Complete)
- Unified `ResourceConfig` across codebase
- TPU/GPU/CPU configurations standardized

**Phase 2: Job Launch** (In Progress)
- Replace `ray_run` with Fray `cluster.launch()`
- Executor uses Fray for job submission
- Ray still runs within jobs

**Phase 3: Streaming RPC Backend**
- Implement `SubmitStreamingTask` / `StreamNext` in Rust coordinator
- Enable Zephyr to use RPC backend for task execution
- Validate streaming performance matches Ray

**Phase 4: Table Introduction**
- Implement `fray.table` with GCS backend
- Use tables for large shuffles (optional optimization)
- Tables complement, don't replace, existing object store

**Phase 5: Cluster Backend**
- Replace Ray cluster management with Fluster
- Native TPU gang scheduling
- Per-user job isolation

## Concerns and Open Questions

### Concern 1: Streaming Performance over RPC

**Issue**: The streaming protocol requires frequent RPC round-trips (`StreamNext` per chunk). This may introduce latency compared to Ray's in-process generator iteration.

**Mitigation Options**:
1. Batch multiple chunks per `StreamNext` call
2. Use bidirectional streaming RPC
3. Keep Ray backend as fallback for latency-sensitive workloads

**Status**: Needs benchmarking before Phase 3.

### Concern 2: Table vs Object Store Boundary

**Issue**: The design has both object store (`put/get`) and tables (`write/read`). When should code use which?

**Resolution**:
- Object store: Pass data to workers, retrieve results
- Tables: Only for shuffle operations that need persistence/replay

This is now documented but may need clearer API guidance.

### Concern 3: ForkChunks / Intra-Shard Parallelism

**Issue**: Zephyr's `ForkChunks` spawns multiple parallel tasks within a shard, using `context.run()` recursively. The design doesn't address how nested streaming works.

**Current Behavior** (in Zephyr):
```python
def _execute_fork_join(context, stream, parallel_ops, target_chunks):
    # Split stream into chunks
    chunks = list(chunked(stream, chunk_size))

    # Launch parallel tasks
    futures = [context.run(_run_chunk_ops, chunk, parallel_ops) for chunk in chunks]

    # Merge results as available
    while futures:
        ready, futures = context.wait(futures, num_returns=1)
        for f in ready:
            yield from context.get(f)
```

**Status**: This should work with StreamingRef, but needs explicit testing.

### Concern 4: Join Right-Side Distribution

**Issue**: Zephyr joins require the right-side shard to be passed to each left-side worker via `StageContext.aux_shards`. How is this coordinated?

**Current Behavior**: Right-side shards are executed first, results stored as refs, then passed to left-side workers.

**Status**: Works with object store pattern. No changes needed.

### Concern 5: Namespace Isolation Semantics

**Issue**: The design mentions "namespace" for job isolation but doesn't specify:
- Do namespaces share object store?
- Can actors in different namespaces communicate?
- What happens to detached actors when namespace is cleaned up?

**Proposed Semantics**:
- Namespaces provide separate object stores
- Actors are namespace-scoped (even detached ones)
- Cross-namespace communication requires explicit Tables or external storage

**Status**: Needs detailed design.

### Concern 6: Coordinator Single Point of Failure

**Issue**: The Rust coordinator is a single process. What happens when it fails?

**Mitigation Options**:
1. Coordinator checkpoints state to GCS
2. Workers reconnect to new coordinator on restart
3. In-flight tasks are replayed (Tables enable this)

**Status**: Needs detailed design for production deployment.

## Alternatives Considered

### Alternative 1: Fray Cluster + Ray Backend

Instead of building our own RPC, have Fray cluster start up a Ray backend, booting workers into a configured Ray environment.

**Pros**:
- Incremental: Replace cluster logic without changing job internals
- Lower initial effort

**Cons**:
- **User complexity**: Each user job gets a separate Ray cluster. Without public HTTP interface, tracking logs/status requires tunneling.
- **Limited benefit**: Still using Ray for critical functionality (actors, object store).
- **Operational overhead**: More Ray clusters to manage.

**Decision**: Rejected. The complexity of managing multiple Ray clusters negates the benefit.

### Alternative 2: Implement Full Ray Semantics

Implement Ray's generator protocol and reference counting in Fray.

**Pros**:
- Full compatibility with existing Ray code
- No migration effort for job internals

**Cons**:
- **Complexity**: Ray's refcount semantics are complex and partly broken. Bug-for-bug compatibility is expensive.
- **Limited upside**: We only need the streaming pattern, not full generator semantics.

**Decision**: Rejected. We implement streaming only, not full Ray generator protocol.

### Alternative 3: Keep Ray

Accept Ray's limitations and continue containment strategy (Zephyr, Fray abstractions).

**Pros**:
- No migration effort
- Battle-tested infrastructure

**Cons**:
- **Cluster management pain**: Job isolation, cross-cluster scheduling remain difficult.
- **Observability gaps**: No clean HTTP APIs for job management.
- **Technical debt**: Ray workarounds scattered throughout codebase.

**Decision**: Rejected for long-term, but Ray remains supported during migration.

### Alternative 4: Tables for Everything (Rejected)

Replace object store with Tables entirely.

**Pros**:
- Single abstraction for all data exchange
- Uniform replay semantics

**Cons**:
- **Semantic mismatch**: Zephyr's shard data is not key-value. Forcing it into Tables adds complexity.
- **Performance**: Tables add partitioning overhead for simple pass-through data.

**Decision**: Rejected. Tables complement object store for shuffle operations only.

## References

- [Piccolo: Building Fast, Distributed Programs with Partitioned Tables](https://www.usenix.org/conference/osdi10/piccolo-building-fast-distributed-programs-partitioned-tables) - OSDI 2010
- [Fray Design Document](./fray-migration.md)
- [Zephyr Migration Patterns](./zephyr-migration.md)
- Existing Fray implementation: `lib/fray/`
- Zephyr backend: `lib/zephyr/src/zephyr/backends.py`
