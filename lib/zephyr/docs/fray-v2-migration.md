# Zephyr Execution Architecture

**Related**: [fray-lite-design.md](../../../lib/fray/docs/fray-lite-design.md)

## Overview

Zephyr executes distributed data pipelines using long-lived fray v2 actors in a
coordinator-worker pattern. Workers pull tasks from a central coordinator,
execute shard operations, and report results. This architecture provides:

- **Persistent workers** that cache state (models, tokenizers) across pipeline stages
- **Automatic fault tolerance** with transient error recovery and shard retry
- **First-class shared data broadcast** for distributing read-only context to all workers
- **Backend-agnostic execution** that works on Ray, Iris, and local (in-process)
- **Live progress tracking** with per-worker, per-shard status visibility

## Historical Context

Prior to the fray v2 migration, Zephyr used a `BackendContext` protocol with
stateless Ray tasks. Each `run_stage` call created fresh Ray tasks that could
not hold state across invocations, had no fault tolerance for node preemption,
lacked a shared data model, and tightly coupled Zephyr to Ray. The fray v2
architecture addresses all of these limitations while enabling execution on
multiple backends (Ray, Iris, local).

## Architecture: Pull Model

Workers pull tasks from a coordinator actor in a continuous loop. The coordinator
is the single source of truth for task state, worker state, and progress. All
coordinator state mutations are serialized by the actor framework, eliminating
the need for locks.

```
User code
    │
    ▼
ZephyrContext(client, num_workers, resources)
    │
    ├── .put(name, obj)                    # broadcast shared data
    ├── .execute(dataset, hints)           # run pipeline
    │       │
    │       ▼
    │   ZephyrCoordinator (fray actor, "zephyr-controller")
    │       │   preemptible=False
    │       │
    │       ├── task queue per stage
    │       ├── worker liveness via last_seen timestamps
    │       ├── pull_task() / report_result() / report_error()
    │       ├── check_heartbeats() → re-queue orphaned shards
    │       └── get_status() → live progress
    │
    │   ZephyrWorker actors (fray actor_group, "zephyr-worker-{i}")
    │       │
    │       └── run_loop(coordinator_handle):
    │             loop { pull_task → execute → report_result }
    │             background heartbeat thread keeps last_seen fresh
    │
    └── User process polls coordinator.get_status() for progress display
```

## Detailed Design

### ZephyrContext

The user-facing entry point. Replaces `Backend.execute()`.

```python
@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Creates and manages a coordinator actor and a pool of long-lived worker
    actors. Workers persist across pipeline stages, allowing cached state
    (tokenizers, models) to be reused. Shared data broadcast via put() is
    delivered to workers through the coordinator.
    """

    client: Client
    num_workers: int
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    max_parallelism: int = 1024

    # Shared data: name → object. Delivered to all workers via coordinator.
    _shared_data: dict[str, Any] = field(default_factory=dict)
    _coordinator: ActorHandle | None = None
    _workers: list[ActorHandle] = field(default_factory=list)

    def put(self, name: str, obj: Any) -> None:
        """Register shared data to broadcast to all workers.

        Must be called before execute(). The object must be picklable.
        Workers access it via shard_ctx().get_shared(name).
        """
        self._shared_data[name] = obj

    def execute(
        self,
        dataset: Dataset,
        hints: ExecutionHint = ExecutionHint(),
        verbose: bool = False,
        dry_run: bool = False,
    ) -> Sequence:
        """Execute a dataset pipeline on the worker pool."""
        plan = compute_plan(dataset, hints)
        if dry_run:
            _print_plan(dataset.operations, plan)
            return []

        coordinator = self._get_or_create_coordinator()
        coordinator.set_shared_data(self._shared_data)

        # Start worker loops (non-blocking — workers run as actor tasks)
        for worker in self._workers:
            worker.run_loop.remote(coordinator)

        # Execute stages sequentially
        shards = _build_source_shards(plan.source_items)
        for stage in plan.stages:
            if stage.stage_type == StageType.RESHARD:
                shards = _reshard(shards, stage.output_shards)
                continue

            tasks = _shards_to_tasks(shards, stage, hints)
            coordinator.start_stage(stage.stage_name(), tasks)

            # Poll until stage completes
            while True:
                coordinator.check_heartbeats()
                status = coordinator.get_status()
                _display_status(status)

                # Check for fatal application errors
                if status.get("fatal_error"):
                    raise ZephyrWorkerError(status["fatal_error"])
                if status["completed"] >= status["total"]:
                    break
                time.sleep(1.0)

            shards = coordinator.collect_results()

        return _materialize(shards)

    def _get_or_create_coordinator(self) -> ActorHandle:
        if self._coordinator is None:
            self._coordinator = self.client.create_actor(
                ZephyrCoordinator,
                name="zephyr-controller",
                resources=ResourceConfig(preemptible=False),
            )
            group = self.client.create_actor_group(
                ZephyrWorker,
                name="zephyr-worker",
                count=self.num_workers,
                resources=self.resources,
            )
            self._workers = group.wait_ready()
        return self._coordinator

    def shutdown(self) -> None:
        if self._coordinator is not None:
            # Workers will exit their run_loop when coordinator signals shutdown
            for w in self._workers:
                try:
                    w.shutdown()
                except Exception:
                    pass
            self._coordinator = None
            self._workers = []

    def __enter__(self) -> ZephyrContext:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
```

**Why a context object?** With long-lived actors, lifecycle management is essential.
A context object owns the coordinator and worker actors and can be reused across
multiple `execute()` calls (e.g. multi-stage recipes), allowing workers to cache
state between pipelines.

### ZephyrCoordinator (Actor)

Central coordinator actor. Workers pull tasks from it. All state mutations
happen through actor method calls, which are serialized — no concurrent access,
no locks needed.

```python
class ZephyrCoordinator:

    def __init__(self):
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, list] = defaultdict(list)
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._shared_data: dict[str, Any] = {}
        self._stage_name: str = ""
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._in_flight: dict[str, ShardTask] = {}  # worker_id → task
        self._done: bool = False
        self._fatal_error: str | None = None

    def set_shared_data(self, data: dict[str, Any]) -> None:
        self._shared_data = data

    def get_shared_data(self) -> dict[str, Any]:
        return self._shared_data

    def start_stage(self, stage_name: str, tasks: list[ShardTask]) -> None:
        """Load a new stage's tasks into the queue."""
        self._task_queue = deque(tasks)
        self._results = defaultdict(list)
        self._stage_name = stage_name
        self._total_shards = len(tasks)
        self._completed_shards = 0
        self._retries = 0
        self._in_flight = {}
        self._done = False
        self._fatal_error = None

    def pull_task(self, worker_id: str) -> ShardTask | None:
        """Called by workers to get next task. Returns None when no work available.

        Each pull_task call implicitly acts as a heartbeat.
        """
        self._last_seen[worker_id] = time.monotonic()
        self._worker_states[worker_id] = WorkerState.READY

        if self._done or self._fatal_error:
            return None

        if not self._task_queue:
            return None

        task = self._task_queue.popleft()
        self._in_flight[worker_id] = task
        self._worker_states[worker_id] = WorkerState.BUSY
        return task

    def report_result(self, worker_id: str, shard_idx: int, result: list) -> None:
        self._last_seen[worker_id] = time.monotonic()
        self._results[shard_idx] = result
        self._completed_shards += 1
        self._in_flight.pop(worker_id, None)
        self._worker_states[worker_id] = WorkerState.READY

        if self._completed_shards >= self._total_shards:
            self._done = True

    def report_error(
        self, worker_id: str, shard_idx: int, error_info: str, is_transient: bool
    ) -> None:
        """Worker reports a task failure.

        Transient errors re-queue the shard. Application errors set fatal_error
        so the user process can raise immediately on next poll.
        """
        self._last_seen[worker_id] = time.monotonic()
        task = self._in_flight.pop(worker_id, None)

        if is_transient:
            if task is not None:
                self._task_queue.append(task)
                self._retries += 1
            self._worker_states[worker_id] = WorkerState.READY
        else:
            self._fatal_error = error_info
            self._worker_states[worker_id] = WorkerState.DEAD

    def heartbeat(self, worker_id: str) -> None:
        """Update last_seen. Called by worker heartbeat thread."""
        self._last_seen[worker_id] = time.monotonic()

    def check_heartbeats(self, timeout: float = 30.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        now = time.monotonic()
        for worker_id, last in list(self._last_seen.items()):
            if (
                now - last > timeout
                and self._worker_states.get(worker_id) != WorkerState.DEAD
            ):
                logger.warning(
                    f"Worker {worker_id} heartbeat timeout ({now - last:.1f}s)"
                )
                self._worker_states[worker_id] = WorkerState.FAILED
                task = self._in_flight.pop(worker_id, None)
                if task is not None:
                    self._task_queue.append(task)
                    self._retries += 1

    def get_status(self) -> dict:
        return {
            "stage": self._stage_name,
            "completed": self._completed_shards,
            "total": self._total_shards,
            "retries": self._retries,
            "in_flight": len(self._in_flight),
            "queue_depth": len(self._task_queue),
            "fatal_error": self._fatal_error,
            "workers": {
                wid: {
                    "state": state.value,
                    "last_seen_ago": time.monotonic()
                    - self._last_seen.get(wid, 0),
                }
                for wid, state in self._worker_states.items()
            },
        }

    def collect_results(self) -> dict[int, list]:
        """Return results for the completed stage."""
        return dict(self._results)
```

### ZephyrWorker (Actor)

The actor class that runs on each worker. Holds shared context data and
executes shard tasks pulled from the coordinator.

```python
class ZephyrWorker:
    """Long-lived worker actor. Pulls tasks from coordinator, executes, reports."""

    def __init__(self):
        self._shared_data: dict[str, Any] = {}
        self._shutdown_event = threading.Event()

    def get_shared(self, name: str) -> Any:
        return self._shared_data[name]

    def run_loop(self, coordinator: ActorHandle) -> None:
        """Main worker loop. Pulls tasks from coordinator until done."""
        self._shared_data = coordinator.get_shared_data()
        worker_id = f"worker-{os.getpid()}"

        # Background heartbeat thread keeps last_seen fresh while main
        # thread is blocked in _execute_shard (which can take minutes).
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator, worker_id),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            self._work_loop(coordinator, worker_id)
        finally:
            self._shutdown_event.set()
            heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(
        self,
        coordinator: ActorHandle,
        worker_id: str,
        interval: float = 5.0,
    ) -> None:
        while not self._shutdown_event.is_set():
            try:
                coordinator.heartbeat(worker_id)
            except Exception:
                pass  # Coordinator unreachable; main loop will handle it
            self._shutdown_event.wait(timeout=interval)

    def _work_loop(self, coordinator: ActorHandle, worker_id: str) -> None:
        while not self._shutdown_event.is_set():
            task = coordinator.pull_task(worker_id)
            if task is None:
                # No task available — stage may be done or queue temporarily empty.
                status = coordinator.get_status()
                if status["completed"] >= status["total"] or status.get("fatal_error"):
                    break
                time.sleep(0.5)
                continue

            try:
                result = self._execute_shard(task)
                coordinator.report_result(worker_id, task.shard_idx, result)
            except Exception as e:
                coordinator.report_error(
                    worker_id,
                    task.shard_idx,
                    _serialize_error(e),
                    is_transient=_is_transient_error(e),
                )

    def _execute_shard(self, task: ShardTask) -> list[tuple[dict, bytes]]:
        """Execute a stage's operations on a single shard.

        Returns a list of (header_dict, data) tuples — one per output chunk.
        For file-based pipelines, data is GCS paths (already the case in
        Zephyr's LoadFileOp/WriteOp). For in-memory pipelines, data is
        inline.
        """
        _shard_ctx_var.set(self)

        shard = _reconstruct_shard(task.shard_data, task.shard_idx)
        stage_ctx = StageContext(
            shard=shard,
            shard_idx=task.shard_idx,
            total_shards=task.total_shards,
            chunk_size=task.chunk_size,
            aux_shards=_reconstruct_aux(task.aux_data) if task.aux_data else {},
            execution_context=_InProcessContext(),
        )

        results = []
        for item in run_stage(stage_ctx, task.operations):
            if isinstance(item, ChunkHeader):
                current_header = item
            else:
                results.append((current_header.to_dict(), item))

        return results

    def shutdown(self) -> None:
        self._shutdown_event.set()
```

### shard_ctx() — Worker-Side Context Access

```python
_shard_ctx_var: ContextVar[ZephyrWorker | None] = ContextVar(
    "zephyr_shard_ctx", default=None
)

def shard_ctx() -> ZephyrWorker:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _shard_ctx_var.get()
    if ctx is None:
        raise RuntimeError("shard_ctx() called outside of a worker task")
    return ctx
```

User code accesses shared data inside map/filter/flatmap functions:

```python
from zephyr import shard_ctx

tokenizer = AutoTokenizer.from_pretrained("gpt2")

ctx = ZephyrContext(client, num_workers=8, resources=ResourceConfig(cpu=4, ram="16g"))
ctx.put("tokenizer", tokenizer)

ds = Dataset.from_files("gs://data/*.jsonl").load_jsonl().map(tokenize_fn)
results = ctx.execute(ds)

def tokenize_fn(record):
    tok = shard_ctx().get_shared("tokenizer")
    return {"tokens": tok.encode(record["text"])}
```

### Progress Display

```
[tokenize] 42/100 shards (42%) | 2 retries | 6 workers active
  worker-0: shard-47 [2.3s ago ⣾]  worker-1: shard-48 [0.1s ago ⣷]
  worker-2: shard-49 [1.0s ago ⣯]  worker-3: idle
  worker-4: shard-50 [0.5s ago ⣟]  worker-5: FAILED (heartbeat timeout)
```

The coordinator's `get_status()` returns all information needed for this display.
The spinner character rotates based on `last_seen` timestamps advancing.

### Data Flow

**Shard data transfer:** The coordinator holds shard task descriptions (paths,
metadata). Workers receive these via `pull_task()` and read actual data from
GCS. Results (chunk headers + GCS paths or inline data) are reported back via
`report_result()`. Between stages, the user process collects results from the
coordinator and passes them as tasks for the next stage.

**Shared data:** Broadcast once via `coordinator.set_shared_data()`. Workers
fetch it via `coordinator.get_shared_data()` at the start of `run_loop()`.
For large shared objects (models, tokenizers), this is one RPC per worker.

**Generator streaming:** `run_stage()` returns an iterator, but `_execute_shard()`
collects all chunks and returns them at once. For file-based pipelines (the
common case), chunks are metadata + GCS paths — not raw data — so memory is not
a concern. For in-memory pipelines with very large outputs, the worker writes
intermediate results to GCS and returns paths.

## Design Rationale

### Why Pull Model?

The pull model was chosen over push-based alternatives for several reasons:

- **Natural load balancing**: Fast workers automatically get more tasks
- **Implicit heartbeat**: `pull_task()` calls act as liveness signals, no separate thread needed
- **No locks**: Coordinator state mutations are serialized by the actor framework
- **Simple polling**: User process only polls one actor (`get_status()`) at 1s intervals

Push-based dispatch would require separate heartbeat threads (→ locking), round-robin
load balancing (→ fast workers idle), and polling N futures (→ O(N) overhead per tick).

### Data Serialization

`PhysicalOp` contains arbitrary callables (user-defined map/filter functions)
that must be picklable for actor RPC. This requirement already existed for Ray
execution; fray v2 uses cloudpickle for Iris backend compatibility.

### Shard Data Ownership

Between stages, shard data must be accessible to different workers. Output data
is returned to the coordinator via `report_result()`, the user process collects
results, and passes them as tasks for the next stage. This explicit flow doesn't
depend on a shared object store.

### Join Operations

Sorted merge joins require co-located left and right shard data. The right
sub-plan executes first, results are collected, then passed as auxiliary data
in left-side shard tasks.

### Coordinator Failure

The coordinator runs on a non-preemptible node. If it dies, workers timeout on
their next `pull_task` call and exit cleanly. The user process sees the
coordinator's `JobHandle` fail and surfaces the error.

## Migration History (Completed)

The migration from `Backend`/`BackendContext` to fray v2 actors followed a
spiral design with five phases:

1. **Core Implementation**: Implemented `ZephyrContext`, `ZephyrCoordinator`,
   `ZephyrWorker` with tests on `LocalClient`
2. **Zephyr Tests**: Migrated all zephyr tests from `Backend.execute()` to
   `ZephyrContext.execute()`
3. **CLI + Cleanup**: Updated CLI to use `ZephyrContext`, deleted
   `BackendContext`/`Backend` classes and all Ray-specific machinery
4. **Marin Callers**: Migrated all downstream callers (processing, download,
   transform modules) to use `ZephyrContext`
5. **Validation**: Verified integration tests pass, removed dead code

Key implementation file: `lib/zephyr/src/zephyr/execution.py`

## Component Summary

| Component | Responsibility |
|-----------|---------------|
| `ZephyrContext` | User-facing API. Creates coordinator + workers. Broadcasts shared data. Polls progress. |
| `ZephyrCoordinator` | Actor (non-preemptible). Task queue, worker liveness, result collection. Single source of truth. |
| `ZephyrWorker` | Actor per worker. Pull loop: pull task → execute → report result. Holds shared state. |
| `shard_ctx()` | Worker-side access to shared data. |

## Key Features

- Workers are long-lived and persist across stages
- Shared data broadcast via coordinator `set_shared_data()`
- Transient vs. application error classification (done in worker, reported to coordinator)
- Worker state machine: INIT → READY → BUSY → FAILED (via coordinator)
- Shard retry on transient failure (re-queued in coordinator)
- Application error propagation (coordinator sets `fatal_error`, user process raises)
- Status display with per-worker spinner and heartbeat age
- Works with LocalClient (in-process), RayClient, FrayIrisClient
- No direct Ray imports in Zephyr
- Join operations supported via explicit aux data passing
- Pull model: natural load balancing, implicit heartbeat, no locks
- Coordinator is externally inspectable via `get_status()` (dashboard-friendly)
- Coordinator non-preemptible; workers can be preempted and recovered
