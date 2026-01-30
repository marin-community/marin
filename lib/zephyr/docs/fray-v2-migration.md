# Zephyr Fray v2 Migration Design

**Status**: Draft
**Related**: [fray-lite-design.md](../../../lib/fray/docs/fray-lite-design.md)

## Problem

Zephyr currently uses a `BackendContext` protocol with three implementations:
`SyncBackendContext`, `ThreadBackendContext`, and `RayBackendContext`. The Ray
implementation directly wraps `ray.put`, `ray.get`, `ray.remote`, and `ray.wait`
— it treats each shard task as a stateless Ray remote function. This means:

1. **No persistent workers.** Each `run_stage` call creates a fresh Ray task.
   Workers cannot hold state (caches, loaded models, tokenizers) across
   invocations.
2. **No fault tolerance for workers.** If a Ray node is preempted, all tasks on
   it fail. There is no retry-at-shard-level or worker replacement.
3. **No shared data model.** Users who want to pass a tokenizer to all workers
   must rely on Ray's object store (`ray.put`) — there is no first-class
   mechanism for broadcasting read-only context data.
4. **Direct Ray coupling.** Zephyr cannot run distributed workloads on Iris
   without reimplementing the dispatch layer.

The fray-lite design doc (Phase 4) calls for migrating Zephyr to use `fray.v2`
actors via `create_actor_group`. This document designs that migration.

## Goals

- Replace `Backend` class and `BackendContext` protocol entirely with
  `ZephyrContext` using fray v2 actor-based dispatch
- Long-lived workers that persist across stage executions
- First-class shared data broadcast (e.g. tokenizer, config)
- Worker lifecycle management with state tracking (INIT / READY / FAILED)
- Transient error recovery (preemption, connection errors) vs. application error
  propagation
- Status display showing per-worker, per-shard progress
- Backend-agnostic: works on Ray, Iris, and local (in-process)
- Delete `BackendContext`, `SyncBackendContext`, `ThreadBackendContext`,
  `RayBackendContext`, `Backend`, and all associated plumbing

## Architecture Overview (Pull Model)

The pull model has workers pull tasks from a coordinator actor. The coordinator
is the single source of truth for task state, worker state, and progress. All
coordinator state mutations are serialized by the actor framework — no locks.

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

**Why a context object rather than static methods?** The current `Backend.execute()`
is stateless — it creates a `BackendContext` and throws it away. With long-lived
actors, we need lifecycle management. A context object owns the coordinator and
workers and can be reused across multiple `execute()` calls (e.g. multi-stage
recipes).

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

### Error Classification

The critical distinction: **transient errors** (preemption, connection loss)
trigger shard retry. **Application errors** (bugs in user code) propagate
immediately via the coordinator's `fatal_error` field.

Workers classify errors locally and report the classification to the coordinator.
This avoids serializing arbitrary exception objects across RPC — the coordinator
receives a string error message and a boolean `is_transient` flag.

```python
def _is_transient_error(error: Exception) -> bool:
    """Classify whether an error is transient (recoverable) or permanent.

    We enumerate transient cases and treat everything else as permanent.
    A false positive (treating permanent as transient) wastes time retrying;
    a false negative (treating transient as permanent) kills the pipeline.
    We err toward retrying.
    """
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    # Ray-specific preemption errors
    ray_transient = {
        "NodeDiedError", "ActorDiedError", "OwnerDiedError",
        "WorkerCrashedError", "RayActorError",
    }
    if type(error).__name__ in ray_transient:
        return True

    cause = error.__cause__
    if cause is not None and type(cause).__name__ in ray_transient:
        return True

    return False
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

### ForkChunks (Intra-Shard Parallelism)

The current `ForkChunks` operation spawns sub-tasks via `exec_ctx.run()` for
parallel chunk processing within a shard. With actors, the worker handles this
locally.

The `ZephyrWorker` sets `execution_context=_InProcessContext()` in `StageContext`.
`_InProcessContext` dispatches to a local `ThreadPoolExecutor` within the worker
process. This matches how `ThreadBackendContext` works today — intra-shard
parallelism is always local to the machine processing the shard. No changes
to ForkChunks semantics.

### Data Flow

**Shard data transfer:** The coordinator holds shard task descriptions (paths,
metadata). Workers receive these via `pull_task()` and read actual data from
GCS. Results (chunk headers + GCS paths or inline data) are reported back via
`report_result()`. Between stages, the user process collects results from the
coordinator and passes them as tasks for the next stage.

**Shared data:** Broadcast once via `coordinator.set_shared_data()`. Workers
fetch it via `coordinator.get_shared_data()` at the start of `run_loop()`.
For large shared objects (models, tokenizers), this is one RPC per worker.

**Generator streaming:** The current `run_stage()` returns an iterator. In the
actor model, `_execute_shard()` collects all chunks and returns them at once.
For file-based pipelines (the common case), chunks are metadata + GCS paths —
not raw data — so memory is not a concern. For in-memory pipelines with very
large outputs, the worker writes intermediate results to GCS and returns paths.

## Alternative Designs Considered

### Alternative 1: Keep BackendContext, Add Retry Layer

**Verdict:** Rejected. Patches the symptom (no retries) without addressing
the architectural issue (stateless dispatch, Ray coupling). Workers remain
stateless, no shared data, no caching.

### Alternative 2: Thin Actor Wrapper over BackendContext

**Verdict:** Rejected. The `BackendContext` protocol is shaped around Ray
primitives (opaque refs from `put`, `wait` with `num_returns`). Forcing actors
into it creates an impedance mismatch.

### Alternative 3: Push Model (Controller Dispatches to Workers)

The controller holds the task queue in-process and pushes shards to workers via
RPC. Simpler but has drawbacks vs. pull:

- **Liveness detection** requires a separate heartbeat thread that mutates state
  concurrently with the dispatch loop → needs locking.
- **Load balancing** is round-robin; fast workers idle until next assignment.
- **Future polling** requires polling N futures with short timeouts (no native
  multi-wait on `ActorFuture`). O(N) per tick.

**Verdict:** Rejected in favor of pull model. Pull gives natural load balancing,
implicit heartbeat via pull_task calls, and no locks (coordinator is an actor
with serialized calls).

### Alternative 4: Stateless Functions with GCS Checkpointing

**Verdict:** Rejected. High I/O overhead, no shared state, coarse retry
granularity. Useful as an addition to actor-based execution but not a
replacement.

## Potential Challenges

### 1. Data Serialization Across Backends

`PhysicalOp` contains arbitrary callables (user-defined map/filter functions).
These must be picklable to send via actor RPC.

**Mitigation:** Zephyr already requires functions to be picklable for Ray.
The actor model has the same requirement. For Iris, cloudpickle is used.
No new constraint.

### 2. ActorFuture Polling

In the pull model, the user process only polls one actor (`get_status()`)
at 1s intervals. Workers block on `coordinator.pull_task()`. No N-future
polling problem.

### 3. Shard Data Ownership After Stages

Between stages, shard data must be accessible to a different worker.

**Mitigation:** Output data is returned to the coordinator via `report_result()`.
The user process collects results and passes them as tasks for the next stage.
This is explicit and doesn't depend on a shared object store.

### 4. Join Operations

Sorted merge joins require co-located left and right shard data.

**Mitigation:** Same as current implementation — execute the right sub-plan
first, collect results, then pass them as auxiliary data in shard tasks.

### 5. Coordinator Failure

The coordinator runs on a non-preemptible node. If it still dies, workers
timeout on their next `pull_task` call and exit cleanly. The user process sees
the coordinator's `JobHandle` fail and surfaces the error.

## Migration Plan

Following spiral design: each phase produces a testable, self-contained unit.

### Phase 1: Core Implementation + Test

**Goal:** `ZephyrContext` + `ZephyrCoordinator` + `ZephyrWorker` working with
`LocalClient` on a simple pipeline.

**New files:**
- `lib/zephyr/src/zephyr/execution.py` — `ZephyrContext`, `ZephyrCoordinator`,
  `ZephyrWorker`, `shard_ctx()`, `_is_transient_error`, `ShardTask`,
  `_InProcessContext`
- `lib/zephyr/tests/test_execution.py` — tests using `LocalClient`:
  - `test_simple_map` — `from_list([1,2,3]).map(lambda x: x*2)`, verify `[2,4,6]`
  - `test_shared_data` — `put("key", value)`, verify workers can access via
    `shard_ctx().get_shared("key")`
  - `test_multi_stage` — pipeline with map + filter
  - `test_write_jsonl` — pipeline writing to temp dir
  - `test_join` — sorted merge join

**Modified files:**
- `lib/zephyr/src/zephyr/__init__.py` — export `ZephyrContext`

**Pseudo-code for test:**
```python
def test_simple_map():
    client = LocalClient()
    ctx = ZephyrContext(client=client, num_workers=2)
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = list(ctx.execute(ds))
    assert sorted(results) == [2, 4, 6]
```

### Phase 2: Migrate Zephyr Tests

**Goal:** All existing zephyr tests pass using `ZephyrContext` instead of
`Backend.execute()`.

**Modified files:**
- `lib/zephyr/tests/conftest.py` — replace `backend()` fixture with
  `zephyr_context()` fixture returning `ZephyrContext(LocalClient(), num_workers=N)`
- `lib/zephyr/tests/test_dataset.py` — `Backend.execute(ds)` →
  `ctx.execute(ds)`
- `lib/zephyr/tests/test_groupby.py` — same pattern
- `lib/zephyr/tests/test_vortex.py` — same pattern
- `lib/zephyr/tests/test_backends.py` — remove `BackendContext` tests, add
  `ZephyrContext` equivalents
- `lib/zephyr/tests/test_optimization.py` — update docstring examples
- `lib/zephyr/tests/benchmark_dedup_pipeline.py` — update

### Phase 3: Migrate CLI + Context Management

**Goal:** `zephyr` CLI uses `ZephyrContext`. Delete `BackendContext` and
related machinery.

**Modified files:**
- `lib/zephyr/src/zephyr/cli.py` — `run_local()` creates `ZephyrContext`
  from `fray.v2.current_client()` instead of `create_backend_context()`.
  CLI flags: `--num-workers`, `--max-parallelism`, `--resources`.
  Remove `--backend` flag (backend determined by `FRAY_CLIENT_SPEC`).
- `lib/zephyr/src/zephyr/context.py` — delete `BackendContext`,
  `SyncBackendContext`, `ThreadBackendContext`, `RayBackendContext`,
  `create_backend_context`, `default_backend_context`,
  `get_default_backend_context`. Keep only `ZephyrContext` re-export
  (or merge into execution.py).
- `lib/zephyr/src/zephyr/backends.py` — delete `Backend` class, `Shard`,
  `reshard_refs`, and all the `_run_tasks`/`_execute_shard_parallel`
  machinery. Keep `format_shard_path` (used by write ops).
- `lib/zephyr/src/zephyr/__init__.py` — remove `Backend`, `BackendConfig`
  exports. Add `ZephyrContext`.
- `lib/zephyr/src/zephyr/plan.py` — remove `StageContext.execution_context`
  field (the `BackendContext` reference). `_InProcessContext` from
  `execution.py` handles ForkChunks internally.

**Context variable pattern for tests/scripts:**
```python
# New pattern: set default ZephyrContext via contextvars
_default_zephyr_context: ContextVar[ZephyrContext | None] = ContextVar(
    "zephyr_context", default=None
)

def current_zephyr_context() -> ZephyrContext:
    ctx = _default_zephyr_context.get()
    if ctx is None:
        # Auto-create from fray.v2.current_client()
        from fray.v2.client import current_client
        ctx = ZephyrContext(client=current_client(), num_workers=os.cpu_count() or 1)
    return ctx
```

### Phase 4: Migrate Marin Callers

**Goal:** All marin code using `Backend.execute()` migrates to `ZephyrContext`.

The pattern is mechanical: `Backend.execute(pipeline)` →
`current_zephyr_context().execute(pipeline)` (or explicit `ctx.execute()`).

**Modified files (each follows the same pattern):**

Processing:
- `lib/marin/src/marin/processing/tokenize/tokenize.py`
- `lib/marin/src/marin/processing/classification/decon.py`
- `lib/marin/src/marin/processing/classification/consolidate.py`
- `lib/marin/src/marin/processing/classification/fasttext/train_fasttext.py`
- `lib/marin/src/marin/processing/classification/deduplication/exact.py`
- `lib/marin/src/marin/processing/classification/deduplication/fuzzy.py`
- `lib/marin/src/marin/processing/classification/deduplication/dedup_commons.py`
- `lib/marin/src/marin/processing/classification/deduplication/connected_components.py`
- `lib/marin/src/marin/validate/validate.py`

Download:
- `lib/marin/src/marin/download/wikipedia/download.py`
- `lib/marin/src/marin/download/huggingface/download_hf.py`
- `lib/marin/src/marin/download/ar5iv/download.py`
- `lib/marin/src/marin/download/dclm_hq/download_dclm_hq_html.py`
- `lib/marin/src/marin/download/nemotron_cc/download_nemotron_cc.py`
- `lib/marin/src/marin/download/uncheatable_eval/download.py`
- `lib/marin/src/marin/download/filesystem/transfer.py`
- `lib/marin/src/marin/download/huggingface/stream_remove_columns.py`

Transform:
- `lib/marin/src/marin/transform/conversation/transform_conversation.py`
- `lib/marin/src/marin/transform/conversation/transform_preference_data.py`
- `lib/marin/src/marin/transform/conversation/conversation_to_dolma.py`
- `lib/marin/src/marin/transform/common_pile/filter_by_extension.py`
- `lib/marin/src/marin/transform/dolmino/transform_dclm_hq.py`
- `lib/marin/src/marin/transform/dolmino/filter_dolmino.py`
- `lib/marin/src/marin/transform/huggingface/dataset_to_eval.py`
- `lib/marin/src/marin/transform/evaluation/eval_to_dolma.py`
- `lib/marin/src/marin/transform/wikipedia/transform_wikipedia.py`
- `lib/marin/src/marin/transform/simple_html_to_md/process.py`
- `lib/marin/src/marin/transform/ar5iv/transform.py`
- `lib/marin/src/marin/transform/ar5iv/transform_ar5iv.py`
- `lib/marin/src/marin/transform/medical/lavita_to_dolma.py`
- `lib/marin/src/marin/transform/stackexchange/transform_stackexchange.py`
- `lib/marin/src/marin/transform/stackexchange/filter_stackexchange.py`
- `lib/marin/src/marin/transform/lingoly/to_dolma.py`

Other:
- `lib/levanter/src/levanter/store/cache.py`
- `experiments/train_test_overlap/aggregate_total.py`

Test infrastructure:
- `tests/conftest.py` — reset `_default_zephyr_context` instead of
  `_default_backend_context`
- `tests/transform/conftest.py` — use `ZephyrContext(LocalClient())`
- `tests/download/conftest.py` — same
- `tests/processing/classification/deduplication/conftest.py` — same
- `tests/processing/classification/deduplication/test_connected_components.py`

### Phase 5: Validate + Cleanup

**Goal:** Integration test passes. Dead code removed.

- Run `tests/integration_test.py`
- Delete `lib/zephyr/src/zephyr/context.py` (all contents moved/deleted)
- Remove `BackendContext`, `Backend` from any remaining references
- Run `./infra/pre-commit.py --all-files`

## Summary

| Component | Responsibility |
|-----------|---------------|
| `ZephyrContext` | User-facing API. Creates coordinator + workers. Broadcasts shared data. Polls progress. |
| `ZephyrCoordinator` | Actor (non-preemptible). Task queue, worker liveness, result collection. Single source of truth. |
| `ZephyrWorker` | Actor per worker. Pull loop: pull task → execute → report result. Holds shared state. |
| `shard_ctx()` | Worker-side access to shared data. |

### Review Checklist

- [x] Workers are long-lived and persist across stages
- [x] Shared data broadcast via coordinator `set_shared_data()`
- [x] Transient vs. application error classification (done in worker, reported to coordinator)
- [x] Worker state machine: INIT → READY → BUSY → FAILED (via coordinator)
- [x] Shard retry on transient failure (re-queued in coordinator)
- [x] Application error propagation (coordinator sets `fatal_error`, user process raises)
- [x] Status display with per-worker spinner and heartbeat age
- [x] Works with LocalClient (in-process), RayClient, FrayIrisClient
- [x] No direct Ray imports in Zephyr
- [x] ForkChunks handled via local thread pool within actor
- [x] Join operations supported via explicit aux data passing
- [x] Pull model: natural load balancing, implicit heartbeat, no locks
- [x] Coordinator is externally inspectable via `get_status()` (dashboard-friendly)
- [x] Coordinator non-preemptible; workers can be preempted and recovered
- [x] `Backend` class and `BackendContext` fully deleted — no backward compat
- [x] All marin callers migrated
