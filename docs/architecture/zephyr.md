# Zephyr: Lazy Dataset Processing

Zephyr is Marin's distributed dataset processing library. It provides a declarative `Dataset` API — chainable transformations, lazy evaluation, operation fusion — backed by a pull-based coordinator/worker execution model that runs on Iris or Ray.

The central problem Zephyr solves: data pipelines at Marin's scale involve hundreds of GB to TB of data across millions of files. Ad-hoc scripts to process these were proliferating. Zephyr consolidates the patterns into one library with automatic distributed fan-out, fault tolerance, and efficient inter-stage data flow.

**Source**: `lib/zephyr/src/zephyr/`  
**AGENTS.md**: `lib/zephyr/AGENTS.md`  
**OPS.md**: `lib/zephyr/OPS.md`

---

## Table of Contents

1. [Quick Example](#quick-example)
2. [Dataset API](#dataset-api)
  - [Creation](#creation)
  - [Transformations](#transformations)
  - [Aggregation Operations](#aggregation-operations)
  - [Output](#output)
3. [Operation Fusion and Planning](#operation-fusion-and-planning)
4. [Execution Architecture](#execution-architecture)
  - [Pull-Based Model](#pull-based-model)
  - [Coordinator](#coordinator)
  - [Workers](#workers)
  - [Inter-Stage Data Flow (PickleDiskChunk)](#inter-stage-data-flow-picklediskchunk)
5. [Reshard](#reshard)
6. [Shuffle and Group-By](#shuffle-and-group-by)
7. [Deduplication](#deduplication)
8. [Sorted Merge Join](#sorted-merge-join)
9. [External Sort](#external-sort)
10. [Expression System](#expression-system)
11. [Writers and Partial-Failure Handling](#writers-and-partial-failure-handling)
12. [Counters](#counters)
13. [Fault Tolerance](#fault-tolerance)
14. [Design Decisions](#design-decisions)

---

## Quick Example

```python
from zephyr import Dataset, ZephyrContext
from zephyr.expr import col

ctx = ZephyrContext(max_workers=200)

pipeline = (
    Dataset.from_files("gs://marin-us-central2/raw/", "**/*.jsonl.gz")
    .load_jsonl()
    .filter(col("language") == "en")
    .filter(col("quality_score") > 0.5)
    .map(lambda doc: {**doc, "text": doc["text"].strip()})
    .reshard(num_shards=500)
    .write_jsonl("gs://marin-us-central2/filtered/{shard:05d}-of-{total:05d}.jsonl.gz")
)

result = ctx.execute(pipeline)
print(result.counters)  # {"records_read": 1_234_567, "records_written": 890_123}
```

This pipeline:

1. Discovers all `.jsonl.gz` files under the prefix
2. Fans out across up to 200 workers
3. Each worker loads, filters, and maps its assigned files
4. Reshards the output to exactly 500 output shards
5. Writes gzip-compressed JSONL to GCS atomically per shard

---

## Dataset API

**Source**: `lib/zephyr/src/zephyr/dataset.py`

`Dataset[T]` is a lazy computation graph. No work happens until `ZephyrContext.execute(pipeline)` is called.

### Creation

```python
Dataset.from_files(base_path: str, pattern: str = "**/*",
                   empty_glob_ok: bool = False) -> Dataset[str]
# Returns a Dataset of file paths. Each file becomes one input shard.

Dataset.from_list(items: list[T]) -> Dataset[T]
Dataset.from_iterable(iterable: Iterable[T]) -> Dataset[T]
```

For file-based pipelines, `from_files` is almost always the right entry point. Each matched file path becomes one "shard" — the unit of work assigned to a worker.

### Transformations

All of these are **lazy** — they append a logical operation to the pipeline, no data is processed yet.

```python
# Element-wise transforms (fusible — see Operation Fusion)
.map(fn: Callable[[T], R]) -> Dataset[R]
.flat_map(fn: Callable[[T], Iterable[R]]) -> Dataset[R]
.filter(predicate: Callable[[T], bool] | Expr) -> Dataset[T]
.select(*columns: str) -> Dataset[dict]        # column projection

# File loading (pushes into the source spec for predicate/column pushdown)
.load_file(columns=None) -> Dataset[dict]      # auto-detects format
.load_parquet(columns=None) -> Dataset[dict]
.load_jsonl() -> Dataset[dict]
.load_vortex(columns=None) -> Dataset[dict]

# Shard-level control
.map_shard(fn: Callable[[Iterator[T], ShardInfo], Iterator[R]]) -> Dataset[R]
# fn receives the full iterator for a shard + ShardInfo(shard_idx, total_shards)

.take_per_shard(n: int) -> Dataset[T]          # limit N items per shard
.window(size: int) -> Dataset[list[T]]         # sliding window of size N
.window_by(folder_fn, initial_state) -> Dataset[list[T]]  # custom folding

# Parallelism control
.reshard(num_shards: int | None) -> Dataset[T] # see Reshard section
```

`**filter` with Expressions vs lambdas**: Use `col("field") > 0.5` (an `Expr`) rather than `lambda x: x["field"] > 0.5` when the data source is Parquet — Zephyr will push the filter down into the Parquet reader, skipping entire row groups without deserializing them.

```python
from zephyr.expr import col, lit

# Parquet predicate pushdown — skips row groups entirely
.filter(col("score") > 0.5)
.filter((col("lang") == "en") & (col("tokens") < 10_000))

# Lambda filter — always reads full records
.filter(lambda x: x["score"] > 0.5)
```

### Aggregation Operations

These create **stage boundaries** — the planner inserts a barrier and starts a new execution stage.

```python
# Group-by with reduce (triggers scatter+reduce — two stages)
.group_by(
    key: Callable[[T], K] | str,
    reducer: Callable[[K, Iterator[T]], R],
    sort_by: Callable[[T], Any] | None = None,
    num_output_shards: int | None = None,
    combiner: Callable[[K, Iterator[T]], T] | None = None,
) -> Dataset[R]

# Deduplication (two-phase group-by under the hood)
.deduplicate(
    key: Callable[[T], object],
    num_output_shards: int | None = None,
) -> Dataset[T]

# Aggregate all items to a single value
.reduce(
    local_reducer: Callable[[T, T], T],
    global_reducer: Callable[[T, T], T] | None = None,
) -> Dataset[T]
.count() -> Dataset[int]

# Sorted merge join (requires pre-sorted shards with matching key ranges)
.sorted_merge_join(
    right: Dataset,
    left_key: Callable,
    right_key: Callable,
    combiner: Callable | None = None,
    how: Literal["inner", "left"] = "inner",
) -> Dataset
```

### Output

Output operations also create stage boundaries. They return a `Dataset[str]` of output file paths.

```python
.write_jsonl(
    output_pattern: str,        # e.g. "gs://bucket/{shard:05d}-of-{total:05d}.jsonl.gz"
    skip_existing: bool = False
) -> Dataset[str]

.write_parquet(output_pattern, schema=None, skip_existing=False) -> Dataset[str]
.write_vortex(output_pattern, schema=None, skip_existing=False) -> Dataset[str]
.write_binary(output_pattern, skip_existing=False) -> Dataset[str]
.write_levanter_cache(output_pattern, metadata, skip_existing=False,
                      batch_size=None) -> Dataset[str]
```

Pattern variables: `{shard}`, `{total}`, `{shard:05d}`. Compression is inferred from extension: `.gz` → gzip, `.zst` → zstd.

---

## How MapOp and FilterOp Are Actually Implemented

This section traces the full path from a `.map()` call in user code to items being processed in a worker subprocess.

### Step 1: Logical Ops Are Plain Dataclasses

`dataset.py` defines logical operations as frozen dataclasses that store callables — nothing more:

```python
# dataset.py:88
@dataclass
class MapOp:
    fn: Callable

# dataset.py:98
@dataclass
class FilterOp:
    predicate: Callable
    expr: Expr | None = field(default=None)   # set when using col() expressions

# dataset.py:173
@dataclass
class FlatMapOp:
    fn: Callable
```

When you call `.map(fn)` on a `Dataset`, it appends `MapOp(fn=fn)` to `self.operations` and returns a new `Dataset`. No computation happens. The `Dataset` is just an immutable list of logical ops plus a source iterable.

`FilterOp` carries two representations of the same predicate:

- `predicate`: a callable that works on Python dicts (used during worker execution)
- `expr`: the `Expr` object if you called `.filter(col("x") > 0.5)` (used for Parquet pushdown at the reader)

When you call `.filter(col("x") > 0.5)`, both are populated: `expr = col("x") > 0.5` and `predicate = expr.evaluate` (bound method). When you call `.filter(lambda x: x["score"] > 0.5)`, only `predicate` is set; `expr` is `None` and no pushdown occurs.

### Step 2: Compilation — Fusion Into a Single Closure

`compute_plan()` in `plan.py` compiles the logical op list into a `PhysicalPlan`. The key function is `compose_map()` (line 212), which fuses all consecutive fusible ops into one Python closure:

```python
# plan.py:212
def compose_map(operations: list) -> Callable[[Iterator], Iterator]:
    def pipeline(stream: Iterator, *, shard_idx: int = 0, total_shards: int = 1) -> Iterator:
        for op in operations:
            if isinstance(op, LoadFileOp):
                stream = _load_file_gen(stream)
            elif isinstance(op, MapOp):
                stream = _map_gen(stream, op.fn)
            elif isinstance(op, FilterOp):
                stream = _filter_gen(stream, op.predicate)
            elif isinstance(op, FlatMapOp):
                stream = _flatmap_gen(stream, op.fn)
            elif isinstance(op, MapShardOp):
                stream = op.fn(stream, ShardInfo(shard_idx=shard_idx, total_shards=total_shards))
            elif isinstance(op, TakePerShardOp):
                stream = islice(stream, op.n)
            elif isinstance(op, WindowOp):
                stream = make_windows(stream, op.folder_fn, op.initial_state)
            elif isinstance(op, SelectOp):
                stream = _select_gen(stream, op.columns)
        return stream
    return pipeline
```

The helper generators are simple:

```python
# plan.py:164-177
def _map_gen(stream: Iterator, fn: Callable) -> Iterator:
    for item in stream:
        yield fn(item)

def _filter_gen(stream: Iterator, predicate: Callable) -> Iterator:
    for item in stream:
        if predicate(item):
            yield item

def _flatmap_gen(stream: Iterator, fn: Callable) -> Iterator:
    for item in stream:
        yield from fn(item)
```

Each wraps the previous stream in a new generator. No items are pulled. The result is a chain of nested generators: `_map_gen(_filter_gen(_map_gen(source, fn1), pred), fn2)`. Items flow through lazily when the downstream consumer pulls.

`compose_map` is called from `FusionState.flush_pending()` (line 341), which fires whenever a stage boundary is hit or the operations list ends:

```python
# plan.py:341
def flush_pending(self) -> None:
    if not self.pending_fusible:
        return
    requires_full_shard = any(isinstance(op, MapShardOp) for op in self.pending_fusible)
    self.current_ops.append(
        Map(
            fn=compose_map(self.pending_fusible[:]),
            requires_full_shard=requires_full_shard,
            needs_shard_context=requires_full_shard,
        )
    )
    self.pending_fusible = []
```

The resulting `Map` physical op is a dataclass holding the single fused closure:

```python
# plan.py:84
@dataclass
class Map:
    fn: Callable[[Iterator], Iterator]
    requires_full_shard: bool = False
    needs_shard_context: bool = False
```

### Step 3: Serialization via cloudpickle

The `PhysicalPlan` — including the `Map` closures containing user lambdas — is serialized with `cloudpickle` and written to a temp file. This is how user-defined functions cross the process boundary into the subprocess.

cloudpickle (unlike standard `pickle`) can serialize closures, lambdas, and locally defined functions. This is what makes `ds.map(lambda x: ...)` work without any special registration.

### Step 4: Subprocess Execution

The worker actor calls `_execute_shard()` (execution.py:1203), which:

1. Serializes `(task, chunk_prefix, execution_id)` via cloudpickle to a temp file
2. Spawns `python -u -m zephyr.subprocess_worker task_file result_file`
3. Waits for the subprocess to exit, then reads `(result_or_error, counters)` from the result file

Inside the subprocess (`subprocess_worker.py`), `execute_shard()`:

1. Deserializes the task
2. Sets `pa.set_io_thread_count(1)` and `pa.set_cpu_count(1)` — PyArrow's internal thread pools are redundant since parallelism comes from multiple subprocesses, not threads
3. Calls `run_stage(stage_ctx, task.operations, ...)` and hands the result to `_write_stage_output()`
4. On exit, calls `os._exit()` instead of normal Python shutdown to avoid PyArrow GCS background threads racing with module GC (which would fire `std::terminate`)

### Step 5: `run_stage` Drives the Chain

`run_stage()` (plan.py:727) is the inner loop that actually pulls items through the composed generators:

```python
# plan.py:727
def run_stage(ctx: StageContext, ops: list[PhysicalOp], ...) -> Iterator:
    stream: Iterator = iter(ctx.shard)   # source items for this worker's shard

    for op in ops:
        if isinstance(op, Map):
            if op.needs_shard_context:
                stream = op.fn(stream, shard_idx=ctx.shard_idx, total_shards=ctx.total_shards)
            else:
                stream = op.fn(stream)    # ← wraps stream in fused generator chain
        elif isinstance(op, Write):
            # Drives the stream: pulls all items and writes them to the output file
            output_path = op.output_pattern(ctx.shard_idx, ctx.total_shards)
            result = write_jsonl_file(stream, output_path)["path"]
            yield result
            return
        elif isinstance(op, Scatter):
            yield from stream             # caller handles routing
            return
        ...
    yield from stream
```

The `Map` case just wraps the stream in another generator layer — still lazy. Items are not pulled until `Write` (or `Scatter`, or the final `yield from stream`) drives the iteration. Only at that point do items flow through the entire composed generator chain one at a time.

### Full Call Stack for a `.map().filter().write_jsonl()` Pipeline

```
user code
  Dataset.from_files(...).load_jsonl().filter(col("x")>0.5).map(fn).write_jsonl(...)
  → Dataset.operations = [LoadFileOp, FilterOp(pred, expr), MapOp(fn), WriteOp]

compute_plan()
  → _compute_file_pushdown():  FilterOp.expr pushed into InputFileSpec (Parquet reader)
  → _fuse_operations():
      pending = [LoadFileOp, FilterOp(pred, expr=None*)]  (* expr already consumed)
      flush_pending() → Map(fn=compose_map([LoadFileOp, FilterOp(pred)]))
      add_op(Write(...))
  → PhysicalPlan(stages=[PhysicalStage(ops=[Map, Write])])

ZephyrWorker._execute_shard()
  → cloudpickle.dump((task, ...), task_file)
  → subprocess.run(["python", "-m", "zephyr.subprocess_worker", task_file, result_file])

subprocess_worker.execute_shard()
  → task = cloudpickle.load(task_file)
  → run_stage(StageContext(shard=InputFileSpec(...)), [Map, Write])

run_stage()
  → stream = iter(InputFileSpec(...))         # lazy file reader
  → Map: stream = map_closure(stream)         # wraps in _load_file_gen → _filter_gen chain
  → Write: pulls stream item by item → writes to GCS atomically
```

The `FilterOp.expr` is consumed by `_compute_file_pushdown` and baked into the `InputFileSpec`. The Parquet reader uses it for row-group skipping before records are even deserialized. The remaining `FilterOp(predicate=expr.evaluate)` then does a second-pass Python-level filter on the records that were read (for readers that can't do full predicate pushdown at the page level).

---

## Operation Fusion and Planning

**Source**: `lib/zephyr/src/zephyr/plan.py`

Before execution, Zephyr compiles the logical `Dataset` operation chain into a `PhysicalPlan` via `compute_plan()` (line 528). This involves:

1. **File pushdown** (`_compute_file_pushdown`, line 472): If the first operation is a file load, the planner extracts any immediately following `filter(Expr)` and `select(columns)` operations and bakes them into the `InputFileSpec`. Parquet readers then use predicate pushdown to skip row groups at the file level.
2. **Operation fusion** (`_fuse_operations`): Consecutive fusible operations are composed into a single `Map` physical operation via iterator chaining (`compose_map`, line 212). No intermediate materialization.

**Fusible operations** (run in the same worker pass, no stage boundary):


| Logical Op       | Physical result                             |
| ---------------- | ------------------------------------------- |
| `MapOp`          | fused into `Map`                            |
| `FlatMapOp`      | fused into `Map`                            |
| `FilterOp`       | fused into `Map`                            |
| `LoadFileOp`     | fused into `Map` (or pushed into source)    |
| `MapShardOp`     | fused into `Map`                            |
| `TakePerShardOp` | fused into `Map`                            |
| `WindowOp`       | fused into `Map`                            |
| `SelectOp`       | fused into `Map` (or pushed into source)    |
| `WriteOp`        | included in same stage as the map before it |


**Stage-breaking operations** (each creates a new execution stage with a barrier):


| Logical Op  | Stages created                                            |
| ----------- | --------------------------------------------------------- |
| `ReshardOp` | metadata-only reshard (no worker execution)               |
| `GroupByOp` | Stage 1: `Scatter` + Stage 2: `Reduce`                    |
| `ReduceOp`  | Stage 1: `Fold` + Stage 2: `Reshard(1)` + Stage 3: `Fold` |
| `JoinOp`    | right sub-plan executed as nested stage sequence          |


**Physical stage types**:

```python
class StageType(StrEnum):
    WORKER  = auto()  # Normal distributed execution
    RESHARD = auto()  # Metadata-only shard redistribution
```

**Example**: A pipeline with `from_files → load_jsonl → filter → map → reshard → write_jsonl` compiles to:

- Source items: one per matched file
- Stage 1 (`WORKER`): `[load_jsonl ∘ filter ∘ map ∘ write_jsonl]` — all fused
- Stage 2 (`RESHARD`): metadata redistribution to target shard count

---

## Execution Architecture

**Source**: `lib/zephyr/src/zephyr/execution.py`

```
ZephyrContext.execute(pipeline)
      │
      ├── compute_plan()          ← fusion & planning
      │
      ▼
ZephyrCoordinator               (fray actor, 1 per pipeline)
      │  owns task queue, assignment state, results
      │
      ├── pull_task(worker_id) ◀──── ZephyrWorker 1  (fray actor)
      ├── pull_task(worker_id) ◀──── ZephyrWorker 2
      ├── pull_task(worker_id) ◀──── ZephyrWorker N
      │
      ▼
  Stage barrier: all shards complete → next stage begins
```

Workers are **persistent across stages** — they don't restart between stages. After completing a task, a worker calls `pull_task` again and gets the next assignment, which may be from a different stage.

### Pull-Based Model

The fundamental design choice in Zephyr is **pull-based** work assignment. Compare:

**Push-based** (alternative): Coordinator assigns specific tasks to specific workers upfront. Problems: coordinator must track which workers are available; slow workers block their assigned tasks; worker failure leaves assigned tasks "stuck" until timeout.

**Pull-based** (Zephyr's approach): Workers call `pull_task(worker_id)` and get the next available task. Coordinator doesn't need to track worker state — it just has a queue. Worker failure is automatically handled: a stale worker stops pulling, and after heartbeat timeout its in-flight task is re-queued.

The pull model also naturally implements **work-stealing**: fast workers finish quickly and pull more tasks, slow workers pull fewer. No explicit load balancing needed.

```python
# ZephyrCoordinator.pull_task (execution.py, line 591)
def pull_task(self, worker_id: str) -> tuple[ShardTask, int, RunConfig] | Literal["SHUTDOWN"] | None:
    if self._fatal_error:
        return "SHUTDOWN"
    if not self._task_queue:
        return None    # worker backs off (exponential backoff on client side)
    task = self._task_queue.popleft()
    self._in_flight[worker_id] = (task, attempt)
    return (task, attempt, self._run_config)
```

### Coordinator

**Source**: `execution.py`, `ZephyrCoordinator` (line 350)

The coordinator runs as a Fray actor (a long-lived Iris job). It owns:

```python
class ZephyrCoordinator:
    _task_queue: deque[ShardTask]                     # pending tasks
    _results: dict[int, TaskResult]                   # completed shard results
    _worker_states: dict[str, WorkerState]            # worker liveness
    _in_flight: dict[str, tuple[ShardTask, int]]      # who's working on what
    _task_attempts: dict[int, int]                    # retry counts
    _fatal_error: str | None                          # pipeline abort flag
```

**Stage execution** (`run_pipeline`, line 825): Stages run sequentially as barriers.

```python
for stage_idx, stage in enumerate(plan.stages):
    if stage.stage_type == StageType.RESHARD:
        shards = _reshard_refs(shards, stage.output_shards or len(shards))
        continue  # No worker dispatch — just rearrange chunk references
    
    # Enqueue all tasks for this stage
    for shard_idx, shard in enumerate(shards):
        self._task_queue.append(ShardTask(stage_idx, shard_idx, shard))
    
    # Wait for all tasks to complete
    shards = self._wait_for_stage(stage_idx, num_shards=len(shards))
```

### Workers

**Source**: `execution.py`, `ZephyrWorker` (line ~1100)

Each worker runs `_execute_shard` for each task it pulls. Crucially, each shard executes in a **subprocess** (`python -m zephyr.subprocess_worker`) for memory isolation:

- Prevents OOM from one bad shard killing the worker process
- SIGKILL (returncode -9) is detected and reported as OOM to the coordinator
- The worker process itself stays alive between shards

Workers send heartbeats to the coordinator every 30 seconds (or when they complete a task). The heartbeat carries `worker_id`, `current_task`, and counter deltas.

### Inter-Stage Data Flow (PickleDiskChunk)

**Source**: `execution.py`, `PickleDiskChunk` (line 66)

Between stages, data flows as filesystem-backed references, not in-memory:

```python
@dataclass(frozen=True)
class PickleDiskChunk:
    path: str    # GCS or local path to a pickle file
    count: int   # number of items in this chunk

    @classmethod
    def write(cls, path: str, data: list) -> PickleDiskChunk:
        unique_path = unique_temp_path(path)   # UUID suffix prevents race conditions
        with open_url(unique_path, "wb") as f:
            pickle.dump(data, f)
        return cls(path=unique_path, count=count)
```

Stage 1 workers write their output as `PickleDiskChunk` references. The coordinator collects these references. Stage 2 workers receive a list of `PickleDiskChunk` paths and read them sequentially. Workers process **one chunk at a time**, so memory usage is bounded by the size of a single chunk (~100,000 items by default) rather than the full stage output.

The UUID-unique write path prevents collisions when multiple workers race to write to the same logical output location.

---

## Reshard

**Source**: `dataset.py`, line 605

```python
def reshard(self, num_shards: int | None) -> Dataset[T]:
    """Redistribute data across target number of shards."""
```

Reshard is a **metadata-only operation** at execution time — it rearranges `PickleDiskChunk` references across shard buckets without re-reading or re-writing data.

```python
# execution.py, run_pipeline
if stage.stage_type == StageType.RESHARD:
    shards = _reshard_refs(shards, stage.output_shards or len(shards))
    continue  # No workers involved
```

**When to use reshard**:

- You start with 3 input files but want 200 workers processing the result → `reshard(200)` before your expensive `map`
- You have 10,000 small input shards but want to write 500 output files → `reshard(500)` before `write_jsonl`
- After `group_by`, the output shard count is determined by `num_output_shards` — no manual reshard needed

**What reshard does not do**: It doesn't sort or rebalance data sizes. Shards may be uneven after reshard if the input was uneven.

---

## Shuffle and Group-By

**Source**: `lib/zephyr/src/zephyr/shuffle.py`, `plan.py`

`group_by(key, reducer)` compiles to a two-stage shuffle:

### Stage 1: Scatter

Each worker processes its input shard and **hash-routes items to target shards**:

```python
target_shard = deterministic_hash(key_fn(item)) % num_output_shards
```

Items are buffered per target shard. Optionally, a `combiner_fn` pre-aggregates items with the same key before writing (reduces scatter data size). Items are sorted within each buffer. Results are written as **Parquet with an envelope schema**:

```
Schema: (shard_idx: int, chunk_idx: int, item: binary | <native fields>)
```

For types that convert cleanly to Arrow, items are stored as native Parquet fields. For others (arbitrary Python objects), items are cloudpickled into the `item: binary` column.

**Scatter manifest**: After all scatter writes, each worker writes a sidecar manifest. The coordinator consolidates all manifests into one. Reducers receive only the manifest path — not raw scatter files — so the coordinator doesn't need to track all scatter file locations.

### Stage 2: Reduce

Each reducer is responsible for one output shard. It reads back only the scatter data destined for its shard using **Parquet row-group predicate pushdown**:

```python
# ScatterParquetIterator (shuffle.py, line 134)
# Uses row-group statistics on shard_idx to skip irrelevant row groups
# Avoids pyarrow.dataset memory leak (apache/arrow#39808)
```

`ScatterShard` (line 194) presents per-sorted-chunk iterators across all relevant scatter files. These are k-way merged (`heapq.merge`) to produce a single sorted stream, then grouped by key and fed to the reducer.

```python
# Example group_by usage from marin experiments
domain_counts = (
    Dataset.from_files("gs://bucket/docs/", "**/*.jsonl.gz")
    .load_jsonl()
    .group_by(
        key=lambda doc: doc["domain"],
        reducer=lambda domain, docs: {"domain": domain, "count": sum(1 for _ in docs)},
        num_output_shards=100,
    )
    .write_jsonl("gs://bucket/domain_counts/{shard:05d}-of-{total:05d}.jsonl.gz")
)
```

---

## Deduplication

**Source**: `dataset.py`, line 837

Deduplication is a **two-phase group-by**:

```python
def deduplicate(self, key, num_output_shards=None):
    # Phase 1: intra-shard dedup (streaming, no state accumulation)
    def streaming_dedup(items, _: ShardInfo):
        seen = set()
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                yield item

    # Phase 2: cross-shard dedup via group_by (scatter+reduce)
    def keep_first(k, items):
        return next(iter(items))

    return (
        self.map_shard(streaming_dedup)
            .group_by(key=key, reducer=keep_first,
                      num_output_shards=num_output_shards)
    )
```

Phase 1 runs within each input shard in parallel and eliminates obvious duplicates cheaply. Phase 2 handles cross-shard duplicates: items with the same key are scatter-routed to the same output shard, where `keep_first` retains the first occurrence.

This two-phase approach reduces scatter data volume significantly for datasets with high intra-shard duplicate rates.

---

## Sorted Merge Join

**Source**: `dataset.py`, line 907; `plan.py`, `compose_join` (line 246)

`sorted_merge_join` assumes:

- Left and right datasets have the **same number of shards**
- Corresponding shards contain the **same key ranges** (i.e., both were produced by `group_by` with the same key and `num_output_shards`)
- Items within each shard are **sorted by join key**

These preconditions are met after running `group_by(key=my_key, num_output_shards=N)` on both datasets.

```python
# Correct usage: join two datasets grouped by the same key
left = docs.group_by(key=lambda d: d["url"], reducer=aggregate_doc, num_output_shards=200)
right = scores.group_by(key=lambda s: s["url"], reducer=aggregate_score, num_output_shards=200)

joined = left.sorted_merge_join(
    right=right,
    left_key=lambda d: d["url"],
    right_key=lambda s: s["url"],
    combiner=lambda l, r: {**l, "score": r["score"]},
    how="inner",  # or "left"
)
```

The join function streams through both sorted iterators simultaneously in O(n+m) time — no hash table, no materialization. The right sub-plan is executed by the coordinator as a nested stage sequence before the join stage begins.

---

## External Sort

**Source**: `lib/zephyr/src/zephyr/external_sort.py`

When `group_by` generates more than `EXTERNAL_SORT_FAN_IN = 500` sorted chunk iterators for a single reducer, a multi-pass merge is used to stay within memory limits:

**Pass 1**: Batch chunk iterators into groups of 500, merge each group with `heapq.merge`, spill sorted runs to zstd-compressed pickle files (batches of 10,000 items).

**Pass 2**: `heapq.merge` over the smaller set of run file iterators.

Read batch size is computed from the cgroup memory limit to use at most 25% of available memory for read buffers.

This is transparent to users — Zephyr chooses external sort automatically when needed.

---

## Expression System

**Source**: `lib/zephyr/src/zephyr/expr.py`

Zephyr's expression system, modeled on Vortex, supports filter expressions that can be pushed down to Parquet readers.

```python
from zephyr.expr import col, lit

# Comparisons
col("score") > 0.5
col("lang") == "en"
col("tokens") <= 10_000

# Logical operators
(col("lang") == "en") & (col("score") > 0.5)
(col("lang") == "en") | (col("lang") == "fr")
~(col("lang") == "zh")

# Arithmetic
col("a") + col("b") > lit(100)

# Null checks
col("score").is_null()
col("score").is_not_null()

# Field access (for nested dicts)
col("metadata").field("source") == "wikipedia"
```

Expressions have two evaluation paths:

1. `expr.evaluate(record: dict)` — Python dict evaluation (used for JSONL and intermediate stages)
2. `to_pyarrow_expr(expr)` — converts to a `pyarrow.compute` expression for Parquet predicate pushdown

Use `Expr` objects (via `col()`) rather than lambdas for filters on Parquet data. For JSONL or non-Parquet data, both are equivalent.

---

## Writers and Partial-Failure Handling

**Source**: `lib/zephyr/src/zephyr/writers.py`

All writers use **atomic write-then-rename** to prevent partial shard files:

```python
# writers.py, atomic_rename context manager (line 52)
@contextmanager
def atomic_rename(final_path: str):
    temp_path = unique_temp_path(final_path)   # UUID-unique temp path
    try:
        yield temp_path
        fs.rename(temp_path, final_path)        # atomic on POSIX, best-effort on GCS
    except Exception:
        fs.rm(temp_path, missing_ok=True)       # cleanup on failure
        raise
```

For object stores like GCS that lack atomic rename:

1. Data is written to a local temp directory
2. Uploaded to GCS via `fs.put()` after completion
3. On failure, local temp files are removed

**Format-specific details**:

- `write_jsonl_file`: Uses `msgspec.json.Encoder` (faster than `json`), gzip/zstd compression inferred from extension
- `write_parquet_file`: PyArrow writer with micro-batches of 8 records to control memory, 64MB target row group size
- `write_vortex_file`: Vortex columnar format

`**skip_existing=True`**: Checks if the output file already exists before writing. Useful for resuming interrupted pipelines — completed shards are skipped, incomplete shards are rewritten.

---

## Counters

**Source**: `lib/zephyr/src/zephyr/counters.py`

Workers can increment named counters that are aggregated across all workers and returned in `ZephyrExecutionResult.counters`:

```python
from zephyr import counters

def process_doc(doc):
    counters.increment("docs_processed")
    if not doc.get("text"):
        counters.increment("docs_skipped_empty")
        return None
    counters.increment("tokens", len(doc["text"].split()))
    return doc

pipeline = Dataset.from_files(...).load_jsonl().map(process_doc).filter(...)
result = ctx.execute(pipeline)
print(result.counters)
# {"docs_processed": 1234567, "docs_skipped_empty": 892, "tokens": 987654321}
```

Counters use a `ContextVar` for worker context. Snapshots are sent to the coordinator via heartbeats (only when changed) and as a final snapshot on task completion. The coordinator uses generation numbers to order snapshots correctly.

---

## Fault Tolerance

**Worker failure** (`_check_worker_heartbeats`, line 582): Workers must heartbeat every 120 seconds. A stale worker is marked `WorkerState.FAILED` and its in-flight task is re-queued with incremented attempt count.

**Shard failure limit** (`MAX_SHARD_FAILURES = 3`, line 62): A shard that fails 3 times aborts the entire pipeline with a `ZephyrWorkerError` containing the last error message. This prevents infinite retry loops on bad data or code bugs.

**Worker re-registration**: If a worker's underlying Iris job is preempted and restarted, the new process re-registers with the coordinator. The coordinator updates its handle and re-queues any task the old worker had in-flight.

**Subprocess isolation**: Each shard runs in a subprocess (`python -m zephyr.subprocess_worker`). OOM kills (SIGKILL, returncode -9) are detected and reported. The worker process survives and can pull the next task.

**Coordinator persistence**: The coordinator runs as a single Iris actor. If it dies, the entire pipeline must be re-run (there's no coordinator checkpoint). Because executor steps track completion via `_SUCCESS` files and `skip_existing=True`, re-running a Marin pipeline after coordinator failure typically only re-runs the interrupted stage.

---

## Design Decisions

**Why pull-based instead of push-based?**  
Push-based systems require the coordinator to know which workers are ready to receive work — which means tracking readiness state and handling "not ready yet" responses. Pull-based systems are simpler: workers call when they're ready; the coordinator just maintains a queue. Failure handling is also simpler: a dead worker stops calling, and its task is re-queued after heartbeat timeout. No special failure detection logic needed.

**Why disk-backed inter-stage data flow instead of in-memory streaming?**  
In-memory streaming between stages would require keeping all stage-1 output in memory while stage-2 processes it. At Marin's scale (terabytes of intermediate data), this isn't feasible. Disk-backed chunks bound memory to one chunk per worker (~100K items), allow retrying individual shards without re-running earlier stages, and let stages run at different speeds without backpressure coordination.

**Why subprocess isolation per shard?**  
Python's memory management makes it hard to guarantee that a bad document won't leak memory and eventually OOM the worker. Running each shard in a subprocess provides a clean memory reset between shards. It also provides a clear signal: SIGKILL (returncode -9) means OOM, making it straightforward to detect and report memory issues rather than having the worker process silently die.

**Why two-phase deduplication?**  
The intra-shard streaming dedup (phase 1) is essentially free — it runs in the same pass as the first stage and uses a bounded seen-set per shard. For datasets with high intra-shard duplicate rates (e.g., crawl data where the same URL appears many times in one source shard), this dramatically reduces the data volume sent through the scatter phase. The group_by (phase 2) then handles cross-shard duplicates correctly.

**Why require pre-sorted shards for sorted_merge_join instead of supporting arbitrary joins?**  
A general hash join would require materializing one side of the join in memory or in a hash table, which is expensive at scale. The sorted merge join runs in O(n+m) time with O(1) memory. Marin's use cases for joins (e.g., joining quality scores with documents, joining perplexity scores with text) always follow a `group_by` that produces sorted shards with matching key ranges, so the preconditions are naturally met.

**Why Parquet-based scatter instead of a custom binary format?**  
Parquet row-group statistics enable predicate pushdown in the reduce phase: each reducer only reads row groups with `shard_idx` matching its own shard. This avoids transferring scatter data between reducers and keeps network traffic proportional to output size, not total intermediate size. The envelope schema (`shard_idx, chunk_idx, item`) is simple enough to implement with PyArrow while still getting this benefit.