# zephyr

Simple data processing library for Marin pipelines. Build lazy dataset pipelines that run on Ray clusters, thread pools, or synchronously.

## Quick Start

```python
from zephyr import Dataset, create_backend, load_jsonl

# Read, transform, write
backend = create_backend("ray", max_parallelism=100)
pipeline = (
    Dataset.from_files("gs://input/", "**/*.jsonl.gz")
    .flat_map(load_jsonl)
    .filter(lambda x: x["score"] > 0.5)
    .map(lambda x: transform_record(x))
    .write_jsonl("gs://output/data-{shard:05d}-of-{total:05d}.jsonl.gz")
)
list(backend.execute(pipeline))
```

## Key Patterns

**Dataset Creation:**
- `Dataset.from_files(path, pattern)` - glob files
- `Dataset.from_list(items)` - explicit list

**Loading Files**
- `.load_{file,parquet,jsonl,vortex}` - load rows from a file

**Transformations:**
- `.map(fn)` - transform each item
- `.flat_map(fn)` - expand items (e.g., `load_jsonl`)
- `.filter(fn)` - filter items by function or expression
- `.select(columna, columnb)` - select out the given columns
- `.window(n)` - group into batches
- `.reshard(n)` - redistribute across n shards

**Output:**
- `.write_jsonl(pattern)` - write JSONL (gzip if `.gz`)
- `.write_parquet(pattern, schema)` - write to a Parquet file
- `.write_vortx(pattern)` - write to a Vortex file

**Backends:**
- `RayBackend(max_parallelism=N)` - distributed Ray execution
- `ThreadPoolBackend(max_parallelism=N)` - thread pool
- `SyncBackend()` - synchronous (testing)
- `flow_backend()` - adaptive (infers from CLI args)

## Real Usage

**Wikipedia Processing:**
```python
from zephyr import Dataset, flow_backend, load_jsonl

backend = flow_backend()
pipeline = (
    Dataset.from_list(files)
    .load_jsonl()
    .map(lambda row: process_record(row, config))
    .filter(lambda x: x is not None)
    .write_jsonl(f"{output}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
)
list(backend.execute(pipeline))
```

**Dataset Sampling:**
```python
from zephyr import Dataset, create_backend

backend = create_backend("ray", max_parallelism=1000)
pipeline = (
    Dataset.from_files(input_path, "**/*.jsonl.gz")
    .map(lambda path: sample_file(path, weights))
    .write_jsonl(f"{output}/sampled-{{shard:05d}}.jsonl.gz")
)
list(backend.execute(pipeline))
```

**Parallel Downloads:**
```python
from zephyr import Dataset, flow_backend

tasks = [(config, fs, src, dst) for src, dst in file_pairs]
backend = flow_backend()
pipeline = Dataset.from_list(tasks).map(lambda t: download(*t))
list(backend.execute(pipeline))
```

## CLI Launcher

```bash
# Run in-process (inherits fray client from environment, defaults to local)
uv run zephyr script.py --max-parallelism=100 --memory=2GB

# Submit to Ray cluster
uv run zephyr script.py --cluster=ray --cluster-config=us-central2 --memory=2GB

# Submit to Iris cluster
uv run zephyr script.py --cluster=iris --cluster-config=lib/iris/examples/eu-west4.yaml --memory=2GB

# Show optimization plan
uv run zephyr script.py --dry-run
```

Your script needs a `main()` entry point:
```python
from zephyr import flow_backend, Dataset

def main():
    backend = flow_backend()  # Configured from CLI args
    pipeline = Dataset.from_files(...).map(...).write_jsonl(...)
    list(backend.execute(pipeline))
```

## Installation

```bash
# From Marin monorepo
uv sync

# Standalone
cd lib/zephyr
uv pip install -e .
```

## Running Tests

Zephyr tests run against multiple execution backends to ensure correctness across different environments.

### All Tests on Both Backends (Default)
```bash
uv run pytest lib/zephyr/tests
# Runs all tests on both Local and Iris backends
# Local Iris cluster is started automatically via ClusterManager
```

### Run Specific Backend Only
```bash
uv run pytest lib/zephyr/tests -k "local"
uv run pytest lib/zephyr/tests -k "iris"
uv run pytest lib/zephyr/tests -k "ray"
```

The Iris cluster is started once per test session and reused across all tests for efficiency.

## Design

Zephyr consolidates 100+ ad-hoc Ray/HF dataset patterns in Marin into a simple abstraction. See [docs/design.md](docs/design.md) for details.

**Key Features:**
- Lazy evaluation with operation fusion
- Disk-based inter-stage data flow for low memory footprint
- Chunk-by-chunk streaming to minimize memory pressure
- Distributed execution with bounded parallelism (Ray/Iris/local backends)
- Automatic chunking to prevent large object overhead
- fsspec integration (GCS, S3, local)
- Type-safe operation chaining

## Execution Control Flow

Zephyr uses an actor-based execution model with a pull-based task distribution pattern. This section documents how data flows through the system and how execution is coordinated.

### Architecture Overview

```
User Code
    │
    ▼
ZephyrContext(client, num_workers, resources)
    │
    ├── .put(name, obj)              # broadcast shared data
    ├── .execute(dataset)             # run pipeline
    │       │
    │       ▼
    │   ZephyrCoordinator (fray actor, "zephyr-controller-{instance_id}")
    │       │   - Task queue per stage
    │       │   - Worker liveness tracking
    │       │   - Result collection
    │       │   - pull_task() / report_result() / report_error()
    │       │
    │   ZephyrWorker actors (fray actor_group, "zephyr-worker-{instance_id}")
    │       │   - Persistent across stages
    │       │   - Pull tasks from coordinator
    │       │   - Execute shard operations
    │       │   - Report results back
    │       │
    │   User process polls coordinator.get_status() for progress
```

### Data Flow Between Stages

**Important**: Zephyr passes data between stages via **disk-based chunk references**, not in-memory. Each stage:

1. **Receives input** as `ShardRefs` (references to chunks stored on disk)
2. **Workers stream chunks** one at a time from disk via `_SerializableShard` to minimize memory pressure
3. **Executes operations** on each shard, processing chunks lazily
4. **Writes results** back to disk and returns `ChunkRef` objects (disk references)
5. **Coordinator collects** chunk references and regroups them into new `ShardRefs`
6. **Next stage uses** those disk references as input

**File I/O happens at multiple points:**
- **Source operations** (e.g., `from_files`, `load_jsonl`) read files from GCS/S3/local and write to intermediate chunk storage
- **Each stage** reads chunks from disk, processes them, and writes results back to disk
- **Sink operations** (e.g., `write_jsonl`, `write_parquet`) write final output files to storage
- **Only final materialization** loads data into memory via `shard_refs.load().materialize()`

This design enables:
- **Low memory footprint** - workers stream one chunk at a time instead of loading entire shards
- **Large-scale pipelines** - data size limited by disk, not RAM
- **Fault tolerance** - intermediate results persisted to disk automatically
- **Automatic chunking** to prevent large object overhead in the execution framework

### Execution Sequence

When you call `ctx.execute(dataset)`:

```python
# 1. Create plan from dataset operations
plan = compute_plan(dataset, hints)

# 2. Get or create coordinator + workers
coordinator = self._get_or_create_coordinator()

# 3. Configure chunk storage and broadcast shared data (BLOCKING)
coordinator.set_chunk_config.remote(chunk_prefix, execution_id).result()
coordinator.set_shared_data.remote(self._shared_data).result()

# 4. Start worker run loops (ASYNC - workers pull tasks in background)
self._worker_futures = [w.run_loop.remote(coordinator) for w in self._workers]

# 5. Build source shards and write to disk immediately
source_data = _build_source_shards(plan.source_items)
shard_refs = source_data.write_to_disk(chunk_prefix, execution_id, "source")

# 6. Execute each stage sequentially
for stage_idx, stage in enumerate(plan.stages):
    # 6a. Convert shard_refs (disk references) to tasks
    tasks = _shard_refs_to_tasks(shard_refs, stage, hints)

    # 6b. Load tasks into coordinator queue (BLOCKING - critical!)
    coordinator.start_stage.remote(stage_name, tasks).result()

    # 6c. Poll until stage completes
    while True:
        coordinator.check_heartbeats.remote()  # Monitor worker health
        status = coordinator.get_status.remote().result()
        if status["fatal_error"]:
            raise ZephyrWorkerError(status["fatal_error"])
        if status["completed"] >= status["total"]:
            break
        time.sleep(0.1)

    # 6d. Collect chunk references (not data!) and regroup by output shard
    result_refs = coordinator.collect_results.remote().result()
    shard_refs = _regroup_result_refs(result_refs, len(shard_refs))

# 7. Signal workers to stop (BLOCKING)
coordinator.signal_done.remote().result()

# 8. Wait for workers to finish
for f in self._worker_futures:
    f.result(timeout=10.0)

# 9. Load from disk and materialize final results
return shard_refs.load().materialize()
```

### Worker Loop

Each worker runs this loop concurrently:

```python
while not shutdown:
    # 1. Pull task from coordinator (includes task attempt ID for duplicate detection)
    task_and_attempt = coordinator.pull_task.remote(worker_id).result()

    if task_and_attempt is None:
        # No task available - check if done
        status = coordinator.get_status.remote().result()
        if status["done"] or status["fatal_error"]:
            break
        time.sleep(0.1)
        continue

    task, attempt = task_and_attempt

    # 2. Execute shard operations, streaming chunks from disk
    #    Workers receive _SerializableShard which loads chunks lazily
    result_chunk_refs = self._execute_shard(task)  # Returns list[ChunkRef]

    # 3. Report chunk references (not data!) back to coordinator
    coordinator.report_result.remote(worker_id, task.shard_idx, attempt, result_chunk_refs)
```

### Critical Synchronization Points

**Why `.result()` is essential on certain calls:**

1. **`set_chunk_config.remote().result()`** - Must complete before workers start so they know where to read/write chunks
2. **`set_shared_data.remote().result()`** - Must complete before workers start, otherwise workers get empty shared data
3. **`start_stage.remote().result()`** - Must complete before polling status, otherwise:
   - Main thread polls `get_status()` and sees `total=0` (stage not started yet)
   - Polling loop exits immediately because `completed=0, total=0` → `0 >= 0`
   - Results collected before workers even get tasks
   - Race condition → empty results!
4. **`signal_done.remote().result()`** - Ensures workers see the done signal before context cleanup

**IMPORTANT**: Without these `.result()` calls, the async nature of distributed actors creates races where:
- Main thread thinks stage is done before it starts
- Workers never get tasks because coordinator is already "done"
- Empty results returned despite tasks being created
- Workers may not know where to find chunk files on disk

### Shard Data Format

Shards flow through the system as disk-backed references:

```python
# ShardRefs: disk-based representation (primary inter-stage format)
#   - List of shards, where each shard is a list of ChunkRef objects
#   - ChunkRef points to a pickled chunk file on disk

ShardRefs(
    shards=[
        [ChunkRef(path="/tmp/zephyr/.../chunk-0-0.pkl", count=2),
         ChunkRef(path="/tmp/zephyr/.../chunk-0-1.pkl", count=2)],  # Shard 0
        [ChunkRef(path="/tmp/zephyr/.../chunk-1-0.pkl", count=2)],  # Shard 1
    ]
)
```

**Worker execution** streams chunks from disk:

```python
# Workers receive _SerializableShard which lazily streams chunks
# Process one chunk at a time:
for chunk in shard.iter_chunks():  # Loads one chunk from disk at a time
    results = process_chunk(chunk)
    # Write results to disk immediately
    chunk_ref = ChunkRef.write(path, results)
```

**Worker results** are disk references:

```python
# Worker returns: list[StageResult] where StageResult = tuple[dict, ChunkRef]
results = [
    ({"shard_idx": 0, "count": 2}, ChunkRef(path=".../result-0-0.pkl", count=2)),
    ({"shard_idx": 0, "count": 2}, ChunkRef(path=".../result-0-1.pkl", count=2)),
]
```

**Coordinator collection** aggregates `ChunkRef` objects by output shard index and constructs new `ShardRefs` for the next stage, all without loading data into memory.

### Error Handling

Workers classify errors as **transient** (retryable) or **permanent** (fatal):

- **Transient**: Connection errors, node preemption → task re-queued
- **Permanent**: User code bugs, invalid data → coordinator sets `fatal_error`, main thread raises exception

This enables resilient execution on preemptible infrastructure while surfacing real bugs immediately.



### Notes

#### MacOS

Ray 2.53 enables a `uv run` runtime_env hook by default. When tests are executed
via `uv run pytest`, that hook can (a) start local workers with a different
Python version and (b) attempt to inspect the process tree with psutil, which
may fail in sandboxed Codex environments, at least on MacOS. Disable it for tests.
See https://github.com/ray-project/ray/issues/59639 for (a). (b) seems like a reasonable
design choice.
