# Zephyr

Simple data processing library for Marin pipelines. Build lazy dataset pipelines that run on Iris jobs or a local backend.

## Quick Start

```python
from zephyr import Dataset, ZephyrContext, load_jsonl

# Read, transform, write
ctx = ZephyrContext(max_workers=100)
pipeline = (
    Dataset.from_files("gs://input/", "**/*.jsonl.gz")
    .flat_map(load_jsonl)
    .filter(lambda x: x["score"] > 0.5)
    .map(lambda x: transform_record(x))
    .write_jsonl("gs://output/data-{shard:05d}-of-{total:05d}.jsonl.gz")
)
ctx.execute(pipeline)
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
- `.write_vortex(pattern)` - write to a Vortex file

**Execution (`ZephyrContext`):**
- `ZephyrContext(max_workers=N)` — auto-detects the backend (Iris inside an Iris job, local otherwise) via `fray.current_client()`
- `ZephyrContext(client=LocalClient())` — explicit local backend (testing)
- `ctx.execute(pipeline)` — runs the pipeline; returns a `ZephyrExecutionResult(results, counters)`

Each `execute()` submits its own coordinator + worker job and tears it down when
the pipeline finishes.

**Shared pool (`ZephyrPool` — one pool, many pipelines):**

A `ZephyrPool` starts a single long-lived coordinator + worker pool once, then
serves many pipelines against it — concurrently, and from other drivers/steps.
Open it as a `with` block to get the coordinator endpoint; the pool is torn
down when the block exits (including on exception):

```python
with ZephyrPool(max_workers=200, resources=ResourceConfig(cpu=2, ram="8g"), name="ingest") as endpoint:
    # Any driver (even in another Iris job) connects and submits pipelines.
    # Tasks are scheduled onto whichever workers have free capacity, so several
    # pipelines share the pool. A failing pipeline only fails its own execute();
    # the pool and the other pipelines keep running.
    driver = ZephyrContext(coordinator_endpoint=endpoint, resources=ResourceConfig(cpu=1, ram="2g"))
    result = driver.execute(pipeline)   # blocks until this pipeline completes
# pool is shut down here (workers drained)
```

`ZephyrPool.start()` / `shutdown()` are also available directly if you'd rather
manage the pool's lifetime yourself. Plain `ZephyrContext(...).execute(pipeline)`
with no endpoint is unchanged — a dedicated coordinator per pipeline, as before.

Connecting drivers can pick the endpoint up from the environment instead of a
constructor argument — set `ZEPHYR_COORDINATOR_ENDPOINT` on a step's job (e.g.
Iris `-e ZEPHYR_COORDINATOR_ENDPOINT <endpoint>`), and a plain
`ZephyrContext()` in that step connects to the pool automatically:

```python
# In a step launched with ZEPHYR_COORDINATOR_ENDPOINT set:
ctx = ZephyrContext(resources=ResourceConfig(cpu=1, ram="2g"))  # picks up the env endpoint
ctx.execute(pipeline)
```

The pool's workers are sized by `ZephyrPool`'s `resources` × `max_workers`.
Each connecting pipeline still declares its own per-task cost via the driver's
`map/reduce_task_resources`, so a pipeline needing more memory than a worker can
provide is rejected up front rather than deadlocking unscheduled. Preempted
workers are replenished by Iris automatically; the pool lives until the owner
calls `shutdown()` (or the `with` block exits).

Because the pool's workers are generic — they run every connecting pipeline's
stage code — the pool must be launched with the environment those stages need,
rather than inheriting it per-step. Set `pip_dependency_groups` for uv extras
and `job_env_vars` for environment variables:

```python
ZephyrPool(
    name="datakit",
    max_workers=200,
    resources=ResourceConfig(cpu=2, ram="8g"),
    pip_dependency_groups=["datakit"],        # workers get luxical / faiss / sklearn / scipy
    job_env_vars={"JAX_PLATFORMS": "cpu"},    # jax stages don't probe CUDA on GPU nodes
)
```

Both default to `None`, in which case the pool job inherits its parent's
environment exactly as before.

## Real Usage

**Wikipedia Processing:**
```python
from zephyr import Dataset, ZephyrContext, load_jsonl

ctx = ZephyrContext(max_workers=100)
pipeline = (
    Dataset.from_list(files)
    .load_jsonl()
    .map(lambda row: process_record(row, config))
    .filter(lambda x: x is not None)
    .write_jsonl(f"{output}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
)
ctx.execute(pipeline)
```

**Dataset Sampling:**
```python
from zephyr import Dataset, ZephyrContext

ctx = ZephyrContext(max_workers=1000)
pipeline = (
    Dataset.from_files(input_path, "**/*.jsonl.gz")
    .map(lambda path: sample_file(path, weights))
    .write_jsonl(f"{output}/sampled-{{shard:05d}}.jsonl.gz")
)
ctx.execute(pipeline)
```

**Parallel Downloads:**
```python
from zephyr import Dataset, ZephyrContext

tasks = [(config, fs, src, dst) for src, dst in file_pairs]
ctx = ZephyrContext(max_workers=32)
pipeline = Dataset.from_list(tasks).map(lambda t: download(*t))
ctx.execute(pipeline)
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
```

The Iris cluster is started once per test session and reused across all tests for efficiency.

## Design

Zephyr consolidates ad-hoc distributed and Hugging Face dataset processing patterns in Marin into a simple abstraction.

**Key Features:**
- Lazy evaluation with operation fusion
- Disk-based inter-stage data flow for low memory footprint
- Chunk-by-chunk streaming to minimize memory pressure
- Distributed execution with bounded parallelism (Iris/local backends)
- Automatic chunking to prevent large object overhead
- fsspec integration (GCS, S3, local)
- Type-safe operation chaining

See `AGENTS.md` for execution internals and source layout.
