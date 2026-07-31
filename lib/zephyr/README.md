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

Each `execute()` starts a worker pool sized to the pipeline, runs it, and tears
it down when the pipeline finishes. Workers are released as the final stage
drains, so capacity goes back to the cluster while stragglers finish.

**Standing pool (one pool, many pipelines):**

When many small pipelines each pay that startup cost, stand the pool up once
instead. `mode=PoolMode.HOST` makes a context the pool's host: entering starts
it, leaving tears it down, and any job that names the pool runs on it.

`mode=PoolMode.HOST` makes a context the pool's host: entering starts the pool,
leaving tears it down, and hosting advertises the pool's address to every job
submitted afterwards. Steps need no wiring of their own:

```python
with ZephyrContext(
    mode=PoolMode.HOST,
    pool_name="ingest",
    max_workers=200,
    resources=ResourceConfig(cpu=2, ram="8g"),
) as ctx:
    for step in steps:
        client.submit(JobRequest(name=step, ...))   # inherits the pool
    ctx.execute(pipeline)                            # the host runs on it too
# pool is shut down here (workers drained)
```

Each step is then untouched — it inherits `ZEPHYR_COORDINATOR_ENDPOINT` and
joins:

```python
ZephyrContext(map_task_resources=ResourceConfig(cpu=1, ram="2g")).execute(pipeline)
```

The advertised value is the coordinator's absolute address, not the pool's
name. A name would have to be resolved relative to the caller, which is only
correct one level below the host; an address resolves from any job, at any
depth, and from a job with no parent at all. A driver outside the job tree
passes `coordinator_endpoint=ctx.coordinator_endpoint` explicitly.

Tasks from all active pipelines pack onto whichever workers have free capacity.
A failing pipeline only fails its own `execute()`; the pool and the other
pipelines keep running.

`mode` is how a context accepts or refuses the pool the environment offers:

- `AUTO` (default) — join the offered pool if there is one, else run on a
  one-shot pool of its own
- `INHERIT` — require an offered pool, and fail fast if none is configured
  rather than quietly starting a private one
- `ISOLATED` — never join; always this context's own pool, ignoring the
  environment. The opt-out for a step whose stages need something the shared
  workers lack
- `HOST` — stand up the pool named `pool_name` and own its lifetime



The pool's workers are sized by the host's `resources` × `max_workers`.
Each joining pipeline still declares its own per-task cost via its own
`map/reduce_task_resources`, so a pipeline needing more memory than a worker can
provide is rejected up front rather than deadlocking unscheduled. Preempted
workers are replenished by Iris automatically; the pool lives until the host
calls `shutdown()` (or its `with` block exits).

Because the pool's workers are generic — they run every joining pipeline's
stage code — the host must launch the pool with the environment those stages
need, rather than each step bringing its own. Set `pip_dependency_groups` for
uv extras and `job_env_vars` for environment variables:

```python
ZephyrContext(
    mode=PoolMode.HOST,
    pool_name="datakit",
    max_workers=200,
    resources=ResourceConfig(cpu=2, ram="8g"),
    coordinator_resources=ResourceConfig(cpu=1, ram="4g"),
    pip_dependency_groups=["datakit"],        # workers get luxical / faiss / sklearn / scipy
    job_env_vars={"JAX_PLATFORMS": "cpu"},    # jax stages don't probe CUDA on GPU nodes
)
```

A standing coordinator holds the live state for every concurrent pipeline.
Size `coordinator_resources` for aggregate concurrency; the lightweight default
is intended for ordinary contexts that run one pipeline at a time.

A step whose stages need something the pool's workers lack opts out with
`mode=PoolMode.ISOLATED` and gets its own pool.

Both default to `None`, in which case the pool job inherits its parent's
environment exactly as before.

**Read-only memory stores:**

`ZephyrContext.load_memory_store()` loads an existing partitioned Dataset into
a context-owned group of read-only actors. The handle is picklable, so later
pipelines can use `get()` or order-preserving `get_many()` lookups without
copying the table into every worker.

```python
from fray.types import ActorConfig, ResourceConfig


def document_partition(key: tuple[int, str]) -> int:
    file_index, _ = key
    return file_index


documents = Dataset.from_files("s3://bucket/documents/*.parquet").load_parquet().map(
    lambda row: ((row["file_index"], row["id"]), row["text"])
)

with ZephyrContext(max_workers=100) as ctx:
    document_store = ctx.load_memory_store(
        documents,
        name="documents",
        hash_key=document_partition,
        num_actors=16,
        actor_resources=ResourceConfig(cpu=2, ram="8g"),
        actor_config=ActorConfig(max_task_retries=1_000),
        max_actor_bytes=2_000_000_000,
        recovery_timeout=900,
    )
    result = ctx.execute(Dataset.from_list(document_keys).map(document_store.get))
```

For `P` source shards, every key must already satisfy
`hash_key(key) % P == source_shard_index`. Construction checks every row and
does not insert a shuffle. Readers and shard-local maps can load directly.
Persist and reload the output of a shuffle, join, reshard, reduce, or write
before constructing a store.

Keys must be unique and deterministically encodable by msgspec. Values and the
hash function must be picklable; Python's salted `hash()` is not stable for
string or byte keys. `max_actor_bytes` limits encoded key/value bytes, while
`store.stats()` reports the measured load per actor. Iris reconstructs a
preempted actor from the same source shards, and lookups wait for their owner up
to `recovery_timeout`. Exiting the creating context stops its stores after the
Zephyr worker pool drains.

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
