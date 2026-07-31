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

### Shared worker pools

A plain `execute()` creates a dedicated coordinator and worker group for that
pipeline. Entering a context starts one pool that remains available for every
execution until context exit:

```python
with ZephyrContext(max_workers=100, resources=ResourceConfig(cpu=2, ram="64g")) as ctx:
    first = ctx.execute(first_pipeline)
    second = ctx.execute(second_pipeline)
```

Concurrent executions share the workers according to each execution's map and
reduce task resources. Passing an entered context to another function reuses
the same pool. A serialized child receives a borrowed coordinator handle and
can submit pipelines, but cannot shut down the pool.

### Read-only memory stores

`ZephyrContext.load_memory_store()` loads an existing partitioned dataset into
a context-owned group of read-only actors. The handle is picklable, so later
pipelines and child jobs can use `get()` or order-preserving `get_many()`
lookups without copying the table into every worker.

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
`store.stats()` reports measured load time and size per actor. Iris reconstructs
a preempted actor from the same source shards, and lookups wait for their owner
up to `recovery_timeout`. Exiting the creating context stops its stores after
the Zephyr worker pool drains.

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
