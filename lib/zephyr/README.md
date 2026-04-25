# zephyr

Simple data processing library for Marin pipelines. Build lazy dataset pipelines that run on Ray clusters, thread pools, or synchronously.

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
- `ZephyrContext(max_workers=N)` — auto-detects backend (Ray, Iris, or local) via `fray.current_client()`
- `ZephyrContext(client=LocalClient())` — explicit local backend (testing)
- `ctx.execute(pipeline)` — runs the pipeline; returns a `ZephyrExecutionResult(results, counters)`

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
uv run pytest lib/zephyr/tests -k "ray"
```

The Iris cluster is started once per test session and reused across all tests for efficiency.

## Design

Zephyr consolidates 100+ ad-hoc Ray/HF dataset patterns in Marin into a simple abstraction.

**Key Features:**
- Lazy evaluation with operation fusion
- Disk-based inter-stage data flow for low memory footprint
- Chunk-by-chunk streaming to minimize memory pressure
- Distributed execution with bounded parallelism (Ray/Iris/local backends)
- Automatic chunking to prevent large object overhead
- fsspec integration (GCS, S3, local)
- Type-safe operation chaining

See `AGENTS.md` for execution internals and source layout.
