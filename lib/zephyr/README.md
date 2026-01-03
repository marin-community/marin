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
# Run with Ray backend
uv run zephyr script.py --backend=ray --max-parallelism=100 --memory=2GB

# Submit to cluster
uv run zephyr script.py --backend=ray --cluster=us-central2 --memory=2GB

# Show optimization plan
uv run zephyr script.py --backend=ray --dry-run
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

## Design

Zephyr consolidates 100+ ad-hoc Ray/HF dataset patterns in Marin into a simple abstraction. See [docs/design.md](docs/design.md) for details.

**Key Features:**
- Lazy evaluation with operation fusion
- Ray backend with bounded parallelism
- Automatic chunking to prevent large object overhead
- fsspec integration (GCS, S3, local)
- Type-safe operation chaining



### Notes

#### MacOS

Ray 2.53 enables a `uv run` runtime_env hook by default. When tests are executed
via `uv run pytest`, that hook can (a) start local workers with a different
Python version and (b) attempt to inspect the process tree with psutil, which
may fail in sandboxed Codex environments, at least on MacOS. Disable it for tests.
See https://github.com/ray-project/ray/issues/59639 for (a). (b) seems like a reasonable
design choice.
