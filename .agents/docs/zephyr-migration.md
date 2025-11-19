# Zephyr Migration Patterns

**Last Updated**: 2025-01-07

## Status Summary

**Completed**: 19 files migrated ✅
**Remaining**: 33 files to migrate
**Not Suitable**: ~39 files (TPU orchestration, RL training, stateful orchestration, infrastructure, etc.)

This document describes concrete patterns for migrating Ray boilerplate to zephyr. Each section shows a real codebase example with before/after code.

## Backend Configuration: `flow_backend()` vs `create_backend()`

Zephyr provides two ways to configure backends:

### `flow_backend()` (Recommended)
Use `flow_backend()` to get the backend configured via the zephyr CLI. This is the recommended approach for most scripts:

```python
from zephyr import Dataset, flow_backend

def main():
    backend = flow_backend()  # Get backend from CLI context
    pipeline = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    backend.execute(pipeline)
```

Run with: `uv run zephyr --backend=ray --max-parallelism=1000 --memory=8GB script.py`

**Override specific parameters** while inheriting other settings from CLI:
```python
# CLI: zephyr --backend=ray --max-parallelism=50 --memory=2GB
outer_backend = flow_backend()              # Uses CLI config: 50 workers, 2GB
inner_backend = flow_backend(max_parallelism=16)  # Override: 16 workers, 2GB (inherited)
```

### `create_backend()` (Advanced)
Only use `create_backend()` when you need programmatic backend creation outside the CLI context:

```python
from zephyr import Dataset, create_backend

# Programmatic creation - not recommended for most use cases
backend = create_backend("ray", max_parallelism=1000, memory="8GB")
```

**For all examples below, we use `flow_backend()` as it's the recommended pattern.**

## Pattern 1: Bounded Parallel Map

**What it replaces**: Manual `@ray.remote` + `ray.wait()` loops that maintain bounded parallelism.

**When to use**: Processing a list of items (files, URLs, records) in parallel with a concurrency limit.

### Before

```python
@ray.remote(memory=8 * 1024 * 1024 * 1024)
def count_tokens_in_shard(shard_path: str, output_path: str, tokenizer_name: str):
    """Process one shard and return token counts"""
    # ... processing logic ...
    return num_tokens, num_documents

def count_tokens(input_patterns: list[str], output_path: str, tokenizer_name: str):
    # Get input paths
    input_paths = []
    for pattern in input_patterns:
        input_paths.extend(fsspec_glob(pattern))

    # Manual bounded parallelism
    MAX_CONCURRENT_TASKS = 1000
    num_shards_submitted = 0
    unfinished = []

    # Submit initial batch
    for _ in range(min(MAX_CONCURRENT_TASKS, len(input_paths))):
        shard_path = input_paths.pop()
        output_file = os.path.join(output_path, os.path.basename(shard_path) + ".token_counts")
        unfinished.append(count_tokens_in_shard.remote(shard_path, output_file, tokenizer_name))
        num_shards_submitted += 1

    # Manual ray.wait() loop to maintain bounded parallelism
    num_tokens = 0
    num_documents = 0
    with tqdm(total=len(input_paths) + num_shards_submitted, desc="Counting") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            results = ray.get(finished)
            for shard_num_tokens, shard_num_documents in results:
                num_tokens += shard_num_tokens
                num_documents += shard_num_documents
                pbar.update(1)

            # Submit more tasks to maintain MAX_CONCURRENT_TASKS
            while input_paths and len(unfinished) < MAX_CONCURRENT_TASKS:
                shard_path = input_paths.pop()
                output_file = os.path.join(output_path, os.path.basename(shard_path) + ".token_counts")
                unfinished.append(count_tokens_in_shard.remote(shard_path, output_file, tokenizer_name))
```

### After

```python
from zephyr import Dataset, flow_backend

def count_tokens_in_shard(shard_path: str, output_path: str, tokenizer_name: str):
    """Process one shard and return token counts - no @ray.remote needed"""
    # ... same processing logic ...
    return num_tokens, num_documents

def count_tokens(input_patterns: list[str], output_path: str, tokenizer_name: str):
    # Get input paths
    input_paths = []
    for pattern in input_patterns:
        input_paths.extend(fsspec_glob(pattern))

    # Get backend from CLI context (configured via zephyr --backend=ray --max-parallelism=1000 --memory=8GB)
    backend = flow_backend()

    # Build pipeline
    pipeline = (
        Dataset.from_list(input_paths)
        .map(lambda shard_path: count_tokens_in_shard(
            shard_path,
            os.path.join(output_path, os.path.basename(shard_path) + ".token_counts"),
            tokenizer_name
        ))
    )

    # Execute and aggregate results
    results = list(backend.execute(pipeline))
    num_tokens = sum(r[0] for r in results)
    num_documents = sum(r[1] for r in results)

    logger.info(f"Total tokens: {num_tokens}, documents: {num_documents}")
```

**Key changes**:
- Remove `@ray.remote` decorator - resources specified via CLI flags (e.g., `--memory=8GB`)
- Replace 50 lines of manual coordination with `Dataset.from_list().map()`
- Bounded parallelism handled by `--max-parallelism` CLI flag
- Use `flow_backend()` to get backend configured via CLI
- No manual state tracking or `ray.wait()` loops

## Pattern 2: File-to-Records with flat_map

**What it replaces**: Functions that read files and yield multiple records, coordinated with manual `ray.wait()`.

**When to use**: Transform jobs where each input file produces many output records (JSONL, Parquet, etc.).

### Before

```python
@ray.remote(memory=512 * 1024 * 1024)
def clean_ar5iv_html(file: str, output_path: str, file_size: int):
    """Read file line-by-line, clean HTML, write to output file"""
    outs = ""
    with fsspec.open(file, "rb", compression="gzip") as f:
        for _ in range(file_size):
            line = f.readline()
            if not line:
                break
            html_blob = json.loads(line)
            content = clean_html(html_blob["text"])
            outs += json.dumps({
                "id": html_blob["id"],
                "text": content,
                "source": "ar5iv",
                "added": datetime.datetime.now().isoformat(),
            }) + "\n"

    # Write accumulated output
    output_file = os.path.join(output_path, os.path.basename(file))
    with fsspec.open(output_file, "wt", compression="gzip") as out:
        out.write(outs)
    return True

# Manual coordination
MAX_NUM_PENDING_TASKS = 600
result_refs = []

for html_file in files:
    if len(result_refs) > MAX_NUM_PENDING_TASKS:
        ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
        ray.get(ready_refs)

    result_refs.append(clean_ar5iv_html.remote(html_file, output_path, args.file_size))

ray.get(result_refs)
```

### After

```python
from zephyr import Dataset, flow_backend, load_jsonl

def clean_ar5iv_record(html_blob: dict) -> dict:
    """Process a single record - pure function"""
    content = clean_html(html_blob["text"])
    return {
        "id": html_blob["id"],
        "text": content,
        "source": "ar5iv",
        "added": datetime.datetime.now().isoformat(),
    }

# Build streaming pipeline
backend = flow_backend()  # Backend configured via CLI: zephyr --backend=ray --max-parallelism=600 --memory=512MB
pipeline = (
    Dataset.from_files(html_folder, "*.jsonl.gz")
    .flat_map(load_jsonl)              # Read file, yield records
    .map(clean_ar5iv_record)           # Transform each record
    .write_jsonl(f"{output_path}/{{shard:05d}}.jsonl.gz")
)

list(backend.execute(pipeline))
```

**Key changes**:
- Split file I/O (handled by `load_jsonl`) from business logic (`clean_ar5iv_record`)
- Use `.flat_map()` to read files and yield records - this is critical for file processing
- Remove manual accumulation of output strings
- Remove manual `ray.wait()` coordination
- Streaming: records never fully materialize in memory

**Why flat_map?** A function that reads a file has signature `str -> Iterator[dict]`. Using `.map()` would give `Iterator[Iterator[dict]]` (nested). Using `.flat_map()` automatically flattens to `Iterator[dict]`.

## Pattern 4: Join with In-Memory Attributes

**What it replaces**: Joining large document files with small attribute files using HuggingFace Dataset operations.

**When to use**: Filtering documents based on attributes (scores, labels) stored in separate files that fit in memory.

### Before

```python
from marin.processing.classification.dataset_utils import read_dataset, write_dataset
from functools import partial

def apply_filter_classify(doc: dict, doc_filter: FilterConfig, id_to_attributes: dict) -> bool:
    """Check if document passes threshold"""
    attributes = id_to_attributes[doc["id"]]
    value = attributes[doc_filter.name][doc_filter.label]

    if doc_filter.lower_threshold is not None and value < doc_filter.lower_threshold:
        return False
    if doc_filter.upper_threshold is not None and value > doc_filter.upper_threshold:
        return False
    return True

@ray.remote
def process_file(input_path: str, output_path: str, filters: list[FilterConfig]):
    # Load each attribute file into memory
    attribute_files = []
    for doc_filter in filters:
        table = read_dataset(doc_filter.attribute_path, columns=["id", "attributes"])
        id_to_attrs = {row["id"]: row["attributes"] for row in table}
        attribute_files.append(id_to_attrs)

    # Load entire document file
    dataset = read_dataset(input_path)

    # Apply each filter sequentially - materializes filtered dataset each time
    for doc_filter, id_to_attrs in zip(filters, attribute_files, strict=True):
        dataset = dataset.filter(
            partial(apply_filter_classify,
                   doc_filter=doc_filter,
                   id_to_attributes=id_to_attrs)
        )

    # Write materialized result
    write_dataset(dataset, output_path)
```

### After

```python
from zephyr import Dataset, flow_backend, load_jsonl

def process_file(input_path: str, output_path: str, filters: list[FilterConfig]):
    # Load attribute files into memory (streaming read, but result fits in memory)
    id_to_attributes = {}
    for doc_filter in filters:
        attrs = {}
        for row in load_jsonl(doc_filter.attribute_path):
            attrs[row["id"]] = row["attributes"]
        id_to_attributes[doc_filter.name] = attrs

    # Create filter function that captures all attributes
    def apply_all_filters(doc: dict) -> bool:
        """Closure captures id_to_attributes"""
        for doc_filter in filters:
            if doc_filter.name not in id_to_attributes:
                continue

            attrs = id_to_attributes[doc_filter.name]
            if doc["id"] not in attrs:
                return False

            value = attrs[doc["id"]][doc_filter.name][doc_filter.label]

            if doc_filter.lower_threshold is not None and value < doc_filter.lower_threshold:
                return False
            if doc_filter.upper_threshold is not None and value > doc_filter.upper_threshold:
                return False

        return True

    # Stream documents through composed filter
    backend = flow_backend()  # Backend configured via CLI: zephyr --backend=ray --max-parallelism=1
    pipeline = (
        Dataset.from_list([input_path])
        .flat_map(load_jsonl)
        .filter(apply_all_filters)      # Applies all filters in one pass
        .write_jsonl(output_path)
    )

    list(backend.execute(pipeline))
```

**Key changes**:
- Load attributes by streaming with `load_jsonl()` instead of `read_dataset()`
- Single closure captures all attributes instead of multiple `partial()` calls
- Stream documents - never materialize full dataset or intermediate filtered results
- Memory: only attributes in RAM, documents stream through
- Cleaner composition: one filter function instead of chaining multiple filters

## Pattern 5: Nested Parallelism

**What it replaces**: Outer `@ray.remote` function that spawns inner `@ray.remote` tasks with two levels of bounded parallelism.

**When to use**: Hierarchical processing (e.g., shards → files → records) where both levels need concurrency limits.

### Before

```python
# Inner level: Process individual files
@ray.remote(memory=2 * 1024 * 1024 * 1024, max_retries=5)
def process_file(input_file_path: str, output_file_path: str) -> None:
    """Process a single .json.zst file"""
    with fsspec.open(input_file_path, compression="zstd") as source:
        with fsspec.open(output_file_path, "wt", compression="gzip") as output:
            for line in source:
                row = json.loads(line.strip())
                # Extract HTML from Common Crawl
                html_string = find_html_in_cc(row["metadata"]["WARC-Record-ID"])
                row["html"] = html_string
                print(json.dumps(row), file=output)

# Middle level: Process a shard (directory) of files
@ray.remote(memory=2 * 1024 * 1024 * 1024, max_retries=5)
def process_dclm_shard(input_path: str, output_path: str) -> None:
    """Process all files in a shard with bounded parallelism"""
    result_refs = []
    MAX_CONCURRENT_WORKERS = 16  # Inner parallelism limit

    shard_paths = fsspec_glob(os.path.join(input_path, "*.json.zst"))

    # Inner ray.wait() loop
    for shard_path in shard_paths:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            ray.get(ready_refs)

        output_file = os.path.join(output_path, os.path.basename(shard_path))
        result_refs.append(process_file.remote(shard_path, output_file))

    ray.get(result_refs)

# Outer level: Process all shards
def process_dataset(input_path: str, output_path: str) -> None:
    result_refs = []
    MAX_CONCURRENT_WORKERS = 50  # Outer parallelism limit

    shard_dirs = fsspec_glob(os.path.join(input_path, "*"))

    # Outer ray.wait() loop
    for shard_dir in shard_dirs:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            ray.get(ready_refs)

        output_dir = os.path.join(output_path, os.path.basename(shard_dir))
        result_refs.append(process_dclm_shard.remote(shard_dir, output_dir))

    ray.get(result_refs)
```

### After

```python
from zephyr import Dataset, flow_backend

def process_file(input_file_path: str, output_file_path: str) -> None:
    """Process a single .json.zst file - no @ray.remote needed"""
    with fsspec.open(input_file_path, compression="zstd") as source:
        with fsspec.open(output_file_path, "wt", compression="gzip") as output:
            for line in source:
                row = json.loads(line.strip())
                html_string = find_html_in_cc(row["metadata"]["WARC-Record-ID"])
                row["html"] = html_string
                print(json.dumps(row), file=output)

def process_dataset(input_path: str, output_path: str) -> None:
    # Get all shard directories
    shard_dirs = fsspec_glob(os.path.join(input_path, "*"))

    def get_files_in_shard(shard_dir: str):
        """Expand one shard into its files - yields file info dicts"""
        file_paths = fsspec_glob(os.path.join(shard_dir, "*.json.zst"))
        for file_path in file_paths:
            yield {
                "input": file_path,
                "output": os.path.join(output_path,
                                      os.path.basename(shard_dir),
                                      os.path.basename(file_path))
            }

    # Outer backend: 50 concurrent shards (configured via CLI)
    outer_backend = flow_backend()  # zephyr --backend=ray --max-parallelism=50 --memory=2GB

    # Inner backend: 16 concurrent files per shard (override parallelism)
    inner_backend = flow_backend(max_parallelism=16)

    pipeline = (
        Dataset.from_list(shard_dirs, backend=outer_backend)
        .flat_map(get_files_in_shard)           # Expands shards to files
        .map(lambda f: process_file(f["input"], f["output"]), backend=inner_backend)
    )

    list(outer_backend.execute(pipeline))
```

**Key changes**:
- Remove both `@ray.remote` decorators
- Remove both nested `ray.wait()` loops
- Use `flow_backend()` to get CLI-configured backend, override `max_parallelism` for inner backend
- Use `.flat_map()` to express shard → files relationship naturally
- Same total parallelism: 50 shards × 16 files = 800 concurrent tasks max
- Cleaner error handling and progress tracking built into Dataset

## Migration Checklist

For each file:

1. **Identify the pattern**: Which of the 5 patterns above does this file match?
2. **Remove Ray decorators**: Delete `@ray.remote` - configure resources via CLI instead
3. **Remove manual coordination**: Delete `ray.wait()` loops, `MAX_CONCURRENT_*` tracking
4. **Choose loader**: Use `load_jsonl`, `load_parquet`, or `load_zip_members` for file reading
5. **Use flat_map**: If function yields multiple records per input, use `.flat_map()` not `.map()`
6. **Capture side data**: Use closures for small in-memory data like attributes
7. **Configure backend**: Use `flow_backend()` to get CLI-configured backend (via `zephyr --backend=ray --max-parallelism=N --memory=M`)
   - For nested parallelism: use `flow_backend(max_parallelism=N)` to override specific parameters
   - Only use `create_backend()` if you need programmatic backend creation outside CLI context
8. **Test streaming**: Ensure pipeline doesn't materialize intermediate results
9. **Update imports**: Remove `ray.remote`, `ray.wait`, add `from zephyr import Dataset, flow_backend`
10. **Verify output**: Check that output format and sharding matches original

## Completed Migrations ✅

These files have been successfully migrated to zephyr:

### Downloads
- ✅ `download/uncheatable_eval/download.py` - Uses `Dataset` and `RayBackend`
- ✅ `download/huggingface/download_hf.py` - Uses `Dataset` and `RayBackend`
- ✅ `download/ar5iv/download.py` - Uses `Dataset.from_list().map()` with `flow_backend()`
- ✅ `download/nemotron_cc/download_nemotron_cc.py` - Uses `Dataset.from_list().map()` with `flow_backend()`
- ✅ `download/huggingface/stream_remove_columns.py` - Uses `Dataset.from_list().flat_map().map()` with `flow_backend()`
- ✅ `download/filesystem/transfer.py` - Uses `Dataset.from_list().map()` with `flow_backend()` for random file sampling
- ✅ `download/dclm_hq/download_dclm_hq_html.py` - Migrated with flattening pattern, uses `Dataset.from_list().map()` with `flow_backend()`

### Generation & Datashop
- ✅ `generation/dataset.py` - Uses `Dataset.from_files().flat_map().map()`
- ✅ `generation/chunk_utils.py` - Uses `Dataset` and `RayBackend`
- ✅ `datashop/dataset_processor.py` - Uses `Dataset.from_files().flat_map().map().write_jsonl()`

### Transforms
- ✅ `transform/dolmino/filter_dolmino.py` - Uses `Dataset.from_files().flat_map(load_jsonl).filter().write_jsonl()` with `flow_backend()`
- ✅ `transform/dolmino/transform_dclm_hq.py` - Migrated with flattened parallelism, globs all files upfront, uses `Dataset.from_list().map()` with `flow_backend()`
- ✅ `transform/conversation/transform_preference_data.py` - Uses `Dataset.from_list().flat_map().write_jsonl()` with `flow_backend()`
- ✅ `transform/fineweb/process_parquet_fw.py` - Migrated with top-level parallelism control, pandas/WARC logic preserved, uses `Dataset.from_list().map()` with `flow_backend()`
- ✅ `transform/lingoly/to_dolma.py` - Uses `Dataset.from_list().flat_map(load_zip_members).flat_map().write_jsonl()` with `flow_backend()`, added `load_zip_members()` helper to zephyr for zip file processing

### Classifiers
- ✅ `classifiers/utils.py` - Uses `Dataset` and `RayBackend`

### Crawl & Validation
- ✅ `crawl/count_tokens.py` - Uses `Dataset.from_list().flat_map(load_file).map()` with stateful TokenCounter class and `flow_backend()`
- ✅ `validation/count_total_tokens.py` - Uses `Dataset.from_list().flat_map(load_file).map()` with stateful TokenCounter class and `flow_backend()`
- ✅ `crawl/open_web_math/consolidate_open_web_math_shards.py` - Uses `Dataset.from_list().flat_map(load_jsonl).reshard().write_jsonl()` with `flow_backend()`

## Good Zephyr Migration Targets (Non-Crawl)

These files are **actually good fits** for zephyr - they process collections of files/records in data-parallel patterns:

| Priority | File | LOC | Pattern | Why It Fits |
|----------|------|-----|---------|-------------|
| 1 | `validate/validate.py` | 227 | Bounded parallel map + reduce | Validates list of files, aggregates statistics - classic data-parallel |
| 2 | `processing/classification/consolidate.py` | 417 | Data-parallel map + reduce | Processes files, filters based on attributes, DDSketch aggregation |
| 3 | `processing/classification/dedupe.py` | 1015 | Subprocess wrapper per file | Thin Ray wrapper around Dolma CLI for many files |

## Files NOT Suitable for Zephyr (Training/Orchestration)

These files don't fit zephyr's design - they're single-task orchestration, training jobs, or stateful actor pools:

### Single-Task Orchestration (Not Data Processing)
- `validation/get_env.py` - Single utility task, not a data pipeline
- `classifiers/hf/launch_ray_training.py` - Launches single training job
- `classifiers/fasttext/training.py` - Single training job execution
- `processing/tokenize/tokenize.py` - Orchestrates Levanter (single job), not data processing
- `classifiers/bert/training.py` - Single training job with checkpointing

### Stateful Actor Pools (Require Different Abstraction)
- `processing/classification/inference.py` - Stateful GPU/TPU actor pool for batch inference
- `generation/inference.py` - Stateful vLLM actor pool for serving

### Downloads
- `download/huggingface/download_gated_manual.py` - DEPRECATED, uses ThreadPool


## Files Not Suitable for Migration

These files should NOT be migrated to zephyr due to their specific requirements:

### Threading & Rate Limiting
- `crawl/fetch_links.py` - Threading with per-netloc rate limiting

### Replaced by zephyr
- `processing/classification/autoscaler.py` - zephyr replaces this functionality

### Infrastructure & Deprecated Downloads
- `download/huggingface/download.py` - Uses Google Storage Transfer Service for downloads; Ray only wraps STS job polling. Already deprecated in favor of `download_hf.py`.
- `download/huggingface/upload_gcs_to_hf.py` - Uses Google Cloud Storage's built-in transfer_manager with multiprocessing. No Ray usage. GCS transfer_manager is already optimized for this use case.

### Stateful Orchestration
- `datashop/pipeline.py` - Stateful orchestration and DAG coordination
- `execution/executor.py` - DAG framework
- `execution/status_actor.py` - Status tracking actor

### Alternative Frameworks
- `transform/fasttext/transform_beam.py` - Apache Beam pipeline

### Evaluation & TPU Orchestration
- `evaluation/log_probs.py` - Single Ray task for Levanter evaluation
- `evaluation/visualize.py` - Single Ray task for visualization
- `evaluation/evaluators/levanter_tpu_evaluator.py` - TPU orchestration
- `evaluation/evaluators/vllm_tpu_evaluator.py` - TPU orchestration

### Training Infrastructure
- `training/training.py` - Core training infrastructure
- `classifiers/hf/train_classifier.py` - HuggingFace training (uses HF datasets for data loading)

### RL Training Coordination
- `rl/**/*.py` - RL training coordination (15+ files)
- `rl/rl_job.py` - RL training coordination
- `rl/robust_actor.py` - RL actor wrapper
- `rl/evaluate_environment.py` - RL evaluation
- `rl/weight_transfer/jax.py` - Weight transfer
- `rl/scripts/run_rollout_worker.py` - RL rollout
- `rl/environments/*.py` - RL environment definitions (14 files use HF datasets for loading eval data)

### Cluster Management
- `cluster/cleanup.py` - Cluster cleanup utilities with @ray.remote for worker management

### Utility/Configuration
- `resources.py` - Resource configuration
- `generation/ray_utils.py` - Ray scheduling utilities
- `processing/classification/config/inference_config.py` - Configuration only
- `processing/classification/dataset_utils.py` - Library being replaced by zephyr
