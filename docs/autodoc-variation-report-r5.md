# Autodoc Variation Experiment — Round 5

## Setup
- **Doc generator model**: sonnet (generates module-level docs from source)
- **Coding agent**: haiku (writes script from generated docs)
- **Reviewer**: sonnet
- **Task**: Write a fuzzy dedup script
- **Key constraint**: Docs are generated at module/package level by the actual
  doc pipeline. NO task-specific documentation. Only the MODULE_PROMPT varies.
- **Modules documented**: marin.processing, dupekit, zephyr.dataset

## Results

| # | Variation | Context | Gen Cost | Score | Acc |
|---|-----------|---------|----------|-------|-----|
| 1 | Baseline: Purpose/API/Deps/Abstractions/Gotchas (<150 lines) | 20,099 | $0.0385 | 3/10 | 30% |
| 2 | Minimal: signatures only, no prose (<80 lines) | 15,857 | $0.0257 | 4/10 | 40% |
| 3 | Heavy prose: conceptual overview then API (<200 lines) | 30,624 | $0.0695 | 8/10 | 80% |
| 4 | Code-focused: signatures in fenced code blocks (<120 lines) | 18,068 | $0.0289 | 6/10 | 60% |
| 5 | Hybrid: conceptual group intros + compact signatures (<150 lines) | 26,324 | $0.0399 | 4/10 | 40% |

## Per-module doc sizes

| # | marin.processing | dupekit | zephyr.dataset | Total |
|---|-----------------|---------|----------------|-------|
| 1 | 8,274 | 4,934 | 6,104 | 19,312 |
| 2 | 7,452 | 2,422 | 5,196 | 15,070 |
| 3 | 13,121 | 8,403 | 8,313 | 29,837 |
| 4 | 8,877 | 3,965 | 4,439 | 17,281 |
| 5 | 11,690 | 6,441 | 7,406 | 25,537 |

## Analysis

### Best cost-accuracy tradeoff (≥80%)
**V3** (Heavy prose: conceptual overview then API (<200 lines)): $0.0695 at 80%

### Reviewer caveat

The sonnet reviewer incorrectly flagged some existing APIs as hallucinated
(e.g., `Dataset.from_files` does exist). Manual re-scoring confirms V3's
MinHash defaults are correct (7-8/10 range) but its import path is still wrong.
Future rounds should use an opus-level reviewer with codebase access.

### Key takeaways

1. **Module-level docs are too broad for haiku.** `marin.processing` contains
   dedup, classification, consolidation, tokenization, and more. Haiku gets
   overwhelmed by 15-30K chars of context where only ~2K is relevant.

2. **Import paths are the #1 failure.** 4/5 variations got the import wrong
   (`from marin.processing import ...` instead of the real submodule path).
   Only V4 (code-focused with fenced import blocks) got it right — because the
   code blocks literally showed `from marin.processing.classification.deduplication.fuzzy import ...`.

3. **MinHash defaults are the #2 failure.** Only V3 (heavy prose) passed the
   defaults explicitly. The prose format explained *why* those defaults matter,
   so haiku included them. Other formats listed them in signatures but haiku
   omitted them in the call, assuming the library would use its own defaults.

4. **Code blocks with imports > prose with signatures.** V4 (fenced code blocks
   showing exact imports) was the only one to get the import path right. But it
   missed the defaults because the code block just showed the signature, not
   explaining why the defaults matter.

5. **More context = more cost, not more accuracy.** V3 (30K, $0.07) scored best
   but V4 (18K, $0.03) was more cost-efficient per correct criterion. The sweet
   spot is somewhere between V3's prose and V4's code focus.

6. **Compare to R4:** Hand-crafted task-specific docs (3.2K chars) scored 100%
   at $0.018. Auto-generated module docs (15-30K chars) scored 30-80% at
   $0.03-0.07. Module-level docs are 10x larger and 2-5x less accurate.

### Per-criterion pass rates

| Criterion | Pass Rate |
|-----------|-----------|
| IMPORT_PATH | 1/5 |
| FUNCTION_NAME | 5/5 |
| NUM_PERMS | 1/5 |
| NUM_BANDS | 1/5 |
| NGRAM_SIZE | 1/5 |
| SEED | 1/5 |
| MAX_PARALLELISM | 4/5 |
| KEYWORD_ONLY | 5/5 |
| RUNNABLE | 5/5 |
| NO_HALLUCINATION | 1/5 |

## Generated Docs (V1 baseline sample)

### dupekit
```markdown
## Purpose
`dupekit` is a Rust-backed Python extension providing high-throughput deduplication primitives: hashing, MinHash signature generation, LSH bucketing, text normalization, and Arrow batch processing. Touch it when implementing or tuning near-duplicate document detection pipelines.

## Public API

### Hashing
- `hash_blake2(text: bytes) -> bytes` — 64-byte Blake2b-512 digest. `hashing.rs:34`
- `hash_blake3(text: bytes) -> bytes` — 32-byte Blake3 digest. `hashing.rs:41`
- `hash_xxh3_128(text: bytes) -> int` — XXH3 128-bit hash as u128. `hashing.rs:47`
- `hash_xxh3_64(text: bytes) -> int` — XXH3 64-bit hash as u64. `hashing.rs:52`
- `hash_xxh3_64_batch(texts: list[bytes]) -> list[int]` — Batch XXH3-64 over a list of byte strings. `hashing.rs:58`
- `hash_array(arr: StringArray, algo: HashAlgorithm) -> StringArray` — Hash each string element to hex; nulls preserved. `ops.rs:56`

### MinHash / LSH
- `clean_text(arr: StringArray) -> StringArray` — Lowercase, strip punctuation, normalize whitespace; preprocesses before MinHash. `minhash_ops.rs:15`
- `compute_minhash(arr: StringArray, num_perms: int, ngram_size: int, seed: u64) -> ListArray[uint64]` — Duplodocus-strategy MinHash signatures; returns list-of-uint64 per row. `minhash_ops.rs:40`
- `compute_lsh(input_col: ListArray, num_bands: int) -> ListArray[uint64]` — Band-hash a MinHash signature column into LSH bucket hashes. `minhash_ops.rs:99`

### Arrow Batch Processing
- `process_native(path: str) -> RecordBatch` — Read a local Parquet file projecting `id`+`text`, returns `id`+`head` batch. `marshaling.rs:78`
- `process_arrow_batch(batch: RecordBatch) -> RecordBatch` — Apply `transform_arrow_batch_impl` to an in-memory Arrow batch. `marshaling.rs:137`
- `process_dicts_batch(docs: list[dict]) -> list[dict]` — Batch-process list of `{id, text}` dicts → `{id, head}`. `marshaling.rs:173`
- `process_dicts_loop(doc: dict) -> dict` — Single-doc variant of `process_dicts_batch`. `marshaling.rs:196`
- `process_rust_structs(docs: list[Document]) -> list[Document]` — Truncate text in Document structs via `truncate_text`. `marshaling.rs:162`
- `transform(batch: RecordBatch, steps: list[Transformation]) -> RecordBatch` — Apply a pipeline of `Transformation` steps to a batch; releases the GIL. `pipeline.rs:172`
- `get_string_array(batch: RecordBatch, name: str) -> StringArray` — Extract a named column, casting to Utf8 if needed. `ops.rs:32`
- `add_column(batch: RecordBatch, name: str, col: Array) -> RecordBatch` — Append a new column to a RecordBatch. `ops.rs:71`
- `select_columns(batch: RecordBatch, columns: list[str]) -> RecordBatch` — Project a RecordBatch to the named columns. `ops.rs:82`

### Text / Paragraph Ops
- `split_paragraphs_str(text: str) -> Iterator[SubSpan]` — Split on `\n`, yielding `SubSpan` with byte offsets. `ops.rs:18`
- `split_paragraphs(text_arr: StringArray, id_arr: StringArray) -> (StringArray, StructArray, StringArray)` — Vectorized paragraph split; returns (para_text, span_struct, doc_id). Skips empty paragraphs. `ops.rs:93`

### Structs / Classes
- `Bloom` — Bloom filter backed by a bit array with k LCG-derived hash functions. `bloom.rs:38`
  - (no documented public methods in provided source)
- `Document` — Simple `{id: str, text: str}` struct with Python get/set. `marshaling.rs:146`

## Dependencies
No intra-monorepo dependencies — `dupekit` is a leaf Rust extension with no imports from marin, levanter, haliax, iris, or zephyr.

## Key Abstractions
- `Document` — Minimal `{id, text}` carrier used by the `process_rust_structs` path; `text` is truncated to a `head` on output.
- `SubSpan` — Paragraph slice with character-level `start`/`end` byte offsets into the original string.
- `Transformation` — Opaque pipeline step applied sequentially by `transform`; defined in `pipeline.rs`.
- `HashAlgorithm` — Enum selecting the hash backend for `hash_array`.
- `Bloom` — Probabilistic membership filter; k is derived via LCG seeded from the input hash.

## Gotchas
- `compute_lsh` requires `sig_len % num_bands == 0`; it raises `PyValueError` otherwise — ensure `num_perms` is divisible by `num_bands` before calling.
- `process_native` and `process_arrow_batch` both output `head` (not `text`) — the text field is truncated by `truncate_text`; downstream code expecting `text` will silently get truncated content.
- `split_paragraphs_str` offsets assume UTF-8 byte positions but iterates by `char`; the `+1` for the newline (`ops.rs:18`) is byte-counted, so multi-byte characters in paragraphs will make `start`/`end` inconsistent with Python string indices.
- `compute_minhash` uses Duplodocus single-coefficient permutations seeded from `seed`; changing `seed` produces entirely different signatures — signatures from different seeds are not comparable.
- `process_dicts_batch` and `process_dicts_loop` require both `"id"` and `"text"` keys; missing either raises `PyKeyError`, not a graceful `None`.
```

### marin.processing
```markdown
## Purpose
`marin.processing` provides document processing pipelines for classification, deduplication, decontamination, and tokenization. An agent touches this module when building or modifying data quality filters, running dedup jobs, training fastText classifiers, or preparing tokenizer inputs.

## Public API

### Configuration
- `step_to_lm_mixture_component(step: TokenizerStep | TokenizeConfig, include_raw_paths: bool) -> DatasetComponent` — Converts a tokenizer step into a Levanter `DatasetComponent` for mixture configs. `data_configs.py:34`

### Consolidation / Filtering
- `calculate_percentile_thresholds(config: ConsolidateConfig) -> list[FilterConfig]` — Resolves `keep_fraction` fields in filters to numeric `lower_threshold` by scanning attribute files. `consolidate.py:215`
- `process_file_shard(*, shard, filters, input_base, filetype) -> Iterator[dict]` — Filters documents in one shard against all configured filters; yields passing docs. `consolidate.py:263`
- `consolidate(config: ConsolidateConfig)` — End-to-end filter+write pipeline using Zephyr; outputs Parquet. `consolidate.py:332`
- `extract_id(row: dict, corpus_type: str) -> str` — Extracts document ID via `CORPUS_TYPE_TO_ID_GUIDE`; supports `"dclm"` and `"default"`. `consolidate.py:152`

### Deduplication
- `dedup_exact_document(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, ...) -> dict` — Full-document hash dedup via xxh3-128; writes vortex attribute files. `exact.py:186`
- `dedup_exact_paragraph(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, ...) -> dict` — Paragraph-level hash dedup; writes `dup_spans` attributes. `exact.py:50`
- `dedup_fuzzy_document(*, input_paths, output_path, fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism, ...) -> dict` — MinHash LSH + connected components fuzzy dedup. `fuzzy.py:27`
- `connected_components(ds, ctx, *, output_dir, max_iterations=10, preserve_singletons=True) -> tuple[bool, Sequence[str]]` — Hash-to-Min CC algorithm over Zephyr Dataset; returns `(converged, output_files)`. `connected_components.py:51`
- `group_files(files: list[str], num_groups: int) -> list[list[str]]` — Round-robin file grouping to cap shard count ≤ `num_groups`. `dedup_commons.py:53`
- `make_document_dedup_aggregator(*, idx_to_path, input_paths, output_path, counter_prefix) -> Callable` — Returns a `group_by` reducer that writes vortex output and counts dup/unique. `dedup_commons.py:151`
- `finalize_dedup(shard_results, mode, method, level) -> dict` — Aggregates counters, logs summary, finalizes wandb. `dedup_commons.py:203`

### Decontamination
- `decontaminate(config: DeconConfig)` — Dispatches to decontamination or train/test overlap based on `config.mode`. `decon.py:347`
- `build_filter(input_path, bloom_path, config: DeconConfig) -> str` — Builds and merges a bloom filter from input data into `bloom_path`. `decon.py:128`
- `mark_duplicates_bloom(input_path, bloom_path, output_path, config: DeconConfig) -> list[str]` — Applies bloom filter to annotate duplicate spans in attribute files. `decon.py:206`
- `calculate_paragraph_overlap(paragraph, bloom_filter, ngram_config) -> float` — Returns fraction of n-grams (or exact paragraph) found in bloom filter. `decon.py:186`
- `extract_features(text: str, ngram_config: NGramConfig | None) -> Iterator[str]` — Yields paragraphs or n-grams depending on whether `ngram_config` is set. `decon.py:105`
- `extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]` — Tokenizes by whitespace and yields overlapping n-grams with step `stride+1`. `decon.py:95`

### Inference / Classification
- `run_inference(inference_config: InferenceConfig)` — Distributes Ray remote tasks for classifier inference across all matching files with per-file retry. `inference.py:187`
- `process_file_with_quality_classifier_streaming(input_filename, output_filename, model_name_or_path, attribute_name, model_type, dataset_schema, ..., batch_size=512, resume=True)` — Streams a file through a Ray actor pool with resumption via `get_finished_ids`. `inference.py:49`
- `get_finished_ids(output_filename: str, id_path: tuple[str, ...]) -> set` — Reads already-processed IDs from JSONL or Parquet output for resumption. `checkpoint_utils.py:19`
- `get_id_from_row(row: dict, id_path: tuple[str, ...]) -> str | None` — Traverses a nested key path in a row dict; raises `ValueError` if any key is missing. `checkpoint_utils.py:9`

### Streaming I/O
- `read_dataset_streaming(input_filename: str, columns: list[str] | None = None)` — Yields dicts from `.jsonl(.gz/.zst)` or `.parquet`; avoids HuggingFace filesystem bugs. `dataset_utils.py:76`
- `write_dataset_streaming(rows_iterator, output_filename: str, append: bool = False)` — Writes JSONL via deterministic `/tmp` temp file then uploads; Parquet collected in memory. `dataset_utils.py:108`
- `make_json_serializable(row: dict) -> dict` — Converts `datetime`, `np.ndarray`, and numpy floats in-place. `dataset_utils.py:62`

### fastText Training
- `train(cfg: TrainFasttextClassifierConfig)` — Creates datasets then launches an Iris job to run `train_model`. `train_fasttext.py:342`
- `create_dataset(config: CreateDatasetConfig)` — Runs Zephyr pipeline to produce labeled JSONL shards from documents. `train_fasttext.py:73`
- `train_model(input_path, output_path, seed, val_frac, memory_req, **fasttext_args)` — Merges shards, formats, splits, shuffles, trains via `floret`, uploads model. `train_fasttext.py:268`
- `reservoir_sample(examples, sample_size, seed) -> list` — Standard reservoir sampling. `train_fasttext.py:242`
- `format_example(data: dict) -> str` — Formats `{"text", "label"}` into fastText `__label__X text` format. `train_fasttext.py:34`

## Dependencies
- `marin.processing` (internal submodules) — cross-references `dedup_commons`, `connected_components`, `checkpoint_utils`, `dataset_utils`, `consolidate`
- `zephyr` — `ZephyrContext`, `Dataset` API used throughout for distributed execution
- `dupekit` — bloom filter construction (`Bloom`), MinHash/LSH transforms, `dupekit.transform` pipeline
- `marin.io` / fsspec utilities — `open_url`, `url_to_fs`, `fsspec_glob`, `rebase_file_path`, `load_file`
- `levanter` — `DatasetComponent`, `TokenizerStep`, `TokenizeConfig` consumed by `data_configs.py`
- `iris` — `JobRequest`, `ResourceConfig`, `Entrypoint`, `current_cluster` used in `train_fasttext.py`

## Key Abstractions
- `ConsolidateConfig` — Configures multi-filter consolidation: input/output paths, filetype, and a list of `FilterConfig` objects.
- `FilterConfig` — Describes one filter: type (`CLASSIFY`, `REMOVE_DOC`, `REMOVE_SPANS`), attribute path, thresholds, and `keep_fraction` for percentile-based cutoffs.
- `DeconConfig` — Controls bloom-filter decontamination: mode, `estimated_doc_count`, `false_positive_rate`, optional `NGramConfig`.
- `InferenceConfig` — Wires together model path, attribute name, Ray actor pool settings, and dataset schema for distributed classifier inference.
- `CCInput` / `CCNode` — Internal typed dicts for the connected-components graph algorithm; `CCInput` carries `bucket`+`id`+`file_idx`, `CCNode` carries component state.

## Gotchas
- `get_id_from_row` raises `ValueError` (not returns `None`) when any key in `id_path` is missing mid-traversal — the `None` return only applies when the first key yields `None` via `dict.get`.
- `dedup_fuzzy_document`: `fuzzy_minhash_num_perms` must be divisible by `fuzzy_minhash_num_bands` (default 286/26) — the function raises immediately if not; changing one without the other breaks it.
- `write_dataset_streaming` JSONL append writes to a deterministic `/tmp` file keyed by sha256 of the remote path. If two processes write to the same remote output simultaneously, they will corrupt each other's temp file silently.
- `process_file_shard` infers `corpus_type` as `"dclm"` if `"dclm"` appears anywhere in the file path — any non-dclm path containing that string will use the wrong ID extraction guide.
- `connected_components` returns `(converged=False, files)` without raising when `max_iterations` is exceeded; callers must check the boolean or they silently proceed with incomplete dedup results.
```

### zephyr.dataset
```markdown
## Purpose
`zephyr.dataset` defines the lazy `Dataset[T]` pipeline abstraction for Zephyr's distributed data processing. It stores a source plus a chain of `LogicalOp` dataclasses; actual execution is deferred to `ZephyrContext.execute()`. Touch this module when building, inspecting, or extending data processing pipelines.

## Public API

### Core Types
- `ShardInfo(shard_idx: int, total_shards: int)` — Metadata passed to `map_shard` functions. `dataset.py:24`
- `Dataset(source: Iterable[T], operations: list[LogicalOp] | None = None)` — Lazy pipeline; all methods return new `Dataset` instances. `dataset.py:297`
  - `.from_list(items: list[T]) -> Dataset[T]` — Static factory from list. `dataset.py:330`
  - `.from_iterable(iterable: Iterable[T]) -> Dataset[T]` — Static factory from any iterable. `dataset.py:335`
  - `.from_files(pattern: str, empty_glob_ok: bool = False) -> Dataset[str]` — Glob files; returns dataset of paths; raises `FileNotFoundError` if no matches. `dataset.py:340`
  - `.map(fn) -> Dataset[R]` — Element-wise transform. `dataset.py:383`
  - `.filter(predicate) -> Dataset[T]` — Keep elements; accepts lambda or `Expr`. `dataset.py:400`
  - `.select(*columns: str) -> Dataset[dict]` — Column projection. `dataset.py:423`
  - `.flat_map(fn) -> Dataset[R]` — Map then flatten. `dataset.py:466`
  - `.map_shard(fn: Callable[[Iterator[T], ShardInfo], Iterator[R]]) -> Dataset[R]` — Full-shard stateful transform. `dataset.py:497`
  - `.take_per_shard(n: int) -> Dataset[T]` — Limit per shard (not global). `dataset.py:440`
  - `.window(size: int) -> Dataset[list[T]]` — Fixed-count windows. `dataset.py:455`
  - `.window_by(folder_fn, initial_state=None) -> Dataset[list[T]]` — Custom-predicate windows. `dataset.py:470`
  - `.load_file(columns=None) -> Dataset[dict]` — Auto-detect format (parquet/jsonl/vortex). `dataset.py:482`
  - `.load_parquet(columns=None) -> Dataset[dict]` — Explicit parquet load. `dataset.py:497`
  - `.load_jsonl() -> Dataset[dict]` — Explicit JSONL load. `dataset.py:501`
  - `.load_vortex(columns=None) -> Dataset[dict]` — Explicit Vortex load. `dataset.py:505`
  - `.reshard(num_shards: int | None) -> Dataset[T]` — Best-effort metadata reshard; `None` is a no-op. `dataset.py:530`
  - `.group_by(key, *, reducer, sort_by=None, num_output_shards=None, combiner=None) -> Dataset[R]` — Shuffle-group-reduce. `dataset.py:570`
  - `.deduplicate(key, num_output_shards=None) -> Dataset[T]` — Wraps `map_shard` + `group_by`. `dataset.py:630`
  - `.reduce(local_reducer, global_reducer=None) -> Dataset[R]` — Two-phase reduction; `global_reducer` defaults to `local_reducer`. `dataset.py:650`
  - `.count() -> Dataset[int]` — Wraps `reduce`; returns single-element dataset. `dataset.py:673`
  - `.sorted_merge_join(right, left_key, right_key, combiner=None, how: str = "inner") -> Dataset` — Streaming merge join; requires pre-sorted, co-partitioned inputs. `dataset.py:683`
  - `.write_jsonl(output_pattern, skip_existing: bool = False) -> Dataset[str]` — Append `WriteOp(jsonl)`. `dataset.py:547`
  - `.write_parquet(output_pattern, schema=None, skip_existing: bool = False) -> Dataset[str]` — Append `WriteOp(parquet)`. `dataset.py:562`
  - `.write_vortex(output_pattern, schema=None, skip_existing: bool = False) -> Dataset[str]` — Append `WriteOp(vortex)`. `dataset.py:577`
  - `.write_binary(output_pattern, skip_existing: bool = False) -> Dataset[str]` — Raw bytes, no framing. `dataset.py:555`
  - `.write_levanter_cache(output_pattern, metadata: dict[str, Any], skip_existing: bool = False) -> Dataset[str]` — Levanter TreeStore format; `metadata` is required. `dataset.py:590`

### Utilities
- `format_shard_path(pattern: str, shard_idx: int, total: int) -> str` — Format `{shard}/{total}/{basename}` placeholders; raises `ValueError` if multi-shard write would collide. `dataset.py:36`

### Op Dataclasses (LogicalOp variants)
- `MapOp(fn)`, `FilterOp(predicate, expr=None)`, `SelectOp(columns)`, `FlatMapOp(fn)`, `MapShardOp(fn)` — `dataset.py:88–212`
- `TakePerShardOp(n)`, `WindowOp(folder_fn, initial_state)`, `ReshardOp(num_shards)` — `dataset.py:121–227`
- `GroupByOp(key_fn, reducer_fn, num_output_shards=None, sort_fn=None, combiner_fn=None)` — `dataset.py:228`
- `ReduceOp(local_reducer, global_reducer)`, `JoinOp(left_key_fn, right_key_fn, right_dataset, combiner_fn, join_type)` — `dataset.py:242–276`
- `WriteOp(output_pattern, writer_type, levanter_metadata=None, schema=None, skip_existing=False)` — `dataset.py:150`
- `LoadFileOp(format="auto", columns=None)` — `dataset.py:182`

## Dependencies
- `zephyr.expr` — `Expr` type used in `FilterOp` and `Dataset.filter` for expression pushdown.

## Key Abstractions
- `Dataset[T]` — Immutable lazy pipeline; each transformation returns a new instance with the op appended.
- `LogicalOp` — Union of all op dataclasses; makes the pipeline inspectable as data before execution.
- `WriteOp` — Unified sink for all output formats; `writer_type` selects the backend writer.
- `ShardInfo` — Carried through `map_shard`; exposes `shard_idx` and `total_shards` for shard-aware logic.
- `GroupByOp` — Triggers a distributed shuffle; the most expensive operation in the DAG.

## Gotchas
- `take_per_shard(n)` limits **per shard**, not globally — with k shards you get up to k×n records; there is no global `take`.
- `reshard(None)` is silently a no-op and returns `self`; passing `0` or negative raises `ValueError`.
- `sorted_merge_join` requires both datasets to have **equal shard counts and pre-sorted, co-partitioned keys**; the typical safe path is to produce both sides via `group_by(..., num_output_shards=N)` with the same N.
- `format_shard_path` raises `ValueError` when `total > 1` and the pattern lacks `{shard`} — this fires at write time inside the execution engine, not when you call `.write_jsonl()`.
- `group_by` performance degrades for non-Arrow-serializable items (custom dataclasses fall back to pickle); prefer plain dicts or Arrow-compatible types.
- `write_levanter_cache` requires `metadata` (no default); omitting it is a `TypeError` at pipeline construction time.
```


## Scripts

### V1: Baseline: Purpose/API/Deps/Abstractions/Gotchas (<150 lines)
*Context: 20,099 | Cost: $0.0385 | Score: 3/10*

```python
import argparse
import glob
from marin.processing import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(description="Fuzzy document deduplication")
    parser.add_argument("input_pattern", help="Glob pattern for input Parquet files")
    parser.add_argument("output_path", help="Output path for deduplicated data")
    args = parser.parse_args()

    # Expand glob pattern
    input_files = glob.glob(args.input_pattern, recursive=True)
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_pattern}")

    # Run fuzzy deduplication with default MinHash parameters
    result = dedup_fuzzy_document(
        input_paths=input_files,
        output_path=args.output_path,
    )

    print("Deduplication complete")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import is from 'marin.processing' instead of 'marin.processing.classification.deduplication.fuzzy'. All four MinHash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are omitted entirely rather than passed explicitly. max_parallelism is not provided despite being required with no default. The import path itself is a hallucinated shortcut that does not exist in the codebase.

- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

### V2: Minimal: signatures only, no prose (<80 lines)
*Context: 15,857 | Cost: $0.0257 | Score: 4/10*

```python
import argparse
from marin.processing import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument(
        "input_pattern",
        help="Glob pattern for input Parquet files",
    )
    parser.add_argument(
        "output_path",
        help="Output path for deduplicated documents",
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        filetypes=["parquet"],
        max_parallelism=4,
    )

    print(f"Deduplication completed: {result}")


if __name__ == "__main__":
    main()
```

**Review:** IMPORT_PATH is wrong: script imports from 'marin.processing' instead of 'marin.processing.classification.deduplication.fuzzy'. All four minhash config params (num_perms=286, num_bands=26, ngram_size=5, seed=42) are omitted entirely, so defaults are not being explicitly set as required. NO_HALLUCINATION fails on two counts: the truncated import path is incorrect, and the 'filetypes' keyword argument does not appear in the ground-truth signature and is likely invented.

- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

### V3: Heavy prose: conceptual overview then API (<200 lines)
*Context: 30,624 | Cost: $0.0695 | Score: 8/10*

```python
import argparse
import sys

from marin.processing import dedup_fuzzy_document
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files."
    )
    parser.add_argument(
        "input_pattern",
        help="Glob pattern for input Parquet files (supports cloud paths like gs://)",
    )
    parser.add_argument("output_path", help="Output path for deduplication attribute files")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the text field in input documents (default: text)",
    )

    args = parser.parse_args()

    try:
        dataset = Dataset.from_files(args.input_pattern)
        ctx = ZephyrContext()
        input_files = list(ctx.execute(dataset))
    except FileNotFoundError:
        print(f"No files matched pattern: {args.input_pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} input files")

    result = dedup_fuzzy_document(
        input_paths=input_files,
        output_path=args.output_path,
        text_field=args.text_field,
        max_parallelism=args.max_parallelism,
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
    )

    print(f"Deduplication complete!")
    print(f"Results: {result}")


if __name__ == "__main__":
    main()
```

**Review:** IMPORT_PATH is wrong: script uses 'from marin.processing import dedup_fuzzy_document' instead of the correct 'from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document'. NO_HALLUCINATION fails on two counts: (1) the shortened import path is itself a hallucinated alias, and (2) 'ZephyrContext' / 'Dataset.from_files' / 'ctx.execute(dataset)' are fabricated APIs not present in the Marin/Zephyr codebase. MAX_PARALLELISM is marked 1 because the argument is always forwarded in the call (argparse default of 8 ensures it), even though the ground truth specifies it should be required with no default.

- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

### V4: Code-focused: signatures in fenced code blocks (<120 lines)
*Context: 18,068 | Cost: $0.0289 | Score: 6/10*

```python
#!/usr/bin/env python3
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument("input_pattern", help="Glob pattern for input files")
    parser.add_argument("output_path", help="Path for output directory")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=8,
        help="Maximum parallelism (default: 8)",
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )

    print(f"Deduplication complete. Results: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import path and function name are correct. The script never explicitly passes fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, or fuzzy_minhash_seed=42 — it silently relies on whatever the function defaults are, so criteria 3–6 fail. max_parallelism is passed as a keyword argument (criterion 7 passes), but the script gives it a default of 8 via argparse, making it optional rather than required as the ground truth specifies. All three call-site arguments are keyword-only (criterion 8 passes). The script is runnable with __main__ guard and argparse (criterion 9 passes). No obviously invented APIs are present (criterion 10 passes).

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V5: Hybrid: conceptual group intros + compact signatures (<150 lines)
*Context: 26,324 | Cost: $0.0399 | Score: 4/10*

```python
import argparse
from marin.processing import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument(
        "input_pattern",
        help="Glob pattern for input files"
    )
    parser.add_argument(
        "output_path",
        help="Path where deduplication results will be written"
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=32,
        help="Maximum parallelism for deduplication (default: 32)"
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )

    print(f"Deduplication complete. Result: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import is from 'marin.processing' instead of 'marin.processing.classification.deduplication.fuzzy'. None of the minhash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are passed to the function call — they are silently omitted, relying on defaults that may not match the ground truth. MAX_PARALLELISM is provided but has a hardcoded default of 32 whereas the ground truth treats it as required (no default). The function call invents an 'input_paths' keyword argument whose existence in the real API is unverified, constituting a potential hallucination.

- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

