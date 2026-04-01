# Autodoc Variation Experiment Report

## Objective
Minimize agent token cost while retaining correct answers about
Marin's zephyr deduplication pipeline.

## Test Question
> Explain how to compute duplicate documents from a list of parquet files using Marin's zephyr pipeline. Be concise but specific: which functions to call, what parameters matter (including defaults), and how the pipeline stages work. Include the import paths. Keep it under 200 words.

## Ground Truth Facts Checked
- `dedup_fuzzy_document` — function name
- `num_perms=286` — correct default (not 128)
- `num_bands=26` — correct default
- `ngram_size=5` — correct default
- `seed=42` — correct default
- `connected_components` — clustering step
- `dupekit` — Rust FFI library
- `marin.processing` — import path

## Results

| # | Variation | Context | Cost | In Tokens | Out Tokens | Accuracy | Score |
|---|-----------|---------|------|-----------|------------|----------|-------|
| 1 | Full module docs (3 modules) | 30,557 chars | $0.0688 | 3 | 486 | 100% | 8/8 |
| 2 | Full module docs (7 modules) | 53,330 chars | $0.0954 | 3 | 546 | 100% | 8/8 |
| 3 | API-only (3 modules, no prose/gotchas) | 21,520 chars | $0.0634 | 3 | 759 | 100% | 8/8 |
| 4 | Signature index (one line per function) | 20,736 chars | $0.0591 | 3 | 522 | 100% | 8/8 |
| 5 | Compressed (purpose + 30 API lines + gotchas) | 14,225 chars | $0.0566 | 3 | 990 | 62% | 5/8 |
| 6 | Header + references (summary + gotchas only) | 5,957 chars | $0.0861 | 6 | 1,533 | 75% | 6/8 |
| 7 | Curated snippet (hand-picked dedup functions) | 2,470 chars | $0.0349 | 3 | 461 | 100% | 8/8 |
| 8 | Raw source only (no generated docs) | 23,107 chars | $0.0570 | 3 | 434 | 100% | 8/8 |
| 9 | Single merged doc (3 modules combined) | 30,614 chars | $0.0688 | 3 | 480 | 100% | 8/8 |
| 10 | No context baseline | 0 chars | $0.1846 | 6 | 734 | 88% | 7/8 |

## Analysis

### Best cost-accuracy tradeoff
**Variation 7** (Curated snippet (hand-picked dedup functions)): $0.0349 at 100% accuracy (2,470 chars context)

### Most accurate
**Variation 7** (Curated snippet (hand-picked dedup functions)): 100% accuracy at $0.0349

### Cheapest
**Variation 7** (Curated snippet (hand-picked dedup functions)): $0.0349 at 100% accuracy

### Cost per correct fact

| # | Variation | $/fact | Accuracy |
|---|-----------|--------|----------|
| 1 | Full module docs (3 modules) | $0.0086 | 100% |
| 2 | Full module docs (7 modules) | $0.0119 | 100% |
| 3 | API-only (3 modules, no prose/gotchas) | $0.0079 | 100% |
| 4 | Signature index (one line per function) | $0.0074 | 100% |
| 5 | Compressed (purpose + 30 API lines + gotchas) | $0.0113 | 62% |
| 6 | Header + references (summary + gotchas only) | $0.0143 | 75% |
| 7 | Curated snippet (hand-picked dedup functions) | $0.0044 | 100% |
| 8 | Raw source only (no generated docs) | $0.0071 | 100% |
| 9 | Single merged doc (3 modules combined) | $0.0086 | 100% |
| 10 | No context baseline | $0.0264 | 88% |

## Detailed Answers

### Variation 1: Full module docs (3 modules)
*Context: 30,557 chars | Cost: $0.0688 | Accuracy: 100%*

> Use `marin.processing.exact.dedup_exact_document` (or `dedup_exact_paragraph` for paragraph-level) for exact dedup, or `marin.processing.fuzzy.dedup_fuzzy_document` for fuzzy dedup. All three accept `input_paths: list[str]`, `output_path: str`, `text_field="text"` (default), `filetypes=None`, and `max_parallelism: int`. **Exact document** dedup hashes full text via XXH3-128 using `dupekit.transform`, groups by hash, and keeps one representative per group through a Zephyr `group_by` → `reduce` pipeline, writing deduplicated Parquet to `output_path`. **Fuzzy document** dedup (`marin.processing.fuzzy.dedup_fuzzy_document`) adds MinHash LSH: it calls `dupekit.compute_minhash` (default `num_perms=286`, `ngram_size=5`, `seed=42`) then `dupekit.compute_lsh` (default `num_bands=26`) to assign bucket hashes, runs `connected_components` (`marin.processing.connected_components`) via Zephyr to find near-duplicate clusters, and writes surviving documents. Critical constraint: `fuzzy_minhash_num_perms` must be exactly divisible by `fuzzy_minhash_num_bands` (286 % 26 == 0 ✓); mismatching these raises `ValueError`. `connected_components` returns `(converged: bool, vortex_paths)` — callers must check `converged` since non-convergence only logs a warning. Both exact functions use `DedupMode` from `marin.processing.dedup_commons`.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 2: Full module docs (7 modules)
*Context: 53,330 chars | Cost: $0.0954 | Accuracy: 100%*

> Use `dedup_exact_document` (from `marin.processing.exact`) for exact dedup or `dedup_fuzzy_document` (from `marin.processing.fuzzy`) for near-dedup. For exact document dedup, call `dedup_exact_document(input_paths=[...], output_path="gs://...", text_field="text", max_parallelism=N)` — it hashes each document's full text with XXH3-128 via `dupekit.transform`, then runs a `group_by` shuffle over a `zephyr.dataset.Dataset` built from the input Parquet globs, writing deduplicated Parquet output. For fuzzy dedup, call `dedup_fuzzy_document(input_paths=[...], output_path="gs://...", fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism=N)` — it computes MinHash signatures via `dupekit.compute_minhash`, bands them into LSH buckets via `dupekit.compute_lsh`, then runs `connected_components` (from `marin.processing.connected_components`) over a `ZephyrContext` to find duplicate clusters, writing vortex output. Critical constraint: `fuzzy_minhash_num_perms` (default 286) must be exactly divisible by `fuzzy_minhash_num_bands` (default 26); both exact functions return a summary `dict` of dedup stats via `finalize_dedup` (from `marin.processing.dedup_commons`). Both functions require `max_parallelism` — there is no default.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 3: API-only (3 modules, no prose/gotchas)
*Context: 21,520 chars | Cost: $0.0634 | Accuracy: 100%*

> For **exact document dedup**, call `dedup_exact_document` from `marin.processing` (`exact.py`): it hashes each document's full text with XXH3-128 via `dupekit.transform`, groups by hash using Zephyr's `Dataset.group_by`, and writes surviving documents to `output_path`. Key parameters: `input_paths` (list of parquet paths), `output_path`, `text_field="text"` (column name), `filetypes=None` (auto-detect), and `max_parallelism` (required, controls Zephyr sharding). For **fuzzy document dedup**, call `dedup_fuzzy_document` from `marin.processing` (`fuzzy.py`): it runs `dupekit.compute_minhash` (default `num_perms=286`, `ngram_size=5`, `seed=42`) then `dupekit.compute_lsh` (default `num_bands=26`; `num_perms` must be divisible by `num_bands`) to generate LSH bucket hashes, shuffles candidate pairs via `Dataset.group_by`, then resolves duplicates with `connected_components` from `marin.processing` (`connected_components.py`) using a Hash-to-Min algorithm (default `max_iterations=10`, `preserve_singletons=True`). Both functions return a summary `dict` of dedup stats assembled by `finalize_dedup` from `marin.processing` (`dedup_commons.py`). Import paths: `from marin.processing.dedup.exact import dedup_exact_document`, `from marin.processing.dedup.fuzzy import dedup_fuzzy_document`.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 4: Signature index (one line per function)
*Context: 20,736 chars | Cost: $0.0591 | Accuracy: 100%*

> To deduplicate documents from Parquet files, use `dedup_exact_document` or `dedup_fuzzy_document` from `marin.processing.exact` / `marin.processing.fuzzy`. Both accept `input_paths: list[str]`, `output_path: str`, `text_field="text"` (the column hashed), `filetypes=None`, and `max_parallelism: int`. **Exact dedup** (`dedup_exact_document`) hashes the full document text via XXH3-128 through `dupekit.transform`, groups by hash, and discards duplicates. **Fuzzy dedup** (`dedup_fuzzy_document`) computes MinHash signatures (`fuzzy_minhash_num_perms=286`, `fuzzy_minhash_num_bands=26`, `fuzzy_minhash_ngram_size=5`, `fuzzy_minhash_seed=42`—`num_perms` must be divisible by `num_bands`), bands them into LSH buckets, then runs `connected_components` (from `marin.processing.connected_components`) via Zephyr to cluster near-duplicates, iterating up to `max_iterations=10`. Internally both use `zephyr.dataset.Dataset.from_files` to glob inputs, apply `MapOp`/`GroupByOp` via `make_document_dedup_aggregator`, write Vortex output, and return stats via `finalize_dedup` (from `marin.processing.dedup_commons`). The `DedupMode` enum (`EXACT_DOCUMENT`, `FUZZY_DOCUMENT`) from `marin.processing.dedup_commons` controls dispatch logic shared across both paths.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 5: Compressed (purpose + 30 API lines + gotchas)
*Context: 14,225 chars | Cost: $0.0566 | Accuracy: 62%*

> To compute duplicate documents from Parquet files using the Zephyr + dupekit stack:

**Import paths:** `from zephyr.dataset import Dataset, LoadFileOp, MapShardOp, GroupByOp` and `import dupekit` (for `clean_text`, `compute_minhash`, `compute_lsh`).

**Pipeline:** Start with `Dataset.from_files("gs://bucket/data/*.parquet")` (raises `FileNotFoundError` if empty; pass `empty_glob_ok=True` to suppress). Chain `.map(LoadFileOp(format="parquet", columns=["id","text"]))` to load Arrow batches. Then use `MapShardOp` to apply dupekit in batch: call `dupekit.clean_text(text_col)` → `dupekit.compute_minhash(cleaned, num_perms=286, ngram_size=5, seed=42)` → `dupekit.compute_lsh(minhash_col, num_bands=26)`. Critical: `num_perms` (default 286) **must** be divisible by `num_bands` (default 26); `compute_lsh` raises `PyValueError` at runtime otherwise. The LSH bucket hashes become the grouping key. Finally, use `GroupByOp(key_fn=lambda r: r["lsh_bucket"], reducer_fn=collect_ids)` to cluster candidate duplicates, then a second `MapOp` to apply Jaccard threshold filtering within each bucket. Write results with `WriteOp`. Note: `compute_minhash` uses Duplodocus single-coefficient permutations—results are **not** compatible with standard MinHash libraries.

**Fact check:**
- [MISS] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [MISS] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [MISS] `marin.processing` — import path

### Variation 6: Header + references (summary + gotchas only)
*Context: 5,957 chars | Cost: $0.0861 | Accuracy: 75%*

> Based strictly on the provided documentation, here is the pipeline:

**Load → MinHash → LSH → Connected Components → Filter**

1. **Load**: Use `zephyr.dataset.Dataset.from_files(pattern)` (import: `from zephyr.dataset import Dataset`). Pass `empty_glob_ok=True` to suppress `FileNotFoundError` on empty globs. After loading, `reshard(n)` to control parallelism (`reshard(None)` is a no-op; `reshard(0)` raises).

2. **MinHash + LSH**: Use `dupekit.process_arrow_batch` (or `process_native`) with a `dupekit.Transformation` pipeline containing `compute_minhash` then `compute_lsh` steps (import: `from dupekit import Transformation, compute_minhash, compute_lsh`). Critical: `num_perms % num_bands` must equal zero, or `compute_lsh` raises `PyValueError` at runtime. Defaults in `marin.processing.dedup_fuzzy_document`: `fuzzy_minhash_num_perms=286`, `fuzzy_minhash_num_bands=26` (286 % 26 == 0). Note: `process_arrow_batch` renames `text` → `head`; downstream code must read `head`. `transform()` releases the GIL — steps must not call back into Python.

3. **Cluster**: Call `marin.processing.connected_components` (import: `from marin.processing import connected_components`). It does **not** raise on non-convergence — inspect the returned `bool` before trusting output vortex files.

4. **Dedup**: Call `marin.processing.dedup_fuzzy_document`, passing the component mapping to filter duplicate document IDs.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [MISS] `ngram_size=5` — correct default
- [MISS] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 7: Curated snippet (hand-picked dedup functions)
*Context: 2,470 chars | Cost: $0.0349 | Accuracy: 100%*

> Call `dedup_fuzzy_document` from `classification/deduplication/fuzzy.py` (import path: `marin.processing.classification.deduplication.fuzzy`). Pass `input_paths` (list of parquet file paths), `output_path`, and `max_parallelism` (required). Key optional parameters: `text_field` (default `"text"`), `filetypes` (default `None`), `fuzzy_minhash_num_perms` (default `286`), `fuzzy_minhash_num_bands` (default `26`, must divide `num_perms`), `fuzzy_minhash_ngram_size` (default `5`), `fuzzy_minhash_seed` (default `42`), plus `worker_resources`/`coordinator_resources` for Zephyr task sizing.

Internally, the pipeline runs four stages via dupekit's `transform` (from `rust/dupekit`): **CleanText** (lowercase, strip punctuation, normalize whitespace) → **MinHash** (character n-gram signatures, `num_perms` hash functions) → **MinHashLSH** (band-based bucket hashes, `num_bands` bands) → **SelectColumns**. The function uses `Dataset.from_list(file_groups).flat_map(compute_minhash_lsh_batches)` to yield `{bucket, id}` pairs, then calls `connected_components` (`classification/deduplication/connected_components.py`) which runs Hash-to-Min iterations (up to `max_iterations=10`) writing vortex files. Finally, `finalize_dedup` aggregates counters and writes vortex attribute files marking duplicates per source file. Returns a `dict` of counters.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 8: Raw source only (no generated docs)
*Context: 23,107 chars | Cost: $0.0570 | Accuracy: 100%*

> Call `dedup_fuzzy_document` from `marin.processing.classification.deduplication.fuzzy`. Pass `input_paths` (a string or list of parquet paths), `output_path` (GCS/local dir), and `max_parallelism` (required, no default). Key optional parameters: `text_field` (default `"text"`), `fuzzy_minhash_num_perms=286`, `fuzzy_minhash_num_bands=26` (must divide evenly into `num_perms`), `fuzzy_minhash_ngram_size=5`, `fuzzy_minhash_seed=42`, `worker_resources` (defaults to `ResourceConfig(cpu=1, ram="32g", disk="5g")`). The pipeline runs in three stages: (1) **Map** — each worker loads batches from its file group, runs `dupekit.transform` with `CleanText → MinHash → MinHashLSH`, and emits `{bucket, id}` pairs; (2) **Connected Components** (`connected_components` from `marin.processing.classification.deduplication.connected_components`) — iteratively applies the Hash-to-Min algorithm via Zephyr, grouping documents sharing LSH buckets into components, writing vortex files under `{output_path}/metadata/cc`, running up to `max_iterations=10`; (3) **Aggregation** — `make_document_dedup_aggregator` groups results by `file_idx`, marks documents as duplicates when `component_id != id_norm`, and writes deduplicated output. The function returns a summary dict from `finalize_dedup`.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 9: Single merged doc (3 modules combined)
*Context: 30,614 chars | Cost: $0.0688 | Accuracy: 100%*

> To deduplicate documents from Parquet files, use `dedup_exact_document` or `dedup_fuzzy_document` from `marin.processing.dedup`. For exact dedup: `from marin.processing.exact import dedup_exact_document`, then call `dedup_exact_document(input_paths=[...], output_path="gs://...", text_field="text", max_parallelism=N)` — it hashes each document's full text with XXH3-128 via `dupekit.transform`, then uses a Zephyr `group_by` to find collisions and writes surviving documents as Vortex/Parquet. For fuzzy dedup: `from marin.processing.fuzzy import dedup_fuzzy_document`, call `dedup_fuzzy_document(input_paths=[...], output_path="gs://...", fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism=N)` — this computes MinHash signatures via `dupekit.compute_minhash`, bands them into LSH buckets via `dupekit.compute_lsh`, runs `connected_components` (from `marin.processing.connected_components`) to find near-duplicate clusters (up to `max_iterations=10`; callers **must** check the returned `bool` for convergence), then writes deduplicated output. Critical constraint: `num_perms` (286) must be exactly divisible by `num_bands` (26); mismatching them raises `ValueError`. Both functions return a stats dict from `finalize_dedup` (`marin.processing.dedup_commons`).

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [pass] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

### Variation 10: No context baseline
*Context: 0 chars | Cost: $0.1846 | Accuracy: 88%*

> To deduplicate documents from parquet files, use one of three functions from `marin.processing.classification.deduplication`. For exact document dedup: `from marin.processing.classification.deduplication.exact import dedup_exact_document` — call `dedup_exact_document(input_paths=["gs://…/*.parquet"], output_path="gs://…/out", text_field="text", max_parallelism=256)`, which hashes each document with XXH3-128 (via the `dupekit` Rust library), then runs a two-stage Zephyr pipeline: `Dataset.from_list(file_groups).flat_map(_flat_map_document_hashes).group_by(hash_key, sort_by=id, reducer=annotate_dups).group_by(file_idx, reducer=aggregate_and_write)`. For fuzzy dedup: `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document` — key parameters are `fuzzy_minhash_num_perms=286`, `fuzzy_minhash_num_bands=26` (must divide `num_perms` evenly), `fuzzy_minhash_ngram_size=5`, `fuzzy_minhash_seed=42`; the pipeline computes MinHash signatures, groups by LSH bucket, then runs connected components to find duplicate clusters. For paragraph-level exact dedup: `from marin.processing.classification.deduplication.exact import dedup_exact_paragraph`. All three accept `filetypes` (default `["jsonl", "jsonl.gz", "jsonl.zst", "parquet"]`), `worker_resources` (default `ResourceConfig(cpu=1, ram="32g", disk="5g")`), and require `max_parallelism` explicitly. Results (unique/dup counts) are written as Vortex files under `output_path/data/` and returned as a stats dict.

**Fact check:**
- [pass] `dedup_fuzzy_document` — function name
- [pass] `num_perms=286` — correct default (not 128)
- [pass] `num_bands=26` — correct default
- [pass] `ngram_size=5` — correct default
- [pass] `seed=42` — correct default
- [MISS] `connected_components` — clustering step
- [pass] `dupekit` — Rust FFI library
- [pass] `marin.processing` — import path

