# Train/Test Overlap Pipeline Handoff (Executor + Decon + Aggregation)

This document is a “next agent” handoff for the train/test overlap (“contamination checking”) workflow in
`experiments/train_test_overlap/`.

It summarizes:
- How the Marin **executor framework** works (DAG steps, versioning, output paths, locking).
- How **decontamination/train-test-overlap** is implemented in `decon.py` (Bloom filters + Zephyr).
- How the **experiment scripts** wire it together (`eval_datasets_overlap.py`, `train_test_total.py`, `aggregate_total.py`).
- Practical run/debug notes and a few gotchas observed while reading the code.

## Key Files (start here)

Executor framework (vendored Marin):
- `lib/marin/src/marin/execution/executor.py`
- `lib/marin/src/marin/execution/executor_step_status.py`

Train/test overlap:
- `lib/marin/src/marin/processing/classification/decon.py`
- `experiments/train_test_overlap/eval_datasets_overlap.py`
- `experiments/train_test_overlap/train_test_total.py`
- `experiments/train_test_overlap/aggregate_total.py`

## 1) Executor Framework: How Experiments Run

### Mental model

An experiment is a **DAG** of `ExecutorStep`s. Each step:
- has a stable-ish human name (`step.name`)
- has a Python callable (`step.fn`)
- has a single dataclass config (`step.config`)
- produces a **materialized output directory** on a filesystem (local, `gs://`, `s3://` via fsspec)

Data is passed between steps **by path**, not in-memory. A step that depends on another step receives the upstream
step’s output path (or a subpath under it) in its config and is responsible for reading from it.

### `ExecutorStep` config “special values” that the executor resolves

Configs are dataclasses; fields may contain special objects that are *resolved during execution*:

- `this_output_path()` / `THIS_OUTPUT_PATH`
  - marks “write your output here”
  - resolves to the current step’s output directory (optionally joined with a subpath)

- `output_path_of(step)` / `step.cd("subdir")` / `step / "subdir"`
  - refers to an upstream step’s output directory (optionally with a subpath)

- `versioned(value)`
  - indicates which config values should influence the **step version hash**

Important executor behavior:
- If an `ExecutorStep` object appears inside a config (e.g. in a `list[...]` field), the executor treats it as
  `output_path_of(step)` and resolves it to a concrete string path before calling the step function.
  This is how some configs “carry around steps” but receive paths at runtime.

### Versioning and output paths

For each step, the executor constructs a “version” dict and hashes it:
- `version["name"] = step.name`
- `version["config"] = {...}` from `versioned(...)` fields (plus dependency placeholders)
- `version["dependencies"] = [...]` versions of blocking dependencies
- `version["pseudo_dependencies"] = [...]` versions of non-blocking deps (if any)

Then:
- `output_path = <prefix>/<step.name>-<md5(version)[:6]>`

If `override_output_path` is set on the step, it wins (with a warning if it differs from the computed path).

### Dependencies vs “pseudo-dependencies”

`InputName.nonblocking()` marks a dependency that:
- still affects the version hash (so downstream output paths change),
- but does **not** block execution (and is not launched automatically unless requested elsewhere).

This is used for “soft dependencies” like “consume a checkpoint that might still be running”.

### Execution + cluster integration

Execution is orchestrated by `Executor.run(...)`:
- Computes versions/output paths for the transitive closure of dependencies.
- Writes metadata (`.executor_info` per step, plus a global experiment JSON under `<prefix>/experiments/...`).
- Launches step jobs using **Fray** (`current_cluster()`), not raw Ray tasks.
  - Each step becomes a `JobRequest` with an `Entrypoint.from_callable(step_fn, args=[config])`.
  - The job uses a non-preemptible CPU resource by default for the driver (`ResourceConfig.with_cpu(preemptible=False)`).
  - Step-specific Python deps can be included via `step.pip_dependency_groups` (Fray `EnvironmentConfig.create(extras=...)`).

### Locking / idempotence (critical for shared pipelines)

Each step output directory has:
- `.executor_status` (status token: `RUNNING`, `SUCCESS`, `FAILED`, `DEP_FAILED`)
- `.executor_status.lock` (a JSON lease `{worker_id, timestamp}` refreshed by a heartbeat)

`should_run(...)` uses the lock + status to avoid duplicate work:
- `SUCCESS` → skip
- `FAILED`/`DEP_FAILED` → raise unless `force_run_failed=True`
- `RUNNING` + active lease → wait
- `RUNNING` + stale/no lease → takeover

Local locking uses `fcntl`; GCS locking uses generation-based conditional writes.

### Useful runtime knobs

`ExecutorMainConfig` (CLI via draccus) supports:
- `--prefix` (or `MARIN_PREFIX`)
- `--executor_info_base_path` (defaults to `<prefix>/experiments`)
- `--dry_run` (prints what would run, doesn’t acquire locks/launch jobs)
- `--run_only` (regex list: run only matching steps + their deps, skipping already-successful deps)
- `--force_run_failed` (rerun steps marked FAILED/DEP_FAILED)

## 2) Decontamination / Train-Test Overlap (`decon.py`)

### High-level purpose

`lib/marin/src/marin/processing/classification/decon.py` implements two workflows:

1. `DeconMode.DECONTAMINATE`
   - Build a Bloom filter from a contamination source
   - Apply it to an input dataset, writing “duplicate span” annotations

2. `DeconMode.TRAIN_TEST_OVERLAP`
   - Loop over n-gram sizes
   - Build a Bloom filter from `decontaminate_source`
   - Apply it to `input_path`, writing overlap annotations under `<output_path>/<ngram_len>/...`

### Config (`DeconConfig`)

Key fields:
- `input_path: str | list[str]`
  - path(s) containing data files; `_collect_input_files()` expands directories using a glob for
    `**/*.{jsonl,jsonl.gz,jsonl.zst,parquet}`.
- `decontaminate_source: str | None`
  - required in both modes (but note: codepaths treat this as a *path or list of paths* in practice).
- `text_field: str`
  - which field to read from parquet/json records (default `"text"`)
- `ngram: NGramConfig | None`
  - when provided, features are n-grams (not exact paragraphs)
- `processes: int`
  - also used as Zephyr `max_parallelism` and as the reshard count for Bloom construction
- `false_positive_rate` / `estimated_doc_count`
  - Bloom filter parameters
- `attribute_name`
  - key under `attributes` in the written JSONL

### Feature extraction and overlap scoring

- Text is split into paragraphs by newline (`"\n"`).
- Each paragraph is split into tokens by whitespace (`text.split()`).
- n-grams are `" ".join(tokens[i:i+n])` with stride `stride + 1`.
- Overlap score for a paragraph = `matches / len(ngrams)`.

The output annotations are “spans” in **character offsets** within the paragraph-joined-by-newlines text:
`[start_offset, end_offset, overlap_score]`.

Observation:
- `overlap_threshold` exists on `NGramConfig` but is not used to filter spans; current logic records any `overlap_score > 0`.

### Record IDs

Output records are keyed by:
- `record["id"]` if present, else
- a deterministic hash of the entire record (`msgspec.msgpack.encode(..., order="deterministic")` + blake2b)

### Bloom filter build (`build_filter`)

Pipeline shape:
1. Collect input files from `input_path` (directory globbing).
2. Zephyr pipeline:
   - `Dataset.from_iterable(all_files)`
   - `.reshard(num_shards=config.processes)`
   - `.load_file()` (auto parquet/jsonl)
   - `.select(config.text_field)`
   - `.map_shard(build_shard_bloom)` where each shard creates a Bloom and yields `bf.save_bytes()`
   - `.write_binary(f"{bloom_path}-{shard:05d}-of-{total:05d}.bin", skip_existing=True)`
3. If multiple shard blooms were written, merge them via another Zephyr run and write final `bloom_path`.

Notes:
- Bloom building reads *all* records in the contamination source(s), but does not retain dataset provenance
  (it becomes a single combined filter).

### Applying the Bloom filter (`mark_duplicates_bloom`)

Pipeline shape:
1. Collect input files from `input_path`.
2. Zephyr pipeline:
   - `Dataset.from_iterable(all_files)`
   - `.flat_map(load_file)` to produce records
   - `.map_shard(process_shard_with_bloom)`:
     - loads Bloom filter once per shard (per input file)
     - yields `{id, attributes: {attribute_name: duplicate_spans}}`
   - `.write_jsonl(output_pattern=..., skip_existing=True)`

Output path mapping:
- It uses `rebase_file_path(base_path, file_path, output_path, ...)` to mirror the input file tree under `output_path`.
- `base_path` is the first input path if a list; this assumes all inputs share a common prefix.

Important gotcha:
- The code currently does **not** change file extensions when writing JSONL “attribute” files.
  If an input is `.parquet`, the output path will also end with `.parquet`, but the contents are JSONL.
  This can matter for globbing/aggregation scripts that assume `.jsonl*` extensions.

### What `_run_train_test_overlap` actually does

For each `ngram_len`:
- Build bloom: `bloom/<ngram_len>.bin` under the step output.
- Write attributes under: `<output_path>/<ngram_len>/...`
- The attribute key used is `f"{attribute_name}_{ngram_len}"` (e.g. `ngram_overlap_15`).

## 3) Experiment Scripts in `experiments/train_test_overlap/`

### A) `eval_datasets_overlap.py`: build “decontamination-format” eval datasets

Purpose:
- Define `EVAL_DATASET_STEPS`: a list of `ExecutorStep`s that download and/or convert a bunch of evaluation datasets into a
  Dolma-style JSONL format suitable for decontamination.

Mechanics:
- Many steps use `hf_dataset_to_jsonl(... output_format=OutputFormatOptions("decontamination"))`.
- This file is not “run” directly by the pipeline most of the time; instead, other scripts import `EVAL_DATASET_STEPS` and
  embed them in configs so the executor can resolve/run them as dependencies.

### B) `train_test_total.py`: run overlap detection for multiple “training” datasets

Purpose:
- Define one overlap step per dataset in `DATASET_CONFIGS` and run them with the executor.

What it builds:
- `DEFAULT_NGRAM_CONFIG` uses `ngram_length=[15]` by default.
- For each dataset:
  - Creates `DeconConfig(... mode=TRAIN_TEST_OVERLAP, attribute_name="ngram_overlap", ngram=DEFAULT_NGRAM_CONFIG, ...)`
  - Uses `decontaminate_source=EVAL_DATASET_STEPS`
  - Wraps `decontaminate(config)` in a small `run_train_test_overlap` runner.

Executor-specific subtlety:
- `EVAL_DATASET_STEPS` (a list of `ExecutorStep` objects) is stored inside the config.
  The executor resolves those into actual output paths (strings) before calling `run_train_test_overlap`.
  It also schedules those eval steps to run first because they are dependencies.

Expected output layout (per training dataset step):
- `<prefix>/train_test_overlap/dolma/total/<dataset>-<hash>/`
  - `.executor_status`, `.executor_info`, etc.
  - `bloom/15.bin`
  - `15/...` (attribute files for that n-gram)

Potential conceptual mismatch to keep in mind:
- `decon._run_train_test_overlap` builds a bloom from `decontaminate_source` and applies it to `input_path`.
  In `train_test_total.py`, that means:
  - Bloom built from **evaluation** data
  - Applied to **training** data
  - Output doc IDs correspond to training records

### C) `aggregate_total.py`: aggregate overlap results into CSVs

Purpose:
- Aggregate overlap across all discovered training datasets with:
  1. Per-training-dataset summary
  2. Union summary across all training datasets
  3. An overlap matrix (rows: eval datasets; columns: training datasets)

How it works:
1. Compute eval dataset sizes via Zephyr:
   - counts total examples under each eval dataset output directory
2. Discover training datasets by scanning `attributes_base_path` for files matching:
   - `{base}/*/<ngram>/**/*.jsonl*`
   - The first component after base path is treated as the “training dataset root”
3. For each discovered training dataset root:
   - Find shards with glob: `**/<ngram>/**/*.jsonl*.zst`
   - Create “intermediate” JSONL files with one line per record:
     `{id, test_dataset, training_dataset, has_overlap}`
   - Accumulate `set()`s of unique IDs and overlapping IDs overall + per test dataset
4. Write outputs:
   - `summary.csv` (per training dataset + union)
   - `overlap_matrix.csv` (eval_dataset x training_dataset + union column)

Where “test_dataset” comes from:
- It is parsed from the attribute shard path: the directory segment immediately after `<ngram_size>` (split on `-`).
  Example assumed by the code/docs:
  `.../<training_root>/15/<eval_dataset>/train/c0000.jsonl.zst`

Important: this implies the attribute directory layout includes an explicit eval dataset segment.

Potential mismatches / gotchas:
- The aggregator currently assumes `.jsonl*.zst` attribute shards.
  If `decon.mark_duplicates_bloom()` writes outputs with `.parquet` extensions (because inputs were parquet),
  discovery and aggregation globs may miss them.
- The matrix uses `cont / tot` where:
  - `cont` = count of overlapping **ids** seen in the attribute shards for that eval dataset
  - `tot` = total example count for the eval dataset directory
  This ratio only makes sense if the `id`s in the attribute shards are **eval dataset ids**.
  If the attributes were generated on training docs (as `train_test_total.py` currently suggests),
  the ratio is not well-defined.

## 4) How to Run (typical patterns)

Local-ish run (writes under a local prefix):
```bash
python experiments/train_test_overlap/train_test_total.py --prefix /tmp/marin
python experiments/train_test_overlap/aggregate_total.py --prefix /tmp/marin
```

### Running aggregation on a specific run (example: Proofpile)

If you ran `experiments/train_test_overlap/train_test_proofpile.py` and got an experiment JSON like:

`gs://marin-us-central2/experiments/train_test_proofpile-f8e82f.json`

you can aggregate *that run’s* outputs by pointing the aggregator at the directory that contains the step output.

1) Find the overlap step output directory
- Open the experiment JSON in the data browser and locate the step with name `tmp/train_test_overlap/proofpile`.
- Its `output_path` will look like:
  - `gs://marin-us-central2/tmp/train_test_overlap/proofpile-<hash>`

2) Choose the aggregator `attributes_base_path`
- The aggregator expects `attributes_base_path/*/<ngram>/**/*.jsonl*`.
- For the proofpile run above, use the **parent** directory:
  - `attributes_base_path = gs://marin-us-central2/tmp/train_test_overlap/`

3) Run `aggregate_total` with that base path

`aggregate_total.py` hardcodes its defaults via `build_aggregate_total_step()`, so to override
`attributes_base_path` you typically do one of:

- Quick one-off wrapper (recommended):
  - Create a short script that calls `build_aggregate_total_step(attributes_base_path=...)` and then `executor_main(...)`.
- Or temporarily edit `experiments/train_test_overlap/aggregate_total.py` to change the default argument.

Cluster example (wrapper script assumed at `experiments/train_test_overlap/run_aggregate_proofpile.py`):
```bash
python marin/run/ray_run.py -- \
  python experiments/train_test_overlap/run_aggregate_proofpile.py --prefix gs://marin-us-central2
```

Expected outputs:
- `summary.csv` should give a meaningful “contaminated / total” fraction for the training dataset(s) discovered.
- `overlap_matrix.csv` is only meaningful if the attribute directory structure encodes an evaluation dataset segment
  (see “Open Questions / TODOs” below); with the current merged-bloom implementation, it may be all zeros.

Cluster run (common pattern; uses the Ray+Fray helpers):
```bash
python marin/run/ray_run.py -- python experiments/train_test_overlap/train_test_total.py --prefix gs://my-bucket
python marin/run/ray_run.py -- python experiments/train_test_overlap/aggregate_total.py --prefix gs://my-bucket
```

Debugging executor behavior:
- `--dry_run` to see which steps would run
- `--run_only '^train_test_overlap/...'` to run a subset + deps
- `--force_run_failed` to rerun FAILED steps

## 5) What to Inspect When Debugging

Executor metadata:
- Per step:
  - `<step_output>/.executor_status`
  - `<step_output>/.executor_status.lock`
  - `<step_output>/.executor_info`
- Per experiment (global DAG metadata):
  - `<prefix>/experiments/<entrypoint>-<hash>.json`

Decon outputs for a step:
- `<step_output>/bloom/<ngram>.bin`
- `<step_output>/<ngram>/...` (rebased attribute files)

Aggregation outputs:
- `<aggregate_step_output>/summary.csv`
- `<aggregate_step_output>/overlap_matrix.csv`
- plus `.<output>/.intermediate/<training_name>/...` per training dataset

## 6) Open Questions / TODOs for the Next Agent

These are not fixes; they’re “things to verify” because they affect whether the pipeline behaves as intended:

1. **Directionality vs aggregation expectations**
   - `aggregate_total.py` looks structured to compute “fraction of eval set contaminated by training set” (matrix rows are eval datasets).
   - `train_test_total.py` appears to compute “training docs that overlap eval data” (bloom from eval → applied to training).
   - Verify which direction is desired, and ensure output path layout + IDs match the aggregator’s assumptions.

2. **Per-eval-dataset provenance**
   - Current Bloom construction merges all eval datasets into one filter, losing which eval dataset caused a match.
   - Aggregation tries to break out overlaps per eval dataset based on shard path structure.
   - If per-eval breakdown is required, the decon pipeline likely needs to run per eval dataset (or store provenance in output).

3. **Attribute output extensions**
   - `mark_duplicates_bloom()` writes JSONL but preserves the input file extension in the output path.
   - If inputs are parquet, output files will be “JSONL named .parquet”, which breaks globbing and is confusing.
   - Decide on a convention (e.g. always write `.jsonl.zst`) and align aggregation globs accordingly.

4. **`overlap_threshold` is currently unused**
   - If you want “contaminated if overlap >= threshold”, implement the threshold check where spans are appended.

5. **`DeconConfig.decontaminate_source` typing**
   - It’s annotated as `str | None` but is used as “path or list of paths” in practice (via executor resolution of lists of steps).
   - Consider making the type match reality to reduce footguns.
