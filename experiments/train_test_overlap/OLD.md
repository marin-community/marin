# Old Train/Test Overlap Pipeline (marin2)

This document describes the *old* train/test overlap workflow from
`/Users/ahmed/code/marin2/train_test_overlap/`, end to end. It explains the
exact data flow, the intermediate artifacts, and how sharded runs were
consolidated. The mapping to the current workflow comes **after** this section.

## Where the old implementation lived

Core package (all in `/Users/ahmed/code/marin2/train_test_overlap/`):
- `run_data_overlap.py`: end-to-end pipeline for one training shard or dataset
- `compute_data_overlap_metrics.py`: builds n-gram index from test scenarios and
  scans training docs for overlap
- `compute_metrics_from_ngrams.py`: computes binary/Jaccard/token metrics (uses NLTK)
- `metrics.py`: aggregates instance-level metrics to dataset-level outputs
- `data_overlap_spec.py`: dataclasses for keys and outputs
- `run_overlap_shards.py`: runs one `run_data_overlap` per training shard
- `utils.py`: tokenizer and dataclass serialization helpers

Executor-facing pipelines (all in `/Users/ahmed/code/marin2/experiments/train_test_overlap/train_test/`):
- `overlap_pipeline_*_sharded.py`: launch sharded runs for specific datasets
- `consolidate_sharded_pipeline.py`: merges shard outputs
- `aggregate_test_overlap.py`: final dataset-level summary

## Inputs and data formats (old)

### Scenario data (test set)
`scenario_data` is a JSONL in "scenario" format. Each line describes a dataset
and a list of instances:
- `scenario_key.scenario_spec.class_name`: dataset identifier (HELM scenario)
- `scenario_key.scenario_spec.args`: subset metadata (e.g., `subject`)
- `scenario_key.split`: split (typically `test`)
- `instances`: list of `{id, input, references}`

This is loaded via `load_light_scenarios_from_jsonl` and turned into
`LightScenario` + `LightInstance` objects.

### Training data
`input_data` is a file or directory of JSON/JSONL text data. The pipeline scans
files matching:
`**/*.jsonl.gz`, `**/*.jsonl.zst`, `**/*.jsonl.gs`, `**/*.json.gz`,
`**/*.json.zst`, `**/*.jsonl`.

Each line is JSON with a `text` field. Streaming is done with `fsspec` using
compression inference.

## Old pipeline: single run (one shard)

The main entrypoint is `run_data_overlap.py` (Ray remote task). It writes a
`.SUCCESS` marker next to the output path for idempotence.

### Step 1: Load and normalize the test scenarios
1. Scenario files under `scenario_data` are merged into a local JSONL file.
2. Each line is parsed into `LightScenario` objects.
3. For each scenario and each n-gram size `n`, a `DataOverlapStatsKey` is created
   and a `stats_key_counts` entry tracks how many instances exist for that key.

### Step 2: Build the n-gram index (test side)
`create_ngram_index(...)` builds an in-memory index of n-grams for **test
inputs and references**:
- Tokenization uses `DefaultTokenizer` (lowercase, split on whitespace and
  punctuation).
- For each test instance:
  - All input n-grams are indexed under `part="input"`.
  - All reference n-grams are indexed under `part="references"`.
- Index shape: `ngram_index[n][ngram_tuple] -> set[EntryDataOverlapKey]`

This is an exact, deterministic set-based index (no Bloom filter).

### Step 3: Stream training docs and mark overlaps
`create_compressed_file_iterator(...)` streams documents from training files,
yielding `(document_text, source_info)` tuples. For each document:
- Tokenize with `DefaultTokenizer`.
- For each n-gram in the document:
  - If it exists in `ngram_index[n]`, mark all associated
    `EntryDataOverlapKey`s as overlapping.
  - Track:
    - `stats_key_to_input_ids` and `stats_key_to_reference_ids`
    - `entry_overlap_key_to_ngram_counts` (ngram frequency counts per test instance)

The overlap direction is: **test n-grams indexed, training docs scanned**.
Outputs are keyed by *test instance IDs*.

### Step 4: Write raw outputs (local temp)
The pipeline writes two raw files:
- `stats`: one `DataOverlapStats` record per scenario + n, containing:
  - list of test instance IDs with overlapping inputs
  - list of test instance IDs with overlapping references
  - total number of instances for that scenario
- `raw_ngrams`: `EntryOverlapNgrams` records containing per-instance n-gram
  overlap counts

### Step 5: Compute overlap metrics (NLTK-based)
`compute_metrics_from_ngrams.get_metrics(...)` reads the raw n-grams and the
scenario JSONL to produce instance-level metrics:
- Uses NLTK `ngrams(...)` to compute:
  - binary overlap (any overlap)
  - Jaccard overlap
  - token-level overlap
- Runs each metric twice:
  - no frequency filter (`filter_value=0`)
  - filtered to rare n-grams (`filter_value=10`)
- Optional inverse-frequency weighting for Jaccard/token overlap

Each output line is an `EntryOverlapMetric`.

### Step 6: Aggregate metrics (dataset-level)
`metrics.aggregate_metrics(...)` reads instance-level metrics and produces
`AggregateOverlapMetric` records:
- Aggregation key: `(stats_key, part, metric_protocol_spec)`
- Output keeps:
  - `instance_ids`: overlapping test instance IDs
  - `metric_scores`: metric scores aligned with `instance_ids`
  - `metrics_input_path`: the original training shard path
- Only non-weighted metrics are retained in this aggregation:
  - binary, Jaccard, token (all `filter_value=0`)

### Step 7: Copy outputs to the final output path
The pipeline copies the temp outputs into the requested `output_path`:
- `aggregate_metrics_<n>/aggregate_metrics_<n>` (one file per n)
- `raw_ngrams/raw_ngrams.jsonl`
- `stats/overlap_stats.jsonl`
- `instance_mapping/instance_mapping.json`
- `.SUCCESS` marker

The `instance_mapping.json` file makes it easy to see which test instances
overlap which dataset/n combinations.

## Old pipeline: sharded execution

`run_overlap_shards.py` provides a sharded runner:
- Scans `base_input_dir` for JSONL shards.
- For each shard, runs `run_data_overlap` with:
  - `input_data = shard_path`
  - `scenario_data = consolidated scenario JSONL`
  - `output_path = output_base/<relative shard path>`
- Uses `simple_backpressure(...)` to limit how many Ray tasks are in-flight.

Each shard produces the same output structure as the single-run pipeline.

## Old pipeline: consolidation and final reporting

### Consolidation (`consolidate_sharded_pipeline.py`)
This script recursively merges shard outputs by directory:
- **overlap_stats.jsonl**: union of overlapping instance IDs per scenario/n.
- **raw_ngrams.jsonl**: concatenated (optional; controlled by `only_stats`).
- **instance_mapping.json**: merged per-instance overlap lists.
- **aggregate_metrics_{n}.jsonl**: combined per-n metrics with
  `metrics_input_paths` preserved per instance.

Each consolidated directory gets its own `.SUCCESS` marker.

### Final summaries (`aggregate_test_overlap.py`)
This script reads consolidated metrics and produces dataset-level summaries:
- Filters by `partial_overlap_spec` (binary, jaccard, token).
- Uses the consolidated scenario JSONL to compute total counts per dataset,
  subset, and split.
- Outputs per dataset:
  - `ngram_<n>/<partial>.jsonl` with instance IDs and provenance
  - `summary_input.jsonl` and `summary_references.jsonl`
  - `summary_input_total.jsonl` and `summary_reference_total.jsonl`

These summaries report overlap counts and fractions for each dataset and split.

---

# Mapping to the current workflow (marin)

Below is a 1:1 mapping from the old pipeline to the current implementation in
`/Users/ahmed/code/marin/experiments/train_test_overlap/` and
`/Users/ahmed/code/marin/lib/marin/src/marin/processing/classification/decon.py`.

## Component mapping

| Old pipeline piece | New pipeline piece | Notes / differences |
| --- | --- | --- |
| Scenario JSONL ingestion (`load_light_scenarios_from_jsonl`) | `eval_datasets_overlap.py` + `EVAL_DATASET_STEPS` | Old used structured `input`/`references` with instance IDs. New eval data is decontamination-format JSONL with a single `text` field. |
| Exact n-gram index (`create_ngram_index`) | Bloom filter build (`build_filter`) | Old index was exact and stored per-instance provenance. New uses Bloom filter over all eval data, losing per-eval provenance unless runs are separated. |
| Tokenization (`DefaultTokenizer`) | `text.split()` in `decon.py` | Old lowercased and stripped punctuation. New tokenization is whitespace only and case-sensitive. This changes n-gram boundaries and matches. |
| Training scan (`compute_document_data_overlap`) | `mark_duplicates_bloom` | Old computed per-test-instance hits and counts. New produces paragraph span overlaps with a float score. |
| Metrics (binary/Jaccard/token) | None built-in | New pipeline only emits overlap spans in attributes. If you need old metrics, you have to compute them downstream. |
| Per-shard Ray jobs (`run_overlap_shards.py`) | Executor steps + Zephyr | New runs are scheduled via `ExecutorStep` and Zephyr pipelines rather than manual Ray backpressure. |
| Shard consolidation (`consolidate_sharded_pipeline.py`) | `aggregate_total.py` | New aggregation reads attribute shards and builds summaries; it does not merge raw n-grams or instance mappings. |
| Output layout (stats, raw_ngrams, metrics) | `bloom/<n>.bin` + `<n>/...` attribute JSONL | New writes overlap annotations per document under a rebased path. |

## Directionality mapping (important)

- **Old direction**: Build n-gram index from *test scenarios* and scan *training
  data*. Outputs are keyed by **test instance IDs**.
- **New default direction** (`train_test_total.py`): Build Bloom from *eval data*
  and apply to *training data*. Outputs are keyed by **training document IDs**.

If the desired question is "what fraction of the test set appears in training?",
the old pipeline directly produced that. In the new pipeline, aggregation may
need to be adapted (or the direction reversed) to recover the same semantics.

## Script-level mapping

| Old script | New script(s) | What it corresponds to |
| --- | --- | --- |
| `train_test_overlap/run_data_overlap.py` | `lib/marin/src/marin/processing/classification/decon.py` + `experiments/train_test_overlap/train_test_total.py` | Core overlap detection and output emission. |
| `train_test_overlap/run_overlap_shards.py` | `train_test_total.py` executor steps | Old: per-shard Ray tasks; New: one executor step per dataset. |
| `consolidate_sharded_pipeline.py` | `aggregate_total.py` | Old: merge per-shard outputs. New: compute aggregate summaries from attribute shards. |
| `aggregate_test_overlap.py` | `aggregate_total.py` | Old: dataset-level overlap fractions by test instance. New: overlaps counted by training doc IDs. |
