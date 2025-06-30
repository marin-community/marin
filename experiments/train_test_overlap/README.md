# Dolma Train/Test Overlap

This directory contains helper scripts for running Dolma-based
train–test overlap detection on large datasets.

## Scripts

- `dedupe_total.py` runs the deduplication pipeline on every shard of the
  configured datasets. Each shard is processed with the parameters in
  `ShardedDedupeConfig`.
- `aggregate_total.py` aggregates the per-shard results and produces CSV
  summaries and a contamination matrix.

Both scripts are standard Python programs executed via `python`. They use
Marin's executor framework, which handles Ray job submission and output
paths.

## Supported Input Formats

Dataset shards may end with any of the following extensions:

```
.parquet
.jsonl.gz
.jsonl.zst
.jsonl
.json.gz
.json.zst
```

Parquet shards are automatically converted to `jsonl.gz` before running
the dedupe step.

## Gotchas

- If the provided dataset path contains no files or only empty files, the
  pipeline will exit early to avoid hanging on empty shards.
- The Bloom filter false positive rate defaults to `1e-12` in
  `dedupe_total.py`. For very large datasets you may need to adjust this
  value to manage memory usage.
- Ensure the dataset path you provide actually contains files with one of
  the supported extensions; otherwise no shards will be discovered.
