# N-gram Train/Test Overlap (ngram_new)

This directory contains a Zephyr-based reimplementation of the old train/test
overlap pipeline. It builds a **test-side n-gram index** and streams training
documents to emit overlaps keyed by **test instance IDs**. Tokenization matches
the old DefaultTokenizer behavior (lowercase + split on whitespace/punctuation),
and the reduce stage aggregates per-eval-dataset overlap stats.

## What lives here

- `utils.py`: tokenizer + stable hashing + dataset-name parsing + file discovery.
- `overlap_map.py`: map stage (emit overlap events and test instance counts).
- `overlap_reduce.py`: reduce stage (aggregate per-dataset stats).
- `train_test_overlap_map.py`: executor entrypoint for map stage across datasets.
- `train_test_overlap_reduce.py`: executor entrypoint for map + reduce across datasets.

## Outputs

Map stage (`overlap_map.py`):
- `tmp/test_index.msgpack`: test-side n-gram index (ngram -> instance IDs).
- `tmp/overlap_instances-*.jsonl.gz`: per-shard overlap records with training provenance:
  `{"eval_dataset": str, "n": int, "instance_ids": list[str], "train_path": str, "train_doc_id": str}`.
  `train_doc_id` uses the training record `id` when present, otherwise the file path.
- `tmp/test_instance_counts.jsonl`: total test instance counts per eval dataset.
- `.SUCCESS`: completion marker.

Reduce stage (`overlap_reduce.py`):
- `stats/overlap_stats.jsonl`: records shaped as
  `{"eval_dataset": str, "n": int, "num_instances": int, "instance_ids": list[str], "instance_links": list[str]}`
  where `instance_links` are GCS file paths for the eval dataset shards.
- `stats/overlap_stats_by_train_path.jsonl`: records shaped as
  `{"eval_dataset": str, "n": int, "train_path": str, "train_doc_ids": list[str], "instance_ids": list[str], "instance_links": list[str], "overlap_count": int}`
  where `instance_links` are GCS file paths for the eval dataset shards.
- `.SUCCESS`: completion marker.

## Parallelism and sharding knobs

- **max parallelism** (`processes` / `DEFAULT_MAX_PARALLELISM`):
  Passed to `Backend.execute(..., max_parallelism=...)` for Zephyr. It controls
  how many training shards run concurrently. Each worker loads the test index,
  so higher values increase throughput but multiply memory usage by the index
  size. Lower values reduce peak memory at the cost of runtime.

- **default num shards** (`DEFAULT_NUM_SHARDS`):
  The number of training-file shards created before the map stage runs. This
  controls how many map tasks are launched and how many
  `overlap_instances-*.jsonl.gz` files are produced. More shards means:
  - smaller per-shard output files,
  - more map tasks and scheduling overhead,
  - more total index loads (each worker loads `test_index.msgpack`).
  Fewer shards means:
  - larger per-shard output files,
  - fewer map tasks,
  - less repeated index loading,
  - longer runtime per shard.

## Typical entrypoints

- Map only:
  `uv run python experiments/train_test_overlap/ngram_new/train_test_overlap_map.py --prefix gs://...`
- Map + reduce:
  `uv run python experiments/train_test_overlap/ngram_new/train_test_overlap_reduce.py --prefix gs://...`
