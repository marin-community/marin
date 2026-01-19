# Decon Overlap Plan (Per-File Bloom Filters)

Goal: produce eval-side overlap stats that include **per-training-file provenance**
so we can answer “which FineMath file overlapped this eval instance?” and compare
Bloom-based overlap to exact n-gram overlap.

## Current design (what we are implementing)

We build **one Bloom filter per training file** and apply it to all eval datasets.
This yields per-(eval instance, training file) overlap results and lets the reducer
emit real `train_path` values like:

`gs://marin-us-central2/raw/finemath-7090a5/finemath-3plus/train-00000-of-00128.parquet`

This is not possible with a single Bloom filter over the entire dataset.

### Why not per (eval dataset, train shard) or per (eval file, train shard)?

- The Bloom filter only depends on training data. Rebuilding it per eval dataset
  is redundant and multiplies compute.
- Per eval file is even more redundant; eval provenance already exists in output.

Best practice: **one Bloom per training file**, applied to all eval datasets.

## Pipeline details

### 1) Per-file eval-side overlap runner

Script: `experiments/train_test_overlap/train_test_eval_side_finemath.py`

- Resolve the FineMath training root using the executor + prefix.
- Discover training files via `decon._collect_input_files()`.
- Create **one ExecutorStep per training file**, so steps can run in parallel.
- Each step:
  - Builds a Bloom filter from that single training file.
  - Applies it to each eval dataset.
  - Writes `train_metadata.json` with the full training file path.

Output layout:
```
gs://.../train_test_overlap/dolma/eval_side_per_file/finemath/<train_label>-<hash>/
  train_metadata.json
  bloom/15.bin
  15/<eval_dataset>/*.jsonl*
```

### 2) Per-file reducer

Script: `experiments/train_test_overlap/decon_eval_reduce.py`

- Discover per-file output dirs under:
  `train_test_overlap/dolma/eval_side_per_file/finemath/*`
- Read `train_metadata.json` to emit the real training file path as `train_path`.
- Aggregate overlaps into ngram_new-style stats:
  - `stats/overlap_stats.jsonl`
  - `stats/overlap_stats_by_train_path.jsonl`

## Tradeoffs

- **Compute**: scales with (#training files × #eval datasets).
- **Storage**: per-file bloom + per-file eval-side attributes.
- **Provenance**: exact training-file path is preserved.

---

## CODEX READ THIS IS WHAT I DID

1) **Changed** `experiments/train_test_overlap/train_test_eval_side_finemath.py`:
   - It now resolves the FineMath root using the executor + `--prefix`.
   - It discovers training files at runtime and creates **one ExecutorStep per file**.
   - Each step writes a `train_metadata.json` containing the full training file path.
   - Outputs land under `train_test_overlap/dolma/eval_side_per_file/finemath/<label>`.

2) **Changed** `experiments/train_test_overlap/decon_eval_reduce.py`:
   - It now discovers per-file outputs under
     `train_test_overlap/dolma/eval_side_per_file/finemath/*`.
   - It reads `train_metadata.json` to emit the real training file path in
     `overlap_stats_by_train_path.jsonl`.

These changes are the ones currently in the repo.
