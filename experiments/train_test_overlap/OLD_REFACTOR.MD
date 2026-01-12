# OLD pipeline refactor plan (Zephyr map-reduce)

This plan proposes a Zephyr-based refactor that reproduces the *old*
train/test overlap behavior described in `experiments/train_test_overlap/OLD.md`,
while keeping the same external API shape as
`experiments/train_test_overlap/train_test_proofpile.py`.

**Current goal (Attempt 3):** extend the `ngram_new` map/reduce pipeline to
emit *full overlap provenance* for every matching n-gram between eval and
training data. This includes file paths, row numbers, the exact n-gram text,
and character offsets in the original text for both sides.
If a test row has fewer tokens than a configured `n`, the pipeline should emit
the **largest possible n-gram** for that row using `min(n, token_count)`, and
record the **effective** n-gram length in outputs.

The core idea is a map-reduce pipeline:
1) build an n-gram inventory of all test data
2) map over training shards to compute overlaps
3) reduce to per-test-instance counts and summaries

---

## Assumptions

Confirmed assumptions:
1) Test dataset format is decontamination JSONL with a single `text` field.
   Overlap is computed only on that field (no references or answer reconstruction).
2) Tokenization defaults to the old `DefaultTokenizer` behavior, and we must
   match it exactly:
   - `text.lower()` before tokenization.
   - `re.compile(rf"[\\s{re.escape(string.punctuation)}]+")` as the split pattern.
   - Return the raw `re.split(...)` result without filtering empty strings.
   The pipeline still exposes a pluggable tokenizer for alternate behaviors.
3) We need detailed overlap outputs (not just aggregated counts). Large
   intermediate artifacts are acceptable and should live on GCS.

4) Per-eval-dataset provenance is required, keyed only by dataset name
   (no subset/split).
   Example:
   - If a training doc overlaps with both `mmlu` and `gsm8k`, it must be
     recorded under **both** eval datasets (matrix-style output).

---

## Goals and constraints

- Keep the *same external interface* as `train_test_proofpile.py`:
  `build_*_step()` returns an `ExecutorStep`, and the config shape remains
  aligned with `DeconConfig` (input_path, decontaminate_source, ngram list, etc).
- Use Zephyr for all data processing (no ad-hoc Ray loops).
- Preserve the old directionality: test n-grams indexed, training shards scanned.
- Track full overlap provenance for each eval instance, including eval file
  path + row, training file path + row, n-gram text, and character offsets.
- Keep the implementation simple but scalable (no monolithic in-memory index).

---

## Proposed architecture (high level)

We implement a new Zephyr-based pipeline that:
1) normalizes eval/test data into a stream of test instances
2) builds sharded, hash-partitioned n-gram tables for **test** and **training**
3) performs a *partitioned join* per shard to emit per-instance overlaps
4) reduces overlaps into:
   - `overlap_stats.jsonl` (per eval dataset name, per n, list of instance IDs)
   - optional intermediate overlap shards (for debugging/inspection)

The entrypoint script remains identical to `train_test_proofpile.py` in shape,
but calls the new pipeline implementation internally.

---

## Phase 0: Normalize inputs (test + training)

Define a common record shape for test instances:
```python
@dataclass(frozen=True)
class TestInstance:
    eval_dataset: str
    instance_id: str
    text: str
```

Test records:
- `eval_dataset` derived from the eval dataset step name when available.
  If only paths are available, parse from the output path by:
  1) stripping the executor hash suffix (`-` + 6 hex chars), then
  2) dropping a trailing `-dolma` suffix if present.
  The resulting value is the dataset name only (no subset/split).
- `instance_id` from record `id` if present, else a stable hash of the full
  record using deterministic msgpack + blake2b (same behavior as
  `decon._record_id`).
- `text` from record `text` only

Zephyr pipeline for normalization:
```python
test_instances = (
    Dataset.from_files(eval_paths, "**/*.jsonl*")
    .flat_map(load_file)
    .flat_map(normalize_test_records)  # yields TestInstance dicts
)
```

Training data normalization yields a stream of `{"text": ..., "doc_id": ...}`,
where `doc_id` uses the same `id`-or-hash rule as `instance_id`.

---

## Phase 1: Emit partitioned n-grams (map)

We generate n-gram records for both test and training data and partition them
by a stable hash to enable shard-local joins.

Tokenizer options:
- Default: exact old `DefaultTokenizer` (lowercase + regex split on whitespace
  and punctuation; keep empty tokens from `re.split`).
- Alternate: a pluggable tokenizer selected via config, e.g.
  `tokenizer="whitespace"` or `tokenizer="no_lowercase"`

Common n-gram extraction:
```python
def iter_ngrams(text: str, n: int) -> Iterator[str]:
    tokens = tokenize(text)  # exact old DefaultTokenizer unless overridden
    for i in range(0, len(tokens) - n + 1):
        yield " ".join(tokens[i:i+n])
```

Emit test n-grams:
```python
def emit_test_ngrams(inst: dict, n_values: list[int]) -> Iterator[dict]:
    for n in n_values:
        for ng in iter_ngrams(inst["text"], n):
            yield {
                "n": n,
                "ngram": ng,
                "entry_key": {
                    "eval_dataset": inst["eval_dataset"],
                    "instance_id": inst["instance_id"],
                },
            }
```

Emit training n-grams with counts:
```python
def emit_train_ngrams(rec: dict, n_values: list[int]) -> Iterator[dict]:
    for n in n_values:
        for ng in iter_ngrams(rec["text"], n):
            yield {"n": n, "ngram": ng, "count": 1}
```

Partition by hash:
```python
def shard_key(n: int, ngram: str, num_shards: int) -> int:
    # stable_hash: blake2b digest of UTF-8 bytes, 64-bit big-endian int
    return stable_hash(f"{n}:{ngram}") % num_shards
```

Zephyr pipelines (partitioned writes):
```python
test_ngrams = (
    test_instances
    .flat_map(lambda inst: emit_test_ngrams(inst, n_values))
    .map(lambda r: {"shard": shard_key(r["n"], r["ngram"], num_shards), **r})
    .group_by(key=lambda r: r["shard"], reducer=collect_by_shard)
    .write_jsonl(f"{output}/tmp/test_ngrams-{{shard:05d}}.jsonl.gz")
)

train_counts = (
    Dataset.from_files(train_paths, "**/*.jsonl*")
    .flat_map(load_file)
    .flat_map(lambda rec: emit_train_ngrams(rec, n_values))
    .map(lambda r: {"shard": shard_key(r["n"], r["ngram"], num_shards), **r})
    .group_by(
        key=lambda r: (r["shard"], r["n"], r["ngram"]),
        reducer=lambda key, items: {"shard": key[0], "n": key[1], "ngram": key[2], "count": sum(i["count"] for i in items)},
        num_output_shards=num_shards,
    )
    .write_jsonl(f"{output}/tmp/train_counts-{{shard:05d}}.jsonl.gz")
)
```

---

## Phase 2: Shard-local join (reduce by n-gram)

We join test n-grams with training counts per shard to produce overlap events.
Each shard pair is processed with `map_shard` (no global in-memory index).

Shard join function:
```python
def join_shard(test_path: str, train_path: str) -> Iterator[dict]:
    train_counts = {}
    for rec in load_jsonl(train_path):
        train_counts[(rec["n"], rec["ngram"])] = rec["count"]

    for rec in load_jsonl(test_path):
        key = (rec["n"], rec["ngram"])
        count = train_counts.get(key)
        if count:
            yield {
                "entry_key": rec["entry_key"],
                "n": rec["n"],
                "ngram": rec["ngram"],
                "train_count": count,
            }
```

Zephyr pipeline:
```python
shard_pairs = [{"test": t, "train": tr} for t, tr in zip(test_paths, train_paths)]
overlaps = (
    Dataset.from_list(shard_pairs)
    .flat_map(lambda p: join_shard(p["test"], p["train"]))
    .write_jsonl(f"{output}/tmp/overlaps-{{shard:05d}}.jsonl.gz")
)
```

---

## Phase 3: Aggregate overlaps per test instance (reduce)

Reduce overlap events into per-instance stats (old `DataOverlapStats`).
Optional: also produce `EntryOverlapNgrams` and metric inputs.

Map stage emits training provenance per overlap record:
```json
{
  "eval_dataset": "<name>",
  "n": 15,
  "instance_ids": ["..."],
  "train_path": "gs://.../train-00000-of-00128.jsonl.zst",
  "train_doc_id": "<record.id or train_path when missing>"
}
```

```python
def reduce_instance(key, items):
    counts = {}
    for it in items:
        counts[it["ngram"]] = counts.get(it["ngram"], 0) + it["train_count"]
    return {"entry_key": key, "overlapping_ngram_counts": list(counts.items())}
```

Zephyr pipeline:
```python
instance_overlaps = (
    Dataset.from_files(f"{output}/tmp/overlaps-*.jsonl.gz")
    .flat_map(load_jsonl)
    .group_by(key=lambda r: (r["entry_key"], r["n"]), reducer=reduce_instance)
)
```

From `instance_overlaps`, emit:
- `overlap_stats.jsonl` with records shaped as:
  `{"eval_dataset": str, "n": int, "num_instances": int, "instance_ids": list[str]}`
  where `num_instances` is the total test instances for that dataset (all
  instances, not just overlapping ones).
- optional intermediate overlap shards (for debugging/inspection)

---

## Phase 4: Outputs and layout

Keep the old output layout for compatibility:
```
<output_path>/
  stats/overlap_stats.jsonl
  tmp/overlap_instances-*.jsonl.gz      # intermediate on GCS (with training provenance)
  .SUCCESS
```

Write with `skip_existing=True` to enable resumability and re-runs.
All intermediate files should live under the GCS output prefix.

Updated overlap artifacts (attempt 2):
- `tmp/overlap_instances-*.jsonl.gz` contain training provenance for each overlap,
  including `train_path` and `train_doc_id` (record id when present, else path).

---

## API compatibility with `train_test_proofpile.py`

We keep the existing entrypoint file and shape:
- `build_proofpile_step()` returns an `ExecutorStep`
- `run_train_test_overlap(config: DeconConfig) -> str` remains the entrypoint

Internally:
- replace `decontaminate(config)` with `run_overlap_mapreduce(config)`
- map `DeconConfig.decontaminate_source` to eval/test dataset paths
- map `DeconConfig.input_path` to training data path(s)
- map `NGramConfig.ngram_length` to `n_values`
- add an optional tokenizer selector (defaulting to old `DefaultTokenizer`)
- keep `processes` as Zephyr `max_parallelism`

No external API changes for callers.

---

## Scalability knobs (simple but safe)

- `num_shards`: partition count for n-gram tables (defaults based on dataset size)
- `max_parallelism`: Zephyr backend limit (from `processes`)
- `skip_existing=True`: resume partially completed runs
- `limit_test_instances` (optional debug knob)

---

## Minimal validation plan

- Run the refactored pipeline on a tiny GCS prefix (1-2 eval files, 1-2 training files).
- Compare overlap counts with the old pipeline for the same inputs.
- Verify `overlap_stats.jsonl` contains the expected instance IDs and counts.

---

## ATTEMPT 2: What went wrong and what to fix

Observed failure mode:
- The map stage emitted all training n-grams and then did a global `group_by`
  on `(n, ngram)`. This triggers a massive shuffle and per-worker buffering in
  Zephyr's scatter stage, leading to OOM, node deaths, and infinite retries.
- Logs show `Map + Scatter` with 1024 shards and Ray killing `run_stage` workers
  for memory pressure (workers reaching 150-160 GB+).

Root cause:
- The join strategy is too memory-intensive for large training corpora because
  it materializes the full training n-gram stream before filtering against test.

Fix strategy:
1) Avoid shuffling the full training n-gram stream.
   - Build a test-side n-gram index (partitioned) and stream training docs,
     emitting overlap events only when a training n-gram is present in the test
     index (old pipeline behavior).
2) If a shuffle/join approach is still desired:
   - Lower `num_shards`, lower Zephyr `chunk_size`, and reduce parallelism to
     bound per-worker memory.
   - Prefer shard-local joins where each worker loads only one test shard
     and one training-count shard at a time.

---

## Tracking suggestions

Tracking suggestion:
- Add a lightweight progress pipeline that runs alongside the map stage and
  writes `progress/` JSONL snapshots (e.g., every N files or per input shard):
  - counts of training records processed
  - counts of test records processed
  - total n-grams emitted (train/test)
  - unique overlap events emitted
  This can be a separate Zephyr `Dataset.from_list(file_specs)` run that
  aggregates counters with `reduce` and writes periodic checkpoints
  (`progress/progress-00001.jsonl`, etc.) so long-running jobs show forward
  movement even if the main shuffle is slow.
- Emit a per-shard heartbeat log in `map_shard` (file path + running totals),
  and keep logs structured (JSON) to make interleaved output easier to parse.
- Write a small `progress_summary.json` after each stage with:
  `num_eval_files`, `num_train_files`, `num_shards`, `max_parallelism`,
  and any completed output paths for quick diagnosis on retries.

---

## ATTEMPT 3: Detailed overlap tracing (new goal)

**Goal:** update `experiments/train_test_overlap/ngram_new/train_test_overlap_map.py`
and `experiments/train_test_overlap/ngram_new/train_test_overlap_reduce.py` to
emit *full per-overlap provenance* rather than only aggregated stats.

### Required overlap record (per eval row, train row, n-gram)

Each record must capture a single `(eval_row, train_row, ngram)` relationship
and include lists of character offsets in the original text for both sides.
The same eval row can match multiple train rows, and the same train row can
match multiple different n-grams (each as its own record).

**Eval-side fields:**
- `eval_dataset` (dataset name)
- `eval_path` (GCS link to the eval JSONL/JSON/parquet file)
- `eval_row` (0-based row index within the eval file)
- `eval_text` (full text for the eval row)
- `ngram` (the exact n-gram string)
- `eval_offsets` (list of `[start, end]` character offsets for this n-gram)
- `n` (effective n-gram length after applying `min(n, token_count)` for short rows)

**Train-side fields:**
- `train_path` (GCS link to the training JSONL/JSON/parquet file)
- `train_row` (0-based row index within the training file)
- `train_text` (full text for the training row)
- `train_ngram` (the exact n-gram string; should match `ngram`)
- `train_offsets` (list of `[start, end]` character offsets for this n-gram)
- `train_doc_id` (optional when present in the training record)

### Output layout (attempt 3)

Add a new detailed output file (large is fine):
```
<output_path>/
  stats/overlap_details.jsonl.gz   # per (eval_row, train_row, ngram)
```

Keep existing aggregate outputs (`overlap_stats.jsonl`) unchanged.

### Answered requirements (for continuity)

1) Offsets are **character offsets** in the **original text** (not token offsets).
2) Row numbers are **0-based** and refer to the row within each file (parquet/JSONL).
3) Emit **one record per (eval row, train row, ngram)** with **lists of positions**
   for each side.
4) Output can be large; **use a new detailed output file**.
5) Text fields are **configurable**; default to `text`.
