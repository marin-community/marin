# Snowball ladder datasets

`sample_mix.py` builds fixed-size, pre-mixed token caches for the Snowball scaling ladder. The ladder trains on a cluster that cannot stream the datakit stores from CoreWeave, so we build each cache once in `marin-us-east-02a`, copy it to the cluster's filesystem, and train on it as one flat cache.

| Cache | Source store | Weights | Target tokens | Read at |
|---|---|---|---|---|
| `harrier_main` | `store_4d2e363d` (hero) | September mixture (`main` phase of `harrier_mix_2026_08_18.json`) | 2.135T | 8,192 |
| `harrier_cooldown` | `store_4d2e363d` | Hero cooldown phase | 213.5B | 65,536 (or 8,192) |
| `snowball_phase0` | `store_8ac06c74` (Snowball) | mixture-3 phase 0 | 547.8B | 8,192 |
| `snowball_phase1` | `store_8ac06c74` | mixture-3 phase 1 | 137.0B | 8,192 |

The Harrier targets are Snowball's 2.114T pretraining and 211B cooldown tokens plus a 1% margin. The Snowball-mix targets cover the June-prep d1024 reference run (678B, phase 1 from 80%) plus 1%. Every smaller rung reads a subset of the same cache.

Documents are copied as stored: Marin tokenizer, no retokenization, no length cap, EOS kept, so document masking works as usual. A cell too small for its weight repeats documents; the `plan` log names those cells. `harrier_main` and `harrier_cooldown` share no documents except in cells where the two together need more than one copy.

## Build

Run each phase on CoreWeave with `MARIN_PREFIX=s3://marin-us-east-02a/marin`. Set `OUT` per recipe, e.g. `s3://marin-us-east-02a/marin/snowball_ladder/harrier_v1`.

1. Plan. This reads the cell ledgers and writes `plan.json`. The log shows each cache's target, which cells repeat documents, and how many shards there are to read.

```bash
uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-08a --cpu 4 --memory 16GB --job-name snowball-data-plan -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.grug.snowball_ladder.sample_mix --phase plan --recipe harrier --out $OUT
```

2. Sample, as independent tasks so one failure does not stop the rest. Finished items are skipped on rerun, so resubmit any task that dies.

```bash
for k in $(seq 0 127); do uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-08a --cpu 4 --memory 32GB --disk 16GB --max-retries 2 --no-wait --job-name snowball-data-sample-$k -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.grug.snowball_ladder.sample_mix --phase sample --out $OUT --num-tasks 128 --task-index $k; done
```

3. Merge. This writes each cache's `shard_ledger.json` and `.stats.json`, plus `report.json` with expected and realized tokens per cell.

```bash
uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-08a --cpu 4 --memory 16GB --job-name snowball-data-merge -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.grug.snowball_ladder.sample_mix --phase merge --out $OUT
```

4. Validate. This checks document counts, token ids and EOS, and with `--du` it measures bytes on disk, which tells us how much cluster storage the copy needs.

```bash
uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources --target-cluster cw-us-east-08a --cpu 4 --memory 16GB --job-name snowball-data-validate -e MARIN_PREFIX s3://marin-us-east-02a/marin -- python -m experiments.grug.snowball_ladder.sample_mix --phase validate --out $OUT --du
```

Build the `snowball` recipe the same way with `--recipe snowball` and its own `OUT`. It is about a quarter of the size, so it finishes first and can unblock rung 1. `--target CACHE=TOKENS` overrides a target, and `--seed` draws a different sample.

## Copy

Copy each cache directory (`$OUT/harrier_main`, ...) as is. Each is a sharded Levanter cache: a top-level `shard_ledger.json` and `.stats.json` plus `part-*` subdirectories that the ledger names by relative path, so the tree moves as a unit.

## Train

Read each cache as one flat component, with a full permutation shuffle. The cache is stored grouped by cell, so the default block shuffle would sweep one domain at a time, as #10412 found on a similar cache.

```python
DatasetComponent(source=None, cache_dir=f"{LOCAL}/harrier_main", format=TextLmDatasetFormat(), flat_cache=True)
LmDataConfig(tokenizer=marin_tokenizer, components={"main": main, "cooldown": cooldown, **val}, train_weights=..., shuffle=True, auto_build_caches=False)
```

For a Snowball-style cooldown, launch the cooldown from the pre-cooldown checkpoint with weights `{"cooldown": 1.0}`. Alternatively, give one run two stages, `[(0, {"main": 1, "cooldown": 0}), (start, {"main": 0, "cooldown": 1})]`, with `start` on a mixture-block boundary.
