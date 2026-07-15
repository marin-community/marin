# TPU throughput benchmark (issue #7187)

How fast can the pooled fast-transformer quality classifier score a corpus on a v6e
TPU, how does it scale, and what would a ~10T-token pass cost? The deployed
[`score.py`](../score.py) path runs the forward on CPU under `InlineRunner(cpu=2)` — too
slow to use in practice. This harness measures the TPU path on the real Zephyr-on-Iris
pipeline and compares it to the fasttext baseline it replaced.

## Finding

The v6e forward is essentially free (~1% MXU; ~670 M window-tok/s per v6e-4). Throughput
is **host-bound**, so the pipeline is built around the host, not the chips: a warm
190K-doc shard takes **~10.5s (~18,000 docs/s)** on a v6e-4, split read 2.7s + tokenize/
forward/reduce 4.7s + write 3.1s.

| path | hardware | warm docs/s | note |
|---|---|---|---|
| fast (fork pool + parallel read + stager) | 1× v6e-4 | ~18,000 | `--tok-procs 96 --device-batch 4096` |
| fast, 4 workers | 4× v6e-4 (16 chips) | ~18,000/VM → ~72,000 agg | linear (per-VM independent) |
| fast, GPU | CoreWeave 8× H100 | ~22,700 | `--accelerator H100x8`; faster S3 writes |
| fasttext (baseline) | 32 vCPU | ~6,122 | quality Spearman 0.44 vs 0.69 |

A 10T-token corpus (~11 B docs) is **~2.7 h on 64 v6e-4** (~0.7 h on 256), on preemptible TPU.

The harness is accelerator- and cluster-agnostic: `--accelerator` selects a TPU type
(`v6e-4`) or a `VARIANTxCOUNT` GPU request (`H100x8`), and relative `--corpus` / `--model-dir`
paths root at `marin_prefix()`, so the same command reads the datakit corpus from GCS on the
marin cluster or from S3 on CoreWeave.

## How it's built

The three host costs each get the right tool:

- **Read** — each parquet file's ~11 row groups are read concurrently by a thread pool
  (`id`/`text` columns only, arrow-native; the read + decode release the GIL), so the
  ~400 MB text download runs row-group-parallel instead of as one serial stream.
- **Tokenize** — CPU-heavy Python-and-Rust glue that only scales ~4x across threads (the
  GIL), so it runs in a *fork* process pool (true multi-core); each child packs a block and
  hands it back through `shared_memory`, avoiding a ~750 MB/shard pickle.
- **Forward / stage** — a stager thread copies each packed block out of shared memory and
  ships it to the chips (`device_put`), so the H2D transfer of upcoming blocks overlaps the
  current block's forward + reduce; the forward thread only touches pre-staged device arrays.
  The forward itself is ~free (~0.5s/shard).

## Layout

- `microbench.py` — forward-only device ceiling (pre-tokenized resident batches, async
  launches). Establishes the ~670 M window-tok/s upper bound.
- `build_scorer.py` — build a config-faithful scorer dir (real vocab remap from a corpus
  slice, random weights, percentile calibration). Throughput is weight-independent, so
  this needs no trained checkpoint.
- `fast_stage.py` — the headline pipeline: parallel row-group reads + a fork tokenizer pool
  feeding the device forward through a stager thread.
- `tokenize_worker.py` — fork-pool child: tokenizes a block off the GIL and returns it
  through shared memory (children never touch the TPU).
- `fasttext_stage.py` — the deployed fasttext baseline on the identical doc set.
- `common.py` — shared windowing / tokenize-pack / tokenizer-loading helpers.
- `run_bench.py` — launcher; dispatches to a stage's `run()` (keeps closures importable by
  path for cloudpickle).

## Run

Build the scorer once, then launch a stage as an Iris job (v6e-4 in `europe-west4-a`,
marin cluster). `run_bench.py` is the entry point; `fast` needs an accelerator
(`--enable-extra-resources --extra tpu`, or `--extra gpu` for the GPU path), `fasttext` is a
CPU job. Relative `--corpus` / `--model-dir` / `--out-dir` paths root at `marin_prefix()`;
omit `--corpus` to use the default datakit corpus slice.

```bash
python -m experiments.datakit.cluster.quality.fast_transformer.tpu_bench.run_bench \
    fast --model-dir datakit/quality/ft-tpu-bench --out-dir datakit/quality/ft-tpu-bench-out \
    --accelerator v6e-4 --max-files 24 --max-workers 4 \
    --device-batch 4096 --tok-procs 96 --read-threads 12
```
