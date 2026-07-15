# TPU throughput benchmark (issue #7187)

How fast can the pooled fast-transformer quality classifier score a corpus on a v6e
TPU, how does it scale, and what would a ~10T-token pass cost? The deployed
[`score.py`](../score.py) path runs the forward on CPU under `InlineRunner(cpu=2)` — too
slow to use in practice. This harness measures the TPU path on the real Zephyr-on-Iris
pipeline and compares it to the fasttext baseline it replaced.

## Finding

The v6e forward is essentially free (~1% MXU; ~670 M window-tok/s per v6e-4). Throughput
is **host-bound**: first by tokenization, then — once that is parallelized across the
host with a fork process pool — by the **GCS parquet data path** (reading/decoding the
input, writing the output), which is ~65% of a warm shard while the chips sit ~97% idle.

| path | hardware | warm docs/s | note |
|---|---|---|---|
| fast, fork pool | 1× v6e-4 | ~8,800 | `--tok-procs 48` |
| fast, fork pool ×4 | 4× v6e-4 (16 chips) | 8,345/VM → 33,380 agg | ~94% scaling |
| fast, CPU-only | 32 vCPU | ~1,608 | forward-bound (the "before") |
| fasttext (baseline) | 32 vCPU | ~6,122 | quality Spearman 0.44 vs 0.69 |

A 10T-token corpus (~11 B docs) is ~6 h on 64 v6e-4 (~1.4 h on 256).

## Layout

- `microbench.py` — forward-only device ceiling (pre-tokenized resident batches, async
  launches). Establishes the ~670 M window-tok/s upper bound.
- `build_scorer.py` — build a config-faithful scorer dir (real vocab remap from a corpus
  slice, random weights, percentile calibration). Throughput is weight-independent, so
  this needs no trained checkpoint.
- `fast_stage.py` — the headline pipeline: a per-worker fork tokenizer pool feeds the
  device forward; `--cpu-only` runs the same path with no TPU (the CPU baseline).
- `tokenize_worker.py` — jax-free child module for the fork pool (children never touch
  the TPU).
- `fasttext_stage.py` — the deployed fasttext baseline on the identical doc set.
- `common.py` — shared windowing / tokenize-pack / timing helpers.
- `run_bench.py` — launcher; dispatches to a stage's `run()` (keeps closures importable
  by path for cloudpickle, and forces `TOKENIZERS_PARALLELISM=true`).

## Run

Build the scorer once, then launch a stage as an Iris job (v6e-4 in `europe-west4-a`,
marin cluster). `run_bench.py` is the entry point; `fast` needs a TPU
(`--enable-extra-resources --extra tpu`), `fasttext` and `--cpu-only` are CPU jobs.

```bash
python -m experiments.datakit.cluster.quality.fast_transformer.tpu_bench.run_bench \
    fast --corpus 'gs://.../outputs/main/*.parquet' \
    --model-dir gs://.../ft-tpu-bench --out-dir gs://.../out \
    --max-files 24 --max-workers 4 --device-batch 4096 --tok-procs 48
```
