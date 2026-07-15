# exp117 Batch Study — max global batch & HBM-estimator calibration

A blueprint for measuring, per TPU slice, the largest global batch that trains with no gradient
accumulation, and for checking the sweep's HBM estimator against that measurement. Companion code:
[`exp117_batch_study.py`](./exp117_batch_study.py) (analysis) and the `SMOKE_BATCH` path in
[`exp117_sweep.py`](./exp117_sweep.py) (the probe).

## Objective

For each slice, find the largest power-of-2 **global** batch that fits with
`per_device_parallelism = -1` (the whole per-chip batch in one microbatch, no accumulation). From
that measured ceiling, derive the `(pdp, grad_accum)` needed to reach some target global batch, and
compare it against the `(pdp, grad_accum)` the estimator (`tpu_batch_config`) predicts. Also record
the peak HBM utilization each slice actually reaches.

## Setup

- **Model / data**: contacts-v1 ≈1.5B (Qwen3), `seq_len = 8192`, contacts-v1 tokenizer — the exp117
  smoke recipe. Each probe is a full smoke run (~20 steps + 2 evals + checkpoints), not a single step.
- **Slices** — the 3 smallest *usable* per family (the very smallest of each is skipped so the floor
  is v5litepod-4 / v6e-4 / v5p-8):
  - v5e: `v5litepod-4`, `v5litepod-8`, `v5litepod-16` (16 GiB/chip)
  - v6e: `v6e-4`, `v6e-8`, `v6e-16` (32 GiB/chip)
  - v5p: `v5p-8`, `v5p-16`, `v5p-32` (95 GiB/chip)
  - **v4 excluded**: its only region (us-central2) has no staged raw docs, so it can't tokenize
    region-local without a forbidden cross-region copy.
- **Regions** (where each family's slice exists): v5e → europe-west4, us-west4; v6e → europe-west4,
  us-east1, us-east5; v5p → us-central1, us-east5.
- **Probe mechanism**: `SMOKE_BATCH=<N>` in `exp117_sweep` sets the exact global batch **and** forces
  `per_device_parallelism = -1`, bypassing the estimator entirely — so the run either fits (trains) or
  OOMs. `N` is folded into the run id (`…-b<N>`) and a `smoke_batch=<N>` tag, so every probe is a
  distinct W&B run.

## Method (the blueprint)

1. **Per-slice search** for the max fitting batch: exponential gallop over power-of-2 global batches
   (start at per-chip 1), then bisect once the fit→OOM boundary is bracketed (~5–8 probes/slice).
   Powers of 2 only; the answer is the largest power of 2 that fits.
2. **Region horse-race**: launch each probe in *every* region the slice lives in; the first to reach
   a verdict decides. Availability-first — replicates don't matter, only that one lands.
3. **Fit/OOM detection from FULL logs** (never tails — the marker scrolls out behind
   eval/checkpoint/wandb output):
   - **FIT** = `First train step completed` — step 0 allocated the whole batch on-device and ran a
     training step, which is sufficient proof the batch fits.
   - **OOM** = `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom` (a *compile-time*, deterministic HBM OOM).
   - **Hang** = reached compile (`starting first train step`) but neither completes step 0 nor OOMs
     within a deadline → treat as OOM (batches far over the ceiling stall in compilation).
4. **Kill OOM'd runs; never let them retry.** A `CompileTimeHbmOom` is permanent, and iris
   re-attempts preempted/failed jobs — so on an OOM verdict, kill the whole probe (all regions) to
   stop it burning the slice.
5. **Let every non-OOMing run train to completion.** Any run that launches and does not OOM runs to
   the end (all regions), so it logs a *real multi-step* HBM/telemetry profile. Do **not** stop a run
   after step 0 — that captures only the compile-time allocation and understates peak HBM.
6. **Capture the proof line at detection time.** Store the actual fit/OOM log line with each verdict,
   so a result is verifiable from stored evidence and never depends on re-reading a preempted job's
   volatile (retried, truncated) logs.
7. **HBM utilization** = the max of `system/tpu.<i>.hbmMemoryUsage` (a percentage, one series **per
   chip**) over **all chips and all training steps**, taken from the **completed** max-fit run.
8. **Estimator comparison**: `tpu_batch_config(tpu, target, batch_bytes(target, overhead))` yields the
   predicted `(pdp, grad_accum)`; the measured ceiling yields the actual `(pdp, grad_accum)`. Compare
   at a target global batch too large to fit in one microbatch on any slice (512).

## Results

### Measured ceiling & the overhead that reproduces it

Per-chip ceiling is **constant within a family** (v5e = 4, v6e = 16, v5p = 32) and scales with chip
count — a strong internal-consistency check.

| slice | HBM/chip | max batch | per-chip | ~overhead that matches |
|---|---|---|---|---|
| v5litepod-4 | 16 GiB | 16 | 4 | 0.5 |
| v5litepod-8 | 16 GiB | 32 | 4 | 0.5 |
| v5litepod-16 | 16 GiB | 64 | 4 | 0.5 |
| v6e-4 | 32 GiB | 64 | 16 | 0.25 |
| v6e-8 | 32 GiB | 128 | 16 | 0.25 |
| v6e-16 | 32 GiB | 256 | 16 | <0.0625 † |
| v5p-8 | 95 GiB | 128 | 32 | 0.5 |
| v5p-16 | 95 GiB | 256 | 32 | <0.0625 † |
| v5p-32 | 95 GiB | 512 | 32 | <0.0625 † |

† The estimator is anchored to global batch 128, so for many-chip slices its per-chip prediction
saturates at `128 / chips` (= pdp −1) below the measured per-chip — no overhead can express it. The
overhead comparison is only clean where the prediction is accumulation-bound (v5e, v6e-4/8, v5p-8):
**v5e ≈ 0.5, v6e ≈ 0.25, v5p ≈ 0.5**. The default overhead of 1.0 is 2–4× conservative.

### Target global batch = 512 — measured vs predicted

| slice | chips | HBMtot | HBM/ch | maxUtil% | maxB | pdp_a | gac_a | pdp_p | gac_p |
|---|---|---|---|---|---|---|---|---|---|
| v5litepod-4 | 4 | 64 | 16 | 80.0* | 16 | 4 | 32 | 2 | 64 |
| v5litepod-8 | 8 | 128 | 16 | 72.1* | 32 | 4 | 16 | 2 | 32 |
| v5litepod-16 | 16 | 256 | 16 | 59.8* | 64 | 4 | 8 | 2 | 16 |
| v6e-4 | 4 | 128 | 32 | 99.6* | 64 | 16 | 8 | 4 | 32 |
| v6e-8 | 8 | 256 | 32 | 94.7* | 128 | 16 | 4 | 4 | 16 |
| v6e-16 | 16 | 512 | 32 | 96.8* | 256 | 16 | 2 | 4 | 8 |
| v5p-8 | 4 | 380 | 95 | 62.6* | 128 | 32 | 4 | 16 | 8 |
| v5p-16 | 8 | 760 | 95 | 64.9* | 256 | 32 | 2 | 16 | 4 |
| v5p-32 | 16 | 1520 | 95 | 63.7* | 512 | −1 | 1 | 16 | 2 |

**\* Provisional HBM** — these came from runs stopped shortly after step 0, so they reflect only the
compile-time allocation and likely *understate* the true peak (which can rise during later steps,
eval, and checkpointing). They must be refreshed from completed runs (method step 5) before use.

**Legend** — `chips`: TPU chips in the slice. `HBMtot` / `HBM/ch`: total and per-chip HBM (GiB;
`HBMtot = chips × HBM/ch`). `maxUtil%`: peak `hbmMemoryUsage` over all chips and steps of the max-fit
run. `maxB`: measured max global batch (pdp −1, no accumulation). `pdp_a`/`gac_a`: per-device
parallelism & grad-accum the **measured** ceiling implies to reach global 512. `pdp_p`/`gac_p`: the
same, **predicted** by `tpu_batch_config` at overhead 1.0. **pdp = −1** means the whole per-chip batch
fits with no accumulation.

**Takeaway**: predicted `pdp` is 2–4× below measured everywhere (predicted `gac` correspondingly too
high) — the overhead-1.0 estimate is that conservative. v6e runs right at the edge (95–99.6% at its
ceiling); v5e/v5p ceilings sit lower (60–80%) only because the next power of two would overshoot 100%.

## Reproducing

1. Confirm raw docs are staged region-local for each target region; the tokenized cache is built once
   per region on first probe.
2. For each slice, run its max-fit batch (from the ceiling table) as a **full** smoke run in every
   region the slice lives in, and let it **finish**:
   ```
   SMOKE=yes TPU=<slice> SMOKE_BATCH=<batch> REGION=<region>  # via the exp117_sweep launch path
   ```
   Kill only runs that OOM; let the rest complete (method steps 4–5).
3. Run the analysis, which reads the completed runs' HBM from W&B and rebuilds both tables:
   ```
   python -m experiments.protein.exp117_batch_study
   ```
   The measured ceilings live in `CEILINGS`; update them there if the search is re-run.
