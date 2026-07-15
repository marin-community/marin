# TPU batch calibration

`tpu_batch_config` predicts, for a slice and target global batch, the per-device microbatch (`pdp`)
and gradient accumulation that fit HBM. Its one free parameter is `overhead_factor` — a scalar on the
byte estimate. **Goal: calibrate that single knob** — one value, tuned against direct measurement on a
small set of slices, that lets the heuristic predict accumulation for any slice.

Ground truth is the measured per-chip microbatch ceiling: the largest global batch that trains with no
accumulation, ÷ chips. Measure it on the smaller slices, tune `overhead_factor` so the estimator
reproduces it, and confirm it generalizes to larger slices in the same family.

Workload: 1.5B dense transformer, `seq_len 8192`, bf16 params, Adam. Code:
[`exp117_batch_calibration.py`](./exp117_batch_calibration.py) (analysis); the `SMOKE_BATCH` path in
[`exp117_sweep.py`](./exp117_sweep.py) (the probe). Core library patches this depends on:
[`exp117_core_patches.md`](./exp117_core_patches.md).

## Parameters

- **Slices** — 3 per family, smallest useful first (drop the very smallest so the floor is a 4-chip slice):
  - v5e (16 GiB/chip): `v5litepod-4/8/16`
  - v6e (32 GiB/chip): `v6e-4/8/16`
  - v5p (95 GiB/chip): `v5p-8/16/32`
  - v4 excluded — its only region has no region-local raw docs (see Nuances).
- **Regions** — v5e: europe-west4, us-west4 · v6e: europe-west4, us-east1, us-east5 · v5p: us-central1, us-east5.
- **Probe** — `SMOKE_BATCH=<N>` sets the exact global batch and forces `per_device_parallelism = -1`
  (whole per-chip batch, no accumulation), bypassing the estimator: the run fits (trains) or OOMs.
  `N` folds into the run id + a tag, so each probe is a distinct W&B run.

## Measuring the ceiling

Per slice, search for the largest power-of-2 global batch that trains with `pdp = -1`. `max batch ÷
chips` = the per-chip ceiling — the largest microbatch the chip holds; accumulation for any target
follows from it. A probe is a **full** smoke run (~20 steps + evals + checkpoints), never a single
step. Search = exponential gallop over powers of 2, then bisect the fit→OOM boundary (~5–8 probes).
Each probe:

- **Fan across every region the slice lives in; first terminal verdict wins.** — tolerates preemption
  and per-region capacity gaps.
- **Verdict from the full job log, not a tail.** Fit = `First train step completed` (step 0 allocated
  the whole batch and ran). OOM = `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom`. The fit line is mid-run
  and scrolls past any tail once eval/checkpoint/W&B output piles up — a tail scan misses it. A
  `Progress on:train` line is **not** a fit; it prints before allocation (treating it as one lets the
  gallop run away to absurd batches).
- **Compile that neither completes step 0 nor OOMs within a deadline → OOM.** Far-over-ceiling batches
  stall in compilation instead of raising cleanly.
- **Kill the probe's jobs on OOM.** `CompileTimeHbmOom` is permanent; iris otherwise re-attempts
  preempted/failed jobs indefinitely and burns the slice.
- **Capture the verdict's log line at detection.** — verifiable without re-reading a preempted job's
  rewritten logs.
- **Let every non-OOMing run finish; the reported max-fit run is iris `succeeded` + W&B `finished`.**
  A run killed after step 0 records only the compile-time allocation; only a completed run gives a
  trustworthy peak HBM (corroboration below).
- **Retry policy** — on a failed job classify: OOM (ceiling moved) / preemption (resubmit; resumes from
  checkpoint) / genuine crash (Python traceback, `preemptions=0` → stop after two, surface).

## Calibrating the knob

`batch_memory_bytes = ⌈(params + Adam state + activations) × overhead⌉`; `tpu_batch_config` returns the
largest microbatch whose scaled estimate fits capacity. Higher overhead → smaller predicted microbatch
→ more accumulation.

- Evaluate the prediction at a **target ≥ 512** so `pdp` is not capped by a small batch basis — below
  that it saturates at `128/chips` for many-chip slices (see Nuances). The per-chip prediction is
  stable for any target in that range.
- Per family, take the overhead interval that reproduces the measured per-chip ceiling (per-chip is
  constant within a family, so the family's smallest slice suffices — this is the calibrate-on-small
  step).
- **Recommended single value = the smallest overhead that never over-predicts on any slice** (predicted
  per-chip ≤ measured, so the heuristic never under-accumulates into an OOM), which also minimizes
  wasted accumulation subject to that safety constraint.

## Nuances (Marin)

- **Estimator basis.** `tpu_batch_config` caps `pdp` at `batch/chips`, so calling it with a small basis
  (the sweep's default 128) saturates the per-chip prediction at `128/chips` for many-chip slices,
  masking the knob. Calibrate and predict against a target ≥ 512.
- **One scalar, multi-slice.** The knob lumps replicated params/optimizer and batch-scaled activations
  into a single factor. HBM-to-activation ratio differs per family, so one overhead can be exact for at
  most a subset — the residual is cross-family, not cross-size.
- **Region-local data.** Tokenized caches and checkpoints are region-scoped; cross-region reads are
  disallowed (cost). Raw docs are staged per region; the cache builds once per region on first probe.
  A family confined to a region without staged docs is unmeasurable (v4).
- **Multi-host slices** (>1 VM host) exercise the sharded checkpoint + HF-export path, which needed a
  levanter fix to run at all (`exp117_core_patches.md`). Peak HBM there includes the multi-host
  checkpoint gather.
- **Run identity = region + slice + batch + smoke-version.** Same-region resubmit resumes from
  checkpoint; bump the smoke-version to fork clean W&B runs after a recipe/library change.
- **`CompileTimeHbmOom` is deterministic** (compile-time) — the OOM boundary is reproducible; only run
  completion is subject to preemption.

## Results

Measured ceilings (ground truth). Per-chip is constant within a family and scales with chip count — an
internal consistency check. **overhead range** = the `overhead_factor` interval (per family) that makes
`tpu_batch_config` reproduce that per-chip ceiling. Peak HBM is the max `hbmMemoryUsage` over all chips
and steps of the completed run.

| slice | chips | GiB/chip | max batch | per-chip | overhead range | peak HBM % |
|---|---|---|---|---|---|---|
| v5litepod-4 | 4 | 16 | 16 | 4 | 0.40 – 0.78 | 87.0 |
| v5litepod-8 | 8 | 16 | 32 | 4 | 0.40 – 0.78 | 72.1 |
| v5litepod-16 | 16 | 16 | 64 | 4 | 0.40 – 0.78 | 68.6 |
| v6e-4 | 4 | 32 | 64 | 16 | 0.20 – 0.39 | 100.0 |
| v6e-8 | 8 | 32 | 128 | 16 | 0.20 – 0.39 | 96.6 |
| v6e-16 | 16 | 32 | 256 | 16 | 0.20 – 0.39 | 97.8 |
| v5p-8 | 4 | 95 | 128 | 32 | 0.30 – 0.58 | 63.9 |
| v5p-16 | 8 | 95 | 256 | 32 | 0.30 – 0.58 | 65.6 |
| v5p-32 | 16 | 95 | 512 | 32 | 0.30 – 0.58 | 64.0 |

The ranges are disjoint — v5e needs ≥0.40, v6e ≤0.39 — so no single value fits all three.
**Recommended single overhead = 0.40** (smallest that never over-predicts): exact on v5e and v5p, 2×
conservative on v6e, safe everywhere. The shipped default `overhead_factor = 1.0` is far too
conservative.

Accumulation to reach a global batch of 512 — measured vs. the estimator at the default and calibrated
overhead:

| slice | chips | per-chip | accum (measured) | accum @1.0 (default) | accum @0.40 (calibrated) |
|---|---|---|---|---|---|
| v5litepod-4 | 4 | 4 | 32 | 64 | 32 |
| v5litepod-8 | 8 | 4 | 16 | 32 | 16 |
| v5litepod-16 | 16 | 4 | 8 | 16 | 8 |
| v6e-4 | 4 | 16 | 8 | 32 | 16 |
| v6e-8 | 8 | 16 | 4 | 16 | 8 |
| v6e-16 | 16 | 16 | 2 | 8 | 4 |
| v5p-8 | 4 | 32 | 4 | 8 | 4 |
| v5p-16 | 8 | 32 | 2 | 4 | 2 |
| v5p-32 | 16 | 32 | 1 | 2 | 1 |

Calibrating `1.0 → 0.40` halves over-accumulation everywhere: exact on v5e and v5p, and v6e drops from
4× to 2× the necessary accumulation. Full accuracy on v6e would need overhead ≤0.39, which over-predicts
(OOMs) v5e — the limit of a single scalar.
