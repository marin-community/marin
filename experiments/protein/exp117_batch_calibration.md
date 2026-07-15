# TPU batch calibration

Measure, per TPU slice, the largest global batch that trains with no gradient accumulation, and check
the `tpu_batch_config` HBM estimator against it. Workload: 1.5B dense transformer, `seq_len 8192`,
bf16 params, Adam. A probe is a **full** smoke run (~20 steps + evals + checkpoints), never a single
step — peak HBM includes eval and checkpoint memory, not just the step-0 allocation.

Code: [`exp117_batch_calibration.py`](./exp117_batch_calibration.py) (analysis); the `SMOKE_BATCH`
path in [`exp117_sweep.py`](./exp117_sweep.py) (the probe). Core library patches this depends on:
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
  `N` is folded into the run id + a tag, so each probe is a distinct W&B run.

## Method

Per slice: exponential gallop over power-of-2 global batches, then bisect the fit→OOM boundary
(~5–8 probes). Result = largest power of 2 that fits. Each probe, and the completion/HBM rules:

- **Fan across every region the slice lives in; first terminal verdict wins.** — tolerates preemption
  and per-region capacity gaps.
- **Verdict from the full job log, not a tail.** Fit = `First train step completed` (step 0 allocated
  the whole batch and ran). OOM = `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom`. The fit line is mid-run
  and scrolls past any tail once eval/checkpoint/W&B output piles up — a tail scan misses it. A
  `Progress on:train` line is **not** a fit; it prints before allocation (treating it as one lets the
  gallop run away to absurd batches).
- **Compile that neither completes step 0 nor OOMs within a deadline → OOM.** Batches far over the
  ceiling stall in compilation instead of raising cleanly.
- **Kill the probe's jobs on OOM.** `CompileTimeHbmOom` is permanent; iris otherwise re-attempts
  preempted/failed jobs indefinitely and burns the slice.
- **Capture the verdict's log line at detection.** — the result stays verifiable without re-reading a
  preempted job's rewritten/truncated logs.
- **Let every non-OOMing run finish** (all regions). A run killed after step 0 records only the
  compile-time allocation and understates peak HBM.
- **Reported max-fit run must be iris `succeeded` + W&B `finished`** — no partial or failed run in the
  numbers. Peak HBM = max `system/tpu.<i>.hbmMemoryUsage` (percent, per chip) over all chips and steps
  of that run.
- **Retry policy** — on a failed job classify: OOM (ceiling moved) / preemption (resubmit; resumes
  from checkpoint) / genuine crash (Python traceback, `preemptions=0` → stop after two, surface). Stops
  a real bug from being hammered as if it were preemption.

Estimator comparison: `tpu_batch_config(tpu, target, batch_bytes(target, oh))` gives predicted
`(pdp, accum)`; the measured ceiling gives actual. Compare at a target batch too large to fit in one
microbatch on any slice.

## Nuances (Marin)

- **Region-local data.** Tokenized caches and checkpoints are region-scoped; cross-region reads are
  disallowed (cost). Raw docs must be staged per region; the cache builds once per region on first
  probe. A family confined to a region without staged docs is unmeasurable (v4).
- **Estimator basis.** `tpu_batch_config` is anchored to global batch 128, so for many-chip slices its
  per-chip prediction saturates at `128/chips` (`pdp = -1`) below the true per-chip ceiling. The
  overhead back-out is only meaningful where the prediction is accumulation-bound.
- **Multi-host slices** (>1 VM host) exercise the sharded checkpoint + HF-export path, which needed a
  levanter fix to run at all (`exp117_core_patches.md`). Their peak HBM includes the multi-host
  checkpoint gather — real and in scope.
- **Run identity = region + slice + batch + smoke-version.** Same-region resubmit resumes from
  checkpoint (preemption tolerance); bump the smoke-version to fork clean W&B runs after a
  recipe/library change.
- **`CompileTimeHbmOom` is deterministic** (compile-time), so the OOM boundary is reproducible; only
  run completion is subject to preemption.

## Results

Per-chip ceiling is constant within a family (v5e 4, v6e 16, v5p 32) and scales with chip count — an
internal consistency check. `pdp·acc` = microbatch × accumulation to reach a global batch of 512.

| slice | chips | GiB/chip | max batch | per-chip | peak HBM % | meas pdp·acc | est pdp·acc | est asks |
|---|---|---|---|---|---|---|---|---|
| v5litepod-4 | 4 | 16 | 16 | 4 | 87.0 | 4·32 | 2·64 | 2× |
| v5litepod-8 | 8 | 16 | 32 | 4 | 72.1 | 4·16 | 2·32 | 2× |
| v5litepod-16 | 16 | 16 | 64 | 4 | 68.6 | 4·8 | 2·16 | 2× |
| v6e-4 | 4 | 32 | 64 | 16 | 100.0 | 16·8 | 4·32 | 4× |
| v6e-8 | 8 | 32 | 128 | 16 | 96.6 | 16·4 | 4·16 | 4× |
| v6e-16 | 16 | 32 | 256 | 16 | 97.8 | 16·2 | 4·8 | 4× |
| v5p-8 | 4 | 95 | 128 | 32 | 63.9 | 32·4 | 16·8 | 2× |
| v5p-16 | 8 | 95 | 256 | 32 | 65.6 | 32·2 | 16·4 | 2× |
| v5p-32 | 16 | 95 | 512 | 32 | 64.0 | full·1 | 16·2 | 2× |

- **max batch** — largest global batch trained with no accumulation (`pdp = -1`); **per-chip** = max
  batch ÷ chips. **full** = whole per-chip batch fits, no accumulation.
- **peak HBM %** — max `hbmMemoryUsage` over all chips and steps of the completed (`succeeded` +
  `finished`) run.
- **meas / est pdp·acc** — per-device parallelism × accumulation to reach global 512; `est` =
  `tpu_batch_config` at overhead 1.0. **est asks** = accum(est) ÷ accum(meas).

The estimator is uniformly conservative — 2× the accumulation on v5e/v5p, 4× on v6e. Overhead that
reproduces the measured ceiling: v5e ≈ 0.5, v6e ≈ 0.25, v5p ≈ 0.5 (only where accumulation-bound; the
128-batch basis saturates for the many-chip slices). Peak HBM explains the ceiling gaps: v6e sits at
97–100%, v5e/v5p at 64–87% because the next power-of-two batch would overshoot 100%.
