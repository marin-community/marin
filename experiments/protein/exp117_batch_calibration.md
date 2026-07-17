# TPU batch calibration

This records the completed calibration. The temporary `SMOKE_BATCH` probe used to collect these
measurements has been removed from `exp117_sweep.py`.

`tpu_batch_config` predicts, for a slice and target global batch, the per-device microbatch (`pdp`)
and gradient accumulation that fit HBM. Its one free parameter is `correction_factor` — a scalar on the
byte estimate. **Goal: calibrate that single knob** — a value, tuned against direct measurement on a
small set of slices, that lets the heuristic predict accumulation for any slice.

Ground truth is the measured per-chip microbatch ceiling: the largest global batch that runs to
completion with no accumulation, ÷ chips. What's calibrated is the peak HBM of a *whole* run — the
training forward/backward **plus the evaluation and checkpoint passes**, each of which allocates its own
memory — not training alone. A run stopped at the first training step would miss that memory and
overstate the ceiling, so every measured run must complete. Measure on the smaller slices, tune
`correction_factor` to reproduce the ceiling, and confirm it generalizes to larger slices in the family.

## Parameters

- **Model** — 1.5B dense transformer, `seq_len 8192`, bf16 params, Adam.
- **Slices** — 3 per family, smallest useful first (drop the very smallest so the floor is a 4-chip slice):
  - v5e (16 GiB/chip): `v5litepod-4/8/16`
  - v6e (32 GiB/chip): `v6e-4/8/16`
  - v5p (95 GiB/chip): `v5p-8/16/32`
  - v4 excluded — its only region has no region-local raw docs, so it can't tokenize without a
    (disallowed) cross-region copy.
- **Regions** — v5e: europe-west4, us-west4 · v6e: europe-west4, us-east1, us-east5 · v5p: us-central1, us-east5.
- **Probe** — `SMOKE_BATCH=<N>` set the exact global batch and forced `per_device_parallelism = -1`
  (whole per-chip batch, no accumulation), bypassing the estimator: the run fits (trains) or OOMs.
  `N` was folded into the run id and a tag, so each probe was a distinct W&B run.

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
  The batch must hold through the eval and checkpoint passes, which allocate beyond the training step —
  a run killed after step 0 misses that memory, overstates the ceiling, and gives an untrustworthy peak
  HBM.
- **Retry policy** — on a failed job classify: OOM (ceiling moved) / preemption (resubmit; resumes from
  checkpoint) / genuine crash (Python traceback, `preemptions=0` → stop after two, surface).

## Calibrating the knob

`batch_memory_bytes = ⌈(params + Adam state + activations) × correction_factor⌉`; `tpu_batch_config`
returns the largest microbatch whose scaled estimate fits capacity. Higher correction factor → smaller
predicted microbatch → more accumulation.

- Evaluate the prediction at a **target ≥ 512** so `pdp` is not capped by a small batch basis —
  `tpu_batch_config` caps `pdp` at `batch/chips`, so a small basis saturates the per-chip prediction at
  `128/chips` for many-chip slices and masks the knob. The per-chip prediction is stable for any target
  in that range.
- Per family, take the correction-factor interval that reproduces the measured per-chip ceiling
  (per-chip is constant within a family, so the family's smallest slice suffices — the calibrate-on-small
  step).
- **Recommended single value = the smallest correction factor that never over-predicts on any slice**
  (predicted per-chip ≤ measured, so the heuristic never under-accumulates into an OOM), which also
  minimizes wasted accumulation subject to that safety constraint.

## Results

Measured ceilings (ground truth). Per-chip is constant within a family and scales with chip count — an
internal consistency check. **correction range** = the `correction_factor` interval (per family) that
makes `tpu_batch_config` reproduce that per-chip ceiling. Peak HBM is the max `hbmMemoryUsage` over all
chips and steps of the completed run.

| slice | chips | GiB/chip | max batch | per-chip | correction range | peak HBM % |
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

The ranges are disjoint — v5e needs ≥0.40, v6e ≤0.39 — so no single value fits all three. Two ways to
use this: a **single safe value = 0.40** (smallest that never over-predicts: exact on v5e/v5p, 2×
conservative on v6e), or **per-family factors**, which the sweep ships — `CORRECTION_FACTORS = {v5e:
0.5, v6e: 0.3, v5p: 0.45}`, each inside its family's range with margin. The uncorrected estimate
(`correction_factor = 1.0`) is far too conservative.

Accumulation to reach a global batch of 512 — measured vs. the estimator uncorrected and at the single
safe factor:

| slice | chips | per-chip | accum (measured) | accum @1.0 (uncorrected) | accum @0.40 (calibrated) |
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

Correcting `1.0 → 0.40` halves over-accumulation everywhere: exact on v5e and v5p, and v6e drops from
4× to 2× the necessary accumulation. Full accuracy on v6e would need a correction factor ≤0.39, which
over-predicts (OOMs) v5e — the limit of a single scalar, hence the per-family factors above.
