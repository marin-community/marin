# MoE LM-head init scale: Research Logbook

Experiment ID prefix: `MOE-LMH`

## Scope
- **Goal:** Determine whether scaling the LM head's init standard deviation by
  2× or 4× over the baseline `cfg.initializer_std` improves the QB-MoE recipe
  versus the v16 compute-optimal baseline at d512 / d768 / d1024 / d1280.
- **Primary metric:** `eval/paloma/macro_loss` (final).
- **Secondary metric:** `throughput/tokens_per_second` (last 100 steps avg).
- **Decision metric:** effective speedup ≥ 1 at all four scales (gate 1 + 2),
  plus projected macro loss below baseline at 1e21 and 1e23 from a refit
  `loss(C) = 1.6 + A · C^(-α)`. See `experiments/grug/moe/agent.md`.
- **Constraints:** Only `model.py:Transformer.init` for `output_proj` is
  changed. All other architecture/optimizer/data choices match
  `experiments/grug/moe/launch.py` baseline. seq_len = 4096.

## Baseline
- **Date:** 2026-04-27
- **Code refs:**
  - `experiments/grug/moe/model.py` (Transformer.init, line 470 area)
  - `experiments/grug/moe/heuristic.py` (`build_from_heuristic`, `MoeAdamHHeuristic`)
  - `experiments/grug/moe/launch.py` (`run_grug_moe_trial`, baseline scale)
- **Baseline numbers** (from `README.md`, v16 isoflop sweep optima):

  | Budget   | Dim    | Layers | macro | tok/s   | runtime |
  |----------|--------|--------|-------|---------|---------|
  | 2.19e17  | d512   | 6      | 3.8104 | 405,630 | 0.6h    |
  | 1.70e18  | d768   | 8      | 3.4339 | 273,532 | 2.8h    |
  | 9.00e18  | d1024  | 11     | 3.1605 | 175,165 | 10.5h   |
  | 2.83e19  | d1280  | 13     | 3.0065 | 128,277 | 26.8h   |

  Scaling law: `loss(C) = 1.6 + 95.18 · C^(-0.0941)`.
  Baseline projections: 2.606 @ 1e21, 2.252 @ 1e23.

## Variant
Add a single config knob `lm_head_init_scale: float = 1.0` to `GrugModelConfig`
and apply it as `cfg.initializer_std * cfg.lm_head_init_scale` only in the
`output_proj` initializer (model.py, `Transformer.init`).

Two variants: `lm_head_init_scale = 2.0` and `lm_head_init_scale = 4.0`.
All other heuristic outputs (LR, beta1, beta2, epsilon, batch, steps,
layers, GQA, etc.) are unchanged so each variant is apples-to-apples with
the v16 compute-optimal baseline.

## Experiment Matrix
- 4 compute-optimal scales × 2 init-scale variants = 8 runs total.
- Gate 1 (small-scale screen): d512 + d768 for 2x and 4x → 4 runs.
- Gate 2 (scaling-law confirm): d1024 + d1280 for 2x and 4x → 4 runs.

## Stop criteria
- **Stop after gate 1** if neither variant achieves effective speedup > 1
  at both d512 and d768 — abandon the variant.
- **Promote** if any variant passes both gates and projected macro at 1e21
  and 1e23 is below the baseline.

## Experiment Log

### 2026-04-27 - kickoff
- **Hypothesis:** A larger LM-head init shifts the early-training softmax
  away from a near-uniform distribution faster, reducing time spent in the
  warmup regime. 2x and 4x are mild perturbations; we expect either a small
  speedup (if the recipe is sub-optimal here) or a small degradation
  (if the v16 LR fit was already calibrated against the small init).
- **Branch:** `research/moe-lm-head-init-scale`
- **Code change:** `experiments/grug/moe/model.py` adds `lm_head_init_scale`
  field; `experiments/grug/moe/lm_head_init_sweep.py` is a new sweep
  launch that runs both scales over `_GATE_1_POINTS` (default) or
  `_GATE_2_POINTS` / `both` via env var `LM_HEAD_INIT_SWEEP_GATE`.
- **Submission command (gate 1):**
  ```
  .venv/bin/iris --config lib/iris/examples/marin.yaml job run \
      --no-wait --cpu 1 --memory 2G --extra cpu \
      -e WANDB_API_KEY "$WANDB_API_KEY" \
      -e LM_HEAD_INIT_SWEEP_GATE "1" \
      -- python -m experiments.grug.moe.lm_head_init_sweep
  ```
- **Wandb group:** `lm-head-init-sweep` in project `marin_moe`.
- **Run ID pattern:** `lm-head-init-{2x|4x}-d{dim}-{budget}` e.g.
  `lm-head-init-2x-d512-2.19e17`.
- **Result:** see "gate 1 final" entry below.

### 2026-04-28 02:24 UTC - gate 1 final, both variants FAIL
- **Confidence:** stable (single-seed but consistent direction across two scales).
- **W&B entity/project:** `understanding-sam/marin_moe`
  (note: not `marin-community/marin_moe` — runs land in the personal entity).

| variant | dim | macro_loss | Δ vs baseline | tok/s   | speedup | verdict |
|---------|-----|------------|---------------|---------|---------|---------|
| 2x      | d512 | 3.8165    | +0.0061       | 406,157 | 0.973   | FAIL    |
| 2x      | d768 | 3.4360    | +0.0021       | 274,354 | 0.991   | FAIL    |
| 4x      | d512 | 3.8382    | +0.0278       | 406,696 | 0.878   | FAIL    |
| 4x      | d768 | 3.4314    | **−0.0025**   | 273,949 | 1.016   | (alone) |

- **Interpretation:** scaling the LM-head init causes a small,
  scale-decreasing loss penalty under the v16 LR fit — costly at d512,
  almost nothing at d768. 4x at d768 marginally beats baseline (−0.003
  macro) but 4x at d512 is clearly worse (+0.028), so 4x as a recipe-wide
  multiplier fails Gate 1. 2x is essentially neutral at d768 and slightly
  negative at d512.
- **Decision:** Gate 1 requires effective speedup > 1 at *both* d512 and
  d768; both variants fail. **Gate 2 not submitted.** Closing the
  experiment with a negative result.
- **Repro bundle:**
  - Branch: `research/moe-lm-head-init-scale`
  - Sweep launch: `experiments/grug/moe/lm_head_init_sweep.py`
  - Hardware: 4 × v5p-8 (one per variant × scale)
  - Iris parent job: `/kaiyue/iris-run-job-20260427-225917`

## Conclusion
- **Outcome:** negative. Larger LM-head init (2x or 4x) does not improve
  the v16 compute-optimal MoE recipe at d512–d768; the harm shrinks with
  scale but Gate 1 still fails for both variants.
- **Confidence:** stable (consistent direction across two scales, monotonic
  in scale factor at the smaller scale).
- **Limitation:** single seed; the v16 LR was fit against the smaller
  default init, so the small d768 4x improvement may simply reflect that
  larger inits become tolerable as model size grows. Acting on it would
  require a co-tuned LR + seed replication.

## Followups (optional, ordered)
1. Refit LR for the 2x/4x variants at d512 — cheap, would tell us whether
   the d512 gap is just LR mis-calibration vs. a real architectural cost.
2. If (1) closes the d512 gap, re-run gate 1 with co-tuned LR; otherwise
   stop.

### 2026-04-28 02:28 UTC - reopen + Gate 2 submitted (user override)
- **Why reopened:** user override of the Gate 1 stop rule. Hypothesis is
  that the 4x init's ~zero d512 cost / +Δ d768 trend continues at d1024 /
  d1280, so the variant may be net-positive at the larger compute
  budgets even though Gate 1 failed at d512.
- **Submitted:** `/kaiyue/iris-run-job-20260428-022733` at 02:27:45 UTC
  via `LM_HEAD_INIT_SWEEP_GATE=2`.
- **Children (4 v5p-8):**
  - `lm-head-init-2x-d1024-9.00e+18` (~10.5h baseline)
  - `lm-head-init-4x-d1024-9.00e+18` (~10.5h)
  - `lm-head-init-2x-d1280-2.83e+19` (~26.8h)
  - `lm-head-init-4x-d1280-2.83e+19` (~26.8h)
- **Wall-clock ETA:** d1024 first (~10.5h), d1280 dominates total (~27h).
- **What we'll compute when Gate 2 lands:** effective speedup at all 4
  scales for each variant + refit `loss(C) = 1.6 + A · C^(-α)` on each
  variant's 4 optima and project to 1e21 / 1e23. Compare to baseline
  projections (2.606 @ 1e21, 2.252 @ 1e23).
- **Note:** 2x-d512 (0.973), 4x-d512 (0.878), and 2x-d768 (0.991) already
  disqualify both variants from promotion under `agent.md`'s Gate 2 rule
  (speedup > 1 at *all four* scales). Gate 2 here is informational — it
  characterizes the scaling behavior of the cost rather than testing
  promotability.

### 2026-04-29 - Gate 2 infra recovery
Gate 2 hit two infra issues during execution:

1. **Original Gate 2 parent (`/kaiyue/iris-run-job-20260428-022733`)** ran from
   17:06 UTC for 9.3h, then crashed when Iris briefly reported
   `unschedulable: no groups in region us-central2` mid-run. d1024 / d1280 had
   permanent checkpoints at step 1k–9k.
2. **Resume parent (`/kaiyue/iris-run-job-20260428-170638`)** picked up cleanly —
   d1024 finished, d1280 ran 14h, then both d1280 children received SIGTERM
   from Iris at step 11271/11269 (96–98%) — last permanent checkpoint at
   step 11000, last temp checkpoint at step 11344. Macros at last logged eval:
   2x=3.036, 4x=3.041 (both at step 11000).
3. **Subsequent resubmits failed in 25–30s** with `RuntimeError: N step(s) failed`
   and no informative parent stderr. After patching the launcher to dump the
   full traceback to `gs://marin-us-east5/grug/lm_head_init_sweep/_debug/`, the
   actual exception turned out to be:
   `connectrpc.errors.ConnectError: Job ... is unschedulable: no groups in
   region us-central2; did you mean us-central1?`
   Root cause: Iris's `iris.client.client.submit()` inherits the parent CPU
   worker's region onto the v5p-8 child request when the child has no explicit
   region constraint. Original Gate 2 parents happened to run in us-east5 →
   children went to us-east5 → success. Post-SIGTERM resubmits landed on
   us-central2 CPU → children inherited us-central2 → `no groups`.
4. **Fix 1** (`launch.py` commit `6f0ef28a1`): in `run_grug_moe_trial`, set
   `resources.regions = ("us-east5", "us-central1")` for any TPU resource
   when not already specified. This is a function-body change → does not
   alter `ExecutorStep` hashes → existing checkpoints resume cleanly.
5. **Fix 2** (submission flag): even with Fix 1, the executor's prefix
   `marin_prefix()` is region-dependent (reads GCS metadata of the parent
   worker), so a parent in us-central2 wrote to `gs://marin-us-central2/.../a1cab3/`
   instead of `gs://marin-us-east5/.../a1cab3/`. Same hash, different bucket →
   no resume. Forced via `iris job run --region us-east5
   -e MARIN_PREFIX=gs://marin-us-east5 ...`. Persistent fix would either pin
   `MARIN_PREFIX` per experiment or have the executor probe alternate prefixes
   for cached state.
6. **Successful resubmit (`/kaiyue/iris-run-job-20260429-182115`)** loaded
   checkpoint from `gs://marin-tmp-us-east5/.../checkpoints/step-11344` and
   trained the remaining ~460 steps (×2 variants) cleanly.

### 2026-04-29 13:10 UTC - Gate 2 final
| variant | dim   | budget    | macro   | Δ vs baseline | tok/s    | speedup |
|---------|-------|-----------|---------|---------------|----------|---------|
| 2x      | d512  | 2.19e17   | 3.8165  | +0.006        | 406,157  | 0.973   |
| 2x      | d768  | 1.70e18   | 3.4360  | +0.002        | 274,354  | 0.991   |
| 2x      | d1024 | 9.00e18   | 3.1637  | +0.003        | 177,465  | 0.991   |
| 2x      | d1280 | 2.83e19   | 3.0133  | +0.007        | 128,273  | 0.950   |
| 4x      | d512  | 2.19e17   | 3.8382  | +0.028        | 406,696  | 0.878   |
| 4x      | d768  | 1.70e18   | 3.4314  | **−0.003**    | 273,949  | **1.016** |
| 4x      | d1024 | 9.00e18   | 3.1713  | +0.011        | 176,683  | 0.937   |
| 4x      | d1280 | 2.83e19   | 3.0181  | +0.012        | 128,497  | 0.918   |

**Scaling-law refit** (`loss(C) = 1.6 + A · C^(-α)`, asymptote pinned at L∞=1.6):

| variant   | A     | α      | macro@1e21    | macro@1e23    |
|-----------|-------|--------|---------------|---------------|
| baseline  | 95.18 | 0.0941 | 2.606         | 2.252         |
| 2x        | 90.99 | 0.0930 | 2.612 (+0.006)| 2.260 (+0.008)|
| 4x        | 96.71 | 0.0944 | 2.608 (+0.002)| 2.253 (+0.001)|

## Conclusion
- **Outcome:** negative for promotion. Neither 2x nor 4x achieves effective
  speedup > 1 at all four scales, so neither passes Gate 2.
- **Confidence:** stable across four scales, single-seed.
- **Quantitative note:** 4x's projected scaling-law loss at 1e21/1e23 is
  within +0.002 of baseline — essentially indistinguishable from baseline at
  scale within fit noise. So while 4x doesn't *help*, it doesn't durably
  *hurt* either; the d512 outlier (0.878 speedup) is the binding constraint.
  2x is consistently a small (~1–5% effective compute) tax across all scales.
- **User hypothesis ("4x init may help more with larger scale") not supported.**
  4x at d768 is the only point that beats baseline (Δ=−0.003, speedup 1.016),
  but the trend doesn't hold at d1024/d1280, suggesting d768 is seed noise
  rather than scale benefit.

## Followups (optional, ordered)
1. Refit LR for the 2x/4x variants at d512 — cheap, would tell us whether
   the d512 cost is just LR mis-calibration.
2. If LR co-tune closes d512 gap, redo the full 4-scale sweep with co-tuned
   LR; otherwise stop.
