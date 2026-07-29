---
topic: d12-remat-fit
description: Compile-only AOT memory probe for all_but_moe at the production EP64 shape
author: mwittmann
---

# D-12 remat fit: Task Logbook

## Scope

- Goal: Measure the compiled temporary arena and largest optimized-HLO buffers for
  `recompute_all` and `all_but_moe` at d5120, 48 layers, EP64, batch 1024, and
  sequence length 4096.
- Primary metrics: `memory_analysis().temp_size_in_bytes` and the largest distinct
  tensor shapes in the optimized HLO.
- Constraints: Compile only; no training steps. Use the homogeneous layer scan
  (`SCALE_SCAN_LAYERS=1`) and the exact production per-device token count of
  65,536. Do not modify shared infrastructure.

## Baseline

- Date: 2026-07-29.
- Starting commit: `6ce4a7e6874e10a6013949000dbb0f7e0a92bcf2`.
- Required source commits: scan support
  `97b53fe0e83a52d87a7b0b8b8815cfda9b945937`; slim residuals and
  `all_but_moe` `01b8e7c92f22faf486bf66a4d3e4b6d1aa7f0236`; axis-type-aware
  residual weight storage `868d9d7e4e20ba1a119450aba7f691d7d1dd91b3`.
- Recovered probe:
  `/home/marin/.claude/jobs/6fc274da/tmp/remat_oom_probe.py`, SHA-256
  `a8a1613ad0399b95a393744e6ccb71bf83ed774a26e8ce4850c1ac8cdf2a96fc`.
  This is the uncommitted compile-only probe used for B200MFU-013 jobs
  3264/3265.
- Prior result: at d2560, 26 layers, eight-way sharding, and 262,144
  tokens/device, `recompute_all` used 36.89 GiB of temporary arena and
  `all_but_moe` later used 134.99 GiB after the slim-residual fix.

## Entry Log

### 2026-07-29 10:19 PDT - D12-001 preregistration

- Hypothesis: At 65,536 tokens/device and EP64, `all_but_moe` will fit. I predict
  about 80 GiB of temporary arena, with a 60-110 GiB plausible range.
  `recompute_all` should remain near 40 GiB. The two irreducible
  `gu`/`out_dispatch` pins should contribute about 30 GiB together, following
  the recorded 0.25 token factor and 48/26 layer factor.
- Falsification threshold: The fit prediction is falsified if the compiled
  `all_but_moe` temporary arena exceeds the approximately 186 GiB of usable
  HBM reported for this GB200 stack. A result above 150 GiB will count as a
  technical fit with poor operational margin, not a safe adoption result.
- Commit Hash: The runnable composition has not been created yet. No existing
  branch contains both `97b53fe0e` and `01b8e7c92`; this is a recorded
  prerequisite, not a probe workaround.
- Command: Not submitted.
- Config: `SCALE_HIDDEN_DIM=5120`, `SCALE_NUM_LAYERS=48`,
  `SCALE_INTERMEDIATE=1280`, `SCALE_NUM_EXPERTS=256`, `SCALE_TOP_K=8`,
  `SCALE_EXPERT_AXIS=64`, `SCALE_SEQ_LEN=4096`, `SCALE_BATCH=1024`,
  `SCALE_SCAN_LAYERS=1`; arms `SCALE_REMAT=recompute_all` and
  `SCALE_REMAT=all_but_moe`.
- Result: Pending.
- Interpretation: The 30 GiB desk estimate covers the two named irreducible
  pins, not the whole arena. The 80 GiB prediction allows roughly 50 GiB for
  the hidden/MLP inputs and other compiled temporaries.
- Next action: Compose the two required branch features, mechanically
  parameterize the recovered probe without changing its lowered train-step
  measurement, and CPU-smoke its imports before submission.

### 2026-07-29 10:28 PDT - D12-002 provenance gate blocked

- Hypothesis: Unchanged from D12-001; no result was inspected.
- Commit Hash: No runnable experiment commit was created.
- Command: No Iris command was submitted.
- Config: The requested production shape remains unchanged.
- Result: Blocked before cluster submission. The #7489 head
  (`2949be3bb`) has the candidate split and homogeneous model scan, but its
  slim `sonic_cute` VJP is a local implementation and is rejected when the
  expert mesh axis is 64. The validated EP port (`59e5fe25f`) implements the
  candidate inside the standalone benchmark, not the requested
  `_make_train_step` lineage, and predates the JAX 0.11 production stack.
  The scan/JAX 0.11 verification branch (`761c03d34`) has neither
  `all_but_moe` nor slim residuals.
- Interpretation: A cherry-pick of `01b8e7c92` onto `761c03d34` conflicts
  across the production `Block` API. Resolving it requires a new candidate
  port that preserves SConv, hoisted expert gathers, MTP, and drop telemetry,
  plus value/full-gradient parity. Selecting either conflict side would
  compile a different graph and violate the D-12 provenance requirement.
- Next action: Port and parity-test the candidate on the scan/JAX 0.11
  production branch, then run the two compile-only arms from one commit.
