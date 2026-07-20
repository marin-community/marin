---
topic: mxfp8-quality-7271
issue: https://github.com/marin-community/marin/issues/7271
description: Fixed-token BF16 versus hybrid MXFP8 quality gate on GB200
author: Matt Wittmann
---

# MXFP8 Quality Gate: Task Logbook

## Current TL;DR

- Two smoke-only defects are fixed: shortened mixtures no longer produce empty finite components, and tagged evaluation now applies the same mixed-precision cast as training.
- The first fully instrumented pair exposed a smoke artifact: setting the trainer horizon to 20 compressed the full run's 314-step LR warmup to zero steps. Shortened runs now reproduce the full optimizer schedule prefix and reject horizons that extend past warmup.
- After correcting the schedule, BF16 remains finite but MXFP8 still becomes NaN on the first backward pass. W&B isolates the first non-finite tensor to `expert_mlp.w_down`; `w_gate` and `w_up` gradients are finite.
- The fused op passes the exact local E16/M262144/D2560/F1280 shape on one GB200 with both unit-normal and loss-scaled random cotangents. Grouped-only and dense-only graph controls are next.
- The primary gate is a matched-token d2560/L26/E128/top-4 run: 31,474 steps, batch 512, sequence length 4096, and 66,005,762,048 tokens per arm.
- The full 1e21-FLOP pair is blocked on the first-backward numerical defect. No quality or performance result is claimed yet.

## Scope

- Goal: determine whether the production hybrid MXFP8 recipe from #7282 preserves training quality while reducing time to a matched loss.
- Primary metrics: paired train-loss delta, final/cooldown-tail loss delta, Paloma macro/domain loss, uncheatable-eval loss, tokens/s, MFU, and time to matched loss.
- Constraints: same seed, data order, schedule, topology, and optimizer; only the FP8 model config may differ. Use region-local `s3://marin-us-east-02a/marin` data and checkpoints.
- Coordinating issue: https://github.com/marin-community/marin/issues/7271

## Baseline

- Date: 2026-07-19
- Code refs: #7282 implementation at `c3cb334f8`; quality launcher at `485e23f7f`.
- Baseline numbers: the corrected #7282 64-GPU production-shape smoke measured 299,894 tok/s BF16 versus 392,287 tok/s MXFP8/XLA, or 1.308x. That was a 50-step throughput result, not a long-run quality result.

## Entry Log

### 2026-07-19 17:23 - MXFP8Q-001 preflight ready

- Hypothesis: the exact d2560 quality cell compiles, trains, evaluates, checkpoints, and resumes on 8xGB200x4 per arm with finite paired losses and a stable MXFP8 throughput advantage.
- Commit Hash: `485e23f7f4d89d1d2f218c777766497d6a8d7ae4`
- Command: submit two CPU coordinators through Iris, each running `python -m experiments.grug.moe.launch_mxfp8_quality` with `MXFP8_QUALITY_PAIR_ID=MXFP8Q-001-smoke`, `MXFP8_QUALITY_STEPS=20`, and arm `bf16` or `mxfp8`. Pass `WANDB_API_KEY` from agenix and `MARIN_PREFIX=s3://marin-us-east-02a/marin` to each coordinator.
- Config: d2560/L26/E128/top-4, f1280, shared-f2560, seq4096, batch512, seed0; 8 replicas x 4 GB200, mesh replica2/data2/expert8; full MuonH/Newton-Schulz; ring MoE, scanned layers, recompute-all; MXFP8 uses fused grouped experts, per-tensor FP8 dense GEMMs, BF16 EP wire, and the XLA producer. Paloma and uncheatable evaluation run at the forced final callback. Checkpoints are hourly plus forced final on S3.
- Result: pending submission.
- Interpretation: the launcher passed 16 focused tests and a final read-only review with no findings. Four CPU/`recipe="auto"` variant-contract failures are inherited unchanged from #7282.
- Next action: submit and babysit both 20-step arms; require finite loss, final evaluation, S3 checkpoint, and exact-ID resume before the full 31,474-step pair.

### 2026-07-19 18:02 - MXFP8Q-001 smoke stopped before training

- Hypothesis: shortening `experiment_budget` to 20 steps while retaining a 10.37T-token `target_budget` truncates at least one rare finite mixture component to zero sequences.
- Commit Hash: fix pending commit; failed launch used `c0be963da` with launcher snapshot `485e23f7f`.
- Command: paired Iris coordinators `/mwittmann/mxfp8q-001-smoke-bf16-coord` and `/mwittmann/mxfp8q-001-smoke-mxfp8-coord`; each launched an 8xGB200x4 child gang.
- Config: both arms used the same 20-step shape-smoke data config with simulated epoching enabled.
- Result: both arms failed before step 0 and before W&B initialization with `MixtureDataset in RESTART_STRATEGY encountered an empty finite dataset (async_len() returned 0)`. The coordinators and child gangs were stopped while retrying.
- Interpretation: `LmDataConfig` slices every component to `int(true_length * experiment_budget / target_budget)`. At 20 steps the ratio is about `4.04e-6`, so sufficiently rare components become empty; restart sampling rejects empty finite datasets. This is shared smoke infrastructure, not a BF16/MXFP8 quality signal.
- Next action: keep simulated epoching for the full 31,474-step comparison, disable it for shortened shape smokes as in the existing GB200 scale-smoke pattern, add regression coverage for both modes, and relaunch the exact pair.

### 2026-07-19 17:58 - MXFP8Q-001 training cleared; forced evaluation failed

- Hypothesis: Grug's tagged evaluator passes FP32 parameter leaves directly to the BF16-only FA4 attention kernel, unlike the training loss path which casts ordinary leaves to compute dtype while preserving FP8 operator state.
- Commit Hash: failed launch used `a36b48c20`.
- Command: corrected `MXFP8Q-001-smoke` pair with `GIT_COMMIT=a36b48c20`; W&B runs [BF16](https://wandb.ai/emcwitt/marin_moe/runs/MXFP8Q-001-smoke-bf16-s20) and [MXFP8](https://wandb.ai/emcwitt/marin_moe/runs/MXFP8Q-001-smoke-mxfp8-s20).
- Config: same 20-step shape smoke as above, with shortened-run simulated epoching disabled.
- Result: the previous empty-dataset failure was cleared. BF16 completed 20/20 train steps with final loss 9.2135 and final-step throughput 799,086 tok/s. MXFP8 logged two finite steps, ending at loss 11.8043; its forced evaluation then began. Both arms failed evaluation with `gpu_fa4_cute_attention currently supports only bf16/fp16, got float32`. Neither wrote the forced final checkpoint. The W&B API key defaulted to the submitter entity because the tracker entity was not explicit.
- Interpretation: the failure is isolated to the final evaluation path, which omitted the mixed-precision cast. The MXFP8 W&B background queue did not flush beyond step 1 after the exception, so this smoke is not a valid throughput comparison.
- Next action: reuse the training-path model cast in tagged evaluation, preserving `OverwriteWithGradient` FP8 scale/amax state in FP32; pin W&B to `marin-community`; validate locally; relaunch both arms under the same IDs.

### 2026-07-19 18:20 - MXFP8Q-001b instrumented smoke invalidated by compressed warmup

- Hypothesis: the treatment's step-2 NaN is caused by shortening the trainer horizon from 31,474 to 20, which also shortens the fractional LR warmup and decay schedule.
- Commit Hash: failed launch used `f70187cd0`.
- Command: fresh paired Iris coordinators `/mwittmann/mxfp8q-001b-smoke-{bf16,mxfp8}-coord`; W&B runs [BF16](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-001b-smoke-bf16-s20) and [MXFP8](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-001b-smoke-mxfp8-s20).
- Config: d2560/L26/E128/top-4, batch512, seq4096, 8xGB200x4 per arm, full MuonH/NS, and the corrected evaluation/W&B configuration. The smoke still passed `steps=20` to the optimizer builder.
- Result: BF16 completed 20 steps, saved `step-20`, and finished with train loss 9.21343, 795,203 tok/s, Paloma macro loss 9.49029, and uncheatable macro loss 9.38663. MXFP8 logged finite loss 11.80426 at global step 1, became NaN on the next update on all eight ranks, forced NaN evaluation, and saved a poisoned `step-2` checkpoint. Iris marked both jobs succeeded because the training loop breaks rather than raises on NaN. W&B correctly recorded both runs in `marin-community/marin_moe`.
- Interpretation: the MXFP8 run's first logged LR was 0.0064745. The full 31,474-step schedule has a 314-step warmup and would use about 2.16e-5 at the same point, roughly 300x smaller. The corrected #7282 production-shape run remained finite for 50 steps with the same MXFP8 recipe and full Newton-Schulz, which further points to the smoke schedule rather than an immediate operator defect.
- Next action: for shortened runs, scale the smoke peak and use an all-warmup schedule so every LR value matches the full schedule prefix; reject shortened horizons beyond the full warmup; rerun with fresh identities.

### 2026-07-19 18:36 - MXFP8Q-001c reproduces NaN at the correct full-run LR

- Hypothesis: preserving the full run's first 20 LR values will remove the treatment's step-2 NaN.
- Commit Hash: `47b74d4ce`.
- Command: paired coordinators `/mwittmann/mxfp8q-001c-smoke-{bf16,mxfp8}-coord`; W&B runs [BF16](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-001c-smoke-bf16-s20) and [MXFP8](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-001c-smoke-mxfp8-s20).
- Config: same full topology and treatment as 001b. The 20-step optimizer uses an all-warmup schedule with scaled endpoints, matching the full schedule prefix to float32 rounding; first treatment LR is 2.16477e-5.
- Result: BF16 completed 20/20 with train loss 11.53079, eval loss 11.50498, Paloma macro loss 11.53831, uncheatable macro loss 11.52123, 797,926 tok/s, and a step-20 checkpoint. MXFP8 again stopped on step 2 and produced all-NaN evaluation plus a poisoned step-2 checkpoint. Its step-0 forward loss was finite at 11.80426, but `grad/norm/total` was already NaN. Per-parameter norms isolate the non-finite value to `expert_mlp.w_down`; `w_gate`=0.02608 and `w_up`=0.02594 were finite. Issue update: https://github.com/marin-community/marin/issues/7271#issuecomment-5018066223.
- Interpretation: schedule compression was a real harness defect but not the numerical root cause. The first backward pass, before MuonH applies an update, is invalid specifically in the fused w2 weight-gradient path.
- Next action: reproduce the local grouped-op shape on one GB200, then split grouped MXFP8 from dense per-tensor FP8 in the full graph.

### 2026-07-19 18:45 - Exact local op shape is finite with synthetic cotangents

- Hypothesis: local expert count 16 and dispatch capacity 262,144 trigger a static layout or fused-kernel defect absent from the original E64 op test.
- Commit Hash: `47b74d4ce`.
- Command: direct one-GB200 jobs `/mwittmann/mxfp8q-op-e16-m262k-uniform` and `/mwittmann/mxfp8q-op-e16-smallcot`, using the existing blackbox op harness with E=16, M=262144, D=2560, F=1280, uniform groups, and XLA producers. The second scales the output cotangent by `1 / (512 * 4096)`.
- Result: both jobs passed. Unit-normal cotangents measured relative Frobenius errors output=0.06555, dx=0.06725, dw13=0.06743, dw2=0.06706 versus the BF16 reference; loss-scaled cotangents produced the same finite errors.
- Interpretation: the static local shape, wgrad layout, and simple cotangent underflow are falsified. The defect requires the training graph or its cotangent distribution.
- Next action: run two-step grouped-only and dense-only controls on the exact quality graph. If grouped-only fails, instrument the grouped custom VJP inputs; if only the hybrid fails, trace the dense FP8 upstream cotangent.
