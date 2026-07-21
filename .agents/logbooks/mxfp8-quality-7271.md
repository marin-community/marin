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
- The fused op passes the exact local E16/M262144/D2560/F1280 shape on one GB200 with unit-normal, loss-scaled, and all-zero synthetic cotangents.
- Grouped-only and dense-only graph controls both complete with finite gradients and evaluation; only the unguarded production hybrid fails. A finite reduction plus conditional BF16 `w_down`-gradient recompute clears the exact failure and is now part of the treatment implementation.
- The primary gate is a matched-token d2560/L26/E128/top-4 run: 31,474 steps, batch 512, sequence length 4096, and 66,005,762,048 tokens per arm.
- The promoted implementation passes a fresh paired 20-step smoke: MXFP8 is finite, tracks BF16 within 0.0038 loss, and is 1.0666x faster on mean throughput.
- Both 31,474-step arms finished successfully with 8/8 Iris tasks exiting 0, zero failures, zero preemptions, finished W&B runs, and logged final `step-31474` checkpoints. At identical terminal tokens, MXFP8 minus BF16 train/tail-100/eval/Paloma/uncheatable losses are +0.00263/+0.00311/+0.00130/+0.00289/+0.00429. Aggregate eval favors BF16 at all 32 paired evaluations, and all three held-out metrics favor BF16 at the final 26 consecutive gates. MXFP8 therefore has a small but persistent quality cost and does not reach BF16's exact final held-out targets within the fixed schedule.
- MXFP8 mean non-compile throughput is 845,920 tok/s versus 788,954 tok/s, a 1.0722x gain. Its measured post-first-sample runtime is 84,179.6s versus 89,640.2s, saving 5,460.6s (91.0 minutes, 6.09%). The speed gain outweighs the quality cost against earlier equal-wall-time BF16 checkpoints, but strict time-to-BF16-final-loss is censored because MXFP8 never reaches that final target.
- Final tables, the held-out trajectory chart, matched-target results, and reconciled CSVs are published in the [W&B comparison run](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-007-full-comparison). An in-cluster audit directly read both final checkpoint metadata objects and verified `step=31474` with `is_temporary=false`.

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

### 2026-07-19 19:03 - MXFP8Q-002 isolation controls show a hybrid-only failure

- Hypothesis: either the grouped MXFP8 expert op or dense per-tensor FP8 independently produces the first-backward NaN.
- Commit Hash: `e468da927`.
- Command: two-step full-graph controls [grouped-only](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-002-diag-mxfp8-grouped-only-s2) and [dense-only](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-002-diag-fp8-dense-only-s2), each on the exact 8xGB200x4 quality topology.
- Result: both controls completed with finite train/eval/checkpoint results. Grouped-only step-0 total grad norm=0.56530 and expert w_down/w_gate/w_up norms=0.03060/0.02003/0.02024; final train loss=11.80210 and eval loss=11.78640. Dense-only total grad norm=0.36285 and expert norms=0.02596/0.02608/0.02594; final train loss=11.80208 and eval loss=11.78690. A separate exact-shape all-zero-cotangent op probe returned finite, exact-zero dx/dw13/dw2.
- Interpretation: neither component fails alone. The NaN requires the hybrid computation graph and is not explained by an entirely zero grouped-op cotangent. The known-finite #7282 d5120 hybrid also has zero step-0 attention/shared weight gradients but finite expert w_down=0.04234, so those zero gradients are expected initialization behavior rather than the direct cause.
- Next action: enable diagnostic-only custom-VJP telemetry on non-finite dw2 to record the real hybrid cotangent, column-quantized values/scales, hidden values/scales, and routing range.

### 2026-07-19 19:34 - MXFP8Q-003 telemetry masks the hybrid NaN

- Hypothesis: conditional finite-value telemetry can observe the invalid fused `w_down` gradient without changing the failing computation.
- Commit Hash: telemetry run and matched control used `08470bfe0`; minimal barrier probe is `11ca6cb63`.
- Command: two-step full-hybrid runs [telemetry](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-003-debug-mxfp8-debug-s2) and [non-instrumented control](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-003-control-mxfp8-s2) on the exact 8xGB200x4 quality topology.
- Result: the telemetry run completed finite with train loss=11.80211, eval loss=11.78691, total grad norm=0.36288, and expert `w_down` grad norm=0.02595; its non-finite callback never fired. The matched non-instrumented control reproduced NaN in total and `w_down` gradient norms on its first backward pass. The only model-config difference is the debug flag, which adds a finite reduction and conditional callback consumer after the fused `dw2` custom call.
- Interpretation: observing `dw2` changes the failure, so the telemetry cannot expose the original invalid inputs. The result is consistent with output liveness, aliasing, or scheduling sensitivity in the hybrid compiled graph rather than invalid synthetic-op numerics. A full weight-gradient finite reduction is too expensive to adopt as the quality treatment.
- Next action: test `jax.lax.optimization_barrier` on `dw2` as the smallest zero-arithmetic compiler intervention. If it clears the matched control, confirm at the original 20-step horizon and compare throughput before making it the production treatment.

### 2026-07-19 19:44 - MXFP8Q-004 compiler barrier is insufficient

- Hypothesis: preventing compiler motion across the fused `dw2` output is sufficient to reproduce the stabilizing effect of the diagnostic consumer without doing arithmetic.
- Commit Hash: `11ca6cb63`.
- Command: two-step full-hybrid [barrier run](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-004-barrier-mxfp8-barrier-s2), exact topology and shortened data realization from MXFP8Q-003.
- Result: the run reproduced NaN in total and expert `w_down` gradient norms on the first backward pass. A pure `jax.lax.optimization_barrier(dw2)` does not stabilize the graph.
- Interpretation: compiler motion/fusion prevention alone is insufficient; the successful telemetry graph's finite reduction and conditional side-effect introduce a stronger data/control dependency. A correctness guard must consume the fused gradient and provide a numerically valid alternative rather than relying on a scheduling hint.
- Next action: reduce `isfinite(dw2)` and conditionally recompute only an invalid `w_down` gradient from the saved BF16 preactivation/cotangent via `ragged_dot_general`; validate the fallback helper on CPU, then run the exact two-step hybrid control.

### 2026-07-19 19:55 - MXFP8Q-005 finite guard clears the exact failure

- Hypothesis: consuming the fused `dw2` with a finite reduction and conditionally recomputing an invalid result from BF16 preactivations/cotangents will make the hybrid backward robust without replacing its normal MXFP8 path.
- Commit Hash: diagnostic arm used `abae0fd03`; promotion into the treatment is pending commit.
- Command: two-step full-hybrid [finite-guard run](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-005-guard-mxfp8-finite-guard-s2), exact topology and shortened data realization from MXFP8Q-003/004.
- Result: the first backward is finite: total grad norm=0.36288 and expert `w_down`=0.02595. The run completed with train loss=11.80212, eval loss=11.78692, and Paloma macro loss=11.78972. Its child and coordinator both succeeded. CPU coverage verifies that a non-finite fused result selects a finite BF16 grouped-wgrad reference.
- Interpretation: the guard supplies the stronger data dependency that the pure barrier lacked and guarantees a valid fallback if the fused result is actually non-finite. The normal forward, dgrad, `w13` wgrad, and finite `w2` wgrad remain MXFP8.
- Next action: remove the temporary debug/barrier/diagnostic-arm surface, make the guard intrinsic to `MxFp8MoeMlpOp`, and rerun the original 20-step treatment to validate stability and measure throughput overhead before launching the full pair.

### 2026-07-19 20:11 - MXFP8Q-006 paired smoke passes

- Hypothesis: the promoted treatment remains finite for 20 steps, preserves the BF16 loss trajectory, completes forced evaluation/checkpointing, and retains a throughput advantage after the finite guard.
- Commit Hash: `f8be94f87`.
- Command: paired [BF16](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-006-smoke-bf16-s20) and [MXFP8](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-006-smoke-mxfp8-s20) runs on 8xGB200x4 per arm, exact d2560/L26/E128/top-4 quality topology and full-run LR prefix.
- Result: both child gangs and coordinators succeeded after 20 finite steps and forced evaluation. Final BF16/MXFP8 train loss=11.53080/11.53457 (delta +0.00377); eval loss=11.50498/11.50874 (+0.00375); Paloma macro=11.53829/11.54153 (+0.00323); uncheatable macro=11.52126/11.52456 (+0.00330). Across 18 non-compile throughput samples, BF16 averaged 795,878 tok/s and MXFP8 averaged 848,859 tok/s: 1.0666x, or +6.66%.
- Interpretation: the first-backward defect is cleared at the original smoke horizon, quality tracks closely in the warmup prefix, and the guarded treatment remains faster. This is a launch gate, not the issue's compute-optimal quality conclusion.
- Next action: launch and babysit the matched 31,474-step, 66.006B-token pair; evaluate same-token quality and wall-time-matched loss from the complete histories.

### 2026-07-19 20:15 - MXFP8Q-007 full pair launched

- Hypothesis: over the compute-optimal 66.006B-token schedule, guarded hybrid MXFP8 remains quality-neutral at matched tokens and reaches the BF16 loss frontier sooner in wall time.
- Commit Hash: `d11d6ac54` (treatment implementation `f8be94f87`).
- Command: coordinators `/mwittmann/mxfp8q-007-full-{bf16,mxfp8}-coord`, each launching 8xGB200x4 non-preemptible replicas with pair ID `MXFP8Q-007-full`; default 31,474-step horizon.
- Config: d2560/L26/E128/top-4/f1280/shared-f2560, seq4096, batch512, seed0; identical data order, optimizer, schedule, ring/scan/remat topology, Paloma and uncheatable eval; BF16 versus hybrid grouped MXFP8 + dense per-tensor FP8. Hourly and forced-final checkpoints use region-local S3.
- Result: both 32-GPU child gangs allocated and passed the first-gradient gate. BF16/MXFP8 total grad norm=0.56620/0.36288 and expert `w_down`=0.03072/0.02595. At step 7, loss=11.76569/11.76785 and instantaneous throughput=805,660/857,373 tok/s. W&B: [BF16](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-007-full-bf16-s31474), [MXFP8](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-007-full-mxfp8-s31474).
- Next action: babysit progress, hourly checkpoints, periodic evaluation, and terminal state without changing the cluster.

### 2026-07-19 21:10 - MXFP8Q-007 step-1,000 gate passes

- Hypothesis: the first scheduled same-token evaluation will remain within the smoke-scale quality delta while preserving a measurable throughput advantage.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: both arms completed 1,000 finite steps and the scheduled Paloma/uncheatable evaluation. BF16/MXFP8 train loss at exactly step 1,000 is 2.341675/2.342448 (delta +0.000773); the trailing 100-step mean delta is +0.001807. Eval loss is 3.060175/3.060911 (+0.000736), Paloma macro is 3.406820/3.407281 (+0.000461), and uncheatable macro is 2.853132/2.852828 (-0.000304). No non-finite train-loss or total-gradient samples were found.
- Performance: through step 1,000, 100 non-compile throughput samples average 789,096 tok/s BF16 versus 835,929 tok/s MXFP8, or 1.0594x. W&B runtime from the first train sample to step 1,000 is 2,830.4s versus 2,638.7s; at the BF16 gate time, MXFP8 had reached step 1,070.
- Interpretation: the treatment is quality-neutral at the first early gate and retains a 5.9% throughput advantage. This gate covers only 3.2% of the 31,474-step schedule and does not establish the final cooldown-tail or matched-loss conclusion.
- Health: both Iris child gangs and coordinators remain running. An error-level log scan found no training or infrastructure errors; only known JAX bootstrap, partitioning, and evaluation data-loader warnings appeared. The first hourly checkpoints completed successfully at BF16 step 1,206 and MXFP8 step 1,264, providing verified region-local restart points.
- Next action: continue babysitting hourly checkpoints, scheduled evaluations, and both runs through step 31,474 before drawing the issue conclusion.

### 2026-07-19 21:55 - MXFP8Q-007 step-2,000 gate passes

- Hypothesis: the early quality delta will remain small after the warmup while the accumulated wall-time advantage grows.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: both arms completed the second scheduled evaluation and resumed training. BF16/MXFP8 train loss at exactly step 2,000 is 2.115574/2.117753 (delta +0.002179); the trailing 100-step mean delta is +0.002089. Eval loss is 2.858761/2.859699 (+0.000938), Paloma macro is 3.197837/3.198338 (+0.000500), and uncheatable macro is 2.630839/2.633275 (+0.002435).
- Performance: through step 2,000, mean non-compile throughput is 789,435 tok/s BF16 versus 847,661 tok/s MXFP8, or 1.0738x. W&B runtime from the first train sample to step 2,000 is 5,693.1s versus 5,330.8s; at the BF16 gate time, MXFP8 had reached step 2,138.
- Interpretation: at 6.4% of the schedule, the treatment continues to track BF16 closely and has accumulated 138 extra steps at matched elapsed training time. The uncheatable delta is larger than at step 1,000 but remains small and requires later-gate confirmation.
- Health: both child gangs and coordinators remain running with finite current loss and gradient norms. Second hourly checkpoints completed successfully at BF16 step 2,477 and MXFP8 step 2,617, so each arm has two verified recovery generations.
- Next action: continue monitoring checkpoints and scheduled evaluations; publish the next issue milestone after materially more of the schedule has completed or if the quality/health conclusion changes.

### 2026-07-19 22:41 - MXFP8Q-007 step-3,000 gate passes

- Hypothesis: the treatment will continue to track BF16 through 10% of the schedule while preserving its accumulated step advantage.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 3,000 is 2.058030/2.059608 (delta +0.001578); the trailing 100-step mean delta is +0.002096. Eval loss is 2.781238/2.783703 (+0.002465), Paloma macro is 3.119459/3.121742 (+0.002284), and uncheatable macro is 2.545758/2.545686 (-0.000072).
- Performance: cumulative mean non-compile throughput is 789,281 tok/s BF16 versus 846,478 tok/s MXFP8, or 1.0725x. At the BF16 arm's step-3,000 W&B elapsed time, MXFP8 had reached step 3,183.
- Interpretation: at 9.5% of the schedule, all three held-out metrics and the short-window train trajectory remain close. The step-2,000 uncheatable gap did not persist at this gate, supporting continued execution without intervention.
- Health: both child gangs and coordinators remain running after the evaluation with finite current loss and gradient norms; second hourly recovery checkpoints remain available.
- Next action: continue scheduled evaluations and checkpoints, with the next coordinating-issue update reserved for a materially later milestone or a changed conclusion.

### 2026-07-19 23:30 - MXFP8Q-007 step-4,000 gate passes

- Hypothesis: held-out quality and the smoothed train trajectory will remain close as the treatment accumulates a larger wall-time lead.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 4,000 is 2.040592/2.046827 (delta +0.006234), while the trailing 100-step mean delta is only +0.002216. Eval loss is 2.736732/2.739722 (+0.002990), Paloma macro is 3.071379/3.069717 (-0.001662), and uncheatable macro is 2.495353/2.499286 (+0.003933).
- Performance: cumulative mean non-compile throughput is 789,038 tok/s BF16 versus 845,893 tok/s MXFP8, or 1.0721x. At the BF16 arm's step-4,000 W&B elapsed time, MXFP8 had reached step 4,266.
- Interpretation: at 12.7% of the schedule, held-out quality remains close and the higher single-step train delta is not supported by the trailing-window mean. Paloma slightly favors MXFP8 at this gate while uncheatable slightly favors BF16; neither is yet a persistent cross-gate trend.
- Health: both child gangs and coordinators remain running after evaluation. Third hourly checkpoints completed successfully at BF16 step 3,745 and MXFP8 step 3,967, giving each arm three verified recovery generations.
- Next action: continue scheduled evaluations and checkpoints; use step 5,000 as the next coordinating-issue milestone unless health or the quality conclusion changes sooner.

### 2026-07-20 00:17 - MXFP8Q-007 step-5,000 gate passes

- Hypothesis: the small, mixed-sign held-out deltas seen through step 4,000 will remain non-directional at the issue's next published milestone while MXFP8 preserves its wall-time advantage.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 5,000 is 2.007334/2.009798 (delta +0.002463); the trailing 100-step mean delta is +0.002335. Eval loss is 2.704660/2.706634 (+0.001974), Paloma macro is 3.037091/3.035517 (-0.001575), and uncheatable macro is 2.462716/2.461128 (-0.001588).
- Performance: cumulative mean non-compile throughput is 789,123 tok/s BF16 versus 845,897 tok/s MXFP8, or 1.0719x. W&B runtime from the first train sample to step 5,000 is 14,214.9s versus 13,341.5s; at the BF16 arm's step-5,000 elapsed time, MXFP8 had reached step 5,329, a 329-step lead.
- Interpretation: at 15.9% of the schedule, the smoothed train gap remains a few thousandths and the held-out metrics are again mixed: BF16 is slightly better on aggregate eval while MXFP8 is slightly better on Paloma and uncheatable. There is still no persistent held-out quality winner across gates, so the experiment remains healthy but inconclusive until the cooldown tail and final evaluation.
- Health: both child gangs and coordinators remain running and resumed training after evaluation. The fourth hourly checkpoints completed successfully at BF16 step 5,011 and MXFP8 step 5,313; no training or infrastructure error was present in the checkpoint-window logs.
- Next action: publish this planned issue milestone, then continue babysitting both arms through all 31,474 steps, including scheduled evaluation, hourly checkpoints, and terminal verification.

### 2026-07-20 01:04 - MXFP8Q-007 step-6,000 gate passes

- Hypothesis: the held-out differences will remain small and non-directional near 20% of the schedule while the treatment maintains its accumulated wall-time lead.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 6,000 is 1.873455/1.874413 (delta +0.000958); the trailing 100-step mean delta is +0.002136. Eval loss is 2.677226/2.677540 (+0.000314), Paloma macro is 3.009296/3.007381 (-0.001916), and uncheatable macro is 2.431415/2.432432 (+0.001017).
- Performance: cumulative mean non-compile throughput is 788,993 tok/s BF16 versus 845,892 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 6,000 is 17,049.2s versus 16,013.3s; at the BF16 arm's step-6,000 elapsed time, MXFP8 had reached step 6,386, a 386-step lead.
- Interpretation: at 19.1% of the schedule, the smoothed train gap remains about two thousandths and the three held-out deltas remain small with mixed signs. This extends the quality-neutral early trajectory while the treatment's roughly 7.2% throughput advantage persists, but does not establish the cooldown-tail or final matched-loss conclusion.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. The fourth hourly checkpoints at BF16 step 5,011 and MXFP8 step 5,313 remain the latest verified recovery points.
- Next action: continue babysitting scheduled evaluations and checkpoints; reserve the next coordinating-issue update for step 10,000 unless health or the quality conclusion changes sooner.

### 2026-07-20 01:18 - MXFP8Q-007 fifth hourly checkpoints pass

- Result: both arms completed their fifth hourly checkpoint write to region-local S3. BF16 saved step 6,284 and MXFP8 saved step 6,664; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-7,000 evaluations.

### 2026-07-20 01:52 - MXFP8Q-007 step-7,000 gate passes

- Hypothesis: the treatment will keep its smoothed train-loss delta near two thousandths while held-out differences continue to vary around a small magnitude.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 7,000 is 1.938809/1.940908 (delta +0.002098); the trailing 100-step mean delta is +0.002196. Eval loss is 2.654647/2.657123 (+0.002476), Paloma macro is 2.984921/2.986131 (+0.001209), and uncheatable macro is 2.405274/2.410152 (+0.004878).
- Performance: cumulative mean non-compile throughput is 789,018 tok/s BF16 versus 846,029 tok/s MXFP8, or 1.0723x. W&B runtime from the first train sample to step 7,000 is 19,891.9s versus 18,691.2s; at the BF16 arm's step-7,000 elapsed time, MXFP8 had reached step 7,451, a 451-step lead.
- Interpretation: at 22.2% of the schedule, all three held-out metrics favor BF16 at this gate, with uncheatable showing the largest gap so far. The gaps remain only a few thousandths and the smoothed train delta is unchanged from prior gates, but the aligned held-out sign is a new signal to track rather than dismiss. Later gates and the cooldown tail are required before deciding whether it persists.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Fifth hourly recovery checkpoints are available at BF16 step 6,284 and MXFP8 step 6,664.
- Next action: continue scheduled evaluations and checkpoints; escalate the issue update before step 10,000 if the aligned held-out regression persists or grows at the next gates.

### 2026-07-20 02:17 - MXFP8Q-007 sixth hourly checkpoints pass

- Result: both arms completed their sixth hourly checkpoint write to region-local S3. BF16 saved step 7,557 and MXFP8 saved step 8,006; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-8,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 02:39 - MXFP8Q-007 step-8,000 gate shows a second aligned held-out gap

- Hypothesis: the step-7,000 aligned held-out differences will either revert toward mixed signs, as earlier gates did, or persist while the smoothed train-loss delta remains near two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 8,000 is 1.852993/1.855775 (delta +0.002782); the trailing 100-step mean delta is +0.002009. Eval loss is 2.634026/2.637180 (+0.003154), Paloma macro is 2.961504/2.963996 (+0.002492), and uncheatable macro is 2.389326/2.390670 (+0.001344).
- Performance: cumulative mean non-compile throughput is 789,085 tok/s BF16 versus 846,082 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 8,000 is 22,728.9s versus 21,363.3s; at the BF16 arm's step-8,000 elapsed time, MXFP8 had reached step 8,514, a 514-step lead.
- Interpretation: at 25.4% of the schedule, all three held-out metrics favor BF16 for a second consecutive gate. The aggregate and Paloma gaps are slightly larger than at step 7,000 while the uncheatable gap is smaller, and the smoothed train delta remains stable near two thousandths. This is evidence of a small persistent early quality difference, not yet a final regression verdict; the cooldown tail and complete paired trajectory remain decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Sixth hourly recovery checkpoints are available at BF16 step 7,557 and MXFP8 step 8,006.
- Next action: publish the changed quality signal to the coordinating issue, then continue babysitting every scheduled evaluation and checkpoint through step 31,474.

### 2026-07-20 03:17 - MXFP8Q-007 seventh hourly checkpoints pass

- Result: both arms completed their seventh hourly checkpoint write to region-local S3. BF16 saved step 8,830 and MXFP8 saved step 9,354; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-9,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 03:29 - MXFP8Q-007 step-9,000 gate extends the aligned held-out gap

- Hypothesis: the small aligned held-out gap seen at steps 7,000 and 8,000 will remain bounded or revert while the smoothed train trajectory stays close.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 9,000 is 1.921814/1.923987 (delta +0.002174); the trailing 100-step mean delta is +0.002420. Eval loss is 2.617103/2.618999 (+0.001896), Paloma macro is 2.939692/2.944503 (+0.004811), and uncheatable macro is 2.370592/2.373052 (+0.002460).
- Performance: cumulative mean non-compile throughput is 789,050 tok/s BF16 versus 846,020 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 9,000 is 25,683.8s versus 24,042.1s; at the BF16 arm's step-9,000 elapsed time, MXFP8 had reached step 9,616, a 616-step lead.
- Interpretation: at 28.6% of the schedule, all three held-out metrics favor BF16 for a third consecutive gate. Aggregate eval narrows relative to step 8,000, while Paloma and uncheatable widen; the smoothed train delta remains a stable few thousandths. This extends the evidence for a small persistent early quality difference without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Seventh hourly recovery checkpoints are available at BF16 step 8,830 and MXFP8 step 9,354.
- Next action: continue to the planned step-10,000 issue milestone and all subsequent scheduled evaluations and checkpoints through step 31,474.

### 2026-07-20 04:16 - MXFP8Q-007 step-10,000 milestone extends the aligned held-out gap

- Hypothesis: the persistent held-out gap will remain small through the planned issue milestone while the smoothed train trajectory and throughput advantage remain stable.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 10,000 is 1.840105/1.840291 (delta +0.000186); the trailing 100-step mean delta is +0.001841. Eval loss is 2.599134/2.602409 (+0.003275), Paloma macro is 2.920813/2.925022 (+0.004209), and uncheatable macro is 2.348996/2.355901 (+0.006905).
- Performance: cumulative mean non-compile throughput is 789,017 tok/s BF16 versus 845,971 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 10,000 is 28,519.6s versus 26,718.4s; at the BF16 arm's step-10,000 elapsed time, MXFP8 had reached step 10,678, a 678-step lead.
- Interpretation: at 31.8% of the schedule, all three held-out metrics favor BF16 for a fourth consecutive gate. The single-step train losses are nearly identical and the smoothed delta remains below two thousandths, but uncheatable widens to the largest held-out gap observed so far. This strengthens the evidence for a small persistent early quality cost without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation.
- Next action: publish the planned coordinating-issue milestone, then continue all scheduled evaluations and checkpoints through step 31,474.

### 2026-07-20 04:17 - MXFP8Q-007 eighth hourly checkpoints pass

- Result: both arms completed their eighth hourly checkpoint write to region-local S3. BF16 saved step 10,051 and MXFP8 saved step 10,704; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry, scheduled evaluations, and hourly recovery points.

### 2026-07-20 05:05 - MXFP8Q-007 step-11,000 gate preserves the aligned held-out gap

- Hypothesis: after the step-10,000 uncheatable widening, the next gate will show whether the largest held-out gap persists while other metrics remain small.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 11,000 is 1.929385/1.931899 (delta +0.002514); the trailing 100-step mean delta is +0.002267. Eval loss is 2.583909/2.586761 (+0.002853), Paloma macro is 2.909213/2.910640 (+0.001427), and uncheatable macro is 2.330275/2.337360 (+0.007085).
- Performance: cumulative mean non-compile throughput is 788,985 tok/s BF16 versus 846,017 tok/s MXFP8, or 1.0723x. W&B runtime from the first train sample to step 11,000 is 31,380.1s versus 29,396.7s; at the BF16 arm's step-11,000 elapsed time, MXFP8 had reached step 11,749, a 749-step lead.
- Interpretation: at 34.9% of the schedule, all three held-out metrics favor BF16 for a fifth consecutive gate. Paloma narrows relative to step 10,000, while uncheatable remains near seven thousandths and slightly exceeds its prior largest gap. The smoothed train delta remains a few thousandths, so this continues the small persistent early quality-cost signal without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Eighth hourly recovery checkpoints are available at BF16 step 10,051 and MXFP8 step 10,704.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 05:17 - MXFP8Q-007 ninth hourly checkpoints pass

- Result: both arms completed their ninth hourly checkpoint write to region-local S3. BF16 saved step 11,314 and MXFP8 saved step 12,045; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-12,000 evaluations.

### 2026-07-20 05:52 - MXFP8Q-007 step-12,000 gate preserves the aligned held-out gap

- Hypothesis: the held-out gap will remain small but aligned, with the largest domain aggregate varying between Paloma and uncheatable rather than growing monotonically.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 12,000 is 1.885655/1.888101 (delta +0.002445); the trailing 100-step mean delta is +0.001963. Eval loss is 2.568757/2.572387 (+0.003629), Paloma macro is 2.888081/2.893800 (+0.005719), and uncheatable macro is 2.318237/2.321133 (+0.002895).
- Performance: cumulative mean non-compile throughput is 789,019 tok/s BF16 versus 845,962 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 12,000 is 34,231.8s versus 32,072.1s; at the BF16 arm's step-12,000 elapsed time, MXFP8 had reached step 12,814, an 814-step lead.
- Interpretation: at 38.1% of the schedule, all three held-out metrics favor BF16 for a sixth consecutive gate. Uncheatable narrows relative to step 11,000 while Paloma widens, and the smoothed train delta remains near two thousandths. The evidence continues to support a small persistent early quality cost without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Ninth hourly recovery checkpoints are available at BF16 step 11,314 and MXFP8 step 12,045.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 06:18 - MXFP8Q-007 tenth hourly checkpoints pass

- Result: both arms completed their tenth hourly checkpoint write to region-local S3. BF16 saved step 12,585 and MXFP8 saved step 13,396; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-13,000 evaluations.

### 2026-07-20 06:40 - MXFP8Q-007 step-13,000 gate preserves the aligned held-out gap

- Hypothesis: the alternating Paloma/uncheatable widening will remain bounded while aggregate eval and smoothed train loss continue to differ by only a few thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 13,000 is 1.831820/1.834646 (delta +0.002825); the trailing 100-step mean delta is +0.002182. Eval loss is 2.555382/2.557729 (+0.002347), Paloma macro is 2.876756/2.878444 (+0.001688), and uncheatable macro is 2.300642/2.306751 (+0.006108).
- Performance: cumulative mean non-compile throughput is 789,033 tok/s BF16 versus 846,025 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 13,000 is 37,071.3s versus 34,745.5s; at the BF16 arm's step-13,000 elapsed time, MXFP8 had reached step 13,878, an 878-step lead.
- Interpretation: at 41.3% of the schedule, all three held-out metrics favor BF16 for a seventh consecutive gate. Paloma narrows relative to step 12,000 while uncheatable widens again, and the smoothed train delta remains near two thousandths. The evidence continues to support a small persistent early quality cost without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Tenth hourly recovery checkpoints are available at BF16 step 12,585 and MXFP8 step 13,396.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 07:18 - MXFP8Q-007 eleventh hourly checkpoints pass

- Result: both arms completed their eleventh hourly checkpoint write to region-local S3. BF16 saved step 13,855 and MXFP8 saved step 14,747; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-14,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 07:25 - MXFP8Q-007 step-14,000 gate preserves a bounded aligned gap

- Hypothesis: after the step-13,000 uncheatable widening, the held-out differences will remain aligned but trade magnitude across aggregates while the smoothed train delta remains near two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 14,000 is 1.803200/1.805586 (delta +0.002385); the trailing 100-step mean delta is +0.002071. Eval loss is 2.541102/2.543746 (+0.002644), Paloma macro is 2.860485/2.863400 (+0.002914), and uncheatable macro is 2.289658/2.292884 (+0.003225).
- Performance: cumulative mean non-compile throughput is 788,979 tok/s BF16 versus 846,052 tok/s MXFP8, or 1.0723x. W&B runtime from the first train sample to step 14,000 is 39,918.3s versus 37,421.0s; at the BF16 arm's step-14,000 elapsed time, MXFP8 had reached step 14,942, a 942-step lead.
- Interpretation: at 44.5% of the schedule, all three held-out metrics favor BF16 for an eighth consecutive gate. Uncheatable narrows relative to step 13,000 while aggregate eval and Paloma remain within three thousandths, and the smoothed train delta remains near two thousandths. The quality difference remains small, aligned, and bounded so far; the final cooldown tail is still required for the experiment conclusion.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Eleventh hourly recovery checkpoints are available at BF16 step 13,855 and MXFP8 step 14,747.
- Next action: continue to the step-15,000 coordinating-issue milestone and all subsequent scheduled evaluations and checkpoints through step 31,474.

### 2026-07-20 08:13 - MXFP8Q-007 step-15,000 milestone preserves the bounded quality signal

- Hypothesis: at the next issue milestone, the aligned held-out gap will remain on the order of a few thousandths while the throughput ratio remains stable.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 15,000 is 1.786857/1.788465 (delta +0.001607); the trailing 100-step mean delta is +0.002122. Eval loss is 2.522805/2.525446 (+0.002641), Paloma macro is 2.840516/2.844012 (+0.003496), and uncheatable macro is 2.271998/2.276894 (+0.004896).
- Performance: cumulative mean non-compile throughput is 788,928 tok/s BF16 versus 846,088 tok/s MXFP8, or 1.0725x. W&B runtime from the first train sample to step 15,000 is 42,755.6s versus 40,093.0s; at the BF16 arm's step-15,000 elapsed time, MXFP8 had reached step 15,991, a 991-step lead.
- Interpretation: at 47.7% of the schedule, all three held-out metrics favor BF16 for a ninth consecutive gate. Aggregate eval is effectively unchanged from step 14,000, Paloma and uncheatable widen modestly, and the smoothed train delta remains near two thousandths. The evidence continues to support a small persistent quality cost alongside a stable throughput gain, but the final cooldown-tail result remains decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Eleventh hourly recovery checkpoints are available at BF16 step 13,855 and MXFP8 step 14,747.
- Next action: publish the planned issue milestone, then continue every scheduled evaluation and checkpoint through step 31,474.

### 2026-07-20 08:18 - MXFP8Q-007 twelfth hourly checkpoints pass

- Result: both arms completed their twelfth hourly checkpoint write to region-local S3. BF16 saved step 15,117 and MXFP8 saved step 16,088; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-16,000 evaluations.

### 2026-07-20 09:00 - MXFP8Q-007 step-16,000 gate narrows the aligned held-out gap

- Hypothesis: after the modest step-15,000 Paloma and uncheatable widening, the aligned held-out gap will remain bounded rather than grow monotonically.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 16,000 is 1.826170/1.827567 (delta +0.001397); the trailing 100-step mean delta is +0.002021. Eval loss is 2.513951/2.514678 (+0.000727), Paloma macro is 2.834051/2.835609 (+0.001558), and uncheatable macro is 2.260833/2.264016 (+0.003183).
- Performance: cumulative mean non-compile throughput is 788,927 tok/s BF16 versus 846,047 tok/s MXFP8, or 1.0724x. W&B runtime from the first train sample to step 16,000 is 45,609.0s versus 42,765.9s; at the BF16 arm's step-16,000 elapsed time, MXFP8 had reached step 17,050, a 1,050-step lead.
- Interpretation: at 50.8% of the schedule, all three held-out metrics favor BF16 for a tenth consecutive gate, but aggregate eval and Paloma narrow to their smallest aligned differences since step 7,000. The smoothed train delta remains near two thousandths and uncheatable also narrows from step 15,000. This supports a small persistent but non-monotonic quality cost; the cooldown tail remains necessary for the final verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Twelfth hourly recovery checkpoints are available at BF16 step 15,117 and MXFP8 step 16,088.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 09:18 - MXFP8Q-007 thirteenth hourly checkpoints pass

- Result: both arms completed their thirteenth hourly checkpoint write to region-local S3. BF16 saved step 16,384 and MXFP8 saved step 17,435; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-17,000 evaluations.

### 2026-07-20 09:21 - Local Iris tunnel recovered without run intervention

- Signal: the resident status loop began receiving connection-refused errors from `127.0.0.1:10080`; the original port-forward session terminated with `lost connection to pod`.
- Root cause: the local `kubectl port-forward` died. The Iris controller pod remained 1/1 ready with zero restarts, and the controller service remained present; this was not a run or cluster failure.
- Result: replaced only the local port-forward and updated both monitoring state files to session 55036. The existing status loop reconnected on its next retry and confirmed both child gangs and coordinators still running.
- Next action: continue normal monitoring; do not restart or resubmit either arm.

### 2026-07-20 09:48 - MXFP8Q-007 step-17,000 gate widens within the prior range

- Hypothesis: the very small aggregate and Paloma differences at step 16,000 will rebound without exceeding the earlier bounded range, while the smoothed train gap remains a few thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 17,000 is 1.749151/1.752017 (delta +0.002866); the trailing 100-step mean delta is +0.003036. Eval loss is 2.499588/2.503356 (+0.003768), Paloma macro is 2.814274/2.816684 (+0.002411), and uncheatable macro is 2.244242/2.250246 (+0.006003).
- Performance: cumulative mean non-compile throughput is 788,917 tok/s BF16 versus 845,967 tok/s MXFP8, or 1.0723x. W&B runtime from the first train sample to step 17,000 is 48,457.3s versus 45,448.1s; at the BF16 arm's step-17,000 elapsed time, MXFP8 had reached step 18,105, a 1,105-step lead.
- Interpretation: at 54.0% of the schedule, all three held-out metrics favor BF16 for an eleventh consecutive gate. The metrics rebound from the unusually narrow step-16,000 gate, but remain within the earlier observed range; the smoothed train difference increases to three thousandths. The evidence continues to support a small persistent, non-monotonic quality cost without establishing the final cooldown-tail verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Thirteenth hourly recovery checkpoints are available at BF16 step 16,384 and MXFP8 step 17,435.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 10:18 - MXFP8Q-007 fourteenth hourly checkpoints pass

- Result: both arms completed their fourteenth hourly checkpoint write to region-local S3. BF16 saved step 17,654 and MXFP8 saved step 18,789; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-18,000 evaluations.

### 2026-07-20 10:36 - MXFP8Q-007 step-18,000 gate preserves the bounded aligned gap

- Hypothesis: the step-17,000 rebound will remain bounded, with Paloma and uncheatable trading magnitude while the smoothed train delta returns toward two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 18,000 is 1.801975/1.804156 (delta +0.002180); the trailing 100-step mean delta is +0.002228. Eval loss is 2.484909/2.487882 (+0.002972), Paloma macro is 2.800546/2.805013 (+0.004466), and uncheatable macro is 2.230099/2.233790 (+0.003691).
- Performance: cumulative mean non-compile throughput is 788,940 tok/s BF16 versus 845,882 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 18,000 is 51,299.7s versus 48,121.6s; at the BF16 arm's step-18,000 elapsed time, MXFP8 had reached step 19,155, a 1,155-step lead.
- Interpretation: at 57.2% of the schedule, all three held-out metrics favor BF16 for a twelfth consecutive gate. Aggregate eval and uncheatable narrow from step 17,000 while Paloma widens, and the smoothed train delta returns near two thousandths. The small persistent quality-cost signal remains bounded and non-monotonic; the final cooldown tail is still decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Fourteenth hourly recovery checkpoints are available at BF16 step 17,654 and MXFP8 step 18,789.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish the next issue update at a materially later milestone or if the quality/health conclusion changes.

### 2026-07-20 11:18 - MXFP8Q-007 fifteenth hourly checkpoints pass

- Result: both arms completed their fifteenth hourly checkpoint write to region-local S3. BF16 saved step 18,925 and MXFP8 saved step 20,129; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-19,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 11:25 - MXFP8Q-007 step-19,000 gate preserves the bounded aligned gap

- Hypothesis: the held-out aggregates will continue trading magnitude within the established range while the smoothed train delta remains near two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 19,000 is 1.743063/1.745672 (delta +0.002608); the trailing 100-step mean delta is +0.002040. Eval loss is 2.470860/2.472553 (+0.001692), Paloma macro is 2.786834/2.791447 (+0.004612), and uncheatable macro is 2.214528/2.218555 (+0.004027).
- Performance: cumulative mean non-compile throughput is 788,966 tok/s BF16 versus 845,898 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 19,000 is 54,144.4s versus 50,797.8s; at the BF16 arm's step-19,000 elapsed time, MXFP8 had reached step 20,253, a 1,253-step lead.
- Interpretation: at 60.4% of the schedule, all three held-out metrics favor BF16 for a thirteenth consecutive gate. Aggregate eval narrows from step 18,000 while Paloma and uncheatable remain around four thousandths, and the smoothed train difference remains near two thousandths. This continues the small persistent, bounded quality-cost signal; the final cooldown tail remains decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Fifteenth hourly recovery checkpoints are available at BF16 step 18,925 and MXFP8 step 20,129.
- Next action: continue to the step-20,000 coordinating-issue milestone and all subsequent scheduled evaluations and checkpoints through step 31,474.

### 2026-07-20 12:10 - MXFP8Q-007 step-20,000 milestone preserves the stable quality signal

- Hypothesis: at the next coordinating-issue milestone, the smoothed train and held-out differences will remain a few thousandths while the throughput ratio stays near 1.072x.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`).
- Result: BF16/MXFP8 train loss at exactly step 20,000 is 1.732926/1.735309 (delta +0.002383); the trailing 100-step mean delta is +0.002042. Eval loss is 2.457578/2.459685 (+0.002107), Paloma macro is 2.772133/2.775098 (+0.002965), and uncheatable macro is 2.201548/2.204651 (+0.003103).
- Performance: cumulative mean non-compile throughput is 788,941 tok/s BF16 versus 845,886 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 20,000 is 56,985.9s versus 53,473.7s; at the BF16 arm's step-20,000 elapsed time, MXFP8 had reached step 21,316, a 1,316-step lead.
- Interpretation: at 63.5% of the schedule, all three held-out metrics favor BF16 for a fourteenth consecutive gate. All deltas return near two to three thousandths, and the smoothed train difference is effectively unchanged from step 19,000. The paired trajectory continues to support a small, stable quality cost alongside a stable throughput gain; the final cooldown tail remains necessary for the verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Fifteenth hourly recovery checkpoints are available at BF16 step 18,925 and MXFP8 step 20,129.
- Next action: publish the planned coordinating-issue milestone, then continue every scheduled evaluation and checkpoint through step 31,474.

### 2026-07-20 12:20 - MXFP8Q-007 sixteenth hourly checkpoints pass

- Result: both arms completed their sixteenth hourly checkpoint write to region-local S3. BF16 saved step 20,184 and MXFP8 saved step 21,481; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and complete the matched step-21,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 13:00 - MXFP8Q-007 step-21,000 gate narrows the aligned held-out gap

- Hypothesis: the step-20,000 quality gap will remain bounded rather than grow monotonically as both arms enter the final third of the schedule.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `2f450c616`.
- Result: BF16/MXFP8 train loss at exactly step 21,000 is 1.722408/1.724770 (delta +0.002362); the trailing 100-step mean delta is +0.002194. Eval loss is 2.443086/2.444647 (+0.001560), Paloma macro is 2.757547/2.759408 (+0.001861), and uncheatable macro is 2.186452/2.188400 (+0.001948).
- Performance: cumulative mean non-compile throughput is 788,916 tok/s BF16 versus 845,871 tok/s MXFP8, or 1.0722x. W&B runtime from the first train sample to step 21,000 is 59,843.8s versus 56,148.3s; at the BF16 arm's step-21,000 elapsed time, MXFP8 had reached step 22,384, a 1,384-step lead.
- Interpretation: at 66.7% of the schedule, all three held-out metrics favor BF16 for a fifteenth consecutive gate, but every held-out delta narrows below two thousandths. Aggregate eval has favored BF16 at all 21 paired gates, yet its range remains bounded and non-monotonic. The trajectory still supports a small persistent quality cost alongside a stable throughput gain; the final cooldown tail remains decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Sixteenth hourly recovery checkpoints are available at BF16 step 20,184 and MXFP8 step 21,481.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 13:18 - MXFP8Q-007 seventeenth hourly checkpoints pass

- Result: both arms completed their seventeenth hourly checkpoint write to region-local S3. BF16 saved step 21,451 and MXFP8 saved step 22,829; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-22,000 evaluations.

### 2026-07-20 13:45 - MXFP8Q-007 step-22,000 gate preserves the bounded quality signal

- Hypothesis: the unusually narrow step-21,000 held-out gap will remain within the established range rather than begin a monotonic increase.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `2f450c616`.
- Result: BF16/MXFP8 train loss at exactly step 22,000 is 1.637891/1.640409 (delta +0.002518); the trailing 100-step mean delta is +0.002197. Eval loss is 2.429433/2.431466 (+0.002032), Paloma macro is 2.740998/2.744033 (+0.003036), and uncheatable macro is 2.171774/2.176112 (+0.004338).
- Performance: cumulative mean non-compile throughput is 788,899 tok/s BF16 versus 845,807 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 22,000 is 62,688.4s versus 58,825.2s; at the BF16 arm's step-22,000 elapsed time, MXFP8 had reached step 23,448, a 1,448-step lead.
- Interpretation: at 69.9% of the schedule, all three held-out metrics favor BF16 for a sixteenth consecutive gate. The held-out differences widen from the narrow step-21,000 gate but remain within the established range, while the smoothed train delta is essentially unchanged. Aggregate eval has favored BF16 at all 22 paired gates, but the quality cost remains small, bounded, and non-monotonic; the final cooldown tail remains decisive.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Seventeenth hourly recovery checkpoints are available at BF16 step 21,451 and MXFP8 step 22,829.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 14:18 - MXFP8Q-007 eighteenth hourly checkpoints pass

- Result: both arms completed their eighteenth hourly checkpoint write to region-local S3. BF16 saved step 22,722 and MXFP8 saved step 24,170; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure. A narrow four-minute log query at 14:23 missed the completion lines by about 50 seconds; the required 20-minute cross-check recovered the successful writes.
- Next action: continue monitoring finite telemetry and complete the matched step-23,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 14:35 - MXFP8Q-007 step-23,000 gate remains within the established range

- Hypothesis: the step-22,000 held-out widening will remain within the earlier observed range while the smoothed train gap stays near two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `2f450c616`.
- Result: BF16/MXFP8 train loss at exactly step 23,000 is 1.633011/1.634263 (delta +0.001252); the trailing 100-step mean delta is +0.002334. Eval loss is 2.417056/2.419630 (+0.002575), Paloma macro is 2.726440/2.730242 (+0.003802), and uncheatable macro is 2.158189/2.163279 (+0.005090).
- Performance: cumulative mean non-compile throughput is 788,899 tok/s BF16 versus 845,783 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 23,000 is 65,527.9s versus 61,497.3s; at the BF16 arm's step-23,000 elapsed time, MXFP8 had reached step 24,511, a 1,511-step lead.
- Interpretation: at 73.1% of the schedule, all three held-out metrics favor BF16 for a seventeenth consecutive gate. Paloma and uncheatable widen from step 22,000 but remain below their earlier maxima, while the smoothed train delta stays near two thousandths. Aggregate eval has favored BF16 at all 23 paired gates; the quality cost remains small and non-monotonic, but the long aligned sequence makes the sign increasingly credible. The final cooldown tail remains decisive for magnitude and the wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Eighteenth hourly recovery checkpoints are available at BF16 step 22,722 and MXFP8 step 24,170.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 15:18 - MXFP8Q-007 nineteenth hourly checkpoints pass

- Result: both arms completed their nineteenth hourly checkpoint write to region-local S3. BF16 saved step 23,993 and MXFP8 saved step 25,524; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-24,000 quality and performance gate once BF16 finishes the immediately following scheduled evaluation.

### 2026-07-20 15:24 - MXFP8Q-007 step-24,000 gate narrows the held-out gap

- Hypothesis: the step-23,000 Paloma and uncheatable widening will remain non-monotonic while the smoothed train gap stays near two thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `2f450c616`.
- Result: BF16/MXFP8 train loss at exactly step 24,000 is 1.709205/1.711612 (delta +0.002407); the trailing 100-step mean delta is +0.002293. Eval loss is 2.403165/2.404554 (+0.001389), Paloma macro is 2.713129/2.715516 (+0.002387), and uncheatable macro is 2.145395/2.149498 (+0.004104).
- Performance: cumulative mean non-compile throughput is 788,890 tok/s BF16 versus 845,768 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 24,000 is 68,374.2s versus 64,172.6s; at the BF16 arm's step-24,000 elapsed time, MXFP8 had reached step 25,575, a 1,575-step lead.
- Interpretation: at 76.3% of the schedule, all three held-out metrics favor BF16 for an eighteenth consecutive gate. Every held-out difference narrows from step 23,000 while the smoothed train delta is nearly unchanged. Aggregate eval has favored BF16 at all 24 paired gates; the quality cost remains small and non-monotonic, but its sign is persistent. The final cooldown tail remains decisive for magnitude and the wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Nineteenth hourly recovery checkpoints are available at BF16 step 23,993 and MXFP8 step 25,524.
- Next action: continue to the step-25,000 coordinating-issue milestone and all subsequent scheduled evaluations and checkpoints through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 16:10 - MXFP8Q-007 step-25,000 milestone preserves the stable tradeoff

- Hypothesis: at the next coordinating-issue milestone, the smoothed train and held-out differences will remain within the established few-thousandths range while the throughput ratio stays near 1.072x.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `2f450c616`.
- Result: BF16/MXFP8 train loss at exactly step 25,000 is 1.616706/1.618859 (delta +0.002153); the trailing 100-step mean delta is +0.002319. Eval loss is 2.391230/2.393480 (+0.002250), Paloma macro is 2.700806/2.704305 (+0.003499), and uncheatable macro is 2.131068/2.136678 (+0.005611).
- Performance: cumulative mean non-compile throughput is 788,899 tok/s BF16 versus 845,718 tok/s MXFP8, or 1.0720x. W&B runtime from the first train sample to step 25,000 is 71,205.2s versus 66,847.3s; at the BF16 arm's step-25,000 elapsed time, MXFP8 had reached step 26,637, a 1,637-step lead.
- Interpretation: at 79.4% of the schedule, all three held-out metrics favor BF16 for a nineteenth consecutive gate. The held-out differences widen from step 24,000 but remain within the established range, while the smoothed train delta stays near two thousandths. Aggregate eval has favored BF16 at all 25 paired gates. The evidence increasingly supports a real but small quality cost alongside the stable throughput gain; the final cooldown tail remains necessary to quantify whether the wall-time benefit compensates for it.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Nineteenth hourly recovery checkpoints are available at BF16 step 23,993 and MXFP8 step 25,524.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; preserve this planned issue milestone locally until external posting is explicitly approved.

### 2026-07-20 16:18 - MXFP8Q-007 twentieth hourly checkpoints pass

- Result: both arms completed their twentieth hourly checkpoint write to region-local S3. BF16 saved step 25,256 and MXFP8 saved step 26,875; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and the scheduled step-26,000 evaluations.

### 2026-07-20 16:55 - MXFP8Q-007 step-26,000 gate preserves the late-run signal

- Hypothesis: entering the final 20% of the schedule will preserve the small aligned quality gap and stable throughput ratio without numerical instability.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Result: BF16/MXFP8 train loss at exactly step 26,000 is 1.719567/1.722362 (delta +0.002795); the trailing 100-step mean delta is +0.002329. Eval loss is 2.375680/2.378331 (+0.002651), Paloma macro is 2.678721/2.683281 (+0.004560), and uncheatable macro is 2.114739/2.120010 (+0.005271).
- Performance: cumulative mean non-compile throughput is 788,908 tok/s BF16 versus 845,754 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 26,000 is 74,046.8s versus 69,516.1s; at the BF16 arm's step-26,000 elapsed time, MXFP8 had reached step 27,699, a 1,699-step lead.
- Interpretation: at 82.6% of the schedule, all three held-out metrics favor BF16 for a twentieth consecutive gate. Aggregate eval and Paloma widen from step 25,000 while uncheatable narrows slightly, and the smoothed train delta is effectively unchanged. Aggregate eval has favored BF16 at all 26 paired gates. The evidence continues to support a real but small quality cost alongside the stable throughput gain; the final cooldown tail remains necessary for the wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Twentieth hourly recovery checkpoints are available at BF16 step 25,256 and MXFP8 step 26,875.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 17:18 - MXFP8Q-007 twenty-first hourly checkpoints pass

- Result: both arms completed their twenty-first hourly checkpoint write to region-local S3. BF16 saved step 26,529 and MXFP8 saved step 28,214; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: continue monitoring finite telemetry and complete the matched step-27,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 17:42 - MXFP8Q-007 step-27,000 gate remains aligned and bounded

- Hypothesis: the late-run held-out gap will remain bounded while the smoothed train difference stays within a few thousandths and throughput remains stable.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Result: BF16/MXFP8 train loss at exactly step 27,000 is 1.651859/1.655300 (delta +0.003441); the trailing 100-step mean delta is +0.002444. Eval loss is 2.361295/2.363271 (+0.001975), Paloma macro is 2.665448/2.668405 (+0.002957), and uncheatable macro is 2.098679/2.103666 (+0.004987).
- Performance: cumulative mean non-compile throughput is 788,917 tok/s BF16 versus 845,802 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 27,000 is 76,893.8s versus 72,193.1s; at the BF16 arm's step-27,000 elapsed time, MXFP8 had reached step 28,764, a 1,764-step lead.
- Interpretation: at 85.8% of the schedule, all three held-out metrics favor BF16 for a twenty-first consecutive gate. All held-out differences narrow from step 26,000 while the smoothed train delta widens slightly but remains within the established range. Aggregate eval has favored BF16 at all 27 paired gates. The evidence continues to support a real but small quality cost alongside the stable throughput gain; the final cooldown tail remains necessary for the wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Twenty-first hourly recovery checkpoints are available at BF16 step 26,529 and MXFP8 step 28,214.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 18:18 - MXFP8Q-007 twenty-second hourly checkpoints pass

- Result: both arms completed their twenty-second hourly checkpoint write to region-local S3. BF16 saved step 27,800 and MXFP8 saved step 29,567; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-28,000 quality and performance gate once BF16 finishes its scheduled evaluation.

### 2026-07-20 18:35 - MXFP8Q-007 step-28,000 gate narrows aggregate eval and Paloma

- Hypothesis: the final 11% of the schedule will preserve a bounded held-out gap while the smoothed train difference stays within a few thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Result: BF16/MXFP8 train loss at exactly step 28,000 is 1.691082/1.693871 (delta +0.002789); the trailing 100-step mean delta is +0.002489. Eval loss is 2.348755/2.349843 (+0.001088), Paloma macro is 2.650136/2.652120 (+0.001984), and uncheatable macro is 2.086016/2.091851 (+0.005835).
- Performance: cumulative mean non-compile throughput is 788,928 tok/s BF16 versus 845,785 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 28,000 is 79,736.4s versus 74,870.8s; at the BF16 arm's step-28,000 elapsed time, MXFP8 had reached step 29,828, a 1,828-step lead.
- Interpretation: at 89.0% of the schedule, all three held-out metrics favor BF16 for a twenty-second consecutive gate. Aggregate eval and Paloma narrow to near their lower late-run range while uncheatable widens modestly, and the smoothed train delta stays within the established range. Aggregate eval has favored BF16 at all 28 paired gates. The evidence continues to support a real but small quality cost alongside the stable throughput gain; the cooldown tail remains necessary for the final magnitude and wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Twenty-second hourly recovery checkpoints are available at BF16 step 27,800 and MXFP8 step 29,567.
- Next action: continue every scheduled evaluation and hourly checkpoint through step 31,474; publish externally only after explicit approval or at the authorized final handoff.

### 2026-07-20 19:18 - MXFP8Q-007 twenty-third hourly checkpoints pass

- Result: both arms completed their twenty-third hourly checkpoint write to region-local S3. BF16 saved step 29,060 and MXFP8 saved step 30,919; the process-0 logs reported `Saved checkpoint` after all eight ranks completed serialization barriers.
- Health: both child gangs and coordinators remained running through the writes and resumed training without an error-level checkpoint failure.
- Next action: analyze the matched step-29,000 gate, then closely monitor the MXFP8 final evaluation/checkpoint and BF16 cooldown tail.

### 2026-07-20 19:22 - MXFP8Q-007 step-29,000 gate preserves the bounded late-run gap

- Hypothesis: with less than 8% of the schedule remaining, the held-out differences will remain small and aligned while the smoothed train gap stays within a few thousandths.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Result: BF16/MXFP8 train loss at exactly step 29,000 is 1.626623/1.629534 (delta +0.002912); the trailing 100-step mean delta is +0.002536. Eval loss is 2.337475/2.338702 (+0.001227), Paloma macro is 2.638402/2.641266 (+0.002864), and uncheatable macro is 2.076971/2.080400 (+0.003428).
- Performance: cumulative mean non-compile throughput is 788,943 tok/s BF16 versus 845,821 tok/s MXFP8, or 1.0721x. W&B runtime from the first train sample to step 29,000 is 82,577.7s versus 77,543.0s; at the BF16 arm's step-29,000 elapsed time, MXFP8 had reached step 30,892, a 1,892-step lead.
- Interpretation: at 92.1% of the schedule, all three held-out metrics favor BF16 for a twenty-third consecutive gate. Aggregate eval remains near the narrow late-run range, Paloma widens modestly, and uncheatable narrows from step 28,000 while the smoothed train gap remains within the established range. Aggregate eval has favored BF16 at all 29 paired gates. The evidence continues to support a real but small quality cost alongside the stable throughput gain; the final 2,474-step cooldown tail remains decisive for the wall-time verdict.
- Health: both child gangs and coordinators remain running with finite telemetry and resumed training after evaluation. Twenty-third hourly recovery checkpoints are available at BF16 step 29,060 and MXFP8 step 30,919.
- Next action: monitor the MXFP8 final evaluation/checkpoint and BF16 cooldown tail through step 31,474, then run the terminal same-token and time-to-matched-loss analysis.

### 2026-07-20 19:44 - MXFP8Q-007 MXFP8 arm completes successfully

- Result: the MXFP8 arm completed all 31,474 optimizer steps. W&B finished with a final train/evaluation row at zero-based `global_step=31473` and final aggregate eval loss 2.31548. Iris marked both the eight-task child gang and its coordinator `succeeded`; all eight tasks exited 0 after 23h30m02.83s, with zero failures and zero preemptions.
- Checkpoint: all eight ranks completed the forced final serialization barrier, and process 0 reported `Saved checkpoint` for durable region-local `step-31474` at 02:43:46 UTC. A local `aws s3api head-object` check could not authenticate, so direct `metadata.json` verification remains pending an in-cluster final audit rather than being inferred.
- Interpretation: the treatment arm is terminally healthy and its W&B/final-checkpoint evidence is complete apart from the direct metadata-object check. This is not yet the paired conclusion because BF16 remains in its cooldown tail.
- Next action: continue babysitting BF16 through its final evaluation/checkpoint, verify both metadata objects from an authorized in-cluster context, then run the paired terminal analysis.

### 2026-07-20 20:05 - MXFP8Q-007 step-30,000 gate establishes a wall-time benefit

- Hypothesis: the stable throughput advantage will outweigh the small same-token quality cost by the time BF16 reaches the late cooldown tail.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Same-token result: BF16/MXFP8 train loss at exactly step 30,000 is 1.623617/1.626835 (delta +0.003219); the trailing 100-step mean delta is +0.002916. Eval loss is 2.326670/2.328074 (+0.001404), Paloma macro is 2.626495/2.629058 (+0.002563), and uncheatable macro is 2.066508/2.070774 (+0.004266).
- Performance: cumulative mean non-compile throughput through step 30,000 is 788,949 tok/s BF16 versus 845,834 tok/s MXFP8, or 1.0721x. Same-step W&B runtime is 85,415.0s versus 80,216.5s.
- Wall-time result: MXFP8's final step-31,474 evaluation completed at W&B runtime 84,556.3s, before BF16 reached step 30,000 at 85,415.0s. At that equal-time boundary, final MXFP8 aggregate eval/Paloma/uncheatable losses are 2.315482/2.614211/2.057221 versus BF16-at-30k 2.326670/2.626495/2.066508, improvements of 0.011187/0.012284/0.009287.
- Interpretation: at identical tokens, all three held-out metrics favor BF16 for a twenty-fourth consecutive gate and aggregate eval has favored BF16 at all 30 paired gates. However, the treatment's throughput gain already more than compensates for the small quality cost on a wall-time basis: MXFP8 finishes the full compute-optimal schedule before BF16 reaches 30k and is materially better on all held-out aggregates at that time. BF16's final 1,474 steps are still required for the definitive same-token result and final time-to-matched-loss table.
- Health: BF16 remains running with finite telemetry; MXFP8 is terminally successful with W&B finished, 8/8 Iris tasks exited 0, and a logged final `step-31474` checkpoint.
- Next action: babysit BF16 through its final evaluation/checkpoint, then run the terminal paired analysis and direct in-cluster metadata verification.

### 2026-07-20 20:18 - MXFP8Q-007 BF16 twenty-fourth hourly checkpoint passes

- Result: the remaining BF16 arm completed its twenty-fourth hourly checkpoint write to region-local S3 at step 30,332; the process-0 log reported `Saved checkpoint` after all eight ranks completed serialization barriers. The MXFP8 arm had already finished successfully with its forced final `step-31474` checkpoint.
- Health: BF16 remained running through the write and resumed training without an error-level checkpoint failure.
- Next action: complete the matched step-31,000 gate and closely monitor BF16's final 474-step evaluation/checkpoint sequence.

### 2026-07-20 20:52 - MXFP8Q-007 step-31,000 gate preserves the terminal tradeoff

- Hypothesis: the final scheduled intermediate gate will preserve the small same-token quality cost and stable throughput advantage before both arms enter the terminal comparison.
- Commit Hash: running jobs use `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Same-token result: BF16/MXFP8 train loss at exactly step 31,000 is 1.631137/1.633317 (delta +0.002180); the trailing 100-step mean is 1.591301/1.594297 (delta +0.002997). Eval loss is 2.318259/2.320333 (+0.002074), Paloma macro is 2.616024/2.619285 (+0.003262), and uncheatable macro is 2.056028/2.059842 (+0.003813).
- Performance: cumulative mean non-compile throughput through step 31,000 is 788,950 tok/s BF16 versus 845,909 tok/s MXFP8, or 1.0722x. Same-step W&B runtime is 88,260.3s versus 82,886.0s. By the BF16 arm's step-31,000 elapsed time, MXFP8 had already completed all 31,474 steps, a capped 473-step lead.
- Interpretation: all three held-out metrics favor BF16 for a twenty-fifth consecutive gate, and aggregate eval has favored BF16 at all 31 paired gates. The quality cost remains small and persistent; the stable throughput advantage remains large enough for MXFP8 to finish before BF16 reaches this gate. BF16's final 473-step tail is now the only missing training evidence for the definitive same-token and time-to-matched-loss result.
- Health: BF16 remained running with finite telemetry and resumed training after the step-31,000 evaluation. MXFP8 remains terminally successful with W&B finished, 8/8 Iris tasks exited 0, and a logged final `step-31474` checkpoint.
- Next action: babysit BF16 through its forced terminal evaluation/checkpoint, verify both terminal artifacts, and run the final paired analysis.

### 2026-07-20 21:25 - MXFP8Q-007 paired run completes with a speed-quality tradeoff

- Hypothesis: the terminal cooldown evaluation will determine whether the stable 1.072x throughput gain preserves BF16's final held-out quality or reaches it sooner.
- Commit Hash: running jobs used `d11d6ac54` (treatment implementation `f8be94f87`); the preceding logbook snapshot is `b7a9059d6`.
- Completion: both arms completed all 31,474 optimizer steps and W&B finished with terminal train/evaluation rows at zero-based `global_step=31473`. Iris marked both child gangs and coordinators `succeeded`; each arm had 8/8 tasks exit 0 with zero failures and zero preemptions. BF16's task duration was 24h59m53s to 24h59m58.26s, while every MXFP8 task took 23h30m02.83s.
- Checkpoints: BF16's eight ranks completed the forced final storage commit, barrier, and serialization error checks; process 0 logged `Saved checkpoint` for region-local `step-31474` at 04:13:22 UTC. The equivalent MXFP8 evidence was recorded at 02:43:46 UTC. Direct `metadata.json` object reads remain pending an authorized in-cluster audit because the local environment has no S3 credentials.
- Same-token result: terminal BF16/MXFP8 train loss is 1.554700/1.557329 (delta +0.002629), trailing-100 mean is 1.581135/1.584242 (+0.003107), and trailing-1,000 mean is 1.594350/1.597402 (+0.003052). Aggregate eval is 2.314181/2.315482 (+0.001301, +0.0562%), Paloma macro is 2.611326/2.614211 (+0.002885, +0.1105%), and uncheatable macro is 2.052934/2.057221 (+0.004287, +0.2088%).
- Trajectory: MXFP8 aggregate eval is higher at all 32 paired evaluation points, with a mean delta of +0.002098 and range +0.000314 to +0.003768. Paloma and uncheatable are higher at 29/32 points, with mean deltas +0.002429 and +0.003694. The final result extends the run of all three held-out metrics favoring BF16 to 26 consecutive gates from step 7,000 through the terminal evaluation.
- Performance: mean non-compile throughput is 788,954 tok/s BF16 versus 845,920 tok/s MXFP8, a 1.072204x ratio (+7.220%). Measured time from the first train sample through the terminal evaluation is 89,640.2s versus 84,179.6s, saving 5,460.6s (91.0 minutes, 6.092%). MXFP8 had already finished before BF16 reached step 30,000; at that equal-wall-time boundary its final aggregate eval/Paloma/uncheatable losses were better than BF16-at-30k by 0.011187/0.012284/0.009287.
- Matched-loss verdict: within the fixed 31,474-step schedule, MXFP8 never reaches BF16's final aggregate eval, Paloma, or uncheatable target at any logged evaluation. Strict time-to-BF16-final-loss is therefore censored rather than improved. The run demonstrates a stable throughput benefit with a small, persistent quality cost, not exact quality preservation.
- Data integrity: W&B's history scanner repeats/drops pagination-boundary rows in a page-size-dependent pattern. Reconciling independent page sizes 10,000 and 997 recovers every expected train row from global step 2 through 31,473 for both arms, with no conflicting values; all 32 paired evaluation rows are present. Final analysis artifacts are staged at `/tmp/mxfp8q-007-final/{summary,time_to_matched_eval,paired_eval_trajectory}.csv` for W&B publication.
- Next action: after explicit approval, perform the read-only in-cluster metadata-object audit, upload the three small comparison artifacts and a primary summary table/chart to W&B, post the final result to issue #7271, and snapshot/push the durable logbook update.

### 2026-07-21 09:55 - Terminal artifacts audited and published to W&B

- Checkpoint audit: Iris job `/mwittmann/mxfp8q-007-final-checkpoint-audit` succeeded after directly opening both region-local `metadata.json` objects with the in-cluster S3 identity. BF16 metadata is `step=31474`, `is_temporary=false`, timestamp `2026-07-21T04:13:22.902387`; MXFP8 metadata is `step=31474`, `is_temporary=false`, timestamp `2026-07-21T02:43:46.764263`. These timestamps agree with the process-0 final-save logs.
- W&B publication: [MXFP8Q-007 final comparison](https://wandb.ai/marin-community/marin_moe/runs/MXFP8Q-007-full-comparison) finished successfully with the primary terminal comparison table, paired held-out trajectory table/chart, matched-BF16-target table, and `mxfp8q-007-final-analysis:v0` evaluation artifact containing the three reconciled CSVs. Its summary records the +7.220% throughput gain, 5,460.6-second runtime saving, terminal held-out deltas, and three unmatched BF16-final-target booleans.
- Provenance: both source W&B runs now include `comparison/run_id=MXFP8Q-007-full-comparison`. The comparison artifact names issue #7271, both source run IDs, job commit `d11d6ac54`, treatment implementation `f8be94f87`, 31,474 optimizer steps, and 66,005,762,048 tokens per arm.
- Next action: push this durable logbook update and replace issue #7271's provisional step-15k summary/status/conclusion with the terminal result, then post the final linked comparison comment.
