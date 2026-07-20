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
- The full 1e21-FLOP pair is running. At step 12,000, its trailing-100-step train delta is +0.00196; eval/Paloma/uncheatable deltas are +0.00363/+0.00572/+0.00290, and mean throughput is 1.0722x BF16. Six consecutive gates now have aligned held-out signs favoring BF16, but this covers only 38.1% of the schedule and is not the final cooldown-tail quality conclusion.

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
