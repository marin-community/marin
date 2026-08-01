---
topic: coupon-clipping-16gb200
issue: https://github.com/marin-community/marin/issues/7836
description: Test shallow knowledge acquisition and layerwise shared-expert capacity on 16 GB200 GPUs.
author: rjpower
---

# Coupon clipping on 16 GB200 GPUs: Research Logbook

## Scope

- Goal: measure whether shallow-to-deep growth or input-adjacent active capacity reaches an interesting early loss faster than uniform full-depth training.
- Primary metrics: held-out Datamix and Paloma loss against allocation wall time, valid tokens, and estimated training FLOPs; factual retrieval versus manipulation probes.
- Constraints: four 4-GPU GB200 nodes per arm, at most 12 hours, with a comparison checkpoint after four hours.
- Design artifact: https://loom.rjp.io/s/9eoiqdub/artifacts/coupon-clipping-experiment-design
- Coordinating issue: https://github.com/marin-community/marin/issues/7836

## Current TL;DR

The first wave did not produce a greater-than-5x model suitable for SFT or RL. Fat-first and fat-middle capacity matched C0. D1 and C-short were only 2.06x and 1.93x faster than C0. WD1 finished 4.77x faster but remained unusable at 4.0338 train loss and 4.8286 Paloma micro loss, versus C0's 1.9903 and 2.8192. The fixed-64-expert WD2 source reached 2.50M tok/s, or 13.14x C0, with zero overflow. A 90/10 WD2 arm is entering its factor-four transition canary before production launch.

## Current Baseline

- Arm: `CC16-C0`, uniform full-depth control.
- Model: `d3072/L48/E64/top4`, routed expert width 1536, shared expert width 1280.
- Provisional horizon: 6,400 updates at batch 256 and sequence length 4,096, or 6.711B valid tokens.
- Hardware: 16 GB200 GPUs across four nodes; production mesh and horizon are gated on a 128-update sustained pilot.

## Hypothesis Queue

### Active

- `CC16-H5`: a `d768/L1/E64/top4` source plus substantially more adaptation than WD1's 320-update tail can retain the measured 13.14x source rate and reach a usable full-model checkpoint above 5x end-to-end. Next test: gate a 5,760-source / 640-target schedule on a 32+16-update hardware transition canary.

### Blocked

- `CC16-H4`: a 12-layer source is a safer growth platform than a one-layer source. Blocker: full-width L1 already tops out at 4.39x, so a deeper source cannot meet the headline speed gate without also narrowing width or the vocabulary head.

### Falsified / Dead End

- `CC16-H1`: D1 finished 2.06x faster than C0 at 2.1351 terminal train loss and lost its initial post-growth advantage after roughly one thousand full-depth updates.
- `CC16-H2`: WD1 finished 4.77x faster but stopped at 4.0338 train loss and 4.8286 Paloma micro loss.
- `CC16-H3`: C0/P1/P2 terminal 64-step means differed by at most 0.0016; placement did not affect this horizon.

### Promoted

- None.

## Decision Log

- 2026-07-31: use approximately 5B active parameters. The earlier 15.8B-active design would receive too few tokens on 16 GPUs.
- 2026-07-31: vary shared-expert width for the primary pyramid. This applies the treatment capacity to every token and removes early router specialization as a required mechanism.
- 2026-07-31: compare fat-first with fat-middle. The middle placement controls for generic layer heterogeneity.
- 2026-07-31: collect a coordinated comparison after four hours; continue healthy runs toward the 12-hour horizon.
- 2026-07-31: use FSDP-only (`expert=1`, `replica=1`, `data=16`). `sonic_cute` is a local SM100 backend and rejects expert-parallel axes larger than one.
- 2026-07-31: preregister D1 at update 4,480 (70%) so the token-matched arm retains 1,920 full-depth updates before the shared 640-update decay.
- 2026-08-01: test WD2 with a 10% target tail. Keep its 1,536-wide experts and 3:1 query/KV-head ratio in the target so every widened axis admits an exact factor-four transform.

## Negative Results Index

- None.

## Entry Log

### 2026-07-31 01:20 - CC16-000 experiment charter

- Hypothesis: shallow data exposure and input-adjacent shared capacity may improve early time-to-loss on an intentionally undertrained 5.2B-active MoE.
- Commit Hash: `2bcabffaa906b6d643988f60aefdf457fda37d03` (pre-implementation base).
- Command: launch commands pending implementation and systems validation.
- Config: five core arms (`CC16-C0`, `CC16-D1`, `CC16-D2`, `CC16-P1`, `CC16-P2`) on 16 GB200 GPUs; provisional batch 256, sequence length 4,096, and 6.711B-token full-depth horizon.
- Result: no run data yet.
- Interpretation: undertraining is intentional; conclusions will be scoped to early training.
- Next action: implement the shared-expert segments and depth-growth checkpoint transform, then run the production-path systems/LR gate.

### 2026-07-31 15:35 - CC16-001 implementation gate

- Hypothesis: matched shared-expert placement and deterministic L1-to-L48 growth can be exercised on the corrected GB200 Grug stack without changing the data, optimizer, or mesh contracts between arms.
- Commit Hash: `9a83fe899c8040b5905251c5348d068ebaa0cb60`.
- Command: `uv run --with pytest --with pytest-timeout --with pytest-asyncio pytest -q experiments/grug/coupon_clipping tests/test_grug_depth_growth.py tests/test_grug_variant_contracts.py`
- Config: P0/P1/P2 use segment lengths `(4, 18, 4, 22)` and exact matched accounting of 46,063,592,448 stored / 5,294,957,568 active parameters. D1 uses L1 through update 4,480 and L48 through update 6,400. All production and pilot stages retain the 6,400-step optimizer and Datamix horizons.
- Result: 25 passed and 1 skipped in the wider Grug contract run. One unrelated base-Grug CPU test failed on pre-existing label-concatenation sharding; the 12 coupon/depth tests pass. Artifact construction passes for all C0/P1/P2/D1 and canary entry points.
- Interpretation: parameter, scan, optimizer-bucket, transition-state, data-offset, and artifact dependency contracts are locally enforced. On-hardware memory, compile time, throughput, router stability, and checkpoint restore remain unmeasured.
- Next action: commit and push the reproducible snapshot, submit the bounded canary wave, then launch C0, D1, P1, and P2 in parallel if the gate passes.

### 2026-07-31 15:53 - CC16-002 GB200 canary admission

- Hypothesis: the FSDP-only 16-GPU topology can run the local `sonic_cute` MoE and preserve finite loss, gradients, and router occupancy across the shallow and full-depth paths.
- Commit Hash: `c965e9fb47`.
- Command: seven Iris coordinators launched the low/center/high C0, P1, P2, L1, and L1-to-L48 growth pilots in parallel; exact job IDs and recovery commands are in `scratch/20260731T153724Z_monitoring_state.json`.
- Config: each trainer uses four replicas of four GB200 GPUs. Pilots retain the production batch, optimizer, and data horizons while stopping after 128 updates; the growth pilot stops its source at update 32 and resumes its L48 target through update 48.
- Result: the 4-GPU SM100/QuACK smoke passed in 1m41s. The growth source finished update 31 at loss 10.0821, gradient norm 0.6325, zero capacity overflow, and 789,388 tok/s. The independent L1 probe reached update 95 at loss 6.3436, gradient norm 0.2316, zero overflow, and 788,994 tok/s. Its L48 growth target was admitted at 15:53 UTC. Full-depth pilots were still compiling without Iris task failure or preemption.
- Interpretation: the shallow platform and source checkpoint are operational. W&B may label a silent compiler interval as crashed while the underlying Iris gang remains healthy, so admission decisions use Iris state plus metric history rather than W&B state alone. A missing tracker-forwarding path for `train/valid_target_fraction` was found; the metric was already computed in-JIT, and the logging-only fix is being pushed before core launch.
- Next action: require a finite post-growth L48 update and finite full-depth pilot updates, select the LR, then admit the four core arms and start the four-hour comparison clock.

### 2026-07-31 15:57 - CC16-003 depth-growth restore gate

- Hypothesis: the L1 checkpoint can be restored as a full state and expanded to the segmented L48 target without resetting the data or optimizer schedules.
- Commit Hash: `3bd655a18c` (failure bundle).
- Command: `/power/cc16-growth-pilot-coord/grug-train-cc16-growth-pilot-l1-to-l48-16` was launched automatically after the 32-step source artifact completed.
- Config: resolve the final source checkpoint under the source artifact's `checkpoints` root, then require source step 32 and data offset 8,192 before expansion.
- Result: the source stage succeeded, but the target's first attempt passed the checkpoint root directly to `load_checkpoint`; TensorStore reported all expected leaves missing because the concrete `step-32` directory had not been resolved. No transform or L48 update ran. Iris retried the deterministic failure once.
- Interpretation: this is a phase-chaining path-resolution bug, not a model-state incompatibility. The production D1 path has the same bug and remains gated. The loader now resolves `latest_checkpoint_path` before reading the source tree, with a regression test for root-to-step resolution.
- Next action: stop the deterministic retry, push the fix, resubmit the same growth coordinator, and require finite post-growth updates before D1 admission.

### 2026-07-31 16:12 - CC16-004 learning-rate and layout gate

- Hypothesis: one of the preregistered learning-rate candidates improves early loss without destabilizing gradients, and the two pyramid layouts retain control-like throughput and routing occupancy.
- Commit Hash: `41a390894b` (canary bundle); the selected-rate configuration is the next commit.
- Command: C0 low/center/high and P1/P2 center-rate pilots, each for 128 updates on 16 GB200 GPUs.
- Config: MuonH/Adam candidate pairs were low `(0.005768679, 0.001331234)`, center `(0.006423539, 0.001482355)`, and high `(0.007210848, 0.001664041)`. P0/P1/P2 retain identical active parameters, stored parameters, batch, data order, and optimizer horizon.
- Result: C0 finished at loss 5.2744 / 5.1301 / 4.9888 for low / center / high. Terminal gradient norms were 0.3467 / 0.3318 / 0.3282, capacity overflow was zero for all three, and throughput was 191,442 / 191,579 / 191,870 tok/s. At center LR, P1 finished at loss 5.1373, gradient norm 0.4852, zero overflow, and 192,583 tok/s; P2 finished at loss 5.1286, gradient norm 0.3750, zero overflow, and 190,439 tok/s.
- Interpretation: select the high candidate for production: it improves terminal pilot loss by 0.1413 versus center without a gradient or throughput penalty. P1/P2 throughput differs from C0 by less than 1%, so placement comparisons are not confounded by a large kernel-speed difference. The 128-update losses do not distinguish P1, P2, and uniform at center LR; that is the question for the longer core wave.
- Next action: commit the selected high rate, finish the offline recovery of the growth canary, then launch C0, P1, P2, and D1 together.

### 2026-07-31 16:30 - CC16-005 depth-growth gate and core admission

- Hypothesis: exact L1-to-L48 state growth remains numerically trainable after preserving the source step, data offset, optimizer schedule, and non-block optimizer state.
- Commit Hash: `1e0dc4f62b`.
- Command: target-only recovery `/power/cc16-growth-target-recovery-coord/grug-train-cc16-growth-pilot-l1-to-l48-16-recovery`, followed by the four production coordinators recorded in `scratch/20260731T153724Z_monitoring_state.json`.
- Config: recovery loaded the completed step-32 L1 checkpoint and trained 16 L48 updates. The core wave assigns four 4-GPU GB200 nodes to each of C0, P1, P2, and D1; all use the selected high learning rate and common 6,400-update optimizer/data horizon.
- Result: every recovery rank reported `step=32`, `data_offset=8192`, 84 copied parameter leaves, 96 reset new-block optimizer leaves, and 19 preserved optimizer leaves. The grown model reached step 48 with loss 8.77 and saved its checkpoint. C0, P1, P2, and the D1 L1 source were all gang-admitted by 16:29:28 UTC with no task failure or preemption.
- Interpretation: the growth implementation passes both structural and short numerical gates. W&B finalization stalled after uploading the recovery run, but Iris completed all four workers successfully; this is telemetry teardown, not a training failure.
- Next action: require first finite metrics from every core arm, then compare loss, throughput, target fraction, overflow, and effective tokens at the 20:29:28 UTC four-hour checkpoint without stopping healthy jobs.

### 2026-07-31 20:29 - CC16-006 four-hour checkpoint

- Hypothesis: shallow data exposure or input-adjacent shared capacity reaches a useful early loss faster than uniform full-depth training on the same 16-GB200 allocation.
- Commit Hash: `6481f682da`.
- Command: C0, P1, P2, and D1 ran concurrently from a common wall-clock origin of 16:29:28 UTC. W&B rows were filtered to timestamps at or before the preregistered 20:29:28 UTC boundary. Loss comparisons use a 64-update trailing mean; C0/P1/P2 matched-token comparisons end at their largest common logged step.
- Config: each arm retains the frozen `d3072/L48/E64/top4` 5.295B-active / 46.064B-stored model accounting, batch 256, sequence length 4,096, selected high learning rate, and 6,400-update horizon. D1 used L1 through step 4,480, then exact-state L48 growth.
- Result:

  | Arm | Boundary step | Tokens | 64-step mean loss | Throughput | MFU | Valid target fraction |
  |---|---:|---:|---:|---:|---:|---:|
  | C0 uniform | 2,474 | 2.595B | 2.20946 | 189.6k tok/s | 17.16% | 99.976% |
  | P1 fat-first | 2,377 | 2.494B | 2.21144 | 189.6k tok/s | 17.16% | 99.976% |
  | P2 fat-middle | 2,366 | 2.482B | 2.21457 | 188.0k tok/s | 17.02% | 99.976% |
  | D1 L1-to-L48 | 5,795 | 6.078B | 2.35113 | 191.0k tok/s after growth | 17.29% after growth | 99.976% |

  At common step 2,366, the 64-step means were 2.21443 / 2.21479 / 2.21457 for C0 / P1 / P2, a total spread of 0.00036. D1 processed 2.34 times as many tokens as C0 but trailed its wall-clock loss by 0.14167. D1 initially transferred strongly: after 127 logged L48 updates, its mean loss was 3.58067 versus C0's 5.78999 at the corresponding L48 update count. The advantage decayed into noise around 925-1,243 L48 updates; at 1,316 logged L48 updates D1 was 0.01847 worse than C0's corresponding 2.33266. C0 reached mean loss 2.5 in 1.30 hours and 2.4 in 1.79 hours; D1 needed 2.98 and 3.48 hours. P1/P2 followed C0 at matched steps but crossed the thresholds about nine minutes later because of additional first-time compilation. All four gangs remained on their first attempts with zero failures and zero preemptions.
- Interpretation: active shared-expert width placement has not changed the learning curve through 2.48B tokens. The L1 source does implant state that gives a newly grown L48 model a large short-lived head start, but that benefit is erased after roughly one thousand L48 updates and does not compensate for the shallow phase in time-to-loss. This checkpoint is training-loss evidence only: in-run evaluation remains disabled because of issue #7712, and production overflow telemetry was not emitted with the low-overhead watch setting. The production-path canaries had zero overflow.
- Next action: continue all healthy jobs toward the 12-hour horizon. Use the completed checkpoints for held-out Datamix, Paloma, factual retrieval, and manipulation evaluations before deciding whether to promote a shorter shallow phase or a less extreme source depth.

### 2026-07-31 22:50 - CC16-007 greater-than-5x rollout gate

- Hypothesis: a `d1536/L1` knowledge phase followed by function-preserving width growth and identity-prefix depth growth can reach a usable 5.295B-active model in less than one fifth of C0's training time.
- Commit Hash: `32e58c7c7a` (pre-rollout base; implementation snapshot pending).
- Command: `uv run --with pytest --with pytest-timeout --with pytest-asyncio pytest -q experiments/grug/coupon_clipping tests/test_grug_depth_growth.py tests/test_grug_variant_contracts.py`
- Config: WD1 uses a 418,701,376-active / 631,038,016-stored `d1536/L1/E64/top4` source through update 5,760, then widens to `d3072` and inserts 47 exact-identity residual blocks before the trained block for the final 640 updates. The trained source block remains target layer 47 so its full-attention role is unchanged. C-short is the `d3072/L48` control with a compressed 3,200-update WSD horizon.
- Result: target/source analytic forward FLOPs differ by 25.93x and active parameters by 12.65x. The width transform preserves embedding-to-logit, attention-projection, and expert-projection functions in the focused numerical contract; width-dependent optimizer buffers reset while the schedule count persists. The focused suite passed 17 tests. The wider Grug contract run passed 28 tests with one skip; its pre-existing base-Grug CPU label-sharding test failed outside this variant.
- Interpretation: the existing untied `d3072/L1` source measured 4.39x C0 throughput, so layer sampling at the same width cannot meet the 5x target. WD1 removes that fixed-width ceiling. Analytic FLOPs are only an admission prior; the 128-update GB200 source canary must measure at least 5x C0's 189.6k tok/s, with 8x preferred, before the growth canary or full arm runs.
- Next action: snapshot and push the implementation, launch the WD1 source canary on the 16-GPU slice released by D1, then admit the transition canary only after finite loss, zero overflow, and the measured throughput gate.

### 2026-07-31 23:20 - CC16-008 aggressive source admission and schedule correction

- Hypothesis: narrowing both width and depth attacks the fixed embedding/head cost enough to make shallow knowledge exposure materially faster than C0.
- Commit Hash: `8ea0b49579` (source canary); the 95/5 schedule correction is the next commit.
- Command: `/power/ccx-wd1-source-pilot-coord/grug-train-ccx-wd1-d1536-l1-pilot128`, followed by `/power/ccx-wd1-growth-pilot-coord` after the source gate passed.
- Config: the canary used the 418,701,376-active / 631,038,016-stored `d1536/L1/E64/top4` source on 16 GB200 GPUs. The production schedule is corrected from 90/10 to 95/5: source through step 6,080, then 320 `d3072/L48` updates spanning the complete terminal decay.
- Result: the source finished at step 127 with loss 6.15569, 1.493M tok/s on the terminal step, 99.976% valid targets, finite gradients, and zero router overflow. Median logged throughput from step 16 onward was 1.431M tok/s, or 7.55x C0's 189.5k tok/s. At that measured rate, a 90/10 split projects to only 4.56x training-loop speed; 95/5 projects to 5.69x before setup and compilation. The independent 32-step transition source also succeeded with finite loss and zero overflow; its widened target canary is running.
- Interpretation: the systems result clears the 5x admission gate and nearly reaches the preferred 8x source rate. Meeting the end-to-end treatment objective requires spending less than 10% of updates at full width and depth; the revised arm makes that constraint explicit rather than claiming the source-only speedup for the entire run.
- Next action: require 16 finite post-growth updates from the transition canary, then launch the 95/5 production WD1 arm on the same four-node slice and record its four-hour checkpoint from gang admission.

### 2026-07-31 23:33 - CC16-009 width/depth transition gate and production admission

- Hypothesis: the narrow source can be expanded without a loss discontinuity that prevents rapid full-model adaptation, and the measured 7.55x source rate can support a greater-than-fivefold end-to-end schedule.
- Commit Hash: `4c100e9287`.
- Command: `/power/ccx-wd1-growth-pilot-coord`, followed by production coordinator `/power/ccx-wd1-coord` on the same four-node slice.
- Config: the transition canary trained 32 `d1536/L1` updates, expanded to `d3072/L48`, and trained through step 48. Production WD1 trains the source through step 6,080, then gives the expanded target 320 updates covering the complete optimizer decay.
- Result: every canary rank restored step 32 and data offset 8,192, copied 84 parameter leaves, reset 108 width/depth-dependent optimizer leaves, and preserved seven optimizer leaves. Logged target loss fell from 10.6 at step 34 to 9.82 at step 45, and every rank saved the step-48 checkpoint. Production source gang `/power/ccx-wd1-coord/grug-train-ccx-wd1-d1536-l1-source-step6080` was admitted at 23:32:52 UTC; its preregistered four-hour boundary is 03:32:52 UTC on 2026-08-01.
- Interpretation: exact width and identity-prefix depth growth is operational on the production topology. At the canary's measured rates, the 95/5 loop projects to 1.73 hours versus 9.84 hours for C0, or 5.69x before setup and compilation. This is a systems projection, not yet a capability result.
- Next action: require finite production source metrics, monitor at 30-minute intervals while healthy, verify the automatic target transition, then run C-short sequentially on the released slice. Evaluate terminal checkpoints on held-out Datamix, Paloma, factual retrieval, and matched manipulation probes.

### 2026-07-31 23:55 - CC16-010 production source health and offline Paloma path

- Hypothesis: capability comparisons can be collected from terminal checkpoints without perturbing training state or consuming more than the existing four-node slice.
- Commit Hash: `4eff246d26`.
- Command: bounded Paloma artifacts are exposed by `experiments.grug.coupon_clipping.paloma_wd1`, `paloma_c_short`, and `paloma_c0`; they remain gated on terminal checkpoints.
- Config: each evaluation restores model parameters without optimizer state, skips the optimization loop, and evaluates at most eight batches of 64 sequences for each of 16 Paloma subsets. The runs are scheduled sequentially on the same 16 GB200 GPUs.
- Result: all three evaluation DAGs lower successfully. Eighteen focused coupon-clipping and growth tests pass, as do the repository lint, format, and type gates. At 23:55 UTC, production WD1 was at source step 1,170/6,080 with loss 4.38 and a recent rate of 1.3-1.4 updates/s on all four workers.
- Interpretation: Paloma is now an executable terminal readout rather than an unspecified follow-up. W&B timed out during the source's initial compile, so Iris finelogs remain authoritative; synchronized finite progress shows the training run itself is healthy.
- Next action: monitor the source at the 30-minute cadence, require a successful automatic growth transition and terminal checkpoint, run C-short, then execute the three checkpoint-only Paloma artifacts sequentially.

### 2026-08-01 00:04 - CC16-011 fixed-expert 10x source gate

- Hypothesis: halving the residual-stream and vocabulary-head width again while widening selected experts can approach the 10x source-rate target without relying on useful routing across more experts.
- Commit Hash: `0c98066b1f`.
- Command: `experiments.grug.coupon_clipping.pilot_extreme_source` is staged for 128 updates after WD1 releases the four-node slice.
- Config: the source is `d768/L1/E64/top4` with routed and shared intermediate widths of 1,536. It has 217,112,128 active and 429,448,768 stored parameters; the target/source analytic forward-FLOP ratio is 48.61x.
- Result: the artifact lowers with the production Datamix, optimizer horizon, mesh, and pilot telemetry. The 12 focused config contracts pass. No hardware result exists yet.
- Interpretation: this is a source-capability and throughput gate, not a preregistered growth arm. Keeping 64 experts and top-4 routing answers the routing-cold-start objection; the larger selected experts preserve more per-token nonlinear capacity. A target and 1-2% deep-tail schedule are justified only if measured throughput materially exceeds WD1's 7.55x source rate.
- Next action: launch on the released slice, require finite gradients, zero overflow, broad expert traffic, and measured throughput, then decide whether to implement a function-preserving target transition or move directly to sampled/factorized vocabulary-head work.

### 2026-08-01 00:12 - CC16-012 success-gated sequential rollout

- Hypothesis: the next wave can preserve the revised 16-GPU budget by reusing one four-node slice and refusing to cascade after a failed coordinator.
- Commit Hash: `570da43fb3`.
- Command: the local recovery record `scratch/20260801_sequential_rollout.zsh` waits for `/power/ccx-wd1-coord`, then submits `/power/ccx-wd2-source-pilot-coord`, waits for success, and submits `/power/ccx-c-short-coord`.
- Config: WD1 remains the 95/5 `d1536/L1` to `d3072/L48` arm. WD2 is the 128-update `d768/L1/E64/top4` source throughput gate with wider selected experts. C-short is the 3,200-update full-depth strongman control. The three jobs cannot overlap on the new-wave slice.
- Result: at 00:09 UTC, WD1 was healthy and synchronized at source step 2,280/6,080, loss 4.14, and a recent rate of 1.2-1.4 updates/s. No fatal, traceback, or resource-exhaustion lines were present. The success-gated handoff is waiting on WD1; WD2 and C-short have not been submitted yet.
- Interpretation: the source is behaving normally, but setup and sustained-rate overhead may reduce WD1 below its 5.69x loop-only projection. WD2 is therefore a necessary measurement of whether the vocabulary/head and residual-stream costs can be pushed into the 10x-source regime without adding experts.
- Next action: verify the step-6,080 growth transform and finite L48 updates, measure WD1 end-to-end allocation speed, then apply the WD2 throughput and routing gate before C-short admission.

### 2026-08-01 15:38 - CC16-013 terminal training and Paloma results

- Hypothesis: shallow knowledge exposure can produce a full-size checkpoint suitable for SFT or RL more than 5x faster than C0, while the fixed-expert narrow source can approach a 10x systems regime without relying on early routing specialization.
- Commit Hash: `e8e383809e`.
- Command: the success-gated chain ran `/power/ccx-wd1-coord`, `/power/ccx-wd2-source-pilot-coord`, and `/power/ccx-c-short-coord` sequentially; checkpoint-only evaluators ran as `/power/cc16-paloma-{wd1,c-short,c0}-coord`. Metrics use deduplicated W&B rows and terminal 64-update train-loss means.
- Config: C0/P1/P2/D1 and WD1 end at update 6,400; C-short ends at 3,200. WD1 spends 6,080 updates at `d1536/L1` and 320 at `d3072/L48`. WD2 is a 128-update `d768/L1/E64/top4` source pilot with routed and shared expert widths 1,536. Paloma evaluates 77 available batches over 16 registered subsets, capped at eight batches of 64 sequences per subset.
- Result:

  | Arm | Coordinator time | Speedup vs C0 | Terminal 64-step train loss | Paloma micro loss | Paloma macro loss |
  |---|---:|---:|---:|---:|---:|
  | C0 uniform | 10h22m | 1.00x | 1.990261 | 2.819231 | 2.948071 |
  | P1 fat-first | 10h34m | 0.98x | 1.991862 | — | — |
  | P2 fat-middle | 10h35m | 0.98x | 1.990978 | — | — |
  | D1 L1-to-L48 | 5h02m | 2.06x | 2.135108 | — | — |
  | C-short L48 | 5h23m | 1.93x | 2.057690 | 2.903026 | 3.036822 |
  | WD1 narrow/shallow-to-full | 2h10m | 4.77x | 4.033804 | 4.828604 | 4.992723 |

  WD1's source median was 1.491M tok/s, 7.84x C0's 190.2k. WD2's median was 2.500M tok/s, 13.14x C0, with terminal throughput 2.359M tok/s. Every reported run had 99.976% valid targets and zero mean capacity overflow. WD2's terminal 64-step source loss was 7.6902; it does not produce a full-size checkpoint. W&B runs: [C0](https://wandb.ai/marin-community/marin/runs/cc16-c0-p0), [C-short](https://wandb.ai/marin-community/marin/runs/ccx-c-short-l48-step3200), [WD1 source](https://wandb.ai/marin-community/marin/runs/ccx-wd1-d1536-l1-source-step6080), [WD1 target](https://wandb.ai/marin-community/marin/runs/ccx-wd1-d1536-l1-to-d3072-l48), [WD2](https://wandb.ai/marin-community/marin/runs/ccx-wd2-d768-l1-i1536-pilot128), and Paloma [C0](https://wandb.ai/marin-community/marin/runs/cc16-paloma-c0), [C-short](https://wandb.ai/marin-community/marin/runs/cc16-paloma-c-short), [WD1](https://wandb.ai/marin-community/marin/runs/cc16-paloma-wd1).
- Interpretation: the headline hypothesis failed in this wave. WD1 missed the systems gate after setup and compilation and missed the capability gate by a wide margin. C-short is close to C0 on Paloma but provides only a 1.93x speedup. WD2 establishes that the shallow source can exceed 10x without adding experts; useful target adaptation remains unmeasured.
- Next action: do not promote WD1, D1, or either pyramid. If continuing, sweep short WD2 target tails or periodic full-model refreshes and measure Paloma after each; attack the vocabulary/output-head floor if those schedules cannot preserve at least a 5x end-to-end speedup.

### 2026-08-01 15:38 - CC16-014 checkpoint-only evaluation recovery

- Hypothesis: terminal checkpoints can be evaluated without rebuilding mutable training dependencies or mutating optimizer state.
- Commit Hash: `e8e383809e`.
- Command: `python -m experiments.grug.coupon_clipping.paloma_{wd1,c_short,c0} --version dev --run`, each submitted through a separate Iris coordinator on the same four-node slice.
- Config: the launchers adopt the completed `users/power/grug/coupon-clipping/.../dev` output as a typed `LevanterCheckpoint`, restore only model parameters, cast them through the bf16 compute policy, and write numeric metrics to `metrics.json`.
- Result: the first WD1 attempt was stopped after mutable `dev` recipe drift began rebuilding its 6.4B-token source. The adopted handle then failed fast once because its source omitted `users/power`; the first forward failed because FA4 rejected float32 inputs; and the completed evaluation initially failed while JSON serialized a JAX `float32`. Commits `c8200bb904`, `f5ea4d42ef`, `70785b17ce`, and `e8e383809e` fix those four boundaries. All three final evaluation coordinators succeeded, W&B is finished, and each output contains `metrics.json`. The reusable `dev` guidance is recorded in [Echo wiki #61](https://echo.oa.dev/wiki/61).
- Interpretation: `dev` is suitable for active iteration but unsafe as an implicit dependency identity for follow-up evaluation. Adopt the realized output or use a fixed calendar version; print the plan before `--run` and reject any unexpected training step.
- Next action: publish the terminal table to issue #7836 and the Weaver artifact, then close the rollout issue.

### 2026-08-01 21:14 - CC16-015 WD2 90/10 transition gate

- Hypothesis: a 640-update full-model tail can adapt the 13.14x WD2 source while retaining greater-than-5x end-to-end speed.
- Commit Hash: `cc3a9be3bc`.
- Command: `uv run --with pytest --with pytest-timeout --with pytest-asyncio pytest -q experiments/grug/coupon_clipping tests/test_grug_depth_growth.py tests/test_grug_variant_contracts.py`; hardware command pending canary submission.
- Config: train `d768/L1/E64/top4` through update 5,760, then expand by four to a 5.484B-active `d3072/L48/E64/top4` target for 640 updates. Routed and shared intermediate widths remain 1,536; query/KV heads expand from 6:2 to 24:8; the target tail contains the complete decay.
- Result: the focused growth/config suite passed 23 tests. The wider Grug run passed 36 tests with one skip and the known base-Grug CPU label-sharding failure. The new numerical contract checks that fixed-intermediate MLP inputs remain unchanged and outputs are duplicated without erroneous scaling.
- Interpretation: the WD2 source can be embedded exactly into the target despite holding expert width constant. The 24:8 target differs slightly from C0's 24:6 attention and has 5.484B rather than 5.295B active parameters; wall-time and Paloma comparisons must retain that caveat.
- Next action: launch the 32-source / 16-target GB200 canary, require a finite post-growth loss and saved step-48 checkpoint, then submit the 5,760+640 production arm.

### 2026-08-01 22:15 - CC16-016 L4 and stochastic-depth 80/20 gate

- Hypothesis: four narrow source layers provide enough reasoning capacity to learn shallow information more effectively than WD1's single layer, and sampling those four updates across the eventual 48 target positions may reduce the late full-depth transition shock.
- Commit Hash: `ad2b25b242`.
- Command: source canaries `/power/ccx-l4-source-pilot-coord` and `/power/ccx-ld4-source-pilot-coord` were submitted together on `cw-us-east-08a`. `scratch/20260801_l4_layerdrop_canary_rollout.zsh` submits the corresponding 32-source / 16-target transition canary after each source succeeds.
- Config: both treatments train through update 5,120 at `d1536` and through update 6,400 at full `d3072/L48`. The physical-L4 control stores and executes four source layers, then inserts 44 identity layers before its trained suffix. The stochastic-depth arm stores 48 narrow layers, executes four uniformly sampled ordered positions per update, and freezes inactive parameter, QB bias, optimizer, and EMA slices before width-only growth. Both tails contain 1,280 updates and the complete decay.
- Result: the focused config and state-growth suite passed 28 tests, including an end-to-end CPU update through the stochastic-depth path. Pre-commit lint, formatting, and type checks passed. Both source coordinators are pending behind the same scheduler backlog as the WD2 transition canary; neither has a hardware metric yet.
- Interpretation: the 20% full-depth tail imposes a 5x idealized ceiling, so these arms cannot satisfy the headline greater-than-5x objective. They are mechanism probes for WD1's capability failure. The physical source stores 1.340B parameters; the stochastic source stores 11.735B and may lose most of its speed advantage to dense Muon/Adam processing even though it executes the same four transformer blocks.
- Next action: compare 64-update median tokens/s and finite loss after each 128-update source canary, require successful transition canaries, and admit production only if the measured source rate justifies the full run. Evaluate any terminal checkpoints on the same bounded Paloma protocol.

### 2026-08-01 22:20 - CC16-017 L4 rollout policy

- Hypothesis: the stochastic-depth source's optimizer overhead is itself an experiment outcome, so source speed should not prevent collecting the requested 80/20 quality result.
- Commit Hash: `a144c61c8e`.
- Command: `scratch/20260801_l4_layerdrop_canary_rollout.zsh` waits on both source canaries in parallel, submits each matching transition canary after source success, and submits each full arm after transition success.
- Config: production job IDs are `/power/ccx-l4-tail20-coord` and `/power/ccx-ld4-tail20-coord`; each retains the 5,120+1,280 update schedule.
- Result: the two source coordinators are assigned, with no worker attempt or metric yet. The local rollout monitor is armed.
- Interpretation: source throughput remains a required measurement, not a production admission threshold. A failed source or transition canary still stops only its corresponding chain.
- Next action: collect source throughput and transition health, then monitor both automatically submitted production arms at the established 30-minute cadence.
