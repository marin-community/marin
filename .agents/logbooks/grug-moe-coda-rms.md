---
topic: grug-moe-coda-rms
description: Test CODA-style RMS scaling in the Grug MoE GatedNorm down-projection epilogue on GB200.
author: dlwh
---

# Grug MoE CODA RMS: Task Logbook

## Scope

- Goal: Apply CODA's producer-epilogue idea to the Grug RMS-GatedNorm reverse pass and measure training throughput on GB200.
- Primary metrics: Forward and gradient error, compile time, steady-state kernel time, training tokens/s, step time, MFU, and peak HBM.
- Constraints: Preserve the stock BF16 forward exactly; compare identical shapes, dtype, seed, and compiler flags; validate both a serial same-node control and the production 64-GPU shape.
- Coordinating issue/PR: Pending publication.

## Current TL;DR

The valid CODA-style seam is the RMS-GatedNorm reverse, not routed expert W2. The production candidate keeps the stock BF16 forward bit-identical, fuses `dq @ Wup.T` with dSiLU, and emits the input-gradient contribution plus RMS row partials from the `dp @ Wdown.T` SM100 GEMM epilogue. The corrected one-node control improved paired median throughput by 3.391% and won 20/20 scored steps. At the full 16-node/64-GPU FSDP64 hero shape, it improved ratio-of-medians throughput by 2.227%, reduced step duration by 2.178%, and saved 3.804 GiB peak HBM. Step-0 loss was exact; the scored rack loss delta remained below 0.449% relative. The rack arms ran sequentially because capacity exposed one rack at a time, so the one-node serial same-allocation pair is the stronger placement control.

## Baseline

- Date: 2026-08-10.
- Code ref: `499340ce18`.
- Hero shape: d6144, 48 layers, 128 experts, top-4, batch 1024, sequence 4096, 64 GB200 GPUs.
- Local norm shape: 65,536 rows by 6,144 hidden dimensions, BF16 compute with FP32 RMS statistics.
- Established FSDP reference: 235,125 tokens/s and 19.3951% median MFU. This historical number is context, not the matched control for this experiment.

## Hypothesis Queue

### Active

- Run a longer stability comparison before making the custom reverse the default for GB200 hero training.

### Blocked

- The rack launcher must use separate top-level roots until JAX coordinator registry keys include full child-job identity. See the resolved [Echo incident](https://echo.oa.dev/wiki/102).

### Falsified / Dead End

- Direct expert-W2 epilogue fusion: routed combine, two shared experts, SConv, and the residual occur before RMSNorm, so W2 is not the producer boundary CODA requires.
- `GMRMS-003`: The reference-VJP prototype did not exceed the one-node noise floor. Evidence: [XLA](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-epilogue-gb200-20260810-xla), [CODA](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-epilogue-gb200-20260810-quack-coda).
- `GMRMS-005`: Saving the normalized rank-128 preactivation and using an analytic reverse reduced exact-shape candidate forward-plus-backward latency from 4.497 ms to 4.114 ms, but XLA remained faster at 3.352 ms.
- `GMRMS-007`: The first competitive treatment had no batch-axis reductions for gradients of replicated `gamma`, `Wdown`, and `Wup`. All four ranks segfaulted before step 0, so the pair was unscorable.
- `GMRMS-009` v2: The delayed-forward treatment improved throughput by 5.20% but changed BF16 rounding sites and produced unacceptable loss drift.

### Promoted

- `GMRMS-001` (component only): Exact-shape forward latency improved 1.180x on one GB200. This does not promote the training implementation.
- `GMRMS-002` (numerics only): FP32/BF16 parity tests pass and the 25-step loss series stayed aligned. This does not promote the reference VJP.
- `GMRMS-011`: The exact-forward one-node treatment improved paired median throughput by 3.391%, won all 20 scored steps, and stayed above the 0.78% placement floor.
- `GMRMS-013`: The exact-forward full-rack treatment improved throughput by 2.227%, reduced step time by 2.178%, and saved 3.804 GiB peak HBM.

## Entry Log

### 2026-08-10 11:45 PDT - GMRMS-001 experiment selected

- Hypothesis: The existing QuACK SM100 normalized-activation GEMM can implement the only valid CODA-style RMS epilogue seam in Grug without adding a dependency.
- Commit Hash: `499340ce18`.
- Command: Repository, Echo, git-history, QuACK 0.6.1 source, and live A08 scheduler inspection.
- Config: XLA reference versus QuACK `GemmNormActSm100`; exact hero-local shape `[65536, 6144]`; GatedNorm rank 128.
- Result: QuACK 0.6.1 already computes `act((A @ B + C) * rstd)` on SM100. Rewriting `RMSNorm(x) @ Wdown` as `(x @ (gamma[:, None] * Wdown)) * rstd` removes the full normalized activation boundary in real arithmetic. The current scheduler has only four free nodes, below the sixteen-node hero requirement.
- Interpretation: Build the JAX bridge and numerical gate first. Use a one-node screen while the rack comparison is blocked.
- Next action: Implement the forward bridge, XLA fallback, and custom VJP.

### 2026-08-10 11:53 PDT - GMRMS-001 exact-shape GB200 component gate

- Hypothesis: Delaying the inverse-RMS row scale into the rank-128 down-projection epilogue reduces the exact hero-local forward latency.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.grug.moe_hero_fsdp.bench_rms_gated_norm --warmup 5 --steps 20 --output-jsonl /tmp/grug_coda_exact_gb200.jsonl`.
- Config: One GB200; BF16; input `[16,4096,6144]`; GatedNorm rank 128; FP32 RMS statistics; QuACK tile `[256,128]`, cluster `[2,1,1]`; 5 warmups and 20 timed iterations.
- Result: Forward improved from 1.3021 ms to 1.1037 ms (1.1798x). Forward plus backward regressed from 3.4178 ms to 4.4968 ms (0.7600x) because the prototype reverse recomputes the JAX CODA oracle. Output max/mean absolute deviation was 0.015625/1.43e-5. Aggregate gradient max/mean absolute deviation was 2.0/8.13e-4 under an unnormalized random cotangent; focused loss-scaled FP32/BF16 parity tests pass.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/exact-gb200-20260810.jsonl`.
- Interpretation: The forward memory-traffic hypothesis is supported. The reference-VJP reverse is not promotable and is the next kernel boundary if whole-step training confirms the regression.
- Next action: Measure serial one-node training throughput before investing in a fused reverse.

### 2026-08-10 11:57 PDT - GMRMS-003 one-node training A/B submitted

- Hypothesis: Whole-step rematerialization may hide enough of the reference-VJP cost for the forward win to remain visible in training throughput.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: CoreWeave Iris coordinator job `/dlwh/coda-rms-epilogue-gb200-20260810-coord`.
- Config: Serial fresh-process arms on one four-GPU GB200 node; d6144/L48/B64/S4096/E8/top4; 25 steps; XLA then QuACK CODA; score median `throughput/duration` at steps 5-24.
- Result: Pending.
- Interpretation: Pending.
- Next action: Babysit through terminal state and compare matched arm medians.

### 2026-08-10 12:29 PDT - GMRMS-003 one-node training A/B complete

- Hypothesis: The CODA-style boundary improvement is large enough to exceed noise in a serial same-node training A/B despite the reference-VJP reverse.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: `uv run iris --cluster=marin job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 5400 --job-name coda-rms-epilogue-gb200-20260810-coord -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_ab --run-id coda-rms-epilogue-gb200-20260810 --num-steps 25 --priority interactive --version dev --run`.
- Config: Same allocation and one four-GPU GB200 node; fresh subprocesses in XLA then CODA order; d6144/L48/B64/S4096/E8/top4; BF16; 25 steps; median metrics over global steps 5-24 (`n=20` each).
- Result: XLA measured 15.2753 s/step, 17,161.3 tok/s, 22.869% MFU, and 156.227 GiB peak HBM. CODA measured 15.2029 s/step, 17,243.0 tok/s, 22.978% MFU, and 151.438 GiB peak HBM. Ratio-of-medians throughput was 1.00476x; paired-step median was 1.00296x with 17/20 treatment-faster steps. Median loss was 6.7676 versus 6.7655. Both arms completed cleanly without retries.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/one-node-ab-20260810.json`; [Iris job](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-epilogue-gb200-20260810-coord); [XLA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-epilogue-gb200-20260810-xla); [CODA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-epilogue-gb200-20260810-quack-coda).
- Interpretation: Exploratory/neutral. The +0.48% ratio-of-medians is below the approximately 0.78% same-code placement spread and far below the 1.57% one-pair promotion threshold. The 4.79 GiB peak-HBM reduction is promising but comes from one pair. Do not spend a full rack on this reference-VJP implementation.
- Next action: If continuing, fuse the reverse RMS partial reduction and direct GatedNorm gradients; require greater than 1.02x exact-shape forward-plus-backward before another whole-model run.

### 2026-08-10 13:49 PDT - GMRMS-005 analytic reverse gate

- Hypothesis: Retaining the rank-128 preactivation in the normalized GEMM epilogue and applying explicit GatedNorm/RMS derivatives avoids enough forward replay to preserve the forward win in training.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: `CUDA_VISIBLE_DEVICES=0 uv run --package marin-levanter --extra=gpu python -m experiments.grug.moe_hero_fsdp.bench_rms_gated_norm --warmup 5 --steps 20 --output-jsonl /tmp/coda-rms-analytic-gb200.jsonl`.
- Config: One GB200; BF16; input `[16,4096,6144]`; GatedNorm rank 128; FP32 RMS statistics and saved preactivation; QuACK tile `[256,128]`, cluster `[2,1,1]`; 5 warmups and 20 timed iterations.
- Result: Forward was 1.280 ms for XLA and 1.097 ms for CODA (1.167x). Forward plus backward was 3.352 ms for XLA and 4.114 ms for CODA (0.815x). The analytic reverse is 8.5% faster than the earlier 4.497 ms reference-VJP candidate but misses the 1.02x promotion gate. Focused FP32/BF16 analytic-reverse parity tests against JAX autodiff pass.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/exact-gb200-analytic-20260810.jsonl`.
- Interpretation: The forward hypothesis remains supported, but JAX-level reverse algebra does not retain the gain. The remaining opportunity requires physical fusion of the dSiLU GEMM and RMS reduction rather than another algebra-only custom VJP.
- Next action: Stop. Do not launch another one-node or rack-scale model A/B for this implementation.

### 2026-08-10 14:29 PDT - GMRMS-006 competitive reverse gate

- Hypothesis: Producing RMS row and norm-weight partials in the `dp @ Wdown.T` epilogue, and fusing `dq @ Wup.T` with dSiLU, can make the full boundary reverse faster than XLA.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: `CUDA_VISIBLE_DEVICES=0 uv run --package marin-levanter --extra=gpu python -m experiments.grug.moe_hero_fsdp.bench_rms_gated_norm --warmup 5 --steps 30 --output-jsonl /tmp/coda-rms-competitive-gb200.jsonl`.
- Config: One GB200; BF16; input `[16,4096,6144]`; GatedNorm rank 128; forward tile `[256,128]`, cluster `[2,1,1]`; backward producer tile `[64,256]`, cluster `[1,1,1]`; FP32 RMS statistics and partial reductions.
- Result: Forward improved from 1.2753 ms to 1.0740 ms (1.1875x). Forward plus backward improved from 3.3273 ms to 3.1211 ms (1.0661x). Counting the extra forward under `recompute_all`, the region improved from 4.6026 ms to 4.1951 ms (1.0971x). Output max/mean absolute deviation was 0.015625/1.43e-5. Focused FP32/BF16 forward and reverse tests pass. The `[128,128]` producer failed the exact-shape gradient gate after the fused dSiLU integration; `[64,256]` produced stable expected deviations and is the default.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/exact-gb200-competitive-20260810.jsonl`.
- Interpretation: The physical CODA backward passes the 1.02x component promotion gate. The rematerialized per-step saving projects to approximately 0.22% on the historical FSDP64 hero and 0.29% on EP64, before whole-model effects.
- Next action: Run one serial same-node training pair. Do not spend a rack unless the one-node result exceeds the measured placement-noise floor.

### 2026-08-10 14:31 PDT - GMRMS-007 competitive one-node A/B submitted

- Hypothesis: The 1.097x rematerialized boundary speedup produces a measurable whole-step gain in the 48-layer Grug screen.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: CoreWeave Iris coordinator job `/dlwh/coda-rms-competitive-gb200-20260810-coord`.
- Config: Serial fresh-process arms on one four-GPU GB200 node; d6144/L48/B64/S4096/E8/top4; 25 steps; XLA then competitive CODA; score median `throughput/duration` at steps 5-24.
- Result: XLA completed 25/25 steps at 14.9160 s/step and 17,574.7 tok/s over steps 5-24. The CODA arm segfaulted on all four ranks before step 0. Its custom VJP returned parameter gradients varying over `replica_dcn`, `data`, and `expert` even though the corresponding primal inputs were replicated.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/one-node-competitive-ab-20260810.json`; [Iris job](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-competitive-gb200-20260810-coord).
- Interpretation: Unscorable. The later PJIT and coordination failures followed the manual-axis contract violation; four simultaneous standalone kernel processes were stable.
- Next action: Sum `dGamma`, `dWdown`, and `dWup` over the batch mesh axes inside the custom VJP, then require an exact four-GPU sharded replay before resubmitting.

### 2026-08-10 14:58 PDT - GMRMS-008 sharded custom-VJP fix

- Hypothesis: Explicit batch-axis `psum`s make gradients of the replicated norm and GatedNorm weights invariant under `shard_map` and remove the pre-step crash.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: Four-device `shard_map` forward-plus-backward replay on one reserved GB200 node.
- Config: Four local GB200s; exact 65,536 rows by 6,144 hidden dimensions per device; rank 128; weights replicated; activations sharded over the four-device `data` axis; ten timed iterations.
- Result: Before the fix, JAX reported that `dWdown` varied over the batch axes while its primal input was replicated. After adding `psum` for `dGamma`, `dWdown`, and `dWup`, the exact replay returned sharding specs `P(batch,None,None)`, `P(None)`, `P(None,None)`, and `P(None,None)`. CODA measured 3.4913 ms versus 3.7073 ms for XLA (1.0619x).
- Interpretation: The structural crash is reproduced and fixed. The required collectives preserve the component speedup.
- Next action: Submit a fresh serial same-node pair with a new run identity.

### 2026-08-10 15:00 PDT - GMRMS-009 fixed competitive one-node A/B submitted

- Hypothesis: The fixed distributed custom VJP completes training and produces a measurable same-node throughput gain.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: CoreWeave Iris coordinator job `/dlwh/coda-rms-competitive-v2-gb200-20260810-coord`.
- Config: Serial fresh-process arms on one four-GPU GB200 node; d6144/L48/B64/S4096/E8/top4; 25 steps; XLA then fixed competitive CODA; score median `throughput/duration` at steps 5-24.
- Result: Both arms completed 25/25 steps without faults. Over steps 5-24, XLA measured 15.25094 s/step, 17,188.72 tok/s, 22.9060% MFU, and 156.2274 GiB peak HBM. CODA measured 14.49700 s/step, 18,082.66 tok/s, 24.0973% MFU, and 152.0130 GiB peak HBM. Ratio-of-medians throughput improved 1.05201x and the treatment won 20/20 paired steps, but the loss series diverged after the initially aligned step 0: scored median absolute delta 0.07218, or 1.0785% relative, and maximum delta 0.22363.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/one-node-competitive-v2-ab-20260810.json`; [Iris job](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-competitive-v2-gb200-20260810-coord); [XLA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-v2-gb200-20260810-xla); [CODA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-v2-gb200-20260810-quack-coda).
- Interpretation: The 5.20% throughput result is well above placement noise but is not correctness-valid. A four-device FP32 replay through the original FSDP parameter shardings matched XLA gradient norms within 1e-7, ruling out the distributed `psum` and outer reduce-scatter as the cause. The remaining error came from changed BF16 rounding sites in the delayed forward and analytic reverse.
- Next action: Restore the stock BF16 forward exactly and isolate each backward rounding site at the full local hero shape.

### 2026-08-10 16:04 PDT - GMRMS-010 exact-forward fused-row gate

- Hypothesis: Keeping the stock BF16 forward and gate reverse while retaining the CODA producer row reduction removes the training-semantics blocker without giving up the custom-VJP whole-model effect.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.grug.moe_hero_fsdp.bench_rms_gated_norm --warmup 5 --steps 30 --output-jsonl /tmp/coda-fused-row.jsonl`.
- Config: One GB200; BF16 input `[16,4096,6144]`; FP32 norm gain; exact stock forward; fused dSiLU; CODA producer tile `[256,128]`, cluster `[2,1,1]`; BF16 producer handoff; fused RMS row partials; separate `dgamma` reduction; 5 warmups and 30 timed iterations.
- Result: Forward output was bit-identical. Relative gradient L2 deviations versus XLA were 0.000398 for `dx`, 0.000388 for `dgamma`, 0.004643 for `dWdown`, and zero for `dWup`. Forward plus backward measured 3.3524 ms versus 3.3587 ms for XLA, a 1.00189x tie. The original `[64,256]` producer makes the row reduction invalid, while `[256,128]` makes the row reduction correct; QuACK's composable row and column reducers cannot both be used correctly under one epilogue schedule, so `dgamma` is reduced separately.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/exact-forward-fused-row-gb200-20260810.jsonl`.
- Interpretation: The candidate now preserves forward semantics and has bounded sub-percent gradient differences expected from a custom BF16 GEMM. Component timing alone is neutral, but it does not measure the custom-VJP/rematerialization effect that produced the earlier whole-step result.
- Next action: Run one serial same-node training pair. Require exact step-0 loss, materially reduced trajectory drift, and throughput above the 0.78% placement floor.

### 2026-08-10 16:07 PDT - GMRMS-011 corrected competitive one-node A/B submitted

- Hypothesis: The exact-forward fused-row candidate retains a measurable whole-step speedup while eliminating the v2 training-semantics failure.
- Commit Hash: `499340ce18` plus the uncommitted `codex/research/grug-moe-coda-rms` prototype.
- Command: CoreWeave Iris coordinator job `/dlwh/coda-rms-competitive-v3-gb200-20260810-coord`.
- Config: Serial fresh-process arms on one four-GPU GB200 node; d6144/L48/B64/S4096/E8/top4; 25 steps; XLA then corrected CODA; score median `throughput/duration` at steps 5-24.
- Result: Both arms completed 25/25 steps; all ranks exited zero and no recovery was needed. Over steps 5-24, XLA measured 15.33262 s/step, 17,097.15 tok/s, 22.7840% MFU, and 156.2274 GiB peak HBM. CODA measured 14.84395 s/step, 17,659.99 tok/s, 23.5340% MFU, and 156.2550 GiB peak HBM. Ratio-of-medians throughput improved 1.03292x; the paired median improved 1.03391x, treatment won 20/20 steps, and every paired step exceeded the 0.78% placement floor. Step-0 loss was identical. The scored median absolute loss delta was 0.005920, or 0.07662% relative; the maximum relative excursion was -0.36482% at step 9 and the final delta was +0.05181%.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/one-node-competitive-v3-ab-20260810.json`; [Iris coordinator](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-competitive-v3-gb200-20260810-coord); [Iris child](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-competitive-v3-gb200-20260810-coord%2Fgrug-train-coda-rms-competitive-v3-gb200-20260810); [XLA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-v3-gb200-20260810-xla); [CODA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-v3-gb200-20260810-quack-coda).
- Interpretation: Promotable one-node performance result. The corrected forward removes v2's semantic change, the loss trajectory is bounded and non-monotonic, and the 3.29% throughput win is consistently above same-node placement noise. Peak HBM is unchanged. This does not by itself justify a 64-node hero launch while the required rack topology is capacity-constrained.
- Next action: Preserve the one-node result and use it as the gate for a matched full-hero pair when a rack allocation is available.

### 2026-08-10 16:58 PDT - GMRMS-012 concurrent full-hero rack A/B submitted

- Hypothesis: The corrected CODA reverse preserves a multi-percent whole-step gain at the production FSDP64 shape, where communication and expert work dilute the norm boundary.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd` plus the user-approved dirty prototype. Tracked diff SHA-256: `8efd5122e9c7e1b24eb823c4c2cd93533ed9317488ce3f511e9f6a916ada30a9`; untracked manifest SHA-256: `96b1eaf32f6602eb72ddb6e62cd014cfb69bd7a84488ec829059c67c26260cc4`.
- Command: Coordinator `/dlwh/coda-rms-competitive-rack-v1-gb200-20260810-coord`, launched with `python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_rack_ab --run-id coda-rms-competitive-rack-v1-gb200-20260810 --num-steps 25 --priority production --version dev --max-concurrent 2 --run`.
- Config: Two independent production-priority 16-node/64-GPU gangs, one XLA and one corrected CODA; d6144/L48/B1024/S4096/E128/top4; 25 steps; no checkpoints; W&B resume `allow`; score steps 5-24. Artifact roots are `users/dlwh/grug/coda-rms-competitive-rack-v1-gb200-20260810-{xla,quack-coda}/dev`.
- Result: Submitted. At 16:55 PDT A08 reported 16 free GPUs, and an existing 16-node gang could fit only 2/16 nodes in `multinode-nvlink-ib`; both rack arms are expected to wait for capacity.
- Artifact: [Iris coordinator](https://iris.oa.dev/#/job/%2Fdlwh%2Fcoda-rms-competitive-rack-v1-gb200-20260810-coord); intended [XLA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v1-gb200-20260810-xla); intended [CODA W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v1-gb200-20260810-quack-coda).
- Interpretation: Pending. Capacity wait is not a benchmark failure; do not mutate or restart the cluster.
- Next action: Babysit at 120 seconds and then 570-second cadence. Require both arms to complete before scoring throughput and loss drift.

### 2026-08-10 17:48 PDT - GMRMS-012 infrastructure retry 1 submitted

- Hypothesis: The first instance's two independent pre-step JAX coordination failures are transient infrastructure faults rather than a training-code failure.
- Source: Same base SHA and training implementation as the first instance. Runtime-code manifest SHA-256: `5830951ab5f5ee3961f142a5a474e3a13b993a32a6b40306f0cd5d7c7347d48b`; the only intervening source change adds a launcher-only `--arm` recovery selector.
- Result before retry: XLA failed during W&B multihost synchronization after global task 31 reconnected with a different incarnation. CODA later failed registration with `DEADLINE_EXCEEDED`. Both supervisors returned `rc=-6`, `last_step=None`; XLA W&B had zero rows and CODA W&B was never created. Iris wrappers masked both arm failures as success.
- Command: Directly resubmitted `/dlwh/coda-rms-competitive-rack-v1-gb200-20260810-coord` with the identical two-arm command, W&B IDs, artifact roots, and production-priority rack gangs.
- Recovery policy: This is the single direct retry permitted for a recoverable pre-step infrastructure fault. Do not retry again if the same coordination failure family repeats.
- Next action: Babysit through arm-level terminal state, ignoring wrapper success unless W&B and supervisor outcomes confirm 25 completed steps.

### 2026-08-10 18:41 PDT - GMRMS-012 blocked by repeated JAX coordination failure

- Result: Retry 1 placed both 16-node gangs concurrently, but both again failed before step 0. CODA aborted through `CoordinationServiceAgent::SetError`; XLA timed out in `CoordinationService/RegisterTask`. Both supervisors reported `rc=-6`, `last_step=None`, and exhausted a zero-restart budget. Both W&B runs finalized `crashed` with zero rows.
- Orchestrator caveat: Iris reported both children and the coordinator as `succeeded` with exit 0 because the supervised sweep wrapper contained the arm failures. The arm supervisor and W&B rows are the decisive signals for this benchmark.
- Interpretation: Infrastructure-blocked and unscorable. The two-rack placement itself worked twice, including confirmed 32-node/128-GPU overlap. No loss, step duration, throughput, or MFU row exists for either implementation.
- Recovery: Stopped after one direct retry, as required for a repeated coordination-family failure. No cluster, node, or pod state was mutated.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/rack-v1-ab-20260810.json`; monitoring state `scratch/20260810-1659_monitoring_state.json`; [Echo incident](https://echo.oa.dev/wiki/102).
- Next action: Do not spend another rack retry on the same launch path. Resolve or bypass the JAX/Iris coordination failure, then rerun the matched FSDP64 pair with new W&B identities.

### 2026-08-10 22:57 PDT - GMRMS-012 JAX rendezvous collision diagnosed

- Root cause: The XLA and CODA arms were sibling child jobs under one Iris root job. `IrisContext.namespace` uses the root job namespace, while JAX coordinator discovery keys only on `jax_coordinator-attempt-<attempt_id>`. Both child jobs were at attempt 0, so they shared one endpoint-registry key despite having different child job IDs.
- First run evidence: XLA registered `10.186.213.93:14807` and all 64 XLA ranks used it. CODA registered `10.186.210.69:54705`, but 7 CODA ranks used the XLA address and 57 used the CODA address. CODA process 31 duplicated XLA process 31 in the XLA world, producing the different-incarnation failure; the split CODA world later timed out in `RegisterTask`.
- Retry evidence: CODA registered `10.186.213.93:60457` and all 64 CODA ranks used it. XLA then registered `10.186.210.69:39537`, but 63 XLA ranks used the CODA address and only XLA rank 0 used its own address. CODA aborted after duplicate XLA process IDs joined; XLA rank 0 timed out alone after 1,800 seconds.
- Kernel disposition: Both failures occurred during JAX registration before W&B step 0 or model compilation. They do not implicate the CODA implementation.
- Secondary issue: `_run_grug_sweep_local` logs failed arm outcomes but does not raise. A failed one-arm sweep therefore exits zero and Iris reports success.
- Fix: Include full child job identity and attempt ID in the coordinator endpoint key. Add a regression test with two sibling child jobs at attempt 0. Until then, isolate the arms under separate top-level root jobs or run them serially.
- Artifact: [Resolved Echo incident](https://echo.oa.dev/wiki/102).

### 2026-08-10 23:04 PDT - GMRMS-013 isolated-root full-rack A/B launch contract

- Goal: Measure the corrected CODA RMS-GatedNorm implementation against XLA at the production FSDP64 shape without the sibling-job endpoint collision.
- DRI: David Hall.
- Source: `a5f0269edc35a3766958adb494cef7d371632ebd` plus the user-approved dirty prototype. Tracked diff SHA-256: `8efd5122e9c7e1b24eb823c4c2cd93533ed9317488ce3f511e9f6a916ada30a9`.
- Isolation: Submit each arm through a separate top-level Marin coordinator. Their Iris root-job namespaces and JAX coordinator registry keys cannot overlap.
- Baseline command: `.venv/bin/python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_rack_ab --run-id coda-rms-competitive-rack-v2-baseline-gb200-20260810 --num-steps 25 --priority production --arm xla --version dev --max-concurrent 1 --run`.
- Treatment command: `.venv/bin/python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_rack_ab --run-id coda-rms-competitive-rack-v2-treatment-gb200-20260810 --num-steps 25 --priority production --arm quack-coda --version dev --max-concurrent 1 --run`.
- Hardware: Each arm requests one 16-node/64-GPU GB200 rack, four JAX processes per node, FSDP64. Arms may overlap when the scheduler can place both gangs.
- Baseline identities: W&B `coda-rms-competitive-rack-v2-baseline-gb200-20260810-xla`; artifact root `users/dlwh/grug/coda-rms-competitive-rack-v2-baseline-gb200-20260810-xla/dev`.
- Treatment identities: W&B `coda-rms-competitive-rack-v2-treatment-gb200-20260810-quack-coda`; artifact root `users/dlwh/grug/coda-rms-competitive-rack-v2-treatment-gb200-20260810-quack-coda/dev`.
- Training contract: `initialize_from=None`; final step 25; W&B resume `allow`; checkpoints disabled because this is a 25-step metrics-only diagnostic.
- Capacity at validation: A08 reported 20 free of 804 GB200 GPUs, with 704 held by production, 64 interactive, and 16 batch. Scheduling wait is expected until two rack-local gangs fit.
- Babysitting: 120-second immediate gate, then 570-second cadence through both arm-level terminal states. Score steps 5-24 only after each W&B run has 25 rows and each supervisor reports completion.

### 2026-08-10 23:06 PDT - GMRMS-013 isolated roots submitted

- Hypothesis: Separate top-level Iris roots give the two 64-rank worlds distinct coordinator registry namespaces and eliminate the sibling endpoint collision.
- Source: Same code and training configuration recorded in the GMRMS-013 launch contract above.
- Command: Submitted baseline coordinator `/dlwh/coda-rms-competitive-rack-v2-baseline-gb200-20260810-coord` and treatment coordinator `/dlwh/coda-rms-competitive-rack-v2-treatment-gb200-20260810-coord` 0.811 seconds apart.
- Config: One production-priority 16-node/64-GPU child per root; baseline `--arm xla`, treatment `--arm quack-coda`; 25 steps; no checkpoints; score steps 5-24 only after 25 W&B rows and completed supervisor outcomes.
- Result: Both top-level coordinators entered `running` at submission. Child placement, distinct coordinator addresses, W&B freshness, and arm-level outcomes are pending the 120-second gate.
- Monitoring state: `scratch/20260810-2306_monitoring_state.json`.
- Next action: Run the 120-second gate, then monitor at 570-second cadence without cluster mutation.

### 2026-08-10 23:30 PDT - GMRMS-013 baseline reached step 5

- Result: The baseline child placed all 16 nodes and formed one 64-rank JAX world. Fifty rank initialization records observed so far all resolve coordinator `10.186.213.93:19343`; no alternate endpoint or coordination fault appears.
- W&B: [baseline run](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v2-baseline-gb200-20260810-xla) has five rows, `global_step` 0-4. Step 4 measured 42.3824 seconds, 98,963.4 tokens/s, 8.2557% MFU, loss 8.138529, and 142.046 GiB peak HBM.
- Placement: Treatment remains a 16-task Kueue-gated gang and has no W&B identity. The treatment coordinator address and cross-root distinctness check remain pending capacity.
- Interpretation: The isolated baseline root has passed the failure window that blocked both sibling-root attempts and is producing real training metrics. This is a liveness milestone, not a final throughput score.
- Next action: Continue to arm-level baseline completion and treatment placement; score only after both runs contain 25 rows and both supervisors report completion.

### 2026-08-10 23:41 PDT - GMRMS-013 baseline completed

- Result: The XLA baseline completed 25/25 steps on 16 nodes. Iris reported 16/16 tasks succeeded, W&B finalized `finished` with 25 rows, and rank logs reported `outcome=completed attempts=1 faults=[]`.
- Score, steps 5-24: Median duration 17.299719 seconds, throughput 242,449.447 tokens/s, MFU 20.225470%, and peak HBM 142.045727 GiB. Loss moved from 11.807305 at step 0 to 6.067012 at step 24; scored median loss was 6.519599.
- W&B: [baseline run](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v2-baseline-gb200-20260810-xla).
- Placement: The treatment child acquired the released rack and reached 16/16 running tasks. Its JAX endpoint and first W&B row are still pending.
- Interpretation: Separate root isolation succeeded for the baseline arm. The prior pre-step coordination failure did not recur.
- Next action: Verify that the treatment world uses an address distinct from `10.186.213.93:19343`, then babysit it to the same 25-row/completed-supervisor contract before comparing arms.

### 2026-08-10 23:52 PDT - GMRMS-013 treatment passed isolation and step gates

- Isolation: All 64 observed treatment rank initialization records use `10.186.210.69:21487`, distinct from the baseline address `10.186.213.93:19343`. No treatment rank used the baseline endpoint.
- Progress: The treatment child is healthy with 16/16 tasks running and has reached `global_step=10`. W&B has 11 fresh rows; step 10 measured 16.895283 seconds, 248,252.964 tokens/s, 20.709607% MFU, loss 6.917175, and 138.242128 GiB peak HBM.
- W&B: [treatment run](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v2-treatment-gb200-20260810-quack-coda).
- Interpretation: Separate top-level roots eliminated the sibling endpoint collision for both arms. The treatment has passed the prior pre-step failure window and is producing a matching loss trajectory.
- Next action: Continue to 25 rows and require `outcome=completed attempts=1 faults=[]` before scoring steps 5-24.

### 2026-08-11 00:05 PDT - GMRMS-013 isolated-root rack A/B completed

- Result: Both arms completed 25/25 steps with 16/16 Iris child tasks exiting zero, 25-row `finished` W&B runs, and supervisor outcomes `completed attempts=1 faults=[]`. The arms trained sequentially because only one rack was initially available; treatment training began after the baseline's final metric row.
- Isolation: Baseline used `10.186.213.93:19343` in 50 observed rank initialization records. Treatment used `10.186.210.69:21487` in all 64 records. No mixed endpoint was observed, confirming that separate top-level roots avoided the sibling registry collision.
- Score, steps 5-24: XLA measured 17.299719 seconds, 242,449.447 tokens/s, 20.225470% MFU, and 142.045727 GiB peak HBM. CODA measured 16.922891 seconds, 247,848.012 tokens/s, 20.675825% MFU, and 138.242128 GiB peak HBM.
- Comparison: CODA improved ratio-of-medians throughput by 2.22668% and reduced median duration by 2.17824%. The paired median throughput ratio was 1.02282x; CODA won 20/20 scored steps, and every step exceeded the 0.78% placement floor. Peak HBM fell by 3.80360 GiB and MFU rose 0.45036 percentage points.
- Loss parity: Step-0 loss was exactly equal at 11.807305. Across steps 5-24, median signed delta was +0.00001597, median absolute delta was 0.00319409 (0.04715%), and the maximum relative excursion was +0.44807% at step 14. Final treatment-minus-baseline loss was +0.00131893.
- Artifact: `.agents/logbooks/artifacts/grug-moe-coda-rms/rack-v2-isolated-ab-20260811.json`; monitoring state `scratch/20260810-2306_monitoring_state.json`; [baseline W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v2-baseline-gb200-20260810-xla); [treatment W&B](https://wandb.ai/marin-community/marin_moe/runs/coda-rms-competitive-rack-v2-treatment-gb200-20260810-quack-coda).
- Interpretation: The CODA RMS-GatedNorm treatment produces a repeatable full-rack speedup well above the placement floor, lowers peak HBM, and preserves the baseline loss trajectory within sub-0.5% relative drift. No recovery or cluster mutation was needed.
- Next action: Preserve this result as the production-shape performance gate for promotion or a longer stability run.
