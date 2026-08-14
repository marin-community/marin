---
topic: grug-moe-coda-rms
issue: https://github.com/marin-community/marin/issues/8152
description: Evaluate CODA-style RMS-GatedNorm fusion for Grug MoE on GB200.
author: dlwh
---

# Grug MoE CODA RMS: Task Logbook

## Scope

- Goal: Improve Grug MoE training throughput by fusing the RMS-GatedNorm boundary on GB200.
- Primary metrics: Steps 5-24 median duration, tokens/s, MFU, peak HBM, and loss trajectory.
- Fixed production case: 16 nodes, 64 GB200 GPUs, EP64, d6144, 48 layers, sequence 4096, batch 1024, 192 experts, top-4.
- Promotion gate: More than 0.78% throughput improvement with acceptable loss and memory behavior.
- Coordinating issue/PR: [#8152](https://github.com/marin-community/marin/issues/8152), [#8153](https://github.com/marin-community/marin/pull/8153).
- Earlier logbook snapshot: [`e6f67c0484`](https://github.com/marin-community/marin/blob/e6f67c0484/.agents/logbooks/grug-moe-coda-rms.md).

## Current TL;DR

No candidate met the EP64 production gate. The first FSDP64 experiment reported a 2.23% throughput gain, but the EP64 port exposed 9.44 GiB of extra executable scratch from full-width tensors crossing opaque custom-call boundaries. Keeping GatedNorm reverse in XLA and fusing only RMS reverse fit EP64 and saved 2.94 GiB, but throughput fell 0.28%.

The best boundary-merged forward treatment required `NCCL_BUFFSIZE=262144` to fit. Its final tuned, concurrent 25-step EP64 A/B improved ratio-of-medians throughput by 0.478%, below the 0.78% floor, and used 7.344 GiB more peak HBM. Loss stayed finite and aligned. The treatment is not promoted, and no clean production branch was extracted.

## Decision Log

- Preserve the full investigation on [`codex/grug-coda-rms-backward`](https://github.com/marin-community/marin/tree/codex/grug-coda-rms-backward).
- Do not promote #8153: the EP64 candidates either regressed throughput, failed memory fit, or missed the 0.78% floor.
- Treat the 256 KiB NCCL buffer as a diagnostic fit workaround, not a production recommendation. It reduced the integrated treatment peak by about 2.73 GiB and allowed training to start, but did not produce a sufficient throughput margin.
- Keep the 1.0% strict four-device numerical threshold. The integrated treatment's `w_up` gradient remained a finite 1.149% scaled-error diagnostic near miss.

## Negative Results Index

| Candidate | EP64 result | Decision |
| --- | --- | --- |
| Split fused GatedNorm/RMS reverse | 2.36% slower, +4.92 GiB HBM | Reject |
| Fused `w_down` gradient | 3.09% slower, +4.29 GiB HBM; loss diverged at step 19 | Reject |
| RMS-only fused reverse | 0.28% slower, -2.94 GiB HBM; stable loss | Memory win only |
| Integrated delayed forward, 1 MiB NCCL | OOM before step 0 at 184.29 GiB | Reject |
| Integrated delayed forward, 256 KiB NCCL | 0.76% faster before tuning, +7.34 GiB HBM | Below floor |
| Tuned integrated delayed forward | 0.48% faster, +7.34 GiB HBM; stable loss | Reject |

## Entry Log

### 2026-08-14 - GMRMS-101 split fused reverse EP64 A/B

- Hypothesis: Merging the backward gate and RMS boundaries removes the scratch overhead identified in #8153 while preserving the earlier FSDP speedup.
- Commit Hash: `2f7d8fbdab9350ef185b58cbe1e21904eb9a9e42`.
- Config: Concurrent 16-node/64-GPU arms; EP64; batch 1024; sequence 4096; 25 steps; `NCCL_BUFFSIZE=1048576`; score steps 5-24.
- Result: XLA measured 252,895.72 tokens/s and 23.61359% MFU. Treatment measured 246,817.46 tokens/s and 23.04604% MFU. Paired throughput fell 2.362%; treatment peak HBM increased 4.921875 GiB. Loss remained finite.
- W&B: [XLA](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-fused-reverse-nccl1m-ab-v1-xla-20260814), [treatment](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-fused-reverse-nccl1m-ab-v1-quack-coda-20260814).
- Interpretation: The merged reverse fits with reduced NCCL staging, but it is slower and uses more HBM.
- Next action: Reduce the custom region to RMS reverse only.

### 2026-08-14 - GMRMS-102 RMS-only reverse EP64 A/B

- Hypothesis: Keep all GatedNorm weight gradients in XLA and use one custom call only for RMS `dx` and `dgamma`.
- Commit Hash: `556cbfc279`.
- Config: Concurrent 16-node/64-GPU arms; 25 steps; `NCCL_BUFFSIZE=1048576`; score steps 5-24.
- Result: XLA measured 256,098.8 tokens/s and 23.91267% MFU. Treatment measured 255,020.6 tokens/s and 23.81199% MFU. Paired throughput fell 0.28285%. Treatment peak HBM fell 2.9375 GiB. Final treatment-minus-XLA loss was +0.00730.
- W&B: [XLA](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-rms-only-nccl1m-ab-v1-xla-20260814), [treatment](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-rms-only-nccl1m-ab-v1-quack-coda-20260814).
- Interpretation: One compiler-opaque boundary fixes the EP64 memory problem, but the implementation does not improve throughput.
- Next action: Recover forward savings without materializing normalized activations.

### 2026-08-14 - GMRMS-103 integrated forward numerical gate

- Hypothesis: Fuse the delayed down projection and gated output so normalized activations and gates never cross custom-call boundaries.
- Commit Hash: `4ff2c3284ba7bb4816f6a3a4591f36f6db46f3af`.
- Config: Four GB200 GPUs; XLA versus `quack_coda`; full forward and gradients.
- Result: Output, `x`, `gamma`, and `w_down` passed the 1.0% scaled-error gate at 0.741%, 0.987%, 0.532%, and 0.781%. `w_up` was finite but missed at 1.149%. The optional two-update loss probe remained finite.
- Interpretation: Eligible only for the previously defined 1.25% diagnostic band. The strict gate remains failed.
- Next action: Test EP64 memory fit without launching a long comparison.

### 2026-08-14 - GMRMS-104 integrated forward EP64 fit failure

- Hypothesis: Removing the forward normalized/gate boundaries closes the remaining EP64 scratch gap.
- Commit Hash: `4ff2c3284ba7bb4816f6a3a4591f36f6db46f3af`.
- Config: 16 nodes/64 GPUs; two steps; `NCCL_BUFFSIZE=1048576`.
- Result: Five built-in attempts failed before step 0 in NCCL AllToAll allocation. Peak sampled HBM was 184.2855 GiB/GPU; W&B had zero training rows.
- Interpretation: Forward custom calls reintroduced enough opaque-boundary memory to exceed EP64 headroom.
- Next action: Test a smaller NCCL staging allocation as one controlled variable.

### 2026-08-14 - GMRMS-105 256 KiB NCCL fit workaround

- Hypothesis: Reducing NCCL staging from 1 MiB to 256 KiB provides the missing 2-3 GiB of runtime headroom.
- Commit Hash: `4ff2c3284ba7bb4816f6a3a4591f36f6db46f3af`.
- Command: Same canonical EP64 two-step command as GMRMS-104 with `NCCL_BUFFSIZE=262144` and a fresh run identity.
- Result: Coordinator and all 16 child tasks succeeded on the first attempt. Step 1 measured 34.9306 seconds, 120,075 tokens/s, and 11.2118% MFU. Peak HBM was 181.5569 GiB, 2.73 GiB below the failed 1 MiB run.
- Interpretation: The NCCL setting is a valid fit workaround for this exact treatment.
- Next action: Run a concurrent 25-step A/B with the same NCCL setting on both arms.

### 2026-08-14 - GMRMS-106 integrated forward EP64 A/B

- Hypothesis: The boundary-fused forward plus RMS-only backward exceeds the 0.78% placement floor after the fit workaround.
- Commit Hash: `4ff2c3284ba7bb4816f6a3a4591f36f6db46f3af`.
- Command: Independent XLA and `quack_coda` coordinators with fresh ports and output roots; both use `--num-steps 25 --schedule-steps 200 --batch-size 1024` and `NCCL_BUFFSIZE=262144`.
- Config: Both 16-node gangs started about one second apart and overlapped. Both completed 25 rows on their first attempts.
- Result: XLA measured 17.065824 seconds, 245,772.148 tokens/s, and 22.94844% MFU. Treatment measured 16.937524 seconds, 247,633.856 tokens/s, and 23.12227% MFU. Ratio-of-medians throughput improved 0.75749%; same-step paired median improved 0.60759%. Treatment peak HBM increased 7.34375 GiB. Loss stayed finite; final treatment-minus-XLA loss was -0.05236.
- W&B: [XLA](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-integrated-delayed-output-nccl256k-ab-v1-xla-20260814), [treatment](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-integrated-delayed-output-nccl256k-ab-v1-quack-coda-20260814).
- Interpretation: The gain missed the 0.78% floor by 0.02251 percentage points and did not justify promotion.
- Next action: Run one bounded forward-kernel autotune.

### 2026-08-14 - GMRMS-107 forward-kernel autotune

- Hypothesis: Independent launch shapes for the two forward kernels create enough margin to clear the placement floor.
- Commit Hash: `4ff2c3284ba7bb4816f6a3a4591f36f6db46f3af`.
- Config: Four GB200 GPUs; exact local shape M65536/D6144/R128; 64 W&B rows; 24 measured and 8 pruned configurations per kernel; two warmups and ten synchronized iterations.
- Result: Down+SiLU improved from 0.373457 ms to 0.358098 ms with tile 128x128 and cluster 2x1x1. Gated output improved from 0.924259 ms to 0.918755 ms with tile 128x256 and cluster 1x1x1. Projected combined latency improved 1.634%. All measured configurations passed strict parity.
- Interpretation: Apply the independent defaults and repeat the paired rack gate once.
- Next action: Re-run the exact concurrent A/B.

### 2026-08-14 - GMRMS-108 tuned integrated forward EP64 A/B

- Hypothesis: The exact-shape launch tuning turns the marginal integrated-forward result into a repeatable win.
- Commit Hash: `b67e6bca46bc91aaee39102bb66178e7ff61fd0b`.
- Command: Independent XLA and `quack_coda` coordinators; both use 16 nodes, 64 GB200 GPUs, batch 1024, 25 steps, schedule 200, and `NCCL_BUFFSIZE=262144`.
- Config: Both arms overlapped, completed on their first attempts, and produced exactly 25 W&B rows. Score window was steps 5-24.
- Result: XLA measured 17.017147 seconds, 246,475.161 tokens/s, and 23.01408% MFU. Treatment measured 16.936176 seconds, 247,653.925 tokens/s, and 23.12415% MFU. Ratio-of-medians throughput improved 0.47825%; same-step paired median improved 0.64169%; timestamp-overlap paired median improved 0.36416%. Treatment peak HBM increased from 174.221 GiB to 181.565 GiB. Loss remained finite; final treatment-minus-XLA loss was +0.01564.
- W&B: [XLA](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-integrated-forward-tuned-nccl256k-ab-v1-xla-20260814), [treatment](https://wandb.ai/marin-community/marin_moe/runs/pr8153-ep-integrated-forward-tuned-nccl256k-ab-v1-quack-coda-20260814).
- Interpretation: The tuned result is below the promotion floor on every scoring view and uses more HBM. Do not extract a production branch.
- Next action: Publish the negative result in #8152 and leave #8153 unpromoted.
