---
topic: mxfp8-wdown-first-backward
issue: https://github.com/marin-community/marin/issues/7271
description: Reproduction boundary for the hybrid MXFP8 first-backward failure on B200
author: Marin
---

# Hybrid MXFP8 `w_down` first-backward failure

## Result

A fresh B200 replay of commit [`abae0fd`](https://github.com/marin-community/marin/commit/abae0fd030b2f02d2fdb2ea7b226f91c2e75072e) reproduced a finite first forward followed by `NaN loss at step 2` on every participating rank. This places the failure in the first backward pass or update. Tracking was disabled for the replay, so it did not independently identify a parameter.

[Historical telemetry from the same code](https://github.com/marin-community/marin/issues/7271#issuecomment-5018066223) identifies `stacked_blocks.stacked.mlp.expert_mlp.w_down` as the first nonfinite gradient; `w_gate` and `w_up` remain finite. Keep these as separate claims: the replay confirms timing, while the earlier telemetry supplies parameter localization.

## Reproduce

1. Check out [`abae0fd`](https://github.com/marin-community/marin/commit/abae0fd030b2f02d2fdb2ea7b226f91c2e75072e).
2. Set `MXFP8_QUALITY_ARM=mxfp8`, `MXFP8_QUALITY_STEPS=2`, and a unique `MXFP8_QUALITY_PAIR_ID`.
3. Run `python -m experiments.grug.moe.launch_mxfp8_quality` in a B200 environment.
4. Observe a finite first loss followed by `NaN loss at step 2` on every participating rank.

The launcher fixes the model and optimizer configuration: d2560, 26 layers, 128 experts, top-4 routing, expert hidden size 1280, shared-expert hidden size 2560, sequence length 4096, global batch 512, and seed 0. The logical mesh is `replica=2,data=2,expert=8`, with ring MoE, scanned layers, recompute-all, the full MuonH schedule prefix, grouped MXFP8 expert GEMMs, per-tensor dense FP8 GEMMs, and the XLA MXFP8 producer.

## Reduction boundary

A disposable single-device reduction tested the exact `E16/M262144/D2560/F1280` grouped expert MLP feeding a per-tensor dense FP8 GEMM. Native execution, an all-finite gradient consumer, unconditional BF16 `w_down` recomputation, and conditional fused-or-BF16 recomputation all produced finite `w_down` gradients. The native and consumer arms retained the same six FP8 custom calls.

The minimized combined graph is therefore a negative boundary, not a reproduction. The trigger still requires some additional property of the full training graph. [Prior stabilization evidence](https://github.com/marin-community/marin/issues/7271#issuecomment-5018421682) also shows that reading the fused `w_down` gradient or conditionally selecting a BF16 recomputation changes the outcome, consistent with a graph-liveness or scheduling interaction rather than a proven operator-local root cause.
