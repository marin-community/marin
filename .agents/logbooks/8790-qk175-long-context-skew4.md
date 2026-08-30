---
topic: qk175-long-context-skew4
issue: 8790
description: Production qk=1.75 context-extension run with 4x long-document sampling
author: held
---

# qk=1.75 4x Long-Context Skew: Task Logbook

## Scope

- Goal: compare a 4x long-document sampling skew with the qk=1.75 262K context-extension control and 2x treatment.
- Primary metrics: Paloma macro loss, training loss, throughput, and numerical stability.
- Constraints: preserve the control's step-156000 checkpoint, 1000-step schedule, optimizer, mesh, batch size, data offset, and phase-1 quality-by-domain weights.
- Coordinating issue: https://github.com/marin-community/marin/issues/8790

## Baseline

- Date: 2026-08-30
- Control run: `moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k`
- 2x treatment: `moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew2`
- 2x source: `1691d0a9afbfe849538e180186bec7759574cf2c`

## Entry Log

### 2026-08-30 - Launch preparation

- Hypothesis: 4x sampling of documents longer than 64K improves long-context adaptation while the phase-1 quality-by-domain distribution remains fixed.
- Commit Hash: `77d2714168eadc071fdb4f39fe1465f41c599730`
- Command: `iris --cluster=marin job run --no-wait --region us-central2 --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY <redacted> -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk175_long_context_skew`
- Config: v4-2048, qk=1.75, sequence length 262144, batch size 256, context parallelism 4, steps 156000 through 157000, 4x long-document skew.
- Result: source checkpoint metadata is present and the resolved output path is unused.
- Interpretation: the 4x treatment changes only long-document sampling relative to the completed 2x source.
- Next action: submit with production priority and verify startup.

### 2026-08-30 - Production launch

- Commit Hash: `3c82e92a3d4c0d6b286038811dc7c330a2253017`
- Command: `iris --config /home/held/marin-lcr/lib/iris/config/marin.yaml job run --no-wait --region us-central2 --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY <redacted> -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk175_long_context_skew`
- Iris job: `/held/iris-run-job-20260830-073840`
- W&B run ID: `moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4`
- Output: `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4-102e3c`
- Result: submitted in `us-central2` with production priority and non-preemptible v4-2048 resources.
- Interpretation: no training task has started. The coordinator is initializing the experiment graph.
- Caveat: the pinned branch contains an older Iris config schema. Submission used the current checkout's controller config while bundling the pinned worktree.
- Next action: verify 256-worker allocation, W&B initialization, and the first training step.
