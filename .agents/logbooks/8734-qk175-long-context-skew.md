---
topic: qk175-long-context-skew
issue: 8734
description: Production qk=1.75 context-extension run with 2x long-document sampling
author: held
---

# qk=1.75 Long-Context Skew: Task Logbook

## Scope

- Goal: compare a 2x long-document sampling skew with the qk=1.75 262K context-extension control.
- Primary metrics: training loss, validation loss, throughput, and numerical stability.
- Constraints: preserve the control's step-156,000 checkpoint, 1,000-step schedule, optimizer, mesh, batch size, data offset, and phase-1 quality-by-domain weights.
- Coordinating issue: https://github.com/marin-community/marin/issues/8734

## Baseline

- Date: 2026-08-27
- Code ref: `origin/june_tpu_67b_a2b:experiments/grug/moe/moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon.py`
- Baseline run: `moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k`

## Entry Log

### 2026-08-27 - Launch preparation

- Hypothesis: 2x sampling of documents longer than 64K improves long-context adaptation while the phase-1 quality-by-domain distribution remains fixed.
- Commit Hash: pending launch commit
- Command: `iris --cluster=marin job run --no-wait --region us-central2 --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY <redacted> -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk175_long_context_skew`
- Config: v4-2048, qk=1.75, sequence length 262,144, batch size 256, context parallelism 4, steps 156,000 through 157,000, 2x long-document skew.
- Result: launcher and production run record prepared.
- Interpretation: pending launch validation.
- Next action: pin the source commit and resolved output path, then submit once.

### 2026-08-27 - Production launch

- Commit Hash: `1691d0a9afbfe849538e180186bec7759574cf2c`
- Iris job: `/held/iris-run-job-20260827-194841`
- W&B run ID: `moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew2`
- Output: `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew2-ec6741`
- Result: submitted in `us-central2` with production priority and non-preemptible v4-2048 resources.
- Interpretation: the coordinator is running and the 256-worker training job is waiting for the reserved v4-2048 scale group to become ready.
- Next action: verify worker startup, W&B initialization, and forward training progress.

### 2026-08-27 - Degraded v4-2048 slice

- Status: the coordinator remains running, but all 256 training tasks are pending.
- Evidence: the scheduler reports that the largest `tpu-name` group has 254 workers. The slice has 255 registered workers; TPU host `t1v-n-25bf471b-w-74` is registering as `worker-0` with empty TPU metadata, colliding with the real `worker-0`.
- Interpretation: this is a worker-identity collision caused by missing TPU worker-index metadata, not an ordinary capacity wait. No training step has started.
- Decision: keep the job unchanged while recovery authorization is pending.
- Next action: replace the degraded v4-2048 slice, then verify that all 256 workers share one TPU name before training starts.
