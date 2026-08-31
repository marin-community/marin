---
topic: qk175-long-context-skew8
issue: 8813
description: Production qk=1.75 context-extension run with 8x long-document sampling
author: held
---

# qk=1.75 8x Long-Context Skew: Task Logbook

## Scope

- Goal: compare an 8x long-document sampling skew with the qk=1.75 262K context-extension control, 2x treatment, and 4x treatment.
- Primary metrics: Paloma macro loss, training loss, throughput, and numerical stability.
- Constraints: preserve the step-156000 checkpoint, 1,000-step schedule, optimizer, mesh, batch size, data offset, and phase-1 quality-by-domain weights.
- Coordinating issue: https://github.com/marin-community/marin/issues/8813

## Baseline

- Date: 2026-08-31
- Control: https://wandb.ai/marin-community/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k
- 2x treatment: https://wandb.ai/held/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew2
- 4x treatment: https://wandb.ai/held/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4
- Length-partitioned store: `datakit/store/june-67b-a2b-length64k/2026.08.24`

## Entry Log

### 2026-08-31 - Launch preparation

- Hypothesis: 8x sampling of documents longer than 65,536 tokens improves long-context adaptation while phase-1 quality-by-domain weights remain fixed.
- Commit Hash: `05ad9f338cb6d461452354509e0773fe3ba542fb`
- Command: `iris --config /home/held/marin-lcr/lib/iris/config/marin.yaml job run --no-wait --region us-central2 --priority production --no-preemptible --max-retries 100 -e WANDB_API_KEY <redacted> -- python -m experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk175_long_context_skew`
- Config: v4-2048, qk=1.75, sequence length 262,144, batch size 256, context parallelism 4, steps 156,000 through 157,000, 8x long-document skew.
- Result: the aggregate long-document token share is 43.405%, compared with 14.943% at proportional weighting. The resolved output is `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695`.
- Interpretation: the 8x treatment changes only long-document sampling relative to the qk=1.75 control.
- Next action: verify the source checkpoint and unused output, then submit with production priority.

### 2026-08-31 - Production launch

- Runtime source: `92514c5358e9cd99c89bdd4a6600ac7bbeddc147`
- Iris job: `/held/iris-run-job-20260831-170205/grug-train-moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8`
- W&B: https://wandb.ai/held/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8
- Result: Iris allocated all 256 non-preemptible workers at production priority. W&B initialized the intended run on task 57, and all workers began loading the step-156000 cooldown checkpoint. At launch verification, the job had no failed or preempted workers.
- Next action: verify the first completed training step, then monitor training and Paloma metrics.
