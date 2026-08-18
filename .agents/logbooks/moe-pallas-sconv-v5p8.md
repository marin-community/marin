---
topic: moe-pallas-sconv-v5p8
issue: https://github.com/marin-community/marin/issues/8377
description: Verify the Pallas depthwise causal convolution in the Grug MoE recipe on v5p-8.
author: held
---

# MoE Pallas SConv v5p-8: Task Logbook

## Current TL;DR

The implementation and v5p-8 smoke run are in progress. No runtime result is available yet.

## Scope

- Goal: verify that the #8331 Pallas convolution compiles and trains in the four-site Inkling SConv configuration on one v5p-8.
- Primary metrics: terminal Iris state, finite training loss, `throughput/tokens_per_second`, and final W&B state.
- Constraints: d512 May Recipe shape, kernel size 4, K/V/attention-output/MLP-output sites, identity initialization, packed-document boundary masking, 25 training steps.
- Coordinating issue: https://github.com/marin-community/marin/issues/8377

## Baseline

- Date: 2026-08-17
- Code refs: #7585 and #8331.
- Prior result: the d1024 four-site shift-and-sum SConv run reached Paloma macro loss 3.0378 versus 3.0610 without convolution. This smoke test has no quality comparator.

## Hypothesis Queue

### Active

- `MOE-PSC-001`: the four-site Pallas SConv compiles and completes 25 d512 training steps on v5p-8 with finite loss. Next test: submit the smoke run.

### Blocked

None.

### Falsified / Dead End

None.

### Promoted

None.

## Background Research Brief

- Effort: low
- Stop rule: stop after the prior Marin ablation, kernel PR, and primary architecture source agree on the placement and kernel width.
- Date: 2026-08-17

### Question

Can Mayank's Pallas TPU depthwise causal convolution replace the prior shift-and-sum SConv implementation without changing the four-site ablation semantics?

### Current Marin Context

#7585 found a positive quality signal for kernel-size-4 SConv at K, V, attention output, and MLP output. #8331 provides a TPU-specific forward/backward kernel and an XLA reference, but its binary attention mask does not encode packed-document boundaries.

### External Prior Art

Inkling uses short causal convolutions with kernel size 4 in the K stream, V stream, attention output, and MLP output. The public model and architecture description are available from [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling) and the [LMSYS implementation overview](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support).

### Negative / Failed Leads

- A binary per-token mask cannot remove every cross-document tap while preserving valid tokens near an internal packed-document boundary. The variant subtracts only the invalid lag contributions after the fused convolution.

### Evidence Map

#### Claim: four-site kernel-size-4 SConv is the correct smoke configuration

- Support:
  - #7585: completed Marin loss ablation across sites and depths.
  - Thinking Machines Inkling model: public architecture uses short convolution.
  - LMSYS Inkling overview: identifies the four sites and kernel size 4.
- Contradictions:
  - #7585 found V-only had the smallest single-site gain, and the latest recipe drops V. This smoke retains V to match the completed four-site comparator.
- Directness to Marin: #7585 uses the same Grug MoE family; #8331 targets TPU.
- Confidence: exploratory until the v5p-8 run completes.
- Action: run `MOE-PSC-001` before any full Gate 1 cells.

### Recommended Next Experiments

#### 1. v5p-8 integration smoke

- Minimum experiment: 25 d512 training steps with all four SConv sites.
- Baseline/control: successful compilation and finite loss; quality comparison is out of scope.
- Expected signal: terminal success with advancing W&B steps and nonzero throughput.
- Falsifier: compilation, forward/backward, non-finite loss, or checkpoint failure.
- Cost/risk: one short v5p-8 allocation; TPU capacity may delay scheduling.
- Sources: #7585, #8331.

### Hypothesis Queue Update

- Add: `MOE-PSC-001` integration smoke.
- Revise: none.
- Falsify / stop: none.
- Promote: full d512/d768 Gate 1 only after smoke success.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Marin SConv ablation | GitHub issue | https://github.com/marin-community/marin/issues/7585 | placement, width, prior quality | high | Direct Marin experiment |
| Pallas convolution | PR | https://github.com/marin-community/marin/pull/8331 | TPU implementation and mask contract | high | Exact branch under test |
| Inkling model | model card | https://huggingface.co/thinkingmachines/Inkling | architecture origin | medium | Public primary model artifact |
| Inkling implementation overview | official engineering post | https://www.lmsys.org/blog/2026-07-15-inkling-day0-support | four sites and kernel width | medium | External implementation description |

### Handoff

- Suggested issue `Prior work` block: #7585 established the quality signal; #8331 supplies the TPU kernel.
- Suggested logbook entry: record the exact Iris command, job ID, W&B run, loss, throughput, and terminal state.
- Open questions: TPU compile behavior and net throughput.
- Stop reason: the sources agree on the minimum integration experiment.

## Entry Log

### 2026-08-17 16:45 PDT - v5p-8 smoke snapshot

- Hypothesis: `MOE-PSC-001` completes 25 d512 training steps on one v5p-8 with finite loss.
- Commit Hash: `07c6833f4d`
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --preemptible --reserve v5p-8 --job-name moe-pallas-sconv-smoke-8377 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_pallas_sconv.launch --version dev --run`
- Config: d512, 6 layers, batch 32, sequence length 8192, 25 steps, 256 experts/top-4, MuonH, kernel-size-4 SConv at K/V/attention-output/MLP-output, v5p-8 child resources, W&B group `MOE-PSC-issue-8377`.
- Result: four segmented value/gradient and optimizer tests passed; 67 XLA reference tests passed; the two Grug variant contract cases passed; the lazy experiment graph resolves with the intended shape, resources, tracker, and step count.
- Interpretation: CPU/XLA behavior and explicit-sharding lowering are validated. TPU compilation and training remain untested until Iris submission.
- Next action: push the snapshot and submit `MOE-PSC-001` to Iris.

### 2026-08-17 18:10 PDT - Iris submission

- Commit Hash: `1619d1cf90`
- Coordinator: `/held/moe-pallas-sconv-smoke-8377`
- Training child: `/held/moe-pallas-sconv-smoke-8377/grug-train-MOE-PSC-001-d512-v5p8-smoke`
- W&B: https://wandb.ai/marin-community/marin_moe/runs/MOE-PSC-001-d512-v5p8-smoke
- Status: the child acquired one v5p-8 in us-central1-a, initialized JAX across four chips, authenticated to W&B, and began loading the training cache.
- Next action: monitor compilation, all 25 steps, final checkpoint publication, Iris success, and W&B completion.

### 2026-08-17 18:12 PDT - Explicit-mesh lowering failure

- Result: the first attempt failed before compilation with `AssertionError: (2, P(('replica_dcn', 'data', 'expert'), None, 'model'))`. Iris retried once after the process aborted.
- Interpretation: the Pallas call received globally sharded arrays under an explicit mesh. Its block mapping operates on shard-local ranks and requires a manual-axis `shard_map` boundary.
- Action: stopped the coordinator, wrapped the Pallas implementation in `shard_map`, sharded convolution channels over the model axis, and added an abstract-mesh lowering regression.
- Validation: the lowering regression and six targeted numerical, gradient, mask, and optimizer tests passed. The full targeted kernel and variant selection also passed.
- Next action: publish the fix snapshot and resubmit the smoke run.

### 2026-08-17 18:40 PDT - Mixed-precision gradient failure

- Commit Hash: `19488ef108`
- Coordinator: `/held/moe-pallas-sconv-smoke-8377-r2-central1`
- Result: the explicit-mesh fix allowed the training child to initialize W&B, load the caches, and enter the 25-step training loop. The first value-and-gradient trace then failed because the custom VJP returned an fp32 weight cotangent for a bf16 weight primal.
- Interpretation: `ShortConv` cast its fp32 parameter to the bf16 activation dtype before calling the Pallas kernel, while the kernel accumulates parameter gradients in fp32.
- Action: stopped the coordinator before its automatic retry, preserved fp32 convolution parameters at the call site, cast custom-VJP cotangents back to their primal dtypes, and extended the explicit-mesh regression through a mixed-precision value-and-gradient trace.
- Capacity: a pinned us-east5-a attempt could not provision a v5p-8 because the zone had no capacity. Subsequent smoke retries use us-central1-a.
- Validation: the mixed-precision explicit-mesh regression and six targeted numerical, gradient, mask, and optimizer tests passed.
- Next action: publish the dtype fix and resubmit in us-central1-a.
