---
topic: moe-hero-ep
issue: https://github.com/marin-community/marin/issues/7279
description: Build and validate a self-contained Grug MoE EP hero baseline on one GB200 NVL72 rack.
author: rav
---

# MoE Hero EP: Task Logbook

## Scope

- Goal: Add `experiments/grug/moe_hero_ep` from PR 7876 and select the smallest high-MFU EP baseline.
- Primary metrics: Successful steps, finite loss, MFU, tokens per second, step time, dropped assignments, and peak HBM.
- Constraints: Use one CoreWeave A08 NVL72 rack for each gate. Use 25 steps for feature gates and 200 steps for the final gate.
- Coordinating issue: [#7279](https://github.com/marin-community/marin/issues/7279).

## Current TL;DR

The branch starts at PR 7876 head `75d5c27e1`. No EP code or rack result exists on this branch yet.

## Baseline

- Date: 2026-08-01.
- Code refs: [PR 7876](https://github.com/marin-community/marin/pull/7876) at `75d5c27e1`.
- FSDP reference: Three two-rack 200-step runs averaged 19.549% MFU and 468,678 tokens/s.
- Comparison limit: The FSDP reference has a different topology and model shape. It is not an EP performance control.

## Background Research Brief

- Effort: Low.
- Stop rule: Stop when one more source does not change the feature order.
- Date: 2026-08-01.

### Question

Which parts of PR 7780 are necessary for an executable EP64 hero baseline, and which parts require separate data gates?

### Current Marin Context

PR 7820 added the self-contained FSDP hero variant. PR 7876 disabled GPU command buffers after repeated B200 hangs.

The current Levanter code already contains a `ragged_all_to_all` EP backend. Thus, the first variant does not require a new dispatch kernel.

### Internal Prior Work

- [PR 7780](https://github.com/marin-community/marin/pull/7780) added a fixed-capacity EP64 path and an EP hero template.
- [Issue 7279 result](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) measured about 12.4% MFU for ragged EP64 and 24.04% for fixed dispatch plus the custom adjoint.
- [Issue 7279 correction](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846) measured spill at capacity factor 1.0625: 20.708% MFU and 1.44% tail drops.
- PR 7780 reported a 22.398% median across three placement draws. The measured build was `c24ccfcc2`, not the PR head.
- PR 7780 reported a 0.427 percentage-point gain from a build-specific manual PGLE profile. The template did not include that profile.

### External Prior Art

- [JAX ragged all-to-all documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html) defines the existing dynamic dispatch primitive.
- [NVIDIA Megatron Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.model_parallel_config.html) exposes overlap for MoE EP communication.

### Negative Results

- Ragged EP64 measured about half the MFU of the fixed path at the target shape.
- Ring EP64 did not finish because `jit_train_step` requested 141.79 GiB.
- Token chunking, rotation, FP8 permutation wires, and weight prefetch did not improve the measured path.
- Host optimizer offload at d5120 required a 135 GiB pinned arena and measured 19.694% MFU.

### Evidence Map

#### Claim: The existing ragged backend is sufficient for the first correctness gate

- Support: PR 7876's base contains the backend and lowering tests.
- Contradiction: Prior EP64 performance was about 12.4% MFU.
- Directness to Marin: Exact repository and target model family.
- Confidence: Replicated for backend behavior, exploratory for the new hero copy.
- Action: Add only the EP variant and do a 25-step rack run.

#### Claim: Fixed dispatch and its custom adjoint require separate gates

- Support: Prior matched tests attributed large MFU changes to gather dispatch and the custom adjoint.
- Contradiction: The source result used a different research build and a manual profile.
- Directness to Marin: Exact target topology and model shape.
- Confidence: Replicated in prior research, unverified on this branch.
- Action: Add one feature per commit and do one 25-step rack run per feature.

### Recommended Next Experiments

#### MHEP-001: Ragged EP64 correctness baseline

- Minimum experiment: Run 25 steps on one NVL72 rack.
- Baseline: PR 7876 plus the copied EP variant.
- Expected signal: Terminal success, finite loss, no task retry, and recorded resource metrics.
- Falsifier: Compile failure, OOM, non-finite loss, task retry, or incomplete step 25.
- Cost or risk: One rack and one compile.
- Sources: PR 7876, PR 7780, and issue 7279.

#### MHEP-002: Fixed-capacity dispatch

- Minimum experiment: Replace only the ragged dispatch and run 25 steps.
- Baseline: MHEP-001.
- Expected signal: Terminal success and a positive MFU delta in the same measured step window.
- Falsifier: Failure or no performance gain.
- Cost or risk: One rack and one compile.
- Sources: PR 7780 and issue 7279.

#### MHEP-003: Gather dispatch and custom adjoint

- Minimum experiment: Add each optimization separately and run 25 steps after each change.
- Baseline: The prior successful feature gate.
- Expected signal: Numerical parity tests and a positive MFU delta.
- Falsifier: Numerical mismatch, failure, or no performance gain.
- Cost or risk: Two rack runs and two compiles.
- Sources: PR 7780 and issue 7279.

#### MHEP-004: Spill and capacity factor

- Minimum experiment: Add three spill attempts at capacity factor 1.0625 and run 25 steps.
- Baseline: The fastest correct fixed-dispatch result.
- Expected signal: Lower drop fraction with a measured MFU cost.
- Falsifier: Numerical mismatch, failure, or no drop improvement.
- Cost or risk: One rack and one compile.
- Sources: PR 7780 and issue 7279.

### Hypothesis Queue

#### Active

- `MHEP-001`: The existing ragged backend can complete the first 25-step EP64 gate.
- `MHEP-002`: Fixed-capacity dispatch improves MFU on this branch.
- `MHEP-003`: Gather dispatch and the custom adjoint keep numerical parity and improve MFU.
- `MHEP-004`: Spill reduces drops enough to justify its measured MFU cost.

#### Blocked

- None.

#### Falsified or Dead End

- None for this branch.

#### Promoted

- None.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| PR 7876 | PR | https://github.com/marin-community/marin/pull/7876 | Exact branch base and command-buffer setting | Stable | Head `75d5c27e1` |
| PR 7780 | PR | https://github.com/marin-community/marin/pull/7780 | EP configuration and feature order | Exploratory | Published branch was not the measured build |
| Issue 7279 | Issue | https://github.com/marin-community/marin/issues/7279 | Rack MFU, drop, failure, and profile evidence | Replicated | Same target hardware and shape |
| JAX docs | Official docs | https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html | Ragged primitive contract | Stable | API contract only |
| Megatron Core docs | Official docs | https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.model_parallel_config.html | EP communication overlap precedent | Stable | Different trainer stack |

## Entry Log

### 2026-08-01 20:39 UTC - Base selected

- Hypothesis: PR 7876 is a clean, direct base for the EP work.
- Commit Hash: `75d5c27e1`.
- Command: `git merge --ff-only origin/pr-7876`.
- Config: No EP code.
- Result: The branch fast-forwarded from main without a merge commit.
- Interpretation: All later results can use PR 7876 as the exact source base.
- Next action: Add the minimal ragged EP64 variant.
