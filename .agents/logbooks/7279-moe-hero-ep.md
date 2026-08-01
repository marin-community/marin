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

The branch starts at PR 7876 head `75d5c27e1`. The MHEP-001 ragged EP64 snapshot is `b0d20062a`. Its local checks pass. Run `mhep-001-ragged-25-20260801-2148` is waiting for A08 admission.

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

### 2026-08-01 21:00 UTC - MHEP-001 local gate passed

- Hypothesis: The copied EP64 variant lowers with the PR 7876 base and keeps Newton-Schulz output on the expert sharding.
- Commit Hash: `b0d20062a`.
- Commands: `./infra/pre-commit.py --changed-files --fix`; focused `uv run pytest`; `uv run pyrefly check`.
- Config: Ragged all-to-all, EP64, batch 1024, 25 steps, no PGLE, and no fixed-capacity dispatch.
- Result: Five focused checks passed. Pyrefly reported zero errors. The full Grug contract file has a separate existing failure in `experiments/grug/base/model.py:227` on the PR 7876 base.
- Interpretation: Local checks support a rack launch. They do not establish accelerator correctness or MFU.
- Next action: Run the 25-step MHEP-001 gate on one A08 NVL72 rack.

### 2026-08-01 21:50 UTC - MHEP-001 launch contract ready

- Hypothesis: The existing ragged EP64 path can complete 25 steps on one A08 NVL72 rack.
- Run ID: `mhep-001-ragged-25-20260801-2148`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 5400 --max-retries 50 --job-name mhep-001-ragged-25-20260801-2148-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-001-ragged-25-20260801-2148 --num-steps 25 --version 2026.08.01 --run`.
- Code snapshot: `b0d20062a`; the clean launch commit will include this log entry.
- Output identity: `grug/mhep-001-ragged-25-20260801-2148/2026.08.01` under the A08-local Marin prefix.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-001-ragged-25-20260801-2148`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode for this gate because the submitter has no W&B key.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor at 120 seconds, then 570 seconds.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator, and monitor it to a terminal state.

### 2026-08-01 22:22 UTC - MHEP-001 submitted and waiting for admission

- Hypothesis: The existing ragged EP64 path can complete 25 steps on one A08 NVL72 rack.
- Job: `/rav/mhep-001-ragged-25-20260801-2148-coord`.
- Launch commit: `120ccfbe2`; code snapshot `b0d20062a`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Result: The A08 peer accepted the handoff. Coordinator task 0 remains pending before its first attempt, with zero failures and zero preemptions.
- Capacity evidence: At 22:22 UTC, A08 was reachable and reported 0 of 820 GB200 GPUs free. Interactive work held 84 GPUs and batch work held 736 GPUs.
- Interpretation: This is a scheduler wait. No setup, compile, training, or GPU cost has started for this run.
- Decision: Keep the single request queued. Do not duplicate it, change priority, or change the cluster.
- Next action: Continue the 570-second monitor cadence until admission or a terminal state.

### 2026-08-01 22:32 UTC - MHEP-001 admission wait unchanged

- Status: Coordinator task 0 is pending before its first attempt.
- Evidence: Zero failures and zero preemptions. A08 remains reachable and reports 0 of 820 GB200 GPUs free, with 84 held by interactive work and 736 held by batch work.
- Decision: Keep the existing request and its queue position.
- Next action: Continue the 570-second monitor cadence.

### 2026-08-01 22:42 UTC - MHEP-001 admission wait unchanged

- Status: Coordinator task 0 remains pending before its first attempt.
- Evidence: Zero failures and zero preemptions. A08 remains reachable and reports 0 of 820 GB200 GPUs free, with 84 held by interactive work and 736 held by batch work.
- Decision: Keep the existing request and its queue position.
- Next action: Continue the 570-second monitor cadence.

### 2026-08-01 22:54 UTC - MHEP-001 coordinator started

- Status: The coordinator runs on A08 and has submitted the 16-task GPU child job.
- GPU job: `/rav/mhep-001-ragged-25-20260801-2148-coord/grug-train-mhep-001-ragged-25-20260801-2148`.
- Evidence: All 16 GPU tasks are pending before their first attempts. Each task requests four GB200 GPUs. The coordinator and GPU job report zero failures and zero preemptions.
- Interpretation: The 64-GPU gang now waits for GPU admission. Setup, compile, and training have not started.
- Decision: Keep the existing request. Do not submit another job or change its priority.
- Next action: Continue the 570-second monitor cadence until the GPU tasks start or reach a terminal state.
