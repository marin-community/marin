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

### 2026-08-01 23:00 UTC - MHEP-001 artifact prefix corrected

- Correction: The coordinator resolved the output to `s3://marin-us-east-02a/marin/grug/mhep-001-ragged-25-20260801-2148/2026.08.01`. The launch contract incorrectly called this an A08-local prefix.
- Input: The coordinator reused the cached `slimpajama-6b@2026.06.28` artifact. It reported recipe drift from fingerprint `b81fb521` to `6d4a8a93` and kept the versioned cached output.
- Interpretation: The artifact prefix and data input are now explicit. The output bucket and A08 are both in US East. All later feature gates must use this same versioned data input.
- Status: The 16 GPU tasks remain pending before their first attempts, with zero failures and zero preemptions.

### 2026-08-01 23:10 UTC - MHEP-001 GPU admission wait unchanged

- Status: The 16 GPU tasks remain pending before their first attempts.
- Evidence: The coordinator runs normally. The coordinator and GPU job have zero failures and zero preemptions. No new worker logs exist.
- Decision: Keep the same request and wait for full gang admission.

### 2026-08-01 23:20 UTC - MHEP-001 GPU admission wait unchanged

- Status: The 16 GPU tasks remain pending before their first attempts.
- Evidence: The coordinator runs normally. The coordinator and GPU job have zero failures and zero preemptions. No new worker logs exist.
- Decision: Keep the same request and wait for full gang admission.

### 2026-08-01 23:22 UTC - MHEP-001 child priority verified

- Evidence: The child job request has `PRIORITY_BAND_INTERACTIVE`, 16 replicas, four GB200 GPUs per replica, zero permitted task failures, and up to 100 preemption retries.
- Interpretation: The coordinator passed the requested interactive priority to the GPU job. The current wait is not caused by a priority downgrade.
- Decision: Keep the same request and priority.

### 2026-08-01 23:30 UTC - MHEP-001 GPU admission wait unchanged

- Status: The 16 GPU tasks remain pending before their first attempts.
- Evidence: The job remains synced to A08 with zero failures and zero preemptions. No new worker logs exist.
- Decision: Keep the same request and wait for full gang admission.

### 2026-08-01 23:39 UTC - MHEP-001 GPU admission wait unchanged

- Status: The 16 GPU tasks remain pending about 53 minutes after child-job submission.
- Evidence: The only scheduler diagnostic is `Pending on peer cw-us-east-08a`. The job has zero failures, zero preemptions, and no worker logs.
- Decision: Keep the verified interactive request and wait for full gang admission.

### 2026-08-02 00:08 UTC - MHEP-001 training started

- Status: All 16 GPU tasks run on A08 with zero failures and zero preemptions.
- Progress: Training reached step 18 of 25. Logged loss was 7.94 at step 6, 7.18 at step 10, 6.70 at step 14, and 6.40 at step 18.
- Evidence: All workers reported the same progress and finite loss. No OOM, non-finite value, traceback, failure, or preemption appeared in the logs.
- Telemetry caveat: A direct Finelog query is not available in this shell because its cached IAP credentials have expired. The final offline W&B artifact is the fallback metric source.
- Next action: Monitor all tasks to a terminal state, then extract steady-step metrics from the saved artifact.

### 2026-08-02 00:21 UTC - MHEP-001 GPU gate completed; coordinator timed out later

- GPU result: The child job succeeded on all 16 workers with exit 0, zero failures, and zero preemptions. Each worker used four GB200 GPUs. Iris reports an 8-minute 16-second task duration.
- Pipeline result: Marin recorded the training step as succeeded at 00:15:09 UTC. JAX coordination-service connection warnings appeared only during worker teardown after rank 0 exited.
- Harness result: The outer coordinator failed about five minutes later with `Execution timeout exceeded`. Its 5,400-second execution window started when the coordinator began and included the child GPU admission wait.
- Interpretation: The training gate completed. The outer failure does not indicate a model, optimizer, distributed-training, or GPU-task failure.
- Decision: Use a 21,600-second outer coordinator timeout for later gates so a long child queue wait cannot hide a successful result.
- Next action: Read `tracker_metrics.jsonl` through an A08 CPU task and verify final step and steady-step metrics before the fixed-capacity feature gate.

### 2026-08-02 00:34 UTC - MHEP-002 local gate passed

- Hypothesis: Fixed-capacity all-to-all can remove ragged collective overhead without gather dispatch, spill, a custom adjoint, or runtime environment switches.
- Code snapshot: `63499c1ce`.
- Config change: Select `fixed_all_to_all` at capacity factor 1.0. All other hero model, optimizer, mesh, batch, and runtime settings stay unchanged.
- Code size: The new backend file is 107 lines.
- Tests: The full Grug MoE backend file passed with 15 tests and 6 skips. The explicit four-device value-and-gradient comparison passed. Five EP-hero and launch-contract tests passed. Changed-file pre-commit checks and Pyrefly passed.
- Dry run: The launch plan resolves fixed all-to-all, 25 steps, batch 1024, EP64, and the same 16 by 4 GB200 resource request.
- Next action: Wait for the MHEP-001 metric reader, record its comparable baseline, then submit the MHEP-002 one-rack gate with a 21,600-second coordinator timeout.

### 2026-08-02 00:41 UTC - MHEP-001 baseline metrics recorded

- Source: The final offline W&B summary in `tracker_metrics.jsonl` at `s3://marin-us-east-02a/marin/grug/mhep-001-ragged-25-20260801-2148/2026.08.01`.
- Completion: `global_step` is 24 for the zero-based 25-step gate. Final loss is 6.0798.
- Performance: Median MFU is 14.9614%, mean MFU is 14.7248%, p10 MFU is 12.8391%, and p90 MFU is 16.0921% over 24 samples. The last sample is 16.0705% MFU and 260,426 tokens/s.
- Routing: Final MoE drop fraction is 2.4099% at capacity factor 1.0.
- Result: MHEP-001 passes its GPU gate. These values are the comparison baseline for MHEP-002.

### 2026-08-02 00:41 UTC - MHEP-002 launch contract ready

- Hypothesis: Fixed-capacity all-to-all improves the ragged baseline MFU and completes 25 steps without gather dispatch, spill, or a custom adjoint.
- Run ID: `mhep-002-fixed-25-20260802-0041`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-002-fixed-25-20260802-0041-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-002-fixed-25-20260802-0041 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `63499c1ce`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-002-fixed-25-20260802-0041/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-002-fixed-25-20260802-0041`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator, and monitor it to a terminal state.

### 2026-08-02 00:42 UTC - MHEP-002 submitted

- Job: `/rav/mhep-002-fixed-25-20260802-0041-coord`.
- Launch commit: `30ac0d033`; code snapshot `63499c1ce`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the federated request. The coordinator waits for peer acceptance, with zero failures and zero preemptions.
- Next action: Keep this one request and monitor it through the 25-step GPU gate.

### 2026-08-02 00:53 UTC - MHEP-002 waits for full-rack admission

- Coordinator: `/rav/mhep-002-fixed-25-20260802-0041-coord` runs on A08.
- GPU job: `/rav/mhep-002-fixed-25-20260802-0041-coord/grug-train-mhep-002-fixed-25-20260802-0041`.
- Status: All 16 GPU tasks wait before their first attempts. The coordinator and GPU job report zero failures and zero preemptions.
- Decision: Keep the one verified request and its queue position.

### 2026-08-02 01:16 UTC - MHEP-002 GPU gate completed

- GPU result: All 16 workers succeeded with exit 0, zero failures, and zero preemptions. Each worker used four GB200 GPUs. Iris reports an 18-minute 16-second task duration.
- Progress: Training reached step 18 with finite loss 6.40. Durable telemetry later recorded zero-based step 22 with MoE drop fraction 10.9523%.
- Comparison: The fixed sender-local capacity has much more routing loss than the ragged baseline final value of 2.4099% at the same capacity factor 1.0.
- Result: The 25-step accelerator gate passes. The final offline summary is still required for the MFU comparison.
- Next action: Read `tracker_metrics.jsonl`, then use both MFU and routing loss to decide whether gather dispatch is the next gate.

### 2026-08-02 01:32 UTC - MHEP-002 metrics select gather dispatch

- Completion: The root coordinator and all GPU tasks succeeded. Zero-based step 23 has finite loss 6.1136, and the offline summary reports `global_step` 24.
- Performance: Median MFU is 19.2735%, mean MFU is 19.1850%, p10 MFU is 18.5687%, and p90 MFU is 19.6132% over 22 samples. The last sample is 19.4899% MFU, 315,838 tokens/s, and 13.2799 seconds.
- Baseline change: Median MFU improves by 4.3121 percentage points, or 28.8% relative, from the ragged MHEP-001 value of 14.9614%. Last-sample throughput improves by 21.3%.
- Routing: The offline summary reports 9.672% final MoE drops. This is 7.262 percentage points above the ragged final value at the same capacity factor 1.0.
- Decision: Keep fixed-capacity all-to-all because its MFU gain is large. MHEP-003 will change only dispatch construction from repeated bf16 activation scatter to int32 source scatter plus activation gather. This does not address capacity loss; it isolates the next source-backed performance feature before a routing-capacity change.

### 2026-08-02 01:38 UTC - MHEP-003 local gate passed

- Code snapshot: `c5f1bd1e2`.
- Change: Fixed dispatch now scatters int32 assignment-source indices and gathers each activation row into the send buffer. It no longer repeats top-k activations or scatters full bf16 rows.
- Scope: The change adds 12 net lines to the backend. Routing, capacity factor, all-to-all shape, combine, model, optimizer, batch, mesh, and runtime settings stay unchanged.
- Tests: The full Grug MoE backend file passed with 15 tests and 6 skips. The explicit four-device value-and-gradient test passed. The four EP hero tests passed when run alone. Changed-file pre-commit checks and Pyrefly passed.
- Test note: The first EP hero run shared the host with three JAX suites and its unrelated Newton-Schulz subprocess hit the 60-second test limit. The isolated rerun passed all four tests in 18.54 seconds.
- Dry run: The plan resolves fixed all-to-all with gather-dispatch tags, 25 steps, batch 1024, EP64, and 16 workers with four GB200 GPUs each.

### 2026-08-02 01:38 UTC - MHEP-003 launch contract ready

- Hypothesis: Gather dispatch improves the MHEP-002 fixed-capacity median MFU of 19.2735% without changing output values, gradients, or routing loss.
- Run ID: `mhep-003-gather-25-20260802-0138`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-003-gather-25-20260802-0138-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-003-gather-25-20260802-0138 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `c5f1bd1e2`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-003-gather-25-20260802-0138/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-003-gather-25-20260802-0138`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator, and monitor it to a terminal state.

### 2026-08-02 01:39 UTC - MHEP-003 submitted

- Job: `/rav/mhep-003-gather-25-20260802-0138-coord`.
- Launch commit: `f3f23ce1c`; code snapshot `c5f1bd1e2`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the federated request. The coordinator waits on the peer, with zero reported failures or preemptions.
- Next action: Keep this one request and monitor it through the 25-step GPU gate.

### 2026-08-02 02:10 UTC - MHEP-002 full artifact corrects partial telemetry

- Correction: The earlier MHEP-002 entry used a 22-sample telemetry summary. The full offline artifact contains 24 MFU samples.
- Final performance: Median MFU is 19.3689%, mean MFU is 19.2053%, p10 MFU is 18.5799%, and p90 MFU is 19.6090%. The last sample is 19.4278% MFU, 314,831 tokens/s, and 13.3224 seconds.
- Final training values: Loss is 6.0933 and MoE drop fraction is 9.6723% at zero-based step 24.
- Comparison rule: Use these full-artifact values for MHEP-002. The prior 19.2735% median and 315,838 tokens/s values remain a partial 22-sample telemetry snapshot.

### 2026-08-02 02:18 UTC - MHEP-003 gather dispatch passes

- Completion: The root coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. The run completed all 25 steps.
- Performance: Median MFU is 21.8766%, mean MFU is 21.6602%, p10 MFU is 20.5804%, and p90 MFU is 22.2728% over 24 samples. The last sample is 22.2428% MFU, 360,450 tokens/s, and 11.6363 seconds.
- MHEP-002 change: Median MFU improves by 2.5077 percentage points, or 12.9% relative. Last-sample throughput improves by 14.5%.
- Training: Final loss is 6.0917. Final MoE drop fraction is 9.7165%, within 0.045 percentage points of MHEP-002.
- Decision: Keep gather dispatch. MHEP-004 will add only structured custom VJPs for the dispatch and combine gathers. This tests the source-backed backward-scatter removal before any routing-capacity change.

### 2026-08-02 02:26 UTC - MHEP-004 local gate passed

- Code snapshot: `9ba891724`.
- Change: Two structured custom VJPs replace generic gather transposes. Dispatch backward gathers kept send slots and sums top-k rows per token. Combine backward uses the injective slot-to-assignment inverse.
- Scope: The backend change adds 58 net lines. The capacity-overflow test adds 24 net lines for dropped-assignment gradient checks. Routing, capacity, forward values, collectives, model, optimizer, batch, mesh, and runtime settings stay unchanged.
- Tests: The full Grug MoE backend file passed with 15 tests and 6 skips. The capacity-overflow value and gradient test passed. The explicit four-device value-and-gradient test passed. Four EP hero tests passed. Changed-file pre-commit checks and Pyrefly passed.
- Dry run: The plan resolves fixed all-to-all with gather-dispatch and custom-adjoint tags, 25 steps, batch 1024, EP64, and 16 workers with four GB200 GPUs each.

### 2026-08-02 02:26 UTC - MHEP-004 launch contract ready

- Hypothesis: Structured gather transposes improve the MHEP-003 median MFU of 21.8766% without changing loss or routing loss.
- Run ID: `mhep-004-adjoint-25-20260802-0226`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-004-adjoint-25-20260802-0226-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-004-adjoint-25-20260802-0226 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `9ba891724`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-004-adjoint-25-20260802-0226/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-004-adjoint-25-20260802-0226`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator, and monitor it to a terminal state.
