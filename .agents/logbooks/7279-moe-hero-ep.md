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

### 2026-08-02 02:27 UTC - MHEP-004 submitted

- Job: `/rav/mhep-004-adjoint-25-20260802-0226-coord`.
- Launch commit: `ee5dd4294`; code snapshot `9ba891724`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: The coordinator waits for A08 peer acceptance, with zero reported failures or preemptions.
- Next action: Keep this one request and monitor it through the 25-step GPU gate.

### 2026-08-02 02:45 UTC - MHEP-004 custom adjoints pass

- Completion: The root coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. The run completed all 25 steps.
- Performance: Median MFU is 24.1231%, mean MFU is 24.1830%, p10 MFU is 23.7355%, and p90 MFU is 24.8348% over 24 samples. The last sample is 23.9739% MFU, 388,503 tokens/s, and 10.7961 seconds.
- MHEP-003 change: Median MFU improves by 2.2465 percentage points, or 10.3% relative. Last-sample throughput improves by 7.8%.
- Training: Final loss is 6.0858. Final MoE drop fraction is 9.6786%, within 0.038 percentage points of MHEP-003.
- Decision: Keep the structured custom VJPs. MHEP-005 will change only the capacity factor from 1.0 to 1.0625. This tests the smallest configuration change that can reduce dropped assignments.

### 2026-08-02 02:48 UTC - MHEP-005 local gate passed

- Code snapshot: `bccd8eb809`.
- Change: The hero capacity factor increases from 1.0 to 1.0625. Model structure, routing, dispatch code, collectives, optimizer, batch, mesh, and runtime settings stay unchanged.
- Tests: The four EP hero tests passed. Changed-file pre-commit checks passed.
- Dry run: The plan resolves capacity factor 1.0625, fixed all-to-all with gather dispatch and custom adjoints, 25 steps, batch 1024, EP64, and 16 workers with four GB200 GPUs each.

### 2026-08-02 02:48 UTC - MHEP-005 launch contract ready

- Hypothesis: Capacity factor 1.0625 reduces the MHEP-004 final drop fraction of 9.6786% with a small MFU cost.
- Run ID: `mhep-005-capacity-25-20260802-0248`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-005-capacity-25-20260802-0248-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-005-capacity-25-20260802-0248 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `bccd8eb809`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-005-capacity-25-20260802-0248/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-005-capacity-25-20260802-0248`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator, and monitor it to a terminal state.

### 2026-08-02 02:49 UTC - MHEP-005 submitted

- Job: `/rav/mhep-005-capacity-25-20260802-0248-coord`.
- Launch commit: `b9997cd24`; code snapshot `bccd8eb809`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the federated request. The coordinator waits on the peer, with zero reported failures or preemptions.
- Next action: Keep this one request and monitor it through the 25-step GPU gate.

### 2026-08-02 03:10 UTC - MHEP-005 larger capacity passes with a high cost

- Completion: The root coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. The run completed all 25 steps.
- Performance: Median MFU is 23.1277%, mean MFU is 23.2566%, p10 MFU is 22.5028%, and p90 MFU is 24.2065% over 24 samples. The last sample is 23.0215% MFU, 373,068 tokens/s, and 11.2427 seconds.
- MHEP-004 change: Median MFU falls by 0.9954 percentage points, or 4.1% relative. Last-sample throughput falls by 4.0%.
- Training: Final loss is 6.0879. Final MoE drop fraction falls from 9.6786% to 7.2507%, a reduction of 2.4279 percentage points.
- Decision: Capacity 1.0625 alone does not reduce drops enough to justify its MFU cost. MHEP-006 will keep capacity 1.0625 and add only the source-backed three-choice spill rule. If spill does not give a large drop reduction with a small added cost, select the MHEP-004 stack for the 200-step gate.

### 2026-08-02 03:39 UTC - MHEP-006 receiver-ECHO local gate passed

- Sequence change: Evaluate receiver-ECHO before spill. Prior work in #7670 found 19.98% MFU and 1.32% drops for static receiver-ECHO on a related EP64 model, but the method has not run on this hero.
- Code snapshot: `fa1f1b03e9dbea89fbf8d325475103f4b2199555`; gate configuration: `4b89ee8cbc487d4f7ba1a035b93b4d09a21e5c2f`.
- Change: Keep each selected expert, retain work on its home rank up to receiver capacity, move overflow to spare receivers, and send sparse copies of the required expert weights. Fixed token all-to-all uses capacity 1.0. `echo_receiver_cute` uses the QuACK grouped expert kernel.
- Scope: The backend and its public dispatch add 561 net code lines. Cross-shard tests add 151 net lines. Model, optimizer, batch, mesh, data, attention, and runtime settings stay unchanged from MHEP-004.
- Tests: The four-device hot-expert test has zero drops and matches a dense selected-expert reference for the loss and gradients of inputs, combine weights, and both expert weight tensors at `rtol=atol=1e-5`. The abstract EP path lowers. The Grug MoE and EP hero suites pass with 20 tests, 6 skips, and 2 tests excluded by the repository marker policy. Changed-file pre-commit and Pyrefly pass.
- Test limit: XLA CPU does not implement the sparse ragged all-to-all primitive. The four-device CPU test replaces only that collective with an all-gather reference. The A08 gate will use the real primitive and QuACK kernel.
- Dry run: The plan resolves 25 steps, batch 1024, EP64, capacity 1.0, `echo_receiver_cute`, and 16 workers with four GB200 GPUs each.

### 2026-08-02 03:39 UTC - MHEP-006 launch contract ready

- Hypothesis: Receiver-ECHO reduces the MHEP-004 final drop fraction of 9.6786% while keeping median MFU near the MHEP-004 value of 24.1231%.
- Run ID: `mhep-006-echo-receiver-25-20260802-0339`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-006-echo-receiver-25-20260802-0339-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-006-echo-receiver-25-20260802-0339 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `4b89ee8cbc487d4f7ba1a035b93b4d09a21e5c2f`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-006-echo-receiver-25-20260802-0339/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-006-echo-receiver-25-20260802-0339`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator once, and monitor it to a terminal state.

### 2026-08-02 03:40 UTC - MHEP-006 submitted

- Job: `/rav/mhep-006-echo-receiver-25-20260802-0339-coord`.
- Launch commit: `d8072e9cd`; receiver-ECHO code snapshot: `fa1f1b03e`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the one federated request. No duplicate request was submitted.
- Next action: Wait 120 seconds for immediate failure, then use the normal monitor cadence until the 25-step gate reaches a terminal state.

### 2026-08-02 03:44 UTC - MHEP-006 admitted on A08

- Coordinator and GPU child state: Running with zero failures and zero preemptions.
- GPU allocation: All 16 tasks run with four GB200 GPUs each; no task is pending or complete.
- Logs: The first 15-minute scan has no matched loss failure, exception, OOM, dead-node, or resource error.
- Decision: Keep the one admitted job and use the normal monitor cadence.

### 2026-08-02 04:06 UTC - MHEP-006 receiver-ECHO is not selected

- Completion: The coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. The run completed all 25 steps. Each GPU task ran for about 9 minutes.
- Performance: Median MFU is 18.2099%, mean MFU is 18.2159%, p10 MFU is 18.1404%, and p90 MFU is 18.2876% over 24 samples. The last sample is 18.2194% MFU, 295,250 tokens/s, and 14.2059 seconds.
- MHEP-004 change: Median MFU falls by 5.9132 percentage points, or 24.5% relative. Last-sample throughput falls by 24.0%.
- Training: Final loss is 6.0874. Final MoE drop fraction falls from 9.6786% to 4.4110%, a reduction of 5.2676 percentage points.
- Code cost: Receiver-ECHO adds 561 net backend and public-dispatch lines, plus 151 net test lines.
- Decision: Do not select receiver-ECHO. Its routing improvement does not justify the MFU and code costs. The gap is large and stable across the 24 MFU samples, so XProf is not necessary for this selection decision. Keep MHEP-004 as the selected stack before the next isolated feature.

### 2026-08-02 04:16 UTC - MHEP-007 three-choice spill local gate passed

- Code snapshot: `93ac949988943d4cce999c40d82cc21c764701c7`; gate configuration: `f0ce49c8a33a63284f9ef45dfc33de12b526ee11`.
- Change: Fixed-capacity overflow gets three attempts on lower-ranked experts selected by the same token. An accepted spill uses the candidate expert and its router weight. The implementation is an explicit `fixed_all_to_all_spill` backend, with no environment switch.
- Scope: Spill adds 106 net backend and public-dispatch lines, plus 101 net test lines. Token transport, capacity 1.0, gather adjoints, model, optimizer, batch, mesh, data, attention, and runtime settings stay unchanged from MHEP-004.
- Tests: Exact planner checks verify increased accepted work, unique capacity slots, the candidate expert, and its weight. The end-to-end fixed backend check verifies output values plus input and router-weight gradients. The abstract EP path lowers. The Grug MoE and EP hero suites pass with 23 tests, 6 skips, and 2 tests excluded by the repository marker policy. Changed-file pre-commit and Pyrefly pass.
- Dry run: The plan resolves 25 steps, batch 1024, EP64, capacity 1.0, three spill attempts, and 16 workers with four GB200 GPUs each.

### 2026-08-02 04:16 UTC - MHEP-007 launch contract ready

- Hypothesis: Three same-token spill attempts reduce the MHEP-004 final drop fraction of 9.6786% with a small cost from its 24.1231% median MFU.
- Run ID: `mhep-007-spill-25-20260802-0416`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-007-spill-25-20260802-0416-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-007-spill-25-20260802-0416 --num-steps 25 --version 2026.08.02 --run`.
- Code snapshot: `f0ce49c8a33a63284f9ef45dfc33de12b526ee11`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-007-spill-25-20260802-0416/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-007-spill-25-20260802-0416`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 25.
- Checkpoint policy: No checkpoints. This gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Commit this contract, submit the coordinator once, and monitor it to a terminal state.

### 2026-08-02 04:17 UTC - MHEP-007 submitted

- Job: `/rav/mhep-007-spill-25-20260802-0416-coord`.
- Launch commit: `fd37ec8d8`; spill code snapshot: `93ac94998`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the one federated request. No duplicate request was submitted.
- Next action: Wait 120 seconds for immediate failure, then use the normal monitor cadence until the 25-step gate reaches a terminal state.

### 2026-08-02 04:20 UTC - MHEP-007 admitted on A08

- Coordinator and GPU child state: Running with zero failures and zero preemptions.
- GPU allocation: All 16 tasks run with four GB200 GPUs each; no task is pending or complete.
- Logs: The first 15-minute scan has no matched loss failure, exception, OOM, dead-node, or resource error.
- Decision: Keep the one admitted job and use the normal monitor cadence.

### 2026-08-02 04:39 UTC - MHEP-007 spill is not selected

- Completion: The coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. The run completed all 25 steps. Each GPU task ran for 7 minutes and 20 seconds.
- Performance: Median MFU is 24.0829%, mean MFU is 24.1721%, p10 MFU is 23.7578%, and p90 MFU is 24.8420% over 24 samples. The last sample is 23.8640% MFU, 386,721 tokens/s, and 10.8458 seconds.
- MHEP-004 change: Median MFU falls by 0.0401 percentage points, or 0.17% relative. Last-sample throughput falls by 0.46%.
- Training: Final loss is 6.1049. Final MoE drop fraction falls from 9.6786% to 5.8806%, a reduction of 3.7980 percentage points.
- Code cost: Spill adds 106 net backend and public-dispatch lines, plus 101 net test lines.
- Decision: Do not select spill. It is a throughput tie but does not exceed MHEP-004 median MFU and adds code. The selection rule prefers the highest MFU with the least code. Remove receiver-ECHO and spill from the final diff, restore MHEP-004, and run its final 200-step gate. XProf is not necessary for this selection decision.

### 2026-08-02 04:44 UTC - MHEP-008 selected stack local gate passed

- Code snapshot: `e2b10a535d7fd7319b5b659d5bbdf45f3b85da8a`.
- Selection: MHEP-004 has the highest 25-step median MFU, 24.1231%, and the least code among the final candidates. MHEP-006 receiver-ECHO and MHEP-007 spill remain in commit and experiment history but are removed from the final file state.
- Exact state: The Levanter MoE implementation and tests match the MHEP-004 result commit `c1112127e`. The hero uses fixed all-to-all, gather dispatch, structured gather adjoints, and capacity 1.0.
- Tests: The selected Grug MoE and EP hero suites pass with 19 tests, 6 skips, and 1 test excluded by the repository marker policy. Changed-file pre-commit and Pyrefly pass.
- Dry run: The plan resolves 200 steps, batch 1024, EP64, fixed all-to-all at capacity 1.0, and 16 workers with four GB200 GPUs each. The 200-step compute heuristic sets MuonH LR 0.0485937, Adam LR 0.0112139, beta2 0.9684911, and epsilon 1.36839e-16.

### 2026-08-02 04:44 UTC - MHEP-008 final gate launch contract ready

- Goal: Verify the selected minimal stack for 200 training steps with finite loss and stable throughput.
- Run ID: `mhep-008-final-200-20260802-0444`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name mhep-008-final-200-20260802-0444-coord -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch --run-id mhep-008-final-200-20260802-0444 --num-steps 200 --version 2026.08.02 --run`.
- Code snapshot: `e2b10a535d7fd7319b5b659d5bbdf45f3b85da8a`; the clean launch commit will include this log entry.
- Output identity: `s3://marin-us-east-02a/marin/grug/mhep-008-final-200-20260802-0444/2026.08.02`.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-008-final-200-20260802-0444`, project `marin_moe`, group `moe-hero-ep`, resume `allow`, and offline mode.
- Initialization: None.
- Final step: 200.
- Checkpoint policy: No checkpoints. This final throughput gate writes metrics only.
- DRI and babysitter: `rav`; `rav/codex` owns the monitor at the normal 570-second cadence after the first 120-second check.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, sustained loss above 1.5 times the expected trend for 10 steps, or incomplete step 200.
- Next action: Commit this contract, submit the coordinator once, and monitor it to a terminal state.

### 2026-08-02 04:45 UTC - MHEP-008 submitted

- Job: `/rav/mhep-008-final-200-20260802-0444-coord`.
- Launch commit: `78365da76`; selected-stack snapshot: `e2b10a535`; clean tree.
- Source bundle: Iris workspace bundle, 9.4 MB. The submit output did not report a content ID.
- Status: A08 has the one federated request. No duplicate request was submitted.
- Next action: Wait 120 seconds for immediate failure, then use the normal monitor cadence until the 200-step gate reaches a terminal state.

### 2026-08-02 04:48 UTC - MHEP-008 admitted on A08

- Coordinator and GPU child state: Running with zero failures and zero preemptions.
- GPU allocation: All 16 tasks run with four GB200 GPUs each; no task is pending or complete.
- Logs: The first 15-minute scan has no matched loss failure, exception, OOM, dead-node, or resource error.
- Decision: Keep the one admitted job and use the normal monitor cadence.

### 2026-08-02 04:58 UTC - MHEP-008 healthy at the first normal check

- State: The coordinator and GPU child continue to run. All 16 GPU tasks run with zero failures and zero preemptions.
- Progress: The latest visible sample is step 15 of 200. Loss fell from 7.57 at step 9 to 6.90 at step 15.
- Logs: No exception, OOM, dead-node, resource error, non-finite value, or retry is present.
- Decision: Continue the same job at the normal monitor cadence.

### 2026-08-02 05:08 UTC - MHEP-008 healthy through step 69

- State: The coordinator and all 16 GPU tasks continue to run with zero failures and zero preemptions.
- Progress: The latest visible sample is step 69 of 200 at about 11.5 seconds per step. Loss fell to 4.88.
- Logs: No exception, OOM, dead-node, resource error, non-finite value, or retry is present.
- Decision: Continue the same job at the normal monitor cadence.

### 2026-08-02 05:18 UTC - MHEP-008 healthy through step 111

- State: The coordinator and all 16 GPU tasks continue to run with zero failures and zero preemptions.
- Progress: The latest visible sample is step 111 of 200 at about 11.5 seconds per step. Loss fell to 3.86.
- Logs: No exception, OOM, dead-node, resource error, non-finite value, or retry is present.
- Decision: Continue the same job at the normal monitor cadence.

### 2026-08-02 05:28 UTC - MHEP-008 healthy through step 153

- State: The coordinator and all 16 GPU tasks continue to run with zero failures and zero preemptions.
- Progress: The latest visible sample is step 153 of 200 at about 11.6 seconds per step. Loss is 3.50.
- Loss check: The small change from 3.64 at step 129 to 3.67 at step 135 was brief; loss then fell to 3.50. It does not meet the sustained-loss stop rule.
- Logs: No exception, OOM, dead-node, resource error, non-finite value, or retry is present.
- Decision: Continue the same job at the normal monitor cadence.

### 2026-08-02 05:48 UTC - MHEP-008 final gate passed

- Completion: The coordinator and all 16 GPU tasks succeeded with exit 0, zero failures, and zero preemptions. All workers reached step 200 of 200. The training loop took about 40 minutes and 35 seconds after startup.
- Shutdown: Coordination-service connection warnings appeared only after step 200 while peers exited. The scheduler records success for every task, so these are normal shutdown warnings.
- W&B artifact: The durable offline `tracker_metrics.jsonl` summary has `_step=199`, `global_step=199`, and 199 MFU samples for the zero-based 200-step run.
- Performance: Median MFU is 23.6969%, mean MFU is 23.6329%, p10 MFU is 22.8856%, and p90 MFU is 24.0928%. The last sample is 23.6283% MFU, 382,902 tokens/s, and 10.9540 seconds.
- Training: Final loss is 3.3119. Final MoE drop fraction is 7.4113%.
- MHEP-004 change: Median MFU is 0.4262 percentage points lower than the 25-step selection gate. Final drop fraction is 2.2673 percentage points lower. The 200-step optimizer uses the recorded compute-scaled learning rates, so this is the required stability gate rather than another selection comparison.
- Decision: The selected fixed all-to-all, gather-dispatch, custom-adjoint, capacity-1.0 stack passes the final gate. Receiver-ECHO and three-choice spill remain rejected. XProf is not necessary.
- Next action: Complete branch review, required checks, and the pull request.

### 2026-08-02 06:10 UTC - Final branch review passed

- Peer review: The independent review returned 21 findings. Safe changes document fixed sender/expert capacity, remove dead checkpoint and profiler configuration, derive run tags, use one hardware-size source, remove an unused ragged flag, share one sharding helper, simplify the EP-only Newton-Schulz layout, prevent test environment leaks, shorten the Newton-Schulz subprocess, and move the four-device all-to-all value-and-gradient test into the default suite.
- Backend rewrites: Do not replace sorting, padding, collective structure, or custom-VJP arguments after the measured gate. These are unproven HLO changes that need a new 25-step performance gate and are not correctness fixes.
- Reshard finding: No change. The alleged second collective is inactive on the fixed hero mesh because `expert` is the only intra-rack axis with size greater than one.
- Out-of-scope findings: No change to the FSDP hero files because they are not in the exact PR-7876 branch diff.
- Logbook finding: No rewrite because this research logbook is append-only.
- Tests: The default Grug MoE and EP hero suites pass with 20 tests and 6 skips. The set now includes the four-device fixed all-to-all forward and gradient test. The 200-step dry run still resolves EP64, capacity 1.0, fixed all-to-all, batch 1024, and the recorded optimizer values.
- Checks: The full changed-file pre-commit and Pyrefly checks pass. The one required advisory rule-catalog review reports no findings.

### 2026-08-04 18:28 UTC - MHEP-009 FSDP-shape local gate passed

- Hypothesis: The FSDP hero model shape runs on the EP64 mesh, and its MFU is directly comparable
  to the FSDP hero result because the analytic FLOP count depends only on the model config.
- Motive: The recorded FSDP reference has a different model shape, thus the baseline section calls
  it out as no EP control. One shape on two sharding strategies removes that limit.
- Change: `heuristic.py` gets a `HeroShape` selector and the FSDP hero model spec. `launch.py` gets
  `--shape` and takes host offload of the optimizer state from the selected shape.
- Forced deltas: `moe_implementation` becomes `fixed_all_to_all` because `sonic_cute` has no EP
  collectives, and `expert_chunks` becomes 1 because `moe_mlp` rejects a larger value when the
  expert axis is larger than one. All other model fields stay equal to the FSDP hero.
- Capacity note: The local FSDP backend computes every assignment. EP drops each assignment above
  its fixed cell capacity, thus the EP drop fraction is part of the result.
- Memory: A 64-device shape check reports 359.64 B parameters, 24.59 GiB of parameters per device,
  and 27.78 GiB of optimizer state per device. With host offload the resident estimate is about 106
  GiB per device. The measured d5120 EP shape estimates about 110 GiB per device without offload,
  thus 128 experts on 64 devices fit with margin. Two whole experts land on each device.
- Mesh selection: EP64 is the lowest-memory mesh for this shape on 64 GPUs. A hybrid mesh such as
  EP32 with two-way FSDP replicates each expert twice, because the EP path shards expert weights
  on the expert axis only. Thus no hybrid mesh was selected.
- Tests: The eight EP-hero and FSDP-hero tests pass. The new parity test fails if the two model
  specs drift apart in more fields than the two forced deltas.
- Dry run: The plan resolves d6144, 128 experts, top-4, two shared experts, SConv, sliding window
  512, EP64, capacity 1.0, batch 1024, and 16 workers with four GB200 GPUs each.

### 2026-08-04 18:31 UTC - MHEP-009 launch contract ready and submitted

- Hypothesis: The FSDP hero shape completes 25 steps on one EP64 rack, and its median MFU gives the
  first same-shape comparison between EP and FSDP sharding.
- Run ID: `mhep-009-fsdp-shape-25-20260804-1831`.
- Job: `/rav/mhep-009-fsdp-shape-25-20260804-1831-coord`.
- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --enable-extra-resources --target-cluster cw-us-east-08a --priority interactive --cpu 2
  --memory 8GB --disk 32GB --timeout 21600 --max-retries 50 --job-name
  mhep-009-fsdp-shape-25-20260804-1831-coord -e WANDB_MODE offline -- python -m
  experiments.grug.moe_hero_ep.launch --run-id mhep-009-fsdp-shape-25-20260804-1831 --num-steps 25
  --shape fsdp --version 2026.08.04 --run`.
- Code snapshot: `01025d80b`; clean tree. Source bundle: Iris workspace bundle, 9.6 MB.
- Hardware: 16 workers with four GB200 GPUs each on `cw-us-east-08a`; 64 GPUs total.
- W&B: ID and display name `mhep-009-fsdp-shape-25-20260804-1831`, project `marin_moe`, group
  `moe-hero-ep`, tag `shape-fsdp`, and offline mode.
- Final step: 25. Checkpoint policy: No checkpoints. This gate writes metrics only.
- Stop criteria: Stop on terminal failure, non-finite loss, task retry, OOM, or incomplete step 25.
- Next action: Monitor to a terminal state, read `tracker_metrics.jsonl`, then run the one-rack FSDP
  control at the same shape and batch.

### 2026-08-04 18:38 UTC - MHEP-010 one-rack FSDP control submitted

- Purpose: Give MHEP-009 a same-shape, same-day control. The recorded FSDP reference is a two-rack
  200-step average, thus it mixes a topology change into the transport comparison.
- Run ID: `mhep-010-fsdp-control-25-20260804-1838`.
- Job: `/rav/mhep-010-fsdp-control-25-20260804-1838-coord`.
- Command: The same coordinator form as MHEP-009, but `python -m experiments.grug.moe_hero_fsdp.launch
  --run-id mhep-010-fsdp-control-25-20260804-1838 --dp-racks 1 --num-steps 25 --no-save-checkpoints
  --version 2026.08.04 --run`.
- Code snapshot: `8a055fec1`; clean tree.
- Change to the FSDP hero: `--no-save-checkpoints` makes this gate metrics-only. The forced
  completion checkpoint writes the parameters and the offloaded optimizer state, about 2.7 TiB at
  this shape, which an MFU gate does not need. All other hero settings stay unchanged.
- Matched between the two gates: model shape, batch 1024, sequence 4096, 25 steps, capacity 1.0,
  MuonH with the same compute-scaled values, host offload of the optimizer state, mixed precision,
  SlimPajama-6B at `2026.06.28`, and 16 workers with four GB200 GPUs each.
- Not matched: the MoE backend, `expert_chunks`, the mesh, and each hero's own runtime environment.
  PGLE is on for FSDP and off for EP. Each variant keeps the runtime that its own gates selected,
  thus this compares two tuned strategies, not one isolated variable.
- Comparison anchor: A prior EP64 arm at d6144, 4-of-128, sliding window 2048, and 120 steps
  measured 24.842% median MFU and 274,954 tokens/s (issue 7279 comment 5095217108). That arm is not
  this shape, but it bounds the expected range.

### 2026-08-04 23:50 UTC - MHEP-009 passed: the FSDP shape runs on EP64

- Completion: All 16 GPU tasks succeeded with exit 0 and zero failures. The run completed all 25
  steps. Admission took nine attempts across 5 hours and 17 minutes, with eight preemptions before
  the successful window. No preemption reached step 0, thus no GPU work was lost.
- Performance: Median MFU is 27.7544%, mean MFU is 26.0191%, p10 MFU is 19.8324%, and p90 MFU is
  29.3997% over 26 samples. The last sample is 26.4006% MFU, 316,473 tokens/s, and 13.2533 seconds.
- Training: Final loss is 6.0498. Final MoE drop fraction is 9.9683%, or 80,275,372 assignments.
- Memory: The run confirms the pre-flight estimate. There was no OOM and no allocator failure at an
  estimated 106 GiB per device with host offload of the optimizer state.
- Variance caveat: The standard deviation is 6.5623 over 26 samples, because the early samples
  include compile and warmup. Use the median. The 200-step EP hero gate had a 0.5656 deviation, so a
  longer run is necessary before any small difference is called real.
- Capacity caveat: The 9.9683% drop fraction means EP does less work than the analytic FLOP count
  credits. The FSDP control computes every assignment. Thus the MFU comparison is not yet
  quality-fair, and MHEP-012 remains necessary.
- Comparison anchors: A prior EP64 arm at d6144, 4-of-128, and sliding window 2048 measured 24.842%
  median MFU. The native d5120 EP hero measured 23.6969% at 200 steps. This shape is above both.
- Next action: Read the MHEP-010 control when it completes, then report the same-shape comparison.

### 2026-08-05 00:00 UTC - MHEP-010 control gives the same-shape EP versus FSDP result

- Completion: All 16 GPU tasks succeeded with exit 0 and zero failures. The run completed all 25
  steps after seven preemptions, on the same rack allocation window as MHEP-009.
- Performance: Median MFU is 19.3951%, mean MFU is 16.7629%, p10 MFU is 2.9288%, and p90 MFU is
  19.6933% over 26 samples. The last sample is 19.6144% MFU, 235,125 tokens/s, and 17.8386 seconds.
- Training: Final loss is 6.0754. Final MoE drop fraction is 1.8779%, or 15,122,972 assignments.
- Result: At one rack and this exact model shape, EP64 measures 8.3593 percentage points more median
  MFU than FSDP64, or 43.1% relative. Last-sample throughput is 34.6% higher.
- Correction to the MHEP-009 entry: The local `sonic_cute` backend does drop assignments. It dropped
  1.8779%, not zero. The earlier claim that the FSDP path computes every assignment is wrong.
- Adjusted comparison: EP completed 90.03% of assignments and FSDP completed 98.12%. A first-order
  correction of the EP median for the work it did not do gives about 25.5% against 19.4%. Thus the
  EP lead is about 6 percentage points, not 8.4, and it survives the correction.
- Variance limit: Both runs have a wide spread because 26 samples include compile and warmup. The
  FSDP p10 of 2.9288% is a warmup sample, not steady state. Use the medians. The 200-step EP hero
  gate had a 0.5656 deviation against 5.5456 and 6.5623 here.
- Cost record: The pair took 5 hours and 30 minutes from submit to result, of which about 20 minutes
  was GPU work. Fifteen preemptions across both jobs, none of which reached step 0.
- Next action: MHEP-012 (capacity sweep) is now necessary rather than optional, because the drop
  gap of 8.1 percentage points funds part of the EP lead. MHEP-011 (FSDP at expert_chunks=1) remains
  the cheapest way to attribute the rest.

### 2026-08-05 01:00 UTC - MHEP-011 to MHEP-016 queued: size ladder and capacity sweep

- Code snapshot: `5c7d9d2aa`. The EP launcher takes `--num-experts`, `--intermediate-dim`, and
  `--capacity-factor`, and rejects a bank that does not divide the 64-way expert axis.
- W&B: All six runs and `mhep-009b` write live to entity `marin-community`, project `rav_moe`.
- Common settings: EP64, one rack, `--shape fsdp`, 25 steps, batch 1024, d6144, 48 layers, top-4,
  two shared experts, host offload of the optimizer state.

Size ladder. Each keeps 256 experts (four per device) and about 20 B active parameters, and each is
larger than any shape measured on this rack before. Estimates come from a 64-device shape check.

| run | experts x width | total | active | resident estimate | headroom |
| --- | --- | --- | --- | --- | --- |
| `mhep-011` | 256 x i2560 | 591.6 B | 19.9 B | 140.6 GiB | 32.6 GiB |
| `mhep-012` | 256 x i2816 | 649.6 B | 20.8 B | 149.0 GiB | 24.2 GiB |
| `mhep-013` | 256 x i3072 | 707.6 B | 21.7 B | 157.5 GiB | 15.8 GiB |

- Expectation: `mhep-011` fits. `mhep-013` is at the estimate boundary and can fail, because the
  estimate omits the FA4 workspace, the cross-entropy logit blocks, NCCL buffers, fragmentation, and
  the Newton-Schulz transient (about 7 GiB at four experts per device). An OOM there is a result.
- A fourth candidate, 512 x i1536 at top-8, estimates 167.2 GiB with 6.1 GiB of headroom. It was not
  queued, because it fails the same estimate by a larger margin.

Capacity sweep at the measured MHEP-009 shape (128 x i3072, top-4), where capacity 1.0 dropped
9.9683% of assignments: `mhep-014` at 1.125, `mhep-015` at 1.25, `mhep-016` at 1.5. Cell capacity
goes from 2,048 rows to 2,304, 2,560, and 3,072. The native EP hero measured a 4.1% relative MFU
cost for capacity 1.0625, thus a cost is expected here as well.

- Purpose: Price the drop correction that the MHEP-009 versus MHEP-010 comparison currently carries.
- Cluster note: Seven of our gangs are queued at once against a saturated A08. They serialize. This
  is a deliberate choice to keep the queue full while 12-rack runs cycle.

### 2026-08-05 02:45 UTC - Size ladder result: the one-rack ceiling is between 592 B and 650 B

- 591.6 B (`mhep-011`, 256 x i2560, top-4, 19.9 B active) PASSES. Median MFU is 24.0032%, tokens/s
  is 307,778, step time is 13.6277 s, final loss is 6.0396, and the drop fraction is 12.2660%.
  Four whole experts land on each device. Nothing this large ran on one rack before.
- 649.6 B (`mhep-012c`, 256 x i2816) FAILS with `JaxRuntimeError: INTERNAL: NCCL operation
  ncclAlltoAll(...)` on several ranks, from `NCCL WARN Cuda failure 2 'out of memory'`. Stopped
  after repeated retries.
- 707.6 B (`mhep-013c`, 256 x i3072) stopped without a clean measurement. It is larger than the
  configuration that already fails, thus it is declared too big for one rack.
- Failure mechanism: XLA reserves the model, then NCCL allocates its all-to-all send and receive
  buffers outside the XLA pool through `cudaMalloc`. With `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`
  XLA grows on demand and keeps no headroom, thus NCCL fails instead of XLA. The buffers are about
  3.2 GiB each for send and receive at this shape, plus internal channels.
- Correction: An earlier entry in this session called the 650 B failure not-an-OOM, because
  `RESOURCE_EXHAUSTED`, allocator, and `XlaRuntimeError` searches returned nothing. That search
  missed the NCCL path. It is an out-of-memory failure.
- Estimate limit: The `resident = 53.8 GiB + 0.1465 x params_B` curve counts XLA-side memory only.
  It omits NCCL buffers, thus it overstates what fits. Use 592 B as the measured pass and 650 B as
  the measured fail, not the arithmetic ceiling.
- Untested lever: Cap XLA with `XLA_PYTHON_CLIENT_MEM_FRACTION` instead of `cuda_async`, or lower
  `NCCL_BUFFSIZE`, to leave the communicator room. Neither was measured.

### 2026-08-05 03:35 UTC - MHEP-019 stopped: lower top-k does not rescue the 641 B tier

- Configuration: 192 experts x i3712 at top-3, 641.4 B total, 20.7 B active, three whole experts per
  device. Dispatch buffers are 2.25 GiB, a quarter less than the 3.00 GiB of the 649.6 B run that
  already failed.
- Result: The rank-0 diagnostic reports `NCCL operation ncclAlltoAll(...)`, the same failure as
  MHEP-012c, with the coscheduled siblings cascading. Stopped after two failures.
- Interpretation: Two different routings now fail near 650 B. Thus the binding constraint is total
  parameter memory that leaves NCCL too little room, not the size of the dispatch buffers. Lower
  top-k reduces the communicator demand but does not offset the parameter growth.
- Ceiling update: One rack fits 591.6 B (measured pass) and does not fit 641.4 B (measured fail).
  The earlier bound of 592 B to 650 B narrows to 592 B to 641 B.
- Still open: MHEP-020 tests the other direction, holding parameters at the 590.7 B of the known
  pass while raising dispatch to 4.50 GiB with top-6. A pass there confirms that parameters, not the
  communicator, set the ceiling.

### 2026-08-05 03:45 UTC - MHEP-020 stopped: top-k is the expensive axis, width is the cheap one

- Configuration: 256 experts x i2560 at top-6, 590.7 B total, 24.5 B active. Parameters are equal to
  the MHEP-011 configuration that passed, so only the routing multiplicity changed.
- Result: `worker_failed` with one failed task and 15 coscheduled cascades, after the run entered
  the training loop. Stopped after four failures.
- Correction: This session first estimated top-6 as about 4.5 GiB more than top-4. That is wrong.
  Six buffers scale with top-k, and one of them is float32:

| buffer | top-4 | top-6 |
| --- | --- | --- |
| dispatch send (bf16) | 3.00 GiB | 4.50 GiB |
| received after all-to-all (bf16) | 3.00 GiB | 4.50 GiB |
| expert w13 output (bf16) | 2.50 GiB | 3.75 GiB |
| expert output and combine (bf16) | 3.00 GiB | 4.50 GiB |
| gathered [T, k, H] (bf16) | 3.00 GiB | 4.50 GiB |
| backward grad_rows (float32) | 6.00 GiB | 9.00 GiB |
| total | 20.50 GiB | 30.75 GiB |

- The true delta is 10.25 GiB, and its peak is in the backward pass. That matches the observed
  failure after the training loop started rather than during compilation.
- Rule for sizing: active routed neurons are top-k multiplied by the expert width. Parameters track
  expert count multiplied by width, and the k-scaled buffers track tokens multiplied by top-k.
  Width is thus the cheap way to buy active compute and top-k is the expensive way.
- Next test: MHEP-021 runs 128 experts x i5120 at top-4. It doubles the active neurons of MHEP-011
  to 20,480 at the same 590.7 B of parameters, for 23.00 GiB of k-scaled buffers.

### 2026-08-05 05:50 UTC - Four GB200 runs still outstanding; ablation moves to H100

- Outstanding on `cw-us-east-08a`, with exact job ids and collection steps recorded in
  [`7279-outstanding-b200-runs.md`](7279-outstanding-b200-runs.md):
  `/rav/mhep-021-wide-591b-e128-i5120-k4-p32579-20260805-coord` (128 x i5120 top-4, 20,480 active
  neurons), `/rav/mhep-022-fine-591b-e512-i1280-k4-p32580-20260805-coord` (512 x i1280 top-4, eight
  experts per device), `/rav/mhep-023-xprof-10-p32581-20260805-coord` (XProf steps 5 and 6 on rank
  0), and `/rav/mhep-024-nsys-10-p32582-20260805-coord` (Nsight Systems on task 0).
- All four were still queued at the snapshot. MHEP-021 and MHEP-022 each carry one real failure, and
  their memory estimates put them near the measured out-of-memory boundary, so read their logs for
  `ncclAlltoAll` before treating a failure as an infrastructure fault.
- Reason the small-scale ablation moves to H100: A08 stays contended, so the GB200 racks are not
  available for a nine-run sweep.
- H100 attention finding: `gpu_fa4_cute` is Blackwell-only. Its MMA op accepts sm_100, sm_103, and
  sm_110, and rejects H100 with `expects arch to be one of [Arch.sm_100a, ...], but got sm_90a`.
  `gpu_fa4_thd` does carry SM90 forward and backward kernels, but it requires fixed-shape THD
  segment metadata that this model does not supply, so it raises instead. Reference attention is the
  remaining option.
- Correction: this session first called the reference-attention cost about 16 times, from the ratio
  of attention span (8192 against a 512 window). That is wrong. Attention is a minority of the FLOP
  budget at these shapes, so losing the window costs 1.39 to 1.43 times the analytic FLOPs.
- H100 target: 8 nodes of 8 GPUs, which is 64 GPUs and an expert axis of 64. One node was rejected
  because capacity is per (sender shard, expert) cell: EP8 gives 4,096-row cells against 512 at
  EP64, so it would drop far less on the same routing and would not reproduce GB200 behavior.

### 2026-08-08 09:26 UTC - MHEP-146 to MHEP-148 allocator watch gate ready

- Hypothesis: An explicit XLA pool limit can leave enough HBM for NCCL while the capacity-2 EP
  arm logs full gradient and parameter norms. Prior full-watch compilation reported a 197.42 GiB
  rematerialization floor on a 184.30 GiB GPU, so failure on the first watch step is expected.
- Code snapshot: `078813ee6`. The launcher accepts `--watch-interval`, with zero as the default.
  Interval 1 selects the existing `WatchConfig` defaults: gradient and parameter targets, global
  and per-parameter norms, scan-layer splitting, and no histograms.
- Common config: 481.1 B total and 23.3 B active parameters; d6144; 48 layers; 192 experts;
  latent dimension 3072 with RMSNorm; expert width 5504; top-4; capacity factor 2.0; cell capacity
  2,731; batch 1,024; sequence length 4,096; EP64 `fixed_all_to_all`; one 64-GPU GB200 rack;
  pinned-host optimizer state; five steps on the 2,000-step schedule; no eval, profile, checkpoint,
  or retry.
- Arms: MHEP-146 sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` for a 147.44 GiB XLA limit and
  36.86 GiB outside the pool. MHEP-147 sets 0.95 for 175.09 GiB and 9.22 GiB. MHEP-148 sets 0.65
  for 119.80 GiB and 64.51 GiB. All arms keep `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
- Run IDs: `mhep-146-w7-ep-cf2p00-fullwatch-memfrac80-p32744-20260808`,
  `mhep-147-w7-ep-cf2p00-fullwatch-memfrac95-p32745-20260808`, and
  `mhep-148-w7-ep-cf2p00-fullwatch-memfrac65-p32746-20260808`.
- Command template: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --enable-extra-resources --target-cluster cw-us-east-08a --priority production --cpu 2
  --memory 8GB --disk 32GB --timeout 28800 --max-retries 0 --job-name <run>-coord
  -e WANDB_API_KEY <secret> -e WANDB_PROJECT rav_moe -e WANDB_ENTITY marin-community
  -e IRIS_PORT_JAX <port> -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async
  -e XLA_PYTHON_CLIENT_MEM_FRACTION <fraction> -- python -m
  experiments.grug.moe_hero_ep.launch --run-id <run> --num-steps 5 --schedule-steps 2000
  --watch-interval 1 --version 2026.08.08 --run --num-experts 192 --intermediate-dim 5504
  --num-experts-per-token 4 --latent-dim 3072 --capacity-factor 2.0 --batch-size 1024`.
- Stop criteria: Keep each first failure and its complete logs. Stop on a terminal state, a
  non-finite loss, or completion of step 5. Do not resubmit an allocator or compilation OOM.
- Next action: Commit this launch contract, submit the three arms, and monitor after 120 seconds.

### 2026-08-08 09:37 UTC - MHEP-146 to MHEP-148 stopped after allocator failures

- All three jobs were stopped at the user's request. Iris reports each coordinator and training
  child as `killed`, with `Terminated by user` as the reason.
- MHEP-147, with memory fraction 0.95, reached the first training execution and failed in
  `ncclAlltoAll` with CUDA out of memory. It did not complete a step or log norms.
- MHEP-146, with memory fraction 0.80, reported that rematerialization could reduce the program
  only from 210.26 GiB to 203.99 GiB against a 176.64 GiB target. It did not complete a step.
- MHEP-148, with memory fraction 0.65, was stopped during start or compilation. Its retained logs
  do not contain a complete root-cause message.
- Decision: Do not run more memory-fraction arms for the full EP64 watch. Raising the fraction
  still leaves too little room for NCCL, while lowering it cannot make the compiled program fit.

### 2026-08-08 10:02 UTC - MHEP-149 and MHEP-150 d768 premise pair ready

- Hypothesis: At fixed active compute, 192 routed experts improve held-out loss compared with 128
  routed experts. The expert count is the only model or training variable in the pair.
- Code snapshot: `42835f581`, after a rebase on the latest `origin/main`. The small-scale launcher
  accepts `--watch-interval`; zero stays the default.
- Common config: d768; 8 layers; expert width 688; latent dimension 384 with RMSNorm; top-4;
  capacity factor 2.0; batch 1,024; sequence length 4,096; 4,194,304 tokens per step; 9,703 steps;
  750 tokens per active parameter; seed 0; EP64 `fixed_all_to_all`; one 64-GPU GB200 rack per arm;
  the two-phase datakit mixture; Paloma and uncheatable evaluation every 1,000 steps; checkpoints
  every 30 minutes; and full gradient and parameter norms every 10 steps without histograms.
- Arms: MHEP-149 uses 192 routed experts. MHEP-150 uses 128 routed experts. The two arms use the
  same expert width, latent width, top-k, capacity factor, batch, data, seed, and schedule.
- Run IDs: `mhep-149-d768-ep-e192-i688-l384-k4-cf2p00-t750-s0-w10-p32747-20260808` and
  `mhep-150-d768-ep-e128-i688-l384-k4-cf2p00-t750-s0-w10-p32748-20260808`.
- W&B: Entity `marin-community`, project `rav_moe`, group `moe-hero-ep-small-abl`.
- Submit command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --enable-extra-resources --target-cluster cw-us-east-08a --priority production --cpu 2
  --memory 8GB --disk 32GB --timeout 7200 --max-retries 2 --job-name <run>-coord
  -e WANDB_API_KEY <secret> -e WANDB_PROJECT rav_moe -e WANDB_ENTITY marin-community
  -e IRIS_PORT_JAX <port> -- python -m experiments.grug.moe_hero_ep.small_scale_abl_launch
  --run-id <run> --size d768 --target gb200-rack --flavor ep --seq-len 4096
  --tokens-per-step 4194304 --capacity-factor 2.0 --num-experts <128-or-192>
  --num-experts-per-token 4 --intermediate-dim 688 --latent-dim 384
  --tokens-per-active-param 750 --watch-interval 10 --version 2026.08.08 --run`.
- Success criteria: Both arms complete step 9,703 with finite losses, final checkpoints, and Paloma
  results. Compare the final and last three evaluation points, drop fractions, throughput, and norm
  curves. Stop a run on a non-finite loss or a repeated model or allocator failure.
- Expected execution time: About 36 minutes per arm, excluding queue time.

### 2026-08-08 10:10 UTC - MHEP-151 to MHEP-153 full-watch size ladder ready

- Goal: Find the largest capacity-2 EP64 model that can send full gradient and parameter norms to
  W&B. Keep the 192-expert hero shape and change only the routed-expert width.
- Code snapshot: `08b80ae88`.
- Common config: d6144; 48 layers; 192 experts; latent dimension 3072 with RMSNorm; top-4;
  capacity factor 2.0; batch 1,024; sequence length 4,096; EP64 `fixed_all_to_all`; one 64-GPU
  GB200 rack; pinned-host optimizer state; five steps on the 2,000-step schedule; and full gradient
  and parameter norms on every step. Eval, profiles, checkpoints, retries, and an explicit XLA
  memory fraction are disabled. The CUDA async allocator stays enabled.
- Size ladder: MHEP-151 uses width 4,096, with 361.47 B total and 20.83 B active parameters.
  MHEP-152 uses width 4,608, with 404.96 B total and 21.74 B active parameters. MHEP-153 uses
  width 4,992, with 437.58 B total and 22.42 B active parameters. The known failed upper bound is
  the 481.1 B width-5,504 model.
- Run IDs: `mhep-151-w7-ep-e192-i4096-cf2p00-fullwatch-p32749-20260808`,
  `mhep-152-w7-ep-e192-i4608-cf2p00-fullwatch-p32750-20260808`, and
  `mhep-153-w7-ep-e192-i4992-cf2p00-fullwatch-p32751-20260808`.
- Submit command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --enable-extra-resources --target-cluster cw-us-east-08a --priority production --cpu 2
  --memory 8GB --disk 32GB --timeout 28800 --max-retries 0 --job-name <run>-coord
  -e WANDB_API_KEY <secret> -e WANDB_PROJECT rav_moe -e WANDB_ENTITY marin-community
  -e IRIS_PORT_JAX <port> -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async -- python -m
  experiments.grug.moe_hero_ep.launch --run-id <run> --num-steps 5 --schedule-steps 2000
  --watch-interval 1 --version 2026.08.08 --run --num-experts 192
  --intermediate-dim <width> --num-experts-per-token 4 --latent-dim 3072
  --capacity-factor 2.0 --batch-size 1024`.
- Decision rule: A model fits only if all five steps finish and W&B receives finite gradient and
  parameter norms. A compile or NCCL memory failure is an upper bound. Use a new midpoint if two
  adjacent ladder points give different results.

### 2026-08-08 10:21 UTC - MHEP-154 exact top-6 target ready

- Question: Can the target EP64 model send full norms with d6144, 48 layers, 192 experts, expert
  width 4,608, latent width 3,072, top-6, and capacity factor 1.42?
- Code snapshot: `b92b5dad5`.
- Size: 404.96 B total and 25.81 B active parameters. The routing cell has 2,909 rows. This is 6.5%
  more routing capacity than the top-4, capacity-2 MHEP-152 cell with 2,731 rows.
- Run ID: `mhep-154-w8-ep-e192-i4608-k6-cf1p42-fullwatch-p32752-20260808`.
- Config: One 64-GPU GB200 rack, batch 1,024, sequence length 4,096, five steps on the 2,000-step
  schedule, and full gradient and parameter norms on every step. Eval, profiles, checkpoints,
  retries, and an explicit XLA memory fraction are disabled. The CUDA async allocator stays
  enabled.
- Decision rule: The exact target fits only if all five steps finish and W&B receives finite
  gradient and parameter norms.

### 2026-08-08 10:25 UTC - The 437.58 B full-watch model passes

- MHEP-151, MHEP-152, and MHEP-153 completed all five steps with finite losses. Each W&B run has
  38 gradient norm metrics and 38 parameter norm metrics.
- Final losses at step 4 are 10.5670 for width 4,096, 10.5501 for width 4,608, and 10.5396 for
  width 4,992.
- The width-4,992 compile warned that rematerialization could reduce the program from 184.44 GiB
  to 178.36 GiB against a 165.23 GiB target. Execution still finished all five watched steps.
- Result: 437.58 B total and 22.42 B active parameters is the largest confirmed full-watch EP64
  model. The known failed upper bound stays 481.1 B at width 5,504.

### 2026-08-08 10:25 UTC - MHEP-155 midpoint watch gate ready

- Goal: Narrow the full-watch size boundary between the successful width-4,992 model and the
  failed width-5,504 model.
- Code snapshot: `799fbb937`.
- Candidate: 192 experts at width 5,248, with 459.32 B total and 22.87 B active parameters. All
  other MHEP-153 settings stay fixed.
- Run ID: `mhep-155-w8-ep-e192-i5248-cf2p00-fullwatch-p32753-20260808`.
- Decision rule: The candidate fits only if all five steps finish and W&B receives 38 finite
  gradient norm metrics and 38 finite parameter norm metrics.

### 2026-08-08 10:27 UTC - The exact top-6 target passes full watch

- MHEP-154 completed all five steps with exit 0 and a finite final loss of 10.5563 at step 4.
- W&B received 38 finite gradient norm metrics and 38 finite parameter norm metrics.
- Conclusion: EP64 on one 16-node GB200 rack supports full norms for d6144, 48 layers, 192
  experts, expert width 4,608, latent width 3,072, top-6, and capacity factor 1.42.

### 2026-08-08 10:34 UTC - The 459.32 B full-watch model fails in NCCL

- MHEP-155 compiled the watched training step, then failed before step 1. NCCL reported CUDA out
  of memory while it allocated the all-to-all buffer. JAX reported the error from `jit_train_step`.
- Result: Width 5,248 is a failed upper bound. No retry is useful for this deterministic memory
  failure.

### 2026-08-08 10:34 UTC - MHEP-156 final size point ready

- Goal: Test the only 128-wide point between the successful width-4,992 model and the failed
  width-5,248 model.
- Code snapshot: `5d7041bc2`.
- Candidate: 192 experts at width 5,120, with 448.45 B total and 22.64 B active parameters. All
  other MHEP-153 settings stay fixed.
- Run ID: `mhep-156-w9-ep-e192-i5120-cf2p00-fullwatch-p32754-20260808`.
- Decision rule: A pass makes width 5,120 the largest confirmed 128-wide model. A failure keeps
  width 4,992 as the largest confirmed model.

### 2026-08-08 10:45 UTC - The full-watch size boundary is 448.45 B to 459.32 B

- MHEP-156 completed all five steps. Iris reports success for the child and coordinator. W&B
  finished at step 4 with a finite loss of 10.5326.
- W&B received 38 finite gradient norm metrics and 38 finite parameter norm metrics.
- Result: Width 5,120, with 448.45 B total and 22.64 B active parameters, is the largest confirmed
  128-wide point. Width 5,248, with 459.32 B total, is the failed NCCL upper bound.
- The discrete size search is complete. The remaining gap is one 128-wide increment, or 10.87 B
  total parameters.
- Teardown note: MHEP-154 finished all five steps and its W&B run finished. Fourteen Iris tasks
  reported success, while two task records stayed pending after a pod disappeared during teardown.
  The coordinator was stopped after the complete W&B result arrived. MHEP-155 tried to restart
  after its deterministic NCCL failure, so its coordinator was stopped to release the rack.

### 2026-08-08 12:05 UTC - The d768 E192 arm has a small one-seed loss gain

- Completion: MHEP-149 and MHEP-150 completed step 9,702. Both Iris coordinators and training
  children succeeded with exit 0, no failures, and no preemptions. Both W&B runs finished.
- Launch snapshot: `08b80ae88`.
- Final training loss: E192 is 2.0151 and E128 is 2.0377. E192 is lower by 0.02261, or 1.11%.
- Final held-out loss: E192 Paloma micro loss is 2.92373 against 2.93427 for E128. This is 0.01054,
  or 0.36%, lower. E192 uncheatable micro loss is 2.69066 against 2.70878. This is 0.01812, or
  0.67%, lower.

| step | E192 Paloma | E128 Paloma | difference | E192 uncheatable | E128 uncheatable | difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7,999 | 2.98517 | 2.99300 | -0.00783 | 2.76149 | 2.77415 | -0.01266 |
| 8,999 | 2.94702 | 2.95611 | -0.00909 | 2.71607 | 2.73233 | -0.01626 |
| 9,702 | 2.92373 | 2.93427 | -0.01054 | 2.69066 | 2.70878 | -0.01812 |

- Routing: E192 ends at 1.3364% drops against 0.7907% for E128. This is 0.5457 percentage points,
  or 69.0% relative, more drops.
- Speed: E192 ends at 9.228 million tokens/s against 9.547 million for E128. E192 is 3.35% slower.
  Median MFU is 5.902% for E192 and 6.127% for E128.
- Norms: Each run has 980 watched samples from step 0 through step 9,700. All global gradient and
  parameter norms are finite. E192 gradient norms start at 0.2116, are 0.3936 at step 4,860, and
  end at 0.1290. E128 values are 0.2122, 0.4123, and 0.1346. E192 parameter norms start at 864.6,
  are 1,831.1 at step 4,860, and end at 1,955.3. E128 values are 730.1, 1,770.4, and 1,898.4.
- Decision: The one-seed result supports the E192 hypothesis. The held-out gain is consistent and
  grows over the last three evaluations, but it is small. Run at least one more matched seed before
  selecting E192 for a larger training run.

### 2026-08-08 12:18 UTC - MHEP-157 to MHEP-160 memory-scheduler watch gates ready

- Goal: Make the MHEP-131 shape report full gradient and parameter norms without changing its
  model, batch, routing capacity, or allocator. Test compiler memory controls before changing the
  training step.
- Common config: d6144; 48 layers; 192 experts; latent dimension 3072 with RMSNorm; expert width
  5504; top-4; capacity factor 2.0; batch 1024; sequence length 4096; EP64 `fixed_all_to_all`;
  one 64-GPU GB200 rack; pinned-host optimizer state; CUDA async allocator; five steps on the
  2000-step schedule; and full gradient and parameter norms on every step. Eval, profiles,
  checkpoints, retries, and an explicit client memory fraction are disabled.
- MHEP-157 sets `JAX_MEMORY_FITTING_EFFORT=1.0` and `JAX_MEMORY_FITTING_LEVEL=O3`. JAX 0.10.1
  accepts both values. The defaults are 0.0 and O2.
- MHEP-158 sets `--xla_gpu_enable_latency_hiding_scheduler=false`. OpenXLA states that disabling
  latency hiding can reduce memory use by giving up compute and communication overlap. The EP
  runtime default is true.
- MHEP-159 lowers `--xla_gpu_experimental_parallel_collective_overlap_limit` from 4 to 1. This
  limits the number of collectives that the scheduler can overlap.
- MHEP-160 sets `--xla_gpu_enable_analytical_sol_latency_estimator=false`. OpenXLA lists this as a
  memory control because the estimator tries to maximize compute and communication overlap.
- Unsupported controls: This JAX/XLA build rejects `--xla_latency_hiding_scheduler_rerun=5` and
  `--xla_memory_scheduler=kBrkga` during local backend startup. Do not spend a rack on them.
- Run IDs: `mhep-157-w10-ep-e192-i5504-cf2p00-fullwatch-fit-o3-p32755-20260808`,
  `mhep-158-w10-ep-e192-i5504-cf2p00-fullwatch-lhs-off-p32756-20260808`,
  `mhep-159-w10-ep-e192-i5504-cf2p00-fullwatch-overlap1-p32757-20260808`, and
  `mhep-160-w10-ep-e192-i5504-cf2p00-fullwatch-sol-off-p32758-20260808`.
- Decision rule: A method passes only if all five steps finish and W&B receives 38 finite gradient
  norm metrics and 38 finite parameter norm metrics. Stop on the first deterministic compile or
  NCCL memory failure. Keep a successful compiler control for a longer throughput check.

### 2026-08-08 12:25 UTC - MHEP-161 separate diagnostic watch gate ready

- Hypothesis: Full watch fails because the gradient tree has the optimizer and the norm reduction
  as concurrent consumers. A separate diagnostic executable can compute the same forward,
  backward, gradient norms, and parameter norms without an optimizer update. After its scalar
  outputs resolve, the known-good no-watch training executable runs and can free gradients as the
  optimizer consumes them.
- Implementation: `--watch-mode diagnostic` selects the separate executable. `inline` remains the
  default. The diagnostic uses the same pre-update parameters, pending QB router biases, batch,
  mixed-precision policy, loss, and z-loss as the following training step. Metric names and values
  use the existing `compute_watch_stats` path. The mode supports gradient and parameter targets.
- Cost: A watched step repeats forward and backward. Interval 1 is a fit gate and should be close
  to twice the step cost. Interval 10 is the intended training setting and adds about 10 percent
  forward and backward work before any compiler overlap effects.
- Local checks: The diagnostic statistics match direct gradient and parameter statistics on a
  small differentiable model. The new test passes. The changed-file pre-commit checks pass.
- Run ID: `mhep-161-w11-ep-e192-i5504-cf2p00-fullwatch-diagnostic-p32759-20260808`.
- Common config and decision rule match MHEP-157 to MHEP-160. No compiler memory controls are set,
  so this arm isolates the separate diagnostic executable.

### 2026-08-08 12:32 UTC - A collective-overlap limit of one fits MHEP-131

- MHEP-159 completed all five steps with the exact MHEP-131 model and full watch on every step.
- W&B received 38 finite gradient norm metrics and 38 finite parameter norm metrics. The final
  loss was 10.5244 at step 4.
- The run reported 230,051 tokens/s. The 200-step MHEP-131 no-watch baseline reported 233,418
  tokens/s. This short comparison puts the full-watch arm 1.44% below the no-watch baseline.
- Result: `--xla_gpu_experimental_parallel_collective_overlap_limit=1` removes the observed memory
  failure without a model or allocator change. A longer matched run is necessary for a stable
  throughput cost.
- W&B: https://wandb.ai/marin-community/rav_moe/runs/mhep-159-w10-ep-e192-i5504-cf2p00-fullwatch-overlap1-p32757-20260808
- MHEP-157, MHEP-158, and MHEP-160 were still active at this result time.

### 2026-08-08 12:36 UTC - MHEP-162 and MHEP-163 larger-model gates ready

- Goal: Find a larger full-watch model after MHEP-159 made the 481.063 B MHEP-131 model fit.
- Common config: d6144; 48 layers; latent dimension 3072 with RMSNorm; top-4; capacity factor 2.0;
  batch 1024; sequence length 4096; EP64 `fixed_all_to_all`; one 64-GPU GB200 rack; pinned-host
  optimizer state; CUDA async allocator; five steps on the 2000-step schedule; and full gradient
  and parameter norms on every step. Eval, profiles, checkpoints, retries, and an explicit client
  memory fraction are disabled.
- Runtime control: Both arms set
  `--xla_gpu_experimental_parallel_collective_overlap_limit=1`, which passed MHEP-159.
- MHEP-162 uses 192 experts and expert width 6016. It has 524.549 B total parameters and 24.227 B
  active parameters.
- MHEP-163 uses 128 experts and expert width 8960. It has 520.906 B total parameters and 29.418 B
  active parameters.
- Run IDs: `mhep-162-w12-ep-e192-i6016-cf2p00-fullwatch-overlap1-p32760-20260808` and
  `mhep-163-w12-ep-e128-i8960-cf2p00-fullwatch-overlap1-p32761-20260808`.
- Decision rule: A candidate passes only if all five steps finish and W&B receives 38 finite
  gradient norm metrics and 38 finite parameter norm metrics. Stop a deterministic compile or
  NCCL memory failure without a retry.
