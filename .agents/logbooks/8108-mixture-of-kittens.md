---
topic: mixture-of-kittens
issue: https://github.com/marin-community/marin/issues/8108
description: Reimplement and measure the Mixture of Kittens MoE schedule and Blackwell forward path in JAX.
author: rav
---

# Mixture of Kittens: Task Logbook

## Scope

- Goal: Build a Marin-owned JAX implementation and find which fused Blackwell mechanisms improve the existing Grug MoE path.
- Primary metrics: Schedule parity, forward and gradient error, dropped assignments, dispatch time, expert compute time, combine time, and tokens per second.
- Constraints: Add one measured mechanism at a time. Use Marin numerical tolerances. Keep the first GPU screens short.
- Coordinating issue: [#8108](https://github.com/marin-community/marin/issues/8108).

## Current TL;DR

The first implementation reproduces the deterministic 256-row-padded schedule. The strict XLA device kernel and 32 updates per peer raise steady MFU from about 2.8% to 19.6% on one four-GPU GB200 worker. A raw JAX FFI path now passes fused BF16 forward and backward checks on four GB200 GPUs. A 32K macro-buffer lets the full model fit, but two 25-step screens became non-finite after 16 to 24 steps. A corrected top-4 test found that the source backward kernel reduces routed weight gradients in BF16 between macro-buffers. The kernel now writes FP32 partials for each macro-buffer, and JAX reduces them before the BF16 output cast. This treatment passes an eight-macro gradient gate and one full E8 training step.

Both measured arms use hash-pinned JAX `0.11.1.dev20260809` wheels. Its XLA pin includes the device kernel that is not in JAX 0.11.0. A full fused kernel can use XLA collective FFI, but jaxlib does not publish its collective context headers.

## Baseline

- Date: 2026-08-10.
- Marin code: `a5f0269edc35a3766958adb494cef7d371632ebd`.
- Reference code: `6438bf48f88094d305972fbe0fa6deba0f7d4d1a`.
- Marin EP baseline: Issue [#7331](https://github.com/marin-community/marin/issues/7331) measured 19.1% MFU and 15.7 seconds per step for `a2a_cute` at its reported shape.
- Comparison limit: The issue #7331 training shape differs from the reference microbenchmark shape.

## Background Research Brief

- Effort: Medium.
- Stop rule: Stop after the reference code, official design note, Marin backends, XLA source, and an adversarial issue pass agree on the first boundary.
- Date: 2026-08-10.

### Question

Which parts of the fused Blackwell MoE design can JAX 0.11 express through public interfaces, and which part needs a collective FFI kernel?

### Current Marin Context

The Grug `ragged_all_to_all` backend dispatches and combines assignments with `jax.lax.ragged_all_to_all`. It runs expert computation as separate `ragged_dot` calls.

The QuACK bridge supplies Blackwell grouped gated and down GEMMs. It does not own rack-wide memory or dispatch.

### External Prior Art

The reference builds a destination-side schedule from all routes. It pads each local expert to 256 rows and interleaves peer assignments within each expert segment.

The BF16 megakernel uses pull-based dispatch, push-based combine, fixed communication SMs, compute clusters, minibatches, and a macrobatch ring buffer. It reuses one schedule for four communication operations.

JAX 0.11.0 pins XLA commit `131bf41acb4650e4391a640c3f1859c1c86ad74b`. That revision supports symmetric buffers and the one-sided Put and Signal mode. It does not contain the device-initiated ragged all-to-all kernel.

The device kernel landed in XLA commit `acb5aaffe4c0d844bacb57ad85234422f0ceaae0`. The 2026-08-09 JAX nightly pins XLA commit `7c3dd1936addd297d7c6fa46f6183986fc4160c3`, which is 457 commits after that change. The kernel uses NCCL LSA within one NVLink domain and requires symmetric input and output buffers.

### Contradictions and Limits

- The reference repository reports large kernel speedups, but it does not isolate schedule, transport, and GEMM changes.
- Its BF16 checks allow `atol=0.5` and `rtol=0.01`. These limits are too loose for Marin regression tests.
- The current XLA public FFI package exposes stage bundles. It does not include the C++ headers for collective memory requests or acquired collective memory.
- The device-initiated kernel does not fuse `ragged_all_to_all`, grouped GEMMs, and combine into one kernel.
- Reference issue 12 showed that an earlier benchmark did not include the required clamped SwiGLU behavior. The current reference commit includes the fix.

### Evidence Map

#### Claim: The current XLA device kernel is usable in the first JAX experiment

- Support: The 2026-08-09 JAX nightly pins an XLA revision after the device-kernel change. Marin pins NCCL 2.30.7, which is newer than the kernel's NCCL 2.29 minimum.
- Contradiction: The path is experimental and needs symmetric allocation for both transfer buffers.
- Directness to Marin: Marin already uses `jax.lax.ragged_all_to_all`. The experiment can install the nightly wheels only on its train tasks.
- Confidence: High for dependency and flag availability. Accelerator behavior is untested in this experiment.
- Action: Compare private and device implementations with the same nightly, executable shape, and routing.

#### Claim: A full fused reimplementation needs collective FFI

- Support: The reference kernel performs remote memory access and expert computation in one persistent kernel.
- Contradiction: XLA can allocate symmetric buffers, so it can supply the memory runtime to a custom kernel.
- Directness to Marin: Marin already builds CUDA FFI extensions at run time.
- Confidence: High for the kernel boundary. Low for the size of the performance gain.
- Action: Add no private XLA dependency until a profile shows removable boundary cost.

## Hypothesis Queue

### Active

- `MOK-JAX-003`: A fused forward kernel can remove enough transfer, compute, and launch boundaries to reach at least 25% MFU.

### Blocked

- None.

### Falsified or Dead End

- None.

### Promoted

- `MOK-JAX-001`: The JAX schedule matches an independent host reference for peer interleaving, rank offsets, padding, and overflow.
- `MOK-JAX-002`: XLA's device-initiated ragged all-to-all raised steady MFU from about 2.8% to 17.3%.
- `MOK-JAX-002B`: Thirty-two updates per peer raised steady MFU from 17.3% to 19.6%.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Mixture of Kittens | Code | https://github.com/cursor/mixture-of-kittens/tree/6438bf48f88094d305972fbe0fa6deba0f7d4d1a | Schedule and fused kernel behavior | Stable | Apache-2.0 |
| XLA at JAX 0.11 pin | Code | https://github.com/openxla/xla/tree/131bf41acb4650e4391a640c3f1859c1c86ad74b | Stable-release boundary | Stable | Does not include the device kernel |
| JAX 2026-08-09 nightly XLA pin | Code | https://github.com/openxla/xla/tree/7c3dd1936addd297d7c6fa46f6183986fc4160c3 | Device kernel, targeted symmetric allocation, and runtime flags | Stable | Pinned by exact wheel URLs and hashes |
| XLA device-kernel change | Code | https://github.com/openxla/xla/commit/acb5aaffe4c0d844bacb57ad85234422f0ceaae0 | First revision with NCCL LSA and GIN ragged transfer | Stable | Requires NCCL 2.29 or newer |
| Marin issue 7331 | Experiment | https://github.com/marin-community/marin/issues/7331 | Current Blackwell EP performance context | Replicated | Different training shape |
| Marin issue 7973 | Issue | https://github.com/marin-community/marin/issues/7973 | Related forward-port evaluation task | Stable | This issue owns the implementation record |

## Entry Log

### 2026-08-10 05:20 UTC - MOK-JAX design selected

- Hypothesis: XLA symmetric ragged all-to-all can measure the transport mechanism before Marin maintains a fused collective FFI kernel.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd`.
- Commands: `gh issue view 8108 --json ...`; `gh search code ... --repo openxla/xla`; source inspection with `rg` and `gh api`.
- Config: Marin JAX 0.11, BF16, XLA private versus symmetric ragged all-to-all, and the reference 256-row schedule padding.
- Result: JAX 0.11 contains both symmetric-memory flags. XLA collective FFI can request symmetric buffers, but its collective C++ context is private.
- Interpretation: Implement schedule parity first. Use the public symmetric ragged all-to-all mode for the first GB200 transport screen.
- Next action: Add behavior tests for the schedule before its JAX implementation.

### 2026-08-10 05:41 UTC - Latest XLA dependency gate resolved

- Hypothesis: A current JAX nightly supplies the device-initiated ragged kernel without a custom XLA build.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd` plus uncommitted task changes.
- Commands: Echo search for ragged all-to-all and XLA dependency records; JAX nightly index inspection; JAX and XLA revision checks through the GitHub API; focused pytest run.
- Config: JAX `0.11.1.dev20260809`, XLA `7c3dd1936addd297d7c6fa46f6183986fc4160c3`, NCCL 2.30.7, CUDA 13, and ARM64 GB200 wheels.
- Result: The nightly XLA pin is 457 commits after the device-kernel change. Four hash-pinned wheels can be installed only on this experiment's train tasks. Four focused dependency and flag tests passed.
- Interpretation: A direct XLA build is not necessary for the first gate. The treatment must use targeted ragged symmetric allocation plus the device-kernel flag. The separate symmetric ragged mode is not the treatment.
- Next action: Run the full local gate, commit the snapshot, and submit matched private and device runs.

### 2026-08-10 05:51 UTC - Local implementation gate passed

- Hypothesis: The schedule, launcher, train-task dependency pin, and XLA arm selection are valid before accelerator use.
- Commit Hash: `b0183783d`.
- Commands: `uv run pytest -n 0 tests/test_mixture_of_kittens.py -q`; `uv run pytest -n 0 tests/test_grug_variant_contracts.py -q`; `./infra/pre-commit.py --all-files --fix`; launcher plan output for a one-node private arm.
- Config: Ten focused tests, the full Grug variant contract suite, repository lint and type checks, and a one-step dry plan.
- Result: Ten focused tests passed. The contract suite passed 18 tests and skipped one test. The all-files check passed. The plan records all four nightly wheel URLs and hashes in its fingerprint.
- Interpretation: The local implementation is ready for one-node GB200 correctness screens. Accelerator behavior remains untested.
- Next action: Push the snapshot, publish the issue update, and submit the private arm before the device arm.

### 2026-08-10 06:34 UTC - One-shot profile isolated the transfer boundary

- Hypothesis: The initial device-kernel-off arm measures the current XLA ragged all-to-all path and can isolate its cost.
- Commit Hash: `07b1c65a6` for the run; `ec83dc2e4` for the next profile configuration.
- Commands: One four-GPU GB200 Iris run; 10 completed steps; five-step XProf capture; local XPlane summary on the worker.
- Config: JAX `0.11.1.dev20260809`, XLA `7c3dd1936addd297d7c6fa46f6183986fc4160c3`, NCCL 2.30.7, E8, top-4, global batch 64, BF16 compute.
- Result: Loss decreased from 11.8 at step 2 to 7.44 at step 10. Steps 3 through 5 took 119 to 123 seconds, or about 2.8% MFU. The trace shows that XLA selected `RaggedAllToAllWithSymmetricMemoryKernelImpl`, its one-shot copy kernel with the NCCL device barrier. The copy kernel used 74% of XLA device time, and the barrier used 14%. The XProf session and normalized summary are under `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mok-jax-002-private-1n-25-20260810-0600`.
- Interpretation: The path name `private` was not correct for the current nightly. The device-kernel comparison remains valid, but both paths must be selected explicitly and NCCL fallback must be off.
- Next action: Run the device kernel with the one-shot path and fallback disabled. Require its distinct device-kernel trace before performance acceptance.

### 2026-08-10 06:55 UTC - Device kernel passed but stayed below the target

- Hypothesis: The strict device-initiated XLA kernel reduces the transfer boundary enough to reach 25% MFU.
- Commit Hash: `683c4211f`.
- Commands: One four-GPU GB200 Iris run; 25 completed steps; W&B metric history; five-step XProf capture.
- Config: JAX `0.11.1.dev20260809`, XLA `7c3dd1936addd297d7c6fa46f6183986fc4160c3`, NCCL 2.30.7, E8, top-4, global batch 64, BF16 compute, one update per peer, device kernel on, one-shot and NCCL fallback off.
- Result: The run succeeded without a retry. Steps 2 through 4 took 20.21 to 20.29 seconds and reached 17.22% to 17.29% MFU. Loss decreased from 11.81 to 6.50 over 25 steps. The XProf session is under `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mok-jax-002-device-1n-25-20260810-0636`.
- Interpretation: The XLA device kernel gives a 6.2x MFU gain over the one-shot result, but one update per peer does not use enough CTAs and remains 7.7 percentage points below the target.
- Next action: Divide each peer transfer into 32 updates, validate unchanged layout and values, and rerun the same device arm.

### 2026-08-10 07:28 UTC - Multi-update device profile set the fusion boundary

- Hypothesis: Thirty-two updates per peer use the device kernel's full 64-CTA grid and reach 25% MFU.
- Commit Hash: `bf171bb39`.
- Commands: Shared Grug behavior tests; one four-GPU GB200 Iris run; 25 completed steps; W&B metric history; five-step XProf capture and normalized XPlane summary.
- Config: The prior device arm with 32 updates per peer. The model shape, data, batch, runtime, and XLA revision stayed fixed.
- Result: The run succeeded without a retry. Steps 2 through 4 took 17.82 to 17.84 seconds and reached 19.59% to 19.60% MFU. Loss decreased from 11.81 to 6.49. The device ragged all-to-all kernel fell from 4.73 to 2.14 seconds per step. All device collectives total about 2.87 seconds per step. The XProf session and normalized summary are under `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mok-jax-002-device-s32-1n-25-20260810-0710`.
- Interpretation: The split treatment improves steady MFU by 13.4%, but transfer tuning alone cannot reach the target. Subtracting all measured device-collective duration gives an optimistic arithmetic limit of about 23.4% MFU. A useful next increment must also remove launch gaps or overlap transfer with expert compute.
- Next action: Prototype the fused forward boundary. Prefer a runtime that owns NCCL symmetric buffers over a private XLA collective-context dependency.

### 2026-08-10 08:33 UTC - Fused JAX FFI forward passed the first GB200 gate

- Hypothesis: A raw CUDA JAX FFI adapter can run the fused BF16 dispatch, expert compute, and combine kernel without PyTorch or a private XLA API.
- Commit Hash: `ca858c2da` plus uncommitted adapter changes.
- Commands: CUDA 13 CPU build probe; one four-GPU GB200 Iris correctness job; independent JAX BF16 reference.
- Config: Mixture-of-Kittens `6438bf48f88094d305972fbe0fa6deba0f7d4d1a`, ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`, SM100a, four GPUs, 512 tokens per GPU, hidden and intermediate dimensions 256, four routed experts at top-1, and one fused shared expert.
- Result: The adapter compiled and linked without PyTorch. The four-GPU job succeeded. The maximum absolute output error was 0.03125, the mean absolute error was 0.00350, and all values passed the BF16 check.
- Interpretation: JAX FFI can host the fused peer-memory kernel on GB200. The remaining correctness work is the training-shape forward path and a JAX fallback gradient.
- Next action: Add a custom gradient, connect the fused forward to the experiment model, and run a training-shape correctness gate before the 25-step profile.

### 2026-08-10 09:00 UTC - Fused custom gradient passed the GB200 gate

- Hypothesis: A custom VJP can use the fused FFI forward and the exact JAX MoE fallback for backward propagation without a change to training gradients.
- Commit Hash: `ca858c2da` plus uncommitted fused-model changes.
- Commands: One four-GPU GB200 Iris correctness job; independent BF16 forward reference; gradient comparison against the training fallback.
- Config: The prior fused-forward gate with model weight layouts, one fused shared expert, and gradients for the input, router, routed expert, and shared expert arrays.
- Result: The job succeeded. The fused forward again had a maximum absolute error of 0.03125 and a mean absolute error of 0.00350. All eight gradient leaves matched the exact JAX fallback bit for bit.
- Interpretation: The FFI boundary and custom VJP are correct at the small four-GPU shape. The next risk is full training shape, memory use, and the nightly JAX FFI ABI.
- Next action: Run the local change gate, commit the fused increment, and submit a short full-model correctness run on the pinned nightly runtime.

### 2026-08-10 09:23 UTC - Full-shape gate found an explicit-sharding gap

- Hypothesis: The small-shape custom VJP also traces under the full Grug explicit mesh.
- Commit Hash: `3e196aaf7`.
- Commands: One four-GPU GB200 Iris run with one requested training step and the pinned nightly JAX runtime.
- Config: E8, top-4, global batch 64, BF16 compute, fused forward, and the 32-update XLA fallback backward.
- Result: The adapter built under the nightly FFI ABI, but training stopped during backward tracing before a step ran. The shared-expert down projection did not set its output sharding, so JAX rejected a contraction where both intermediate dimensions use the model axis. The later coordination error was only a result of the GPU task exit.
- Interpretation: This is an explicit-mesh annotation error, not a kernel, CUDA, memory, or collective error. The standard model shared expert already sets the required batch output sharding.
- Next action: Apply the same output-sharding rule to the fallback shared projection, rerun local checks, and repeat the one-step gate.

### 2026-08-10 09:31 UTC - Worst-case route capacity exceeded the memory target

- Hypothesis: The output-sharding fix lets the four-times-capacity fused forward and fallback backward compile at the full shape.
- Commit Hash: `2c327aa25`.
- Commands: One four-GPU GB200 Iris run with one requested training step and no retry.
- Config: The prior full-shape gate after the shared-down output-sharding fix.
- Result: Tracing passed and the adapter built. XLA estimated 295.75 GiB before rematerialization and could reduce it only to 290.81 GiB, above its 170.60 GiB target. The later device-versus-pinned-host sharding assertion followed that failed memory schedule. No training step ran.
- Interpretation: A four-times worst-case route reserve is not a viable training configuration. The matched XLA arm uses bounded receiver capacity. The fused schedule must also bound capacity, clip safely at full expert blocks, and report dropped routes.
- Next action: Use 1.1-times route capacity plus expert padding, make schedule clipping kernel-safe, return its dropped-route count, and repeat the exact gradient and full-shape gates.

### 2026-08-10 09:45 UTC - Bounded-capacity full-mesh gradient gate passed

- Hypothesis: The 1.1-times bounded schedule and explicit production mesh preserve the fused forward and exact fallback gradient.
- Commit Hash: `11bf9bfbc`.
- Commands: One four-GPU GB200 Iris correctness job with the full `replica_dcn`, `data`, `expert`, and `model` mesh.
- Config: 256 tokens per GPU, hidden and intermediate dimensions 256, four routed experts at top-1, one shared expert, bounded schedule capacity, and production weight sharding.
- Result: The job succeeded without a retry. The fused forward had a maximum absolute error of 0.03125 and a mean absolute error of 0.00350. All eight gradient groups matched the JAX fallback exactly.
- Interpretation: Bounded capacity, the explicit mesh, and sharded shared weights do not change the verified gradient. The remaining gate is full model memory and one completed training step.
- Next action: Repeat the one-step full-model run with the bounded schedule, then profile a longer run if it succeeds.

### 2026-08-10 09:49 UTC - Custom backward needs an internal rematerialization boundary

- Hypothesis: Reducing the fused route reserve from four times to 1.1 times is enough for one full-model step.
- Commit Hash: `11bf9bfbc`.
- Commands: One four-GPU GB200 Iris run with one requested training step and no retry.
- Config: E8, top-4, global batch 64, BF16 compute, fused forward, bounded schedule capacity, and the exact ragged all-to-all JAX fallback gradient.
- Result: XLA reduced its original estimate from 224.04 GiB to 219.11 GiB, still above its 170.60 GiB target. Execution then failed on a 12.66 GiB collective-memory allocation. This is 71.71 GiB lower than the four-times reserve result, but no step ran.
- Interpretation: Route capacity was one large cause, but the custom backward still builds its JAX fallback VJP outside the model block rematerialization boundary. The fallback forward activations stay live inside the custom backward and add about 48.5 GiB above the target.
- Next action: Put an explicit JAX checkpoint around the reference function before its VJP, repeat the exact gradient gate, and retry the one-step full-model run.

### 2026-08-10 10:33 UTC - Native fused backward passed the four-GPU gate

- Hypothesis: The source BF16 backward kernel can replace the memory-heavy JAX fallback VJP and preserve the training gradients.
- Commit Hash: `2ddae13f4` plus uncommitted native-backward changes.
- Commands: Five short four-GPU GB200 build and correctness iterations, followed by one terminal correctness job with no retry.
- Config: 512 tokens per GPU, hidden and intermediate dimensions 256, four routed experts at top-1, one shared expert, explicit production mesh, 256-token minibatches, and a 1,024-token macrobatch.
- Result: The adapter builds and runs the fused BF16 backward without PyTorch. Forward maximum error was 0.03125. Input, router, and routed-weight gradient maximum errors were 0.0625, 0.2632, and at most 0.5. Shared-weight gradients were summed across the expert axis and had maximum error 1.0 after four BF16 local reductions. All checks passed their stated limits with no mismatch.
- Interpretation: Native backward removes the JAX fallback activations and supplies correct distributed gradients. The remaining risk is full-shape compiler memory and training execution.
- Next action: Run one full-model step with the native backward, then submit the 25-step profile if the memory gate passes.

### 2026-08-10 10:50 UTC - Context recomputation reduced the full-model allocation

- Hypothesis: Saving the fused forward context through all 48 rematerialized layers causes the full-shape memory failure.
- Commit Hash: `70e208d23`.
- Commands: Two one-step four-GPU GB200 Iris runs and one four-GPU recomputation gradient gate.
- Config: E8, top-4, global batch 64, BF16 compute, fused forward and native backward, first with a saved context and then with backward context recomputation.
- Result: Saving the context made XLA request 378.61 GiB. Recomputing it reduced the compiler estimate to 216.05 GiB after rematerialization and the runtime allocation to 143.86 GiB. The allocator limit was 138.22 GiB, so no training step ran. The recomputed gradient gate passed before the full-shape run.
- Interpretation: Context lifetime was the main memory fault. The remaining runtime gap is 5.64 GiB, which matches the size of the routed activation ring at a 131,072-token macro-buffer.
- Next action: Reduce the macro-buffer to 32,768 tokens and use the source kernel's ring replay path.

### 2026-08-10 11:06 UTC - Two-pass ring replay passed the four-GPU gate

- Hypothesis: A macro-buffer smaller than the schedule reduces memory while the fused kernel replays routed activations without a gradient change.
- Commit Hash: `70e208d23` plus the ring-replay change.
- Commands: Fifteen focused tests, repository checks, and one four-GPU GB200 forward and backward job.
- Config: A 1,024-row schedule, a 512-row macro-buffer, a 256-row minibatch, and the same independent BF16 reference and gradient limits as the native-backward gate.
- Result: The terminal job `/rav/mok-ffi-ring-replay-4xgb200-20260810-1104` succeeded without a retry. Forward maximum error was 0.03125. All input, router, routed-weight, and shared-weight gradient checks passed with no mismatches.
- Interpretation: The JAX adapter can use the source ring replay mechanism. The full gate can use a 32,768-row buffer, which is one quarter of the prior routed workspace.
- Next action: Repeat the one-step full-model memory gate with the 32K macro-buffer.

### 2026-08-10 11:18 UTC - Full-shape fused training passed the memory gate

- Hypothesis: A 32,768-row macro-buffer lets the full E8 model finish one fused forward and backward step on four GB200 GPUs.
- Commit Hash: `c258d45a2` plus the epilogue-grid change.
- Commands: Two one-step four-GPU GB200 Iris runs with no retry; focused schedule and FFI tests; repository checks for the changed CUDA file.
- Config: E8, top-4, global batch 64, BF16 compute, 48 layers, a 1.1-times bounded route schedule, fused forward and native backward, backward context recomputation, and a 32,768-row macro-buffer.
- Result: The smaller ring reduced the rematerialized compiler estimate from 216.05 GiB to 209.29 GiB and removed the runtime allocation failure. The first run then found that the epilogue used 65,536 CUDA grid rows, one more than the grid-Y limit. A flat one-dimensional epilogue grid fixed the launch. The terminal run `/rav/mok-fused-ring32k-grid-1n-1-20260810-1112-coord` completed one training step without a retry. It reported loss 11.8 and 98,932 dropped route assignments.
- Interpretation: The fused forward and native backward now run at the complete training shape. The ring-memory change and the flat grid are both required for this shape.
- Next action: Run 25 steps with an XProf capture, measure steady MFU, and tune the fused configuration if it is below 25%.

### 2026-08-10 12:17 UTC - Long fused screens found a numerical failure

- Hypothesis: The 32K fused path can train for 25 steps and remove enough transfer and launch cost to reach 25% MFU.
- Commit Hash: `f521d5fee`.
- Commands: Two 25-step four-GPU GB200 Iris runs, W&B history queries, and one five-step XProf capture for each run.
- Config: E8, top-4, global batch 64, BF16 compute, 48 layers, fused forward and native backward, a 32,768-row macro-buffer, and saved MoE output rematerialization in the second run.
- Result: The first run became non-finite at state step 24. The saved-output run became non-finite at state step 16. Both runs reached about 20.3% MFU before the failure. The first profile showed three fused forward calls and one backward call per layer and step. Saving the MoE output did not change the best step time enough to reach the target.
- Interpretation: The fused path is faster than the one-update XLA arm, but it is not yet a correct training implementation. The non-finite result must be fixed before more throughput tuning.
- Next action: Increase the gradient gate from top-1 to top-4 and require more than one real macro-buffer.

### 2026-08-10 12:36 UTC - Corrected ring test isolated routed weight gradients

- Hypothesis: The earlier ring gate exercised more than one real macro-buffer.
- Commit Hash: `f521d5fee` plus diagnostic smoke-test changes.
- Commands: Four top-4 four-GPU GB200 gradient jobs with one, two, and eight real macro-buffers.
- Config: 512 tokens per GPU, hidden and intermediate dimensions 256, all four expert routes per token, 256-row minibatches, and 256- to 2,048-row macro-buffers.
- Result: The earlier gate had one real macro-buffer because only 512 scheduled rows were valid. The corrected one-buffer top-4 case passed. Two macro-buffers caused routed gate and down gradient mismatches. Eight macro-buffers increased routed weight-gradient mean error from about 0.083 to about 0.113. Forward output, input gradient, router gradient, and shared gradients continued to pass.
- Interpretation: The source kernel stores each macro-buffer's routed weight gradient in BF16 and adds later contributions in BF16. The split changes only routed weight gradients and is a concrete cause for long-run numerical drift.
- Next action: Keep the memory-bounded 32K ring. Write each routed weight-gradient partial in FP32 and reduce the partials in JAX.

### 2026-08-10 12:58 UTC - FP32 routed partials passed correctness and memory gates

- Hypothesis: FP32 routed weight-gradient partials remove the macro-buffer split error without the memory cost of a full schedule buffer.
- Commit Hash: `f521d5fee` plus the FP32-partial change.
- Commands: Two- and eight-macro four-GPU GB200 gradient jobs; one full E8 training step; focused tests; repository checks.
- Config: The corrected top-4 smoke shape and the full E8, top-4, global-batch-64 training shape. The training run kept the 32,768-row macro-buffer and wrote nine FP32 partial sets at the static schedule capacity.
- Result: Both gradient jobs passed with zero mismatches. Routed weight-gradient mean errors returned to about 0.083, the same result as the one-macro gate. The full run `/rav/mok-fused-fp32-wgrad-1n-1-20260810-1305-coord` completed one step at loss 11.8 without a retry or memory failure. Fifteen focused tests and all checks for the changed files passed.
- Interpretation: The backward kernel can keep the small routed-activation ring and preserve one-macro gradient quality. The next long run can test whether this correction removes the non-finite result.
- Next action: Run 25 steps with XProf, require finite completion, and measure steady MFU against the 25% gate.
