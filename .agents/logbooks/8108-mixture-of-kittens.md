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

The first implementation reproduces the deterministic 256-row-padded schedule. The first GPU gate compares private ragged all-to-all with XLA's device-initiated NCCL LSA kernel on one four-GPU GB200 worker.

Both arms use hash-pinned JAX `0.11.1.dev20260809` wheels. Its XLA pin includes the device kernel that is not in JAX 0.11.0. A full fused kernel can use XLA collective FFI, but jaxlib does not publish its collective context headers.

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

- `MOK-JAX-001`: A pure JAX schedule can match the reference schedule for balanced, skewed, and padded routes.
- `MOK-JAX-002`: XLA's device-initiated ragged all-to-all reduces dispatch and combine cost on one four-GPU GB200 worker.

### Blocked

- `MOK-JAX-003`: A fused collective FFI kernel is blocked on a positive profile signal from `MOK-JAX-002`.

### Falsified or Dead End

- None.

### Promoted

- None.

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
