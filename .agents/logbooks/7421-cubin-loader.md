---
topic: 7421-cubin-loader
issue: https://github.com/marin-community/marin/issues/7421
description: Instrument the production CUDA module-loader boundary.
author: mcwitt
---

# Issue #7421 CUDA module loader: Task Logbook

## Scope

- Goal: classify the L48/B512 CUBIN failure as a prior asynchronous error, loader API mismatch, pointer or ordering behavior, resource pressure, or image/context state.
- Primary metrics: per-rank loader outcome sequence, synchronization result, retry result, HBM/pool snapshot, and first-step outcome.
- Constraints: preserve the confirmed 16xGB200x4 graph and process topology; no Iris restart; region-local artifacts only.
- Coordinating issue: https://github.com/marin-community/marin/issues/7421

## Hypothesis Queue

### Active

- `H-ASYNC`: a preceding asynchronous CUDA operation failed.
- `H-API`: raw ELF CUBINs fail specifically through `cuModuleLoadFatBinary`.
- `H-ORDER`: concurrent XLA/CuTe loader ordering changes driver state.
- `H-INPUT`: pointer lifetime, mutation, or alignment changes the result.
- `H-PRESSURE`: a live memory or module resource is exhausted.

## Entry Log

### 2026-07-21 - CUBIN7421-001 probe implementation start

- Hypothesis: module-boundary outcomes can distinguish the active hypotheses.
- Commit Hash: `0a3785463`
- Command: local fake-driver integration tests, then one-GPU JAX smoke.
- Config: trace, sync, Data-direct, and pressure profiles.
- Result: implementation in progress.
- Interpretation: no new runtime evidence yet.
- Next action: make private-handle `dlsym` interception fail under test.

### 2026-07-21 - CUBIN7421-001 local instrumentation checkpoint

- Hypothesis: a `dlsym` interposer can observe XLA's private-driver-handle lookup and preserve enough loader state to distinguish async errors, API behavior, input ownership, overlap, and a paired pressure treatment.
- Commit Hash: based on `0a3785463`; checkpoint commit pending.
- Commands:
  - `uv run ... pytest experiments/grug/test_dispatch.py experiments/grug/moe/standalone/test_cuda_module_probe.py experiments/grug/moe/standalone/test_cuda_module_probe_tools.py`
  - `./infra/pre-commit.py <instrumentation files>`
  - `uv run --with pyrefly pyrefly check <instrumentation Python files>`
  - `g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared ...`
- Config: private-handle fake CUDA driver; trace, sync, Data-direct, and pressure profiles; two-thread overlap; task-zero content-addressed capture.
- Result: 16 focused tests passed in 14.83 seconds. Scoped repository lint passed. Pyrefly reported zero errors. The probe compiled with warnings treated as errors. Source SHA-256: `f79b8e1d5090f6a4c42605cca2557ff335f70dc1482df9e8417f9b10d86c2100`.
- Interpretation: the local harness validates the event and treatment contracts. It does not establish that the interposer reaches JAX/XLA in the production CUDA environment.
- Next action: run a one-GB200 JAX smoke and require both a FatBinary symbol redirect and a raw ELF load before the 16-host diagnostic.

### 2026-07-21 - CUBIN7421-002 one-GB200 smoke attempt 1

- Hypothesis: the locally validated interposer reaches JAX/XLA unchanged on an aarch64 GB200 worker.
- Commit Hash: `db61c4a4f`.
- Job: `/mwittmann/cubin7421-probe-smoke-001`.
- Config: one GB200; JAX 0.10.1; trace profile; required FatBinary and raw ELF coverage; task-zero capture.
- Result: failed before JAX output with no probe events. The worker built the expected source hash and binary, then the child returned before the coverage check.
- Interpretation: `real_dlsym` requested only the x86-64 symbol version `GLIBC_2.2.5`; the aarch64 worker uses `GLIBC_2.17`. This is an instrumentation portability defect, not evidence about the CUDA failure.
- Next action: resolve both glibc symbol versions, rerun local tests, and repeat the same smoke.

### 2026-07-21 - CUBIN7421-002b one-GB200 smoke attempt 2

- Hypothesis: after resolving the aarch64 glibc symbol version, the interposer reaches the actual raw-ELF loader path used by JAX 0.10.1 on GB200.
- Commit Hash: `3ec1f64cc`.
- Job: `/mwittmann/cubin7421-probe-smoke-002`.
- Config: one GB200; JAX 0.10.1; trace profile; required FatBinary and raw ELF coverage; task-zero capture.
- Result: succeeded. JAX returned `cuda:0` and sum `1048575.875`. The probe observed 10 `cuModuleLoadFatBinary` calls on raw ELF inputs, all with original result 0; eight unique CUBIN hashes; maximum in-flight loads 1. Source SHA-256: `974d807b893db6642522bb15c16a40e72173947a3f9d031ea09c199f52cdae21`. Binary SHA-256: `253648ba9e9b4b46a318982cb777bb5c0068bcf169670fb527b759fef201f436`.
- Artifacts: `s3://marin-us-east-02a/marin/scratch/mwittmann/cubin-diag/CUBIN7421-002b/task-0/`.
- Interpretation: the probe builds and loads on the target architecture, intercepts the production XLA loader API, parses the raw ELF inputs, and preserves a successful JIT. The instrumentation is ready for the multi-host positive control.
- Next action: run the exact archived L48/B512 configuration with task-index split trace/sync profiles and automatic retries disabled.

### 2026-07-21 - CUBIN7421-003 split trace/sync diagnostic

- Hypothesis: comparing the same failing graph across trace and pre-load synchronization ranks will distinguish a loader-local failure from an earlier asynchronous CUDA error.
- Commit Hash: `7ae727a82`.
- Jobs: `/mwittmann/cubin7421-trace-sync-l48-b512-coord` and `/mwittmann/cubin7421-trace-sync-l48-b512-002-coord`.
- Config: 16 GB200x4 hosts; d5120, L48, sequence 4096, 128 experts, top-5, batch 512; intermediate and shared intermediate 2560; ring EP8, replica axis 1; `gpu_fa4_cute`; scanned layers; CuTe MXFP8 producer; liger CE; MuonH with all three distribution knobs; `cuda_async`; local checkpoints; one training step. Even task indices use trace, odd task indices synchronize immediately before each load. Probe and Iris failure retries are zero.
- Result: the first coordinator failed before creating a GPU child because `bash -lc` reset the virtual-environment path and selected a Python without Fray. The corrected coordinator launched all 16 hosts and reproduced `CUDA_ERROR_INVALID_VALUE` at `jit_train_step`. Current Iris exports `IRIS_TASK_ID`, not `IRIS_TASK_INDEX`; every worker therefore defaulted to trace, and artifact upload raised `KeyError` before preserving the local files.
- Interpretation: the positive control remains reproducible with the probe loaded, but this attempt cannot compare trace with sync or classify the CUDA failure. The missing task identity is an instrumentation integration defect.
- Next action: derive the task index from canonical `IRIS_TASK_ID` in both the preload library and uploader, verify parity and task-zero capture in the fake-driver harness, then repeat after the GPU cooldown.

### 2026-07-21 - CUBIN7421-003b task-identity correction

- Hypothesis: canonical Iris task IDs provide the stable rank index required for profile parity and per-task artifact paths.
- Commit Hash: `30fdb93e7`.
- Commands: focused fake-driver and tooling tests; strict C++ compilation; scoped repository lint.
- Config: `IRIS_TASK_ID=/user/job/1:0` selects sync; `IRIS_TASK_ID=/user/job/0:0` enables task-zero capture; legacy `IRIS_TASK_INDEX` remains accepted for direct tools.
- Result: 19 focused tests passed. The C++ source compiled with `-Wall -Wextra -Werror`; scoped repository lint passed.
- Interpretation: both the interposer and Python uploader now derive the same index from the runtime identity current Iris actually exports.
- Next action: commit the correction and repeat the exact CUBIN7421-003 split run after cooldown.

### 2026-07-21 - CUBIN7421-003c corrected split trace/sync diagnostic

- Hypothesis: if a preceding asynchronous CUDA operation failed, odd task indices will surface that error from `cuCtxSynchronize` before entering the module loader.
- Commit Hash: `30fdb93e7`.
- Jobs: `/mwittmann/cubin7421-trace-sync-l48-b512-003-coord`, child `/mwittmann/cubin7421-trace-sync-l48-b512-003-coord/grug-train-cubin7421-trace-sync-l48-b512-003`, and collector `/mwittmann/cubin7421-probe-collect-003`.
- Config: the same 16-host L48/B512 positive control as CUBIN7421-003. Even task indices used trace; odd task indices synchronized before every module load. Retries remained disabled.
- Result: all 16 tasks uploaded complete artifacts. The probe recorded 94,116 FatBinary calls: 94,052 succeeded and 64 returned `CUDA_ERROR_INVALID_VALUE`, with four failures per task. All 47,024 sync-profile calls returned `CUDA_SUCCESS` from the pre-load synchronization. The 64 failures used an unknown, page-aligned input and occurred once per local device. Post-failure context and device queries succeeded, while `cuMemGetInfo` returned 201, so no HBM snapshot was available at the failing boundary. Maximum observed loader concurrency was four.
- Artifacts: `s3://marin-us-east-02a/marin/users/root/experiments/grug-moe-cw/grug-moe-cw-d5120-L48-e128-r16-cubin7421-trace-sync-l48-b512-003/dev/cuda-module-probe/`.
- Interpretation: a stale asynchronous CUDA error is not the direct source of the reported loader error. The original FatBinary call itself returns invalid value after a successful synchronization. OOM remains possible only through an upstream mechanism that produces an absent or malformed image, or through loader-local resource state; this run did not measure free memory at the failure.
- Next action: record whether unknown inputs are null. If they are non-null, add a bounded prefix identity before selecting Data-direct or paired-pressure treatments.
