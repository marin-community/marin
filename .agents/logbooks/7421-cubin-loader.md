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
