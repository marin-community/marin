---
date: 2026-07-28
system: iris
severity: degraded
resolution: mitigated
pr: none
issue: https://github.com/marin-community/marin/issues/7691
---

## TL;DR

- Two GB200 8-node, 4-GPU JAX gangs stopped before optimizer update zero.
- GPUs remained at 100 percent utilization but drew about 190-235 W against a 1200 W limit; InfiniBand carried only keepalives.
- NCCL RAS initially reported healthy communicators, then stopped answering live status queries.
- Mixed `nvidia-nccl-cu12` and `nvidia-nccl-cu13` installs shared one path, allowing different CUDA-major NCCL libraries to win by install order.
- Iris now restores CUDA 13 cuDNN and NCCL after GPU dependency synchronization and retains bounded diagnostic evidence before an operator takes action.

## Original problem report

Two 8-node x 4-GPU JAX gangs wedged before optimizer update zero. Python main threads were shallow in `pxla.py __call__ -> pjit -> train.py` dispatch. Pods and nodes were healthy.

## Investigation path

1. Thread dumps, GPU telemetry, and InfiniBand counters established that the processes were alive but not making useful accelerator progress.
2. Initial NCCL RAS `RUNNING/OK` output did not rule out the fault; later status commands themselves stopped returning.
3. An instrumented smoke separated compilation from the hang: `backend_compile_and_load` had zero GPU utilization, millisecond RAS responses, four healthy eight-rank communicators, and matching AllReduce counts.
4. Package inspection found both CUDA 12 and CUDA 13 NCCL distributions writing `nvidia/nccl`; matching arms consequently loaded different NCCL builds by symlink order.

## User course corrections

- The request required a standalone Iris/Marin change from main, not the nested-model experiment branch, preserving the diagnostic work as independent infrastructure.
- The request prohibited a public Telltale callback and required the authenticated `ProfileTask` path with durable `iris.profile` persistence.
- The request prohibited automatic kicks or restarts, preserving the live evidence for operators.

## Root cause

The task environment synchronized `nvidia-nccl-cu12==2.28.9` and `nvidia-nccl-cu13==2.28.9` into the same `nvidia/nccl` namespace. Installation order selected the loaded library. CUDA 13 cuDNN has the same shared-path property.

## Fix

`lib/iris/src/iris/cluster/setup_scripts.py` reinstalls the resolved CUDA 13 cuDNN and NCCL wheels last when present. `iris process profile distributed` records a bounded, partial-result NCCL/CUDA bundle through the existing authenticated profile path.

## How OPS.md could have shortened this

- Add the bounded distributed profile command to the Process Inspection & Profiling examples, with the instruction to capture it before restarting a stalled gang.
- Document high GPU utilization with a low power-limit ratio as a collective-spin clue, not proof of a collective fault.

## Artifacts

- `lib/iris/OPS.md`
- `docs/ops/training-stall-alert-contract.md`
