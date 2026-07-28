---
date: 2026-07-28
system: iris
severity: degraded
resolution: mitigated
pr: https://github.com/marin-community/marin/pull/7692
issue: https://github.com/marin-community/marin/issues/7691
---

## TL;DR

- Two GB200 8-node, 4-GPU JAX gangs stopped before optimizer update zero.
- GPUs remained at 100 percent utilization but drew about 190-235 W against a 1200 W limit; InfiniBand carried only keepalives.
- NCCL RAS initially reported healthy communicators, then stopped answering live status queries.
- Mixed `nvidia-nccl-cu12` and `nvidia-nccl-cu13` installs shared one path, allowing different CUDA-major NCCL libraries to win by install order.
- Iris now restores CUDA 13 cuDNN and NCCL after GPU dependency synchronization, publishes power-limit and training-progress metrics, and warns on stale optimizer progress.

## Original problem report

Two 8-node x 4-GPU JAX gangs wedged before optimizer update zero. Python main threads were shallow in `pxla.py __call__ -> pjit -> train.py` dispatch. Pods and nodes were healthy.

## Investigation path

1. Thread dumps, GPU telemetry, and InfiniBand counters established that the processes were alive but not making useful accelerator progress.
2. Initial NCCL RAS `RUNNING/OK` output did not rule out the fault; later status commands themselves stopped returning.
3. An instrumented smoke separated compilation from the hang: `backend_compile_and_load` had zero GPU utilization, millisecond RAS responses, four healthy eight-rank communicators, and matching AllReduce counts.
4. Package inspection found both CUDA 12 and CUDA 13 NCCL distributions writing `nvidia/nccl`; matching arms consequently loaded different NCCL builds by symlink order.

## User course corrections

- The request required a standalone Iris/Marin change from main, not the nested-model experiment branch, preserving the diagnostic work as independent infrastructure.
- The request prohibited automatic kicks or restarts, preserving the live evidence for operators.
- Review found the first `ProfileTask` collector too coupled to worker-side embedded scripts. It moved to a separate Loom session for a child-side module design.

## Root cause

The task environment synchronized `nvidia-nccl-cu12==2.28.9` and `nvidia-nccl-cu13==2.28.9` into the same `nvidia/nccl` namespace. Installation order selected the loaded library. CUDA 13 cuDNN has the same shared-path property.

## Fix

`lib/iris/src/iris/cluster/setup_scripts.py` reinstalls the resolved CUDA 13 cuDNN and NCCL wheels last when present. GPU jobs enable NCCL RAS and targeted INFO subsystems at process startup. Levanter progress and Iris GPU power-limit metrics feed a warning-only Grafana stalled-training rule.

## Follow-up

The [child-side distributed diagnostic session](https://loom.rjp.io/s/m07nj02f) owns the authenticated on-demand capture design. It must avoid embedded worker-side scripts and prove the task-namespace execution path before opening a separate PR.

## How OPS.md could have shortened this

- Add a bounded distributed profile command after the child-side collector has a tested task-namespace execution contract.
- Document high GPU utilization with a low power-limit ratio as a collective-spin clue, not proof of a collective fault.

## Artifacts

- `lib/iris/OPS.md`
- `docs/ops/training-stall-alert-contract.md`
- [Loom follow-up session](https://loom.rjp.io/s/m07nj02f)
