# CoreWeave Grug MoE Accelerator Perf Logbook

## Scope

- Goal: characterize current Grug MoE on CoreWeave H100, GH200, and B200.
- Branch: `research/cw-grug-moe-accelerator-perf`, based on #5428.
- GitHub issue: https://github.com/marin-community/marin/issues/5458.
- Current status: precondition validation only; single-node benchmark matrix is
  intentionally on hold.

## 2026-05-05T22:56Z - CW-GRUG-007 Direct Runtime Smokes on Current #5428 Branch

- Confidence: `exploratory`.
- Scope: direct one-GPU JAX/CUDA 13 smokes on
  `research/cw-grug-moe-accelerator-perf` commit `2574d5c55`, based on current
  #5428 head `7b53485c`.
- H100 job `/romain/cw-grug-direct-h100-20260505-2253`: passed on
  `NVIDIA H100 80GB HBM3`, driver `595.45.04`, backend `gpu`, matmul sum
  `512.0`.
- GH200 job `/romain/cw-grug-direct-gh200-20260505-2253`: passed on
  `NVIDIA GH200 480GB`, driver `595.45.04`, backend `gpu`, matmul sum `512.0`.
- B200 job `/romain/cw-grug-direct-b200-20260505-2253`: passed on
  `NVIDIA B200`, driver `595.45.04`, backend `gpu`, matmul sum `512.0`.
- All three reported `jax==0.10.0`, `jaxlib==0.10.0`,
  `jax-cuda13-plugin==0.10.0`, `jax-cuda13-pjrt==0.10.0`,
  `nvidia-cublas==13.4.1.1`, and `nvidia-nccl-cu13==2.28.9`.
- The first `20260505-2251` GH200/B200 probe failed before JAX/device execution
  because the probe looked up `nvidia-cublas-cu13` package metadata; the actual
  installed distribution is `nvidia-cublas`. H100 `20260505-2251` was stopped
  while pending to avoid spending GPU time on a known-bad probe.
- Logs are under `/tmp/marin-cw-grug-moe-accelerator-perf/direct-*-20260505-2253.log`.

## 2026-05-05T22:05Z - CW-GRUG-004/005/006 Short Training Smokes

- Confidence: `exploratory`.
- Scope: 50-step synthetic-data current Grug MoE smokes with profiler enabled on
  staged #5428 runtime.
- GH200 job `/romain/cw-grug-smoke-gh200-20260505-2153`: passed on
  `coreweave-rno2a`, request `H200x1`, run id `CW-GRUG-004-gh200-batchseq`,
  parent duration `195006ms`, final state `succeeded`, exit code `0`.
- H100 job `/romain/cw-grug-smoke-h100-20260505-2157`: passed on
  `coreweave-ci`, request `H100x8`, run id `CW-GRUG-005-h100x8`, parent
  duration `250008ms`, final state `succeeded`, exit code `0`.
- B200 job `/romain/cw-grug-smoke-b200-20260505-2157`: passed on
  `coreweave-usw09b`, request `B200x8`, run id `CW-GRUG-006-b200x8`, parent
  duration `275010ms`, final state `succeeded`, exit code `0`.
- All three reached 50/50 train steps, saved a step-50 checkpoint, emitted a
  profiler window, and reported W&B summary `backend gpu`, `global_step 49`.
- GH200 earlier attempts exposed sharding errors in current Grug MoE:
  `CW-GRUG-002-gh200` failed in the QB beta path with a `shard_map` input spec
  mismatch, and `CW-GRUG-003-gh200-reshard` exposed duplicate/incompatible
  batch/sequence sharding. The local fix adds explicit batch-sequence sharding
  in `experiments/grug/moe/model.py`.
- W&B ran offline, so profiler artifacts were produced inside the job container
  but are not durable cloud artifacts.
- Logs are under `/tmp/marin-cw-grug-moe-accelerator-perf/grug-smoke-*-20260505-*.log`.

## 2026-05-05T21:44Z - CW-GRUG-001 Direct Runtime Smokes

- Confidence: `exploratory`.
- Scope: direct one-GPU JAX/CUDA 13 smokes on staged #5428 runtime.
- H100 job `/romain/cw-grug-direct-h100-20260505-2138`: passed on `NVIDIA H100 80GB HBM3`, driver `595.45.04`, JAX backend `gpu`.
- GH200 job `/romain/cw-grug-direct-gh200-20260505-2138`: passed on `NVIDIA GH200 480GB`, driver `595.45.04`, JAX backend `gpu`.
- B200 job `/romain/cw-grug-direct-b200-20260505-2138`: passed on `NVIDIA B200`, driver `595.45.04`, JAX backend `gpu`.
- All three reported `jax=0.10.0`, `jax-cuda13-plugin=0.10.0`, `jax-cuda13-pjrt=0.10.0`, `nvidia-cublas=13.4.1.1`, and `nvidia-nccl-cu13=2.28.9`.
- Logs are under `/tmp/marin-cw-grug-moe-accelerator-perf/direct-*-20260505-2138.log`.
- Added a synthetic-data Grug smoke launcher plus GH200/B200 MFU mappings to prepare for training smokes.
- Local checks passed:
  - `uv run --group fray_test pytest tests/test_device_flops.py` from `lib/fray`.
  - `WANDB_MODE=offline uv run python -c "import experiments.ferries.cw_grug_accelerator_smoke"`.

## 2026-05-05T21:37Z - CW-GRUG-000 Setup

- Branch: `agent/20260505-accelerator-stack-validation`.
- Base HEAD: `749fd0790`.
- Runtime patch: #5428 head `96e3bb07a` is staged locally; not yet on `origin/main` (`72da6ca24`).
- Status: proceeding with first approval batch as provisional PR-head evidence.
- Dependency check: `uv lock --check` passed.
- Local limitation: macOS cannot install CUDA 13 JAX wheels, so GPU package versions are verified inside CoreWeave jobs.
