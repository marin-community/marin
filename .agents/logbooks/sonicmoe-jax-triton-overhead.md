# SonicMoE jax-triton Overhead: Research Logbook

## Scope
- Goal: measure whether `jax_triton.triton_call` can launch a pure Sonic-style Triton token gather/sum path with less fixed overhead than the prior Pallas/FFI attempts.
- Primary metrics: steady per-call wall time for JAX/jax-triton versus direct PyTorch/Triton; PyTorch CUDA event time as the same kernel-body baseline.
- Constraints: run on GH200 via `coreweave-rno2a`; do not disturb the production QuACK backend PR.

## Baseline
- Date: 2026-05-13
- Code refs: SonicMoE commit `cfbd65f39b980b85b878b3cccdacb09191e24993`, prior production branch commit `5ec6613ec`.
- Baseline numbers:
  - Real Sonic token gather/sum: about `0.0528-0.056 ms` CUDA event, `0.064-0.068 ms` wall.
  - Prior Pallas/FFI attempts: best JAX-callable paths about `0.105-0.128 ms` wall for the same microkernel.
  - Full production backend already mostly closes the block-level gap; this probe is specifically for Tri's pure-Sonic/jax-triton launch-overhead question.

## Experiment Log

### 2026-05-13 23:15 - Add jax-triton overhead probe
- Hypothesis: `jax_triton.triton_call` may use JAX's Triton custom call path with less fixed host/runtime overhead than the earlier Pallas/FFI route, so a pure Sonic Triton body might get closer to the real Sonic launch path.
- Command:
  - Local syntax/import checks first.
  - GH200 command to be launched from `/Users/dlwh/.codex/worktrees/moe-jax-triton-overhead`:
    ```bash
    RUN_ID="sonicmoe-jax-triton-overhead-$(date -u +%Y%m%d-%H%M%S)"
    IRIS_CLUSTER=coreweave-rno2a uv run --package marin --group dev iris --cluster=coreweave-rno2a job run \
      --job-name "$RUN_ID" --no-wait --max-retries 0 \
      --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 96g --disk 96g --timeout 3600 \
      -e XLA_PYTHON_CLIENT_PREALLOCATE false \
      -- uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 python \
        .agents/scripts/sonicmoe_jax_triton_overhead_probe.py \
        --tokens 8192 --hidden 2048 --topk 2 --warmup 20 --repeats 200
    ```
- Config: clone of Sonic `token_gather_sum_kernel` without autotune wrapper; fixed-config sweep over `BLOCK_H={512,1024,2048}`, `BLOCK_K={1,2}`, `num_warps={4,8}`.
- Result: first run `/dlwh/sonicmoe-jax-triton-overhead-20260514-034326` failed before useful app logs. Debug rerun `/dlwh/sonicmoe-jax-triton-smoke-20260514-034552` showed `jax-triton==0.3.0` is incompatible with the Triton 3.6 line: `AttributeError: 'CUDABackend' object has no attribute 'get_arg_specialization'`.
- Interpretation: `jax-triton==0.3.0` is not a viable test of this path with our current GPU extra. Move to `jax-triton==0.3.1`, whose published metadata targets `jax>=0.8.2` and `triton>=3.6`.
- Next action: rerun a smoke with `jax-triton==0.3.1`, then launch the full sweep if the lowering path works.

### 2026-05-13 23:48 - jax-triton 0.3.1 smoke setup fixes
- Hypothesis: `jax-triton==0.3.1` should be the compatible package line for JAX 0.10 and Triton 3.6.
- Command:
  ```bash
  uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
    --job-name sonicmoe-jax-triton031-smoke-20260514-034806 --no-wait --max-retries 0 \
    --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 96g --disk 96g --timeout 1800 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE false -e JAX_TRACEBACK_FILTERING off \
    -- bash -lc 'set -euxo pipefail; pwd; ls -la .agents/scripts; uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 --with triton==3.6.0 python .agents/scripts/sonicmoe_jax_triton_overhead_probe.py --tokens 8192 --hidden 2048 --topk 2 --warmup 2 --repeats 5 --block-h 2048 --block-k 2 --num-warps 4'
  ```
- Config: single fixed Sonic gather config plus tiny-store launch floor.
- Result: setup failed in direct PyTorch/Triton after `jax_triton` import because `jax-triton==0.3.1` sets `TRITON_CACHE_DIR=""`; Triton then raises `RuntimeError: Could not create or locate cache dir`.
- Interpretation: this is not a kernel result. Force a real `TRITON_CACHE_DIR` after importing `jax_triton` before running direct Triton baselines.
- Next action: rerun the same smoke after the cache-dir fix.

### 2026-05-13 23:55 - jax-triton 0.3.1 overhead sweep
- Hypothesis: if jax-triton primarily pays the same fixed JAX executable dispatch tax as prior Pallas/FFI probes, standalone microkernel wall time will remain around `0.11 ms`, but a multi-call chain inside one JIT should show that the fixed dispatch cost amortizes.
- Commands:
  ```bash
  # Single-gather fixed-config sweep.
  uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
    --job-name sonicmoe-jax-triton031-sweep-20260514-035043 --no-wait --max-retries 0 \
    --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 96g --disk 96g --timeout 3600 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE false -e JAX_TRACEBACK_FILTERING off \
    -- bash -lc 'set -euxo pipefail; uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 --with triton==3.6.0 python .agents/scripts/sonicmoe_jax_triton_overhead_probe.py --tokens 8192 --hidden 2048 --topk 2 --warmup 20 --repeats 200'

  # Tiny dependent custom-call chain inside one JIT.
  uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
    --job-name sonicmoe-jax-triton031-chain-20260514-035319 --no-wait --max-retries 0 \
    --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 96g --disk 96g --timeout 1800 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE false -e JAX_TRACEBACK_FILTERING off \
    -- bash -lc 'set -euxo pipefail; uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 --with triton==3.6.0 python .agents/scripts/sonicmoe_jax_triton_overhead_probe.py --warmup 20 --repeats 200 --chain-lengths 1,4,16,64 --skip-gather'
  ```
- Config: GH200, JAX `0.10.0`, jax-triton `0.3.1`, Triton `3.6.0`, PyTorch `2.11.0+cu128`, shape `tokens=8192 hidden=2048 topk=2 bf16`.
- Results:

  | probe | config | direct Triton event | direct Triton wall | jax-triton wall | JAX minus direct wall |
  | --- | --- | ---: | ---: | ---: | ---: |
  | tiny store | 1 kernel | `0.0073 ms` | `0.0076 ms` | `0.0690 ms` | `+0.0614 ms` |
  | Sonic gather clone | best standalone, `BH=2048 BK=1 warps=8` | `0.0316 ms` | `0.0317 ms` | `0.1096 ms` | `+0.0779 ms` |
  | Sonic gather clone | `BH=2048 BK=2 warps=8` | `0.0316 ms` | `0.0317 ms` | `0.1142 ms` | `+0.0825 ms` |
  | increment chain | 1 dependent call | `0.0090 ms` | `0.0093 ms` | `0.0726 ms` | `+0.0633 ms` |
  | increment chain | 4 dependent calls | `0.0307 ms` | `0.0309 ms` | `0.0767 ms` | `+0.0458 ms` |
  | increment chain | 16 dependent calls | `0.1100 ms` | `0.1102 ms` | `0.0963 ms` | `-0.0140 ms` |
  | increment chain | 64 dependent calls | `0.4241 ms` | `0.4243 ms` | `0.1990 ms` | `-0.2253 ms` |

- Interpretation:
  - Standalone jax-triton has a fixed `~60-80 us` wall-time tax, matching the prior Pallas/FFI microkernel ceiling. As a standalone token gather launch, it is not a full-speed pure-Sonic route.
  - The dependent custom-call chain shows that most of the standalone delta is one JAX executable dispatch. Once many jax-triton custom calls are inside one compiled executable, the incremental per-call overhead is much smaller than the standalone microbenchmark suggests.
  - For whole-block JAX, this points away from "jax-triton emits much worse cubins" and toward "standalone microbenchmarks are dominated by JAX executable dispatch." The remaining block-level gap should be pursued at whole-block scheduling/layout boundaries, not by expecting single gather microbenchmarks to match PyTorch wall time.
  - Confidence: exploratory but internally consistent across tiny and gather probes.
- Follow-up: posted the result to issue #5328 at
  https://github.com/marin-community/marin/issues/5328#issuecomment-4447353963.
- Next action: do not add jax-triton to the production GPU extra yet unless we have a production backend that benefits from the amortized path.

### 2026-05-14 01:35 - Raw Sonic jax-triton inside train-step-shaped JIT
- Hypothesis: the standalone `~60-80 us` jax-triton gap is mostly one outer JAX executable dispatch, so raw Sonic Triton kernels should be viable when embedded inside a single train-step executable.
- Command:
  ```bash
  # Combine-only proxy, full production combine shape.
  uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
    --job-name sonicmoe-jt-trainstep-full-20260514-082827 --no-wait --max-retries 0 \
    --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 128g --disk 96g --timeout 3600 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE false -e JAX_TRACEBACK_FILTERING off \
    -- bash -lc 'set -euxo pipefail; uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 --with triton==3.6.0 python .agents/scripts/sonicmoe_jax_triton_trainstep_probe.py --tokens 8192 --hidden 2048 --topk 2 --warmup 20 --repeats 100 --block-h 2048 --block-k 1 --num-warps 8 --bwd-block-h 2048 --bwd-num-warps 8 --multi-counts 1,2,4,8'

  # Full-MoE-shaped proxy using ragged_dot W13/down plus raw Sonic combine.
  uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
    --job-name sonicmoe-jt-moe-full-20260514-083332 --no-wait --max-retries 0 \
    --enable-extra-resources --gpu GH200x1 --cpu 8 --memory 160g --disk 96g --timeout 3600 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE false -e JAX_TRACEBACK_FILTERING off \
    -- bash -lc 'set -euxo pipefail; uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 --with triton==3.6.0 python .agents/scripts/sonicmoe_jax_triton_trainstep_probe.py --tokens 8192 --hidden 2048 --intermediate 3072 --num-experts 8 --topk 2 --warmup 5 --repeats 20 --moe-repeats 20 --block-h 2048 --block-k 1 --num-warps 8 --bwd-block-h 2048 --bwd-num-warps 8 --multi-counts 1 --run-moe'
  ```
- Config: GH200, JAX `0.10.0`, jax-triton `0.3.1`, Triton `3.6.0`, Torch `2.11.0+cu128`, shape `T=8192 H=2048 I=3072 E=8 K=2 bf16`.
- Result:

  | probe | raw Sonic jax-triton | XLA reference | delta raw-XLA | notes |
  | --- | ---: | ---: | ---: | --- |
  | direct Triton raw gather fwd+bwd | `0.0843 ms` event / `0.0867 ms` wall | - | - | same raw kernels, host-launched |
  | embedded combine forward loss | `0.1407 ms` | `0.1199 ms` | `+0.0208 ms` | raw forward inside one JIT |
  | embedded combine fwd/bwd/update | `0.2603 ms` | `0.5345 ms` | `-0.2742 ms` | raw custom VJP beats XLA autodiff combine |
  | embedded multi-combine forward, 1 call | `0.1402 ms` | `0.1192 ms` | `+0.0210 ms` | no standalone dispatch tax |
  | full MoE forward loss | `1.3641 ms` | `1.4643 ms` | `-0.1002 ms` | ragged_dot W13/down plus raw Sonic combine |
  | full MoE fwd/bwd/update | `4.5289 ms` | `5.0019 ms` | `-0.4729 ms` | same full-MoE-shaped proxy |

- Interpretation:
  - This confirms the standalone `jax_triton.triton_call` number was pessimistic for train-step use. Inside a single compiled JAX executable, raw Sonic gather/combine does not pay the large standalone dispatch floor per kernel.
  - The raw custom VJP for gather/combine is a real win over XLA autodiff for this boundary in the proxy: about `0.27 ms` for combine-only fwd/bwd/update and about `0.47 ms` in the full-MoE-shaped proxy.
  - The full-MoE proxy is not the exact production `MoEExpertMlp` path yet; it uses the same shape and `ragged_dot` W13/down, but synthetic balanced routing and a simple SGD-style update. Use the delta as evidence for the raw Sonic combine boundary, not as a replacement for the earlier Sonic/Grug whole-block numbers.
- Next action: consider a production backend that uses raw Sonic jax-triton gather/combine with a custom VJP, then re-benchmark against `sonic_xla_interleaved_w13_quack_down`.
