# Megatron Qwen-Scale MoE: Research Logbook

## Scope
- Goal: reproduce the MoE-layer benchmarking methodology used in Megatron-LM's `moe_perf` module, then measure how dispatcher behavior changes with Qwen3-shaped scale in batch tokens, hidden size, expert size, number of experts, and top-k.
- Primary metric(s): per-iteration forward and backward wall time on H100x8 for a full MoE layer benchmark, plus derived ratios between `alltoall`, `deepep`, and `hybridep`.
- Constraints:
  - Use CoreWeave Iris H100x8 without restarting or reconfiguring the cluster.
  - Keep the methodology close to Megatron's `tests/functional_tests/test_cases/common/moe_perf`.
  - Use Qwen3 MoE shapes as the scale anchors rather than inventing arbitrary proxy sizes.
- Experiment issue: https://github.com/marin-community/marin/issues/3666

## Baseline
- Date: 2026-03-14
- Code refs:
  - Megatron-LM `tests/functional_tests/test_cases/common/moe_perf/__main__.py`
  - Megatron-LM `tests/functional_tests/test_cases/common/moe_perf/test_cases.py`
  - Megatron-LM `megatron/core/transformer/moe/README.md`
  - Qwen `Qwen/Qwen3-30B-A3B` and `Qwen/Qwen3-235B-A22B` config.json
- Prior related seals:
  - `#3641` torch-side DeepEP / Hybrid-EP fixed-shape H100x8 seal: `moe-deepep-hybrid-ep-seal-20260314`
  - `#3665` JAX custom-call DeepEP layout H100x8 seal: `moe-deepep-jax-layout-ffi-h100-matrix-20260314`

## Initial Hypotheses
- The previous fixed-shape results may have understated where `deepep` or `hybridep` become attractive because they did not follow Megatron's full MoE-layer benchmark methodology.
- Megatron's use of a full `MoELayer`, fixed routed inputs, force-balanced random routing, manual GC, and a dummy GEMM before each timed iteration will likely reduce router noise and better expose the compute/communication crossover points.
- Qwen3-30B-A3B and Qwen3-235B-A22B provide a clean pair of official scale anchors:
  - `30B-A3B`: `hidden=2048`, `moe_ffn_hidden=768`, `num_experts=128`, `topk=8`, `layers=48`
  - `235B-A22B`: `hidden=4096`, `moe_ffn_hidden=1536`, `num_experts=128`, `topk=8`, `layers=94`
- `hybridep` is the most likely winner at the larger Qwen-like shapes, but the crossover may depend more on batch tokens and expert hidden size than on expert count alone.

## Stop Criteria
- Run a Megatron-style MoE-layer benchmark on CoreWeave H100x8 with at least `alltoall`, `deepep`, and `hybridep`.
- Record at least one complete Qwen-shaped scaling sweep that varies batch tokens, hidden size, expert size, num experts, and top-k in a controlled way.
- Update the issue with enough evidence to say where the dispatcher ranking changes, or document a precise blocker if the Megatron path cannot be made runnable.

## Experiment Log
### 2026-03-14 19:20 - Scope the new Megatron/Qwen scaling thread
- Hypothesis: the right follow-up to the sealed JAX layout negative result is not another metadata-only benchmark, but a full MoE-layer benchmark that matches Megatron's own methodology and Qwen3 model shapes.
- Commands:
  ```bash
  git worktree add -b research/moe-megatron-qwen-scale \
    /Users/romain/marin-wt/moe-megatron-qwen-scale \
    moe-deepep-jax-layout-ffi-h100-matrix-20260314

  sed -n '1,240p' /Users/romain/llms/cw_ops_guide.md

  git clone --depth=1 https://github.com/NVIDIA/Megatron-LM.git /tmp/megatron-lm.<tmp>
  git clone --depth=1 https://github.com/deepseek-ai/DeepEP.git /tmp/deepep-upstream.<tmp>
  ```
- Config:
  - new branch: `research/moe-megatron-qwen-scale`
  - new worktree: `/Users/romain/marin-wt/moe-megatron-qwen-scale`
  - CoreWeave kubeconfig from ops guide: `~/.kube/coreweave-iris`
- Result:
  - Confirmed the ops flow for Iris H100x8 from `~/llms/cw_ops_guide.md`.
  - Identified Megatron's exact benchmark reference: `tests/functional_tests/test_cases/common/moe_perf`.
  - Confirmed the two official Qwen3 MoE anchor configs from Megatron and Hugging Face:
    - `Qwen3-30B-A3B`: `hidden=2048`, `moe_ffn_hidden=768`, `num_experts=128`, `topk=8`
    - `Qwen3-235B-A22B`: `hidden=4096`, `moe_ffn_hidden=1536`, `num_experts=128`, `topk=8`
- Interpretation:
  - The new experiment should benchmark a full MoE layer, not just transport.
  - The first matrix can stay one-dimensional per axis while still matching the official Qwen endpoints.
- Planned first matrix:
  - Dispatchers: `alltoall`, `deepep`, `hybridep`
  - Anchor cases:
    - `qwen3_30b_a3b`: `seq=4096`, `micro_batch=1`, `hidden=2048`, `moe_ffn=768`, `experts=128`, `topk=8`
    - `qwen3_235b_a22b`: `seq=4096`, `micro_batch=1`, `hidden=4096`, `moe_ffn=1536`, `experts=128`, `topk=8`
  - One-axis sweeps around the anchors:
    - batch tokens: `micro_batch in {1, 2, 4}` at `seq=4096`
    - hidden size: `hidden in {2048, 3072, 4096}`
    - expert size: `moe_ffn_hidden in {768, 1152, 1536}`
    - expert count: `num_experts in {32, 64, 128}`
    - top-k: `topk in {2, 4, 8}`
- Next action:
  - Create the experiment issue and link it back here.
  - Decide whether to run Megatron's benchmark directly in the job image or mirror it in a repo-local standalone harness with the same methodology.

### 2026-03-14 19:36 - First CoreWeave H100x8 smoke: Megatron harness reaches TE build
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-1945
  ```
- Result:
  - Megatron and DeepEP source tarballs unpacked successfully.
  - `nv-grouped-gemm` built.
  - The local HybridEP H100 `space_cluster` patch applied cleanly.
  - `transformer_engine_torch` failed before benchmark execution with:
    ```text
    fatal error: cudnn.h: No such file or directory
    ```
- Interpretation:
  - The benchmark logic was not the blocker yet; the pod needed explicit cuDNN include/lib wiring for TE.
- Next action:
  - Discover cuDNN from the pip-installed NVIDIA package layout and export `CUDNN_PATH` / include / lib paths before TE installation.

### 2026-03-14 19:46 - TE bring-up fixed; Megatron benchmark fails on `RandomSTE.generator`
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-2006
  ```
- Result:
  - `transformer-engine[pytorch]`, `nv-grouped-gemm`, and `deep_ep` all built and imported.
  - Megatron reported `HAVE_DEEP_EP=True` and `HAVE_HYBRIDEP=True`.
  - The first benchmark iteration failed because this pinned Megatron ref's benchmark path still checks `RandomSTE.generator`, but `RandomSTE` at the same ref no longer defines that attribute.
- Interpretation:
  - This was a harness/ref compatibility bug, not a CoreWeave or dependency problem.
- Next action:
  - Guard the `RandomSTE.generator` reset so the smoke can proceed.

### 2026-03-14 19:53 - DeepEP under Megatron needs NVSHMEM-enabled build
- Commands:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-2012
  ```
- Result:
  - The first `deepep` dispatch failed inside Megatron `fused_a2a` with:
    ```text
    RuntimeError: Failed: Assertion error ... "NVSHMEM is disable during compilation"
    ```
  - Root cause: the launcher was still shadowing `nvidia.nvshmem`, so DeepEP compiled with `-DDISABLE_NVSHMEM`.
- Interpretation:
  - Unlike the earlier direct DeepEP transport harness, Megatron's `fused_a2a` path requires the NVSHMEM-enabled DeepEP build even for this one-node H100x8 case.
- Next action:
  - Export `NVSHMEM_DIR` to the wheel-installed NVSHMEM root and stop shadowing `nvidia.nvshmem`.

### 2026-03-14 19:58 - NVSHMEM build path exposes missing RDMA headers, then NVSHMEM host-lib packaging mismatch
- Commands:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-2036

  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-2103
  ```
- Results:
  - With `NVSHMEM_DIR` exported, DeepEP enabled internode sources and then failed on:
    ```text
    fatal error: infiniband/mlx5dv.h: No such file or directory
    ```
  - Installing `libibverbs-dev` / `rdma-core` fixed that compile step.
  - The next build then failed at final link with:
    ```text
    /opt/conda/compiler_compat/ld: cannot find -l:libnvshmem_host.so
    ```
  - Root cause: the NVSHMEM wheel ships `libnvshmem_host.so.3`, while DeepEP's `setup.py` hard-codes the unversioned `libnvshmem_host.so` when `NVSHMEM_DIR` is set explicitly.
- Interpretation:
  - The Megatron path is now clearly runnable with pod-side packaging workarounds.
  - Public upstream signal exists for the host-lib naming issue (`deepseek-ai/DeepEP#383`).
- Next action:
  - Install RDMA dev packages in the pod and create the unversioned `libnvshmem_host.so` symlink before building DeepEP.

### 2026-03-14 20:08 - First complete Megatron-style smoke on H100x8
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_30b_a3b_anchor \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 1 \
    --measure-iters 2 \
    --task-id megatron-qwen-smoke-20260314-2120
  ```
- Pod:
  - `iris-task-b39002fa782c` on `g131086`
- Result:
  - Full bring-up succeeded:
    - TE built with explicit cuDNN paths.
    - DeepEP built with explicit NVSHMEM, RDMA dev packages, and `libnvshmem_host.so` symlink.
    - Megatron reported `HAVE_DEEP_EP=True` and `HAVE_HYBRIDEP=True`.
  - The smoke finished the `qwen3_30b_a3b_anchor` case with exploratory timings:
    - `alltoall`: `forward=23.48 ms`, `backward=40.26 ms`
    - `deepep`: `forward=48.13 ms`, `backward=49.11 ms`
    - `hybridep`: `forward=25.23 ms`, `backward=37.12 ms`
  - Variance was still large because this was only `warmup=1`, `measure=2`.
- Interpretation:
  - The Megatron-style benchmark is operational on CoreWeave H100x8.
  - At this smoke-level evidence, `deepep` looks clearly behind, while `alltoall` and `hybridep` are close.
  - Before the real sweep, one more methodology fix is needed: upstream Megatron perf cases default `moe_router_dtype="fp32"`, and HybridEP warned that float32 probs are required.
- Next action:
  - Set `moe_router_dtype="fp32"` in the harness.
  - Launch the full Qwen-patterned scaling matrix with the longer Megatron-style timing window.
