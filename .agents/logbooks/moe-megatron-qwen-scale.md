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

### 2026-03-14 20:12 - Full fp32-router Qwen-patterned matrix changes the ranking vs `#3633`
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases all \
    --dispatchers all \
    --warmup-iters 5 \
    --measure-iters 20 \
    --task-id megatron-qwen-matrix-20260314-2030
  ```
- Pod:
  - `iris-task-f278733d7cc3`
- Committed harness snapshot:
  - branch: `research/moe-megatron-qwen-scale`
  - commit: `e4b742e9d1d27f5b8f1f2b9ff7a06c58c653ee59`
- Results recovered from live logs before pod cleanup:

  | Case | alltoall total ms | deepep total ms | hybridep total ms | Winner |
  | --- | ---: | ---: | ---: | --- |
  | `qwen3_30b_a3b_anchor` | 34.95 | 12.41 | 18.05 | `deepep` |
  | `qwen3_235b_a22b_anchor` | 25.62 | 11.75 | 13.08 | `deepep` |
  | `qwen3_batch_mb2` | 33.77 | 12.42 | 16.61 | `deepep` |
  | `qwen3_batch_mb4` | 35.27 | 18.09 | 21.09 | `deepep` |
  | `qwen3_hidden_3072` | 45.10 | 10.55 | 19.46 | `deepep` |
  | `qwen3_hidden_4096` | 27.22 | 9.71 | 14.59 | `deepep` |
  | `qwen3_expert_1152` | 30.56 | 10.13 | 12.50 | `deepep` |
  | `qwen3_expert_1536` | 24.62 | 9.59 | 11.38 | `deepep` |
  | `qwen3_topk_2` | 31.90 | 9.04 | 14.41 | `deepep` |
  | `qwen3_topk_4` | 33.48 | 9.40 | 15.73 | `deepep` |
- Interpretation:
  - This is the opposite of the sealed `#3633` fixed-shape dispatch-only result.
  - On both official Qwen anchors and on the batch/hidden/expert-size/top-k sweeps, `deepep` is fastest and `hybridep` is consistently second.
  - The methodology change matters: once the benchmark matches Megatron's full `MoELayer` path with fp32 router probabilities, `alltoall` stops being the default winner for these Qwen-like shapes.
- Limitation:
  - The pod auto-deleted before I re-scraped the expert-count slice cleanly, so the `num_experts in {32, 64}` rows still needed a focused rerun before sealing.

### 2026-03-14 20:44 - Focused expert-count rerun supersedes the rough live scrape
- Motivation:
  - After the full matrix pod completed, `kubectl logs` returned `pods "iris-task-f278733d7cc3" not found` because KubernetesRuntime had already cleaned up the finished pod.
  - I relaunched just the missing expert-count slice with local `tee` logging so the raw `RESULT` rows could not be lost again.
- Command:
  ```bash
  mkdir -p /Users/romain/marin-wt/moe-megatron-qwen-scale/.agents/run-logs && \
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run --package iris python /Users/romain/marin-wt/moe-megatron-qwen-scale/.agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-megatron-qwen-scale \
    --cases qwen3_experts_32,qwen3_experts_64 \
    --dispatchers all \
    --warmup-iters 5 \
    --measure-iters 20 \
    --task-id megatron-qwen-expert-slice-20260314-1344 \
    2>&1 | tee /Users/romain/marin-wt/moe-megatron-qwen-scale/.agents/run-logs/megatron-qwen-expert-slice-20260314-1344.log
  ```
- Pod:
  - `iris-task-3e98021abb0d`
- Result:

  | Case | alltoall total ms | deepep total ms | hybridep total ms | Winner |
  | --- | ---: | ---: | ---: | --- |
  | `qwen3_experts_32` | 16.12 | 22.82 | 11.91 | `hybridep` |
  | `qwen3_experts_64` | 25.38 | 15.52 | 14.55 | `hybridep` |

  DeepEP variance on the rerun:
  - `qwen3_experts_32`: `forward_std=29.98 ms`, `backward_std=15.24 ms`
  - `qwen3_experts_64`: `forward_std=11.44 ms`, `backward_std=15.45 ms`
- Interpretation:
  - The earlier rough live scrape for the expert-count axis was not reliable and should not be used.
  - On the authoritative rerun, `hybridep` wins both small-expert points.
  - `deepep` remains strong at `64` experts on mean time, but it shows large outliers on both `32` and `64` experts, which is important because Megatron's methodology reports mean wall time.

### 2026-03-14 20:58 - Final synthesis for `#3666`
- Final comparison table:

  | Case | alltoall total ms | deepep total ms | hybridep total ms | Winner |
  | --- | ---: | ---: | ---: | --- |
  | `qwen3_30b_a3b_anchor` | 34.95 | 12.41 | 18.05 | `deepep` |
  | `qwen3_235b_a22b_anchor` | 25.62 | 11.75 | 13.08 | `deepep` |
  | `qwen3_batch_mb2` | 33.77 | 12.42 | 16.61 | `deepep` |
  | `qwen3_batch_mb4` | 35.27 | 18.09 | 21.09 | `deepep` |
  | `qwen3_hidden_3072` | 45.10 | 10.55 | 19.46 | `deepep` |
  | `qwen3_hidden_4096` | 27.22 | 9.71 | 14.59 | `deepep` |
  | `qwen3_expert_1152` | 30.56 | 10.13 | 12.50 | `deepep` |
  | `qwen3_expert_1536` | 24.62 | 9.59 | 11.38 | `deepep` |
  | `qwen3_topk_2` | 31.90 | 9.04 | 14.41 | `deepep` |
  | `qwen3_topk_4` | 33.48 | 9.40 | 15.73 | `deepep` |
  | `qwen3_experts_32` | 16.12 | 22.82 | 11.91 | `hybridep` |
  | `qwen3_experts_64` | 25.38 | 15.52 | 14.55 | `hybridep` |
- Final takeaways:
  - For the Qwen-like `128`-expert regime, the Megatron-style benchmark on H100x8 strongly favors `deepep`, with `hybridep` second and `alltoall` last.
  - The expert-count sweep is the exception: once the expert count drops to `32` or `64`, `hybridep` becomes the best measured dispatcher on mean wall time.
  - This experiment therefore overturns the narrow implication one might draw from `#3633`: the old fixed-shape dispatch-only GPU benchmark is not a good proxy for the full Megatron `MoELayer` path.
  - The remaining caveat is variance, not functionality: `deepep` shows large timing spikes on the small-expert slice, so the correct conclusion is "use `deepep` for the Qwen-like `128`-expert path, but watch `hybridep` for smaller expert counts or when jitter matters."
- Stop criteria:
  - Met. The benchmark ran on H100x8 with Megatron-style methodology, covered the requested scale axes, and produced a concrete dispatcher ranking plus one meaningful crossover regime.
