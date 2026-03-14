# MoE DeepEP / Hybrid-EP: Research Logbook

## Scope
- Goal: benchmark DeepEP and Hybrid-EP torch-targeted MoE dispatch/combine kernels on GPU from Marin research code, starting from the sealed `#3633` snapshot.
- Primary metric(s): steady-state dispatch/combine wall time and derived `tokens/s` on H100x8.
- Constraints:
  - Do not restart or reconfigure the CoreWeave Iris cluster without explicit approval.
  - Keep the first experiment intranode and torch-side; do not imply a production-ready JAX bridge before one exists.
  - Reuse the sealed `moe-gpu-ragged-all-all-h100-seal-20260313` snapshot as the benchmark baseline.
- Experiment issue: https://github.com/marin-community/marin/issues/3641

## Baseline
- Date: 2026-03-13
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/scripts/bench/bench_deepep_torch.py`
  - DeepEP mainline `README.md`
  - DeepEP hybrid branch `docs/README_Hybrid-EP.md`
- Prior GPU baseline from `#3633`:
  - fixed shape: `tokens=32768`, `hidden=2048`, `mlp_dim=768`, `experts=128`, `shared_expert_dim=2048`
  - `current` beat `ragged_a2a` on every measured `EP > 1` point on H100x8

## Initial Hypotheses
- DeepEP's torch kernels may beat the JAX `ragged_a2a` path because they use a more specialized GPU dispatch/combine implementation.
- Hybrid-EP may improve intranode efficiency further by reducing SM pressure and integrating permute/unpermute more tightly.
- The first meaningful result is likely op-level dispatch/combine performance, not a full end-to-end MoE MLP replacement.

## Stop Criteria
- A repo-local benchmark script can successfully call DeepEP / Hybrid-EP kernels on H100x8.
- We collect at least one steady-state intranode benchmark table for DeepEP and Hybrid-EP on a controlled routing distribution.
- If build or runtime blockers appear, leave a reproducible setup command and a documented blocker state.

## Experiment Log
### 2026-03-13 17:05 - Follow-up kickoff from sealed GPU ragged snapshot
- Hypothesis: the right starting point is to branch the new DeepEP work from the sealed `#3633` snapshot so the follow-up has an explicit baseline and reproducible parent.
- Command:
  ```bash
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all add \
    .agents/logbooks/moe-gpu-ragged-all-all.md \
    .agents/projects/moe-gpu-ragged-all-all-issue.md
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all commit -m "Seal GPU ragged all-to-all experiment artifacts"
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all tag -a \
    moe-gpu-ragged-all-all-h100-seal-20260313 \
    -m "Sealed GPU ragged all-to-all experiment snapshot"
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all push -u origin research/moe-gpu-ragged-all-all
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all push origin moe-gpu-ragged-all-all-h100-seal-20260313
  git worktree add -b research/moe-deepep-hybrid-ep \
    /Users/romain/marin-wt/moe-deepep-hybrid-ep \
    moe-gpu-ragged-all-all-h100-seal-20260313
  ```
- Config:
  - seal tag: `moe-gpu-ragged-all-all-h100-seal-20260313`
  - follow-up branch: `research/moe-deepep-hybrid-ep`
- Result:
  - The prior experiment is sealed and the follow-up worktree now starts from that immutable snapshot.
- Interpretation:
  - The DeepEP thread can now reference a stable parent baseline in both local artifacts and GitHub.
- Next action:
  - Create the follow-up experiment issue.
  - Add a repo-local torch benchmark for DeepEP / Hybrid-EP.

### 2026-03-13 17:12 - DeepEP source inspection and scope cut
- Hypothesis: a practical first step is a torch-side benchmark harness, because DeepEP is exposed as a PyTorch extension and the hybrid branch adds a separate `HybridEPBuffer` wrapper with a richer dispatch/permute path.
- Command:
  ```bash
  git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP-codex
  git clone --depth 1 --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP-hybrid-codex
  sed -n '1,260p' /tmp/DeepEP-codex/README.md
  sed -n '1,260p' /tmp/DeepEP-hybrid-codex/docs/README_Hybrid-EP.md
  sed -n '1,260p' /tmp/DeepEP-hybrid-codex/deep_ep/hybrid_ep_buffer.py
  sed -n '1,320p' /tmp/DeepEP-hybrid-codex/tests/test_hybrid_ep.py
  ```
- Config:
  - DeepEP mainline API: `deep_ep.Buffer`
  - Hybrid-EP API: `deep_ep.HybridEPBuffer`
- Result:
  - DeepEP mainline is a torch extension around `Buffer.dispatch` / `Buffer.combine`.
  - The hybrid branch adds `HybridEPBuffer.dispatch`, `combine`, `dispatch_with_permute`, and `combine_with_unpermute`.
  - Intranode bring-up on a single H100x8 host is supported without the full multi-node RDMA path.
- Interpretation:
  - A true JAX-to-Torch or JAX custom-call bridge is not the first step.
  - The highest-leverage first experiment is a repo-local torch benchmark script that drives DeepEP / Hybrid-EP directly on the same H100x8 worker used in `#3633`.
- Next action:
  - Implement the torch benchmark harness in this worktree.
  - Run an intranode H100x8 smoke/benchmark via Iris.

### 2026-03-14 06:53 - Direct KubernetesRuntime launcher reaches a working DeepEP install/import on H100x8
- Hypothesis: bypassing the default Iris task image with a direct `KubernetesRuntime` pod on a CUDA-devel image should resolve the remaining bring-up blockers (`CUDA_HOME`, missing `nvcc`, and the earlier `-lnvtx3interop` link failure) without changing the cluster.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --warmup 1 \
    --iters 3
  ```
- Config:
  - launcher: `.agents/scripts/deepep_krt_smoke.py`
  - pod image: `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`
  - shape: `tokens_per_rank=4096`, `hidden=2048`, `local_experts=16`, `topk=2`, `distribution=random`, `world_size=8`
  - pod IDs during bring-up: `iris-task-f891bd28306d`, `iris-task-b2859b96b39e`
- Result:
  - The direct pod path now stages the benchmark script reliably and sees all 8 H100s.
  - `nvcc` is present under `/usr/local/cuda/bin/nvcc` and `CUDA_HOME=/usr/local/cuda` is valid.
  - Shadowing `nvidia.nvshmem` during `pip install /tmp/DeepEP` keeps the build intranode-only and avoids the earlier `mlx5dv.h` path.
  - The NVTX shim (`libnvtx3interop.so -> libnvToolsExt.so.1`) is sufficient for `pip install --no-build-isolation /tmp/DeepEP` to succeed.
  - `deep_ep` imports successfully from the installed wheel and exposes `HybridEPBuffer`.
  - First failure after import: the `deep_ep.Buffer` benchmark path still calls `get_rdma_buffer_size_hint(...)`, which asserts with `NVSHMEM is disable during compilation`.
- Interpretation:
  - The cluster, pod image, and build environment are no longer the active blockers.
  - The remaining failures are now inside the kernel paths themselves.
  - The benchmark harness should treat `deep_ep.Buffer` as blocked for this intranode-only build unless/until the library exposes a no-RDMA path that avoids the NVSHMEM assertion.
- Next action:
  - Run `HybridEPBuffer` directly (`--kernel hybrid_ep`) to separate the main DeepEP path from the hybrid path.

### 2026-03-14 06:57 - Hybrid-EP reaches runtime JIT but fails to compile kernels on H100 / CUDA 12.8
- Hypothesis: if the install/import layer is solved, `HybridEPBuffer` should either produce the first H100x8 timings or fail with a kernel-specific compile/runtime error that is narrower than the earlier environment blockers.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --kernel hybrid_ep \
    --warmup 1 \
    --iters 3
  ```
- Config:
  - pod ID: `iris-task-ea6a2781fd40`
  - same fixed shape as above: `tokens_per_rank=4096`, `hidden=2048`, `local_experts=16`, `topk=2`, `distribution=random`, `world_size=8`
- Result:
  - `HybridEPBuffer` gets past installation and import, starts the `torchrun` benchmark, and reaches the runtime JIT compiler.
  - The JIT compile fails on every rank with `cuda::ptx::cp_async_bulk` overload mismatches in `deep_ep/backend/hybrid_ep_backend.cuh`.
  - Representative error:
    - `RuntimeError: Failed to compile the code, compile command: /usr/local/cuda/bin/nvcc ... /root/.deepep/hybrid_ep/jit/2048-4096-16-8-1-uint16_t-10-8-128-32-1-1-...`
    - `error: no instance of overloaded function "cuda::ptx::cp_async_bulk" matches the argument list`
  - Upstream docs still claim Hopper support on CUDA `12.3+`, so this H100 / CUDA 12.8 failure is not explained by an obviously unsupported baseline.
- Interpretation:
  - The current blocker has narrowed to a kernel/toolchain mismatch rather than cluster bring-up.
  - Inference: the hybrid JIT source in the current `hybrid-ep` branch appears incompatible with the CUDA 12.8 PTX header signature for `cp_async_bulk` at this shape.
  - There are still no steady-state kernel timings; the experiment has advanced from environment failure to kernel compile failure.
- Next action:
  - Check whether the upstream fallback knobs (`DISABLE_SM90_FEATURES` / related JIT settings) actually apply to the hybrid runtime path.
  - If not, try a different CUDA minor/toolchain that is still consistent with the upstream Hopper support claim and rerun the same fixed H100x8 smoke command.

### 2026-03-14 07:28 - Intranode RDMA hint skip unblocks the first DeepEP timing on H100x8
- Hypothesis: the remaining `deep_ep.Buffer` blocker is local to the benchmark harness calling `get_rdma_buffer_size_hint(...)` even for an intranode-only H100x8 build with NVSHMEM intentionally disabled, so treating the RDMA buffer size as zero in that narrow case should allow the NVLink path to run.
- Command:
  ```bash
  python -m py_compile lib/levanter/scripts/bench/bench_deepep_torch.py
  uv run ruff check lib/levanter/scripts/bench/bench_deepep_torch.py
  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --kernel deep_ep \
    --torch-cuda-arch-list '9.0' \
    --warmup 1 \
    --iters 3
  ```
- Config:
  - harness change: skip `get_rdma_buffer_size_hint(...)` only when `world_size <= 8` and the exception contains `NVSHMEM is disable during compilation`
  - pod ID: `iris-task-6fe87405ea0e`
  - shape: `tokens_per_rank=4096`, `hidden=2048`, `local_experts=16`, `topk=2`, `distribution=random`, `world_size=8`
- Result:
  - The patched harness now gets through `deep_ep.Buffer` initialization and completes the benchmark loop.
  - First steady-state timing:
    - `RESULT kernel=deep_ep ep=8 distribution=random topk=2 pass=dispatch_combine time_s=0.000463 tokens_per_s=70742659.76`
- Interpretation:
  - The intranode-only NVSHMEM assertion was a harness-level bring-up bug, not a proof that the DeepEP path itself was unusable on H100x8.
  - We now have a working torch-side DeepEP baseline on the same CoreWeave H100x8 environment used earlier in this thread.
- Next action:
  - Probe Hybrid-EP again with the documented compatibility knobs.
  - Add at least one more DeepEP datapoint from the `#3633`-style routing matrix.

### 2026-03-14 07:31 - Hybrid-EP still fails under both documented compatibility knobs
- Hypothesis: the upstream Hopper compatibility knobs might redirect Hybrid-EP away from the failing `cp_async_bulk` code path enough to make the runtime JIT compile succeed on H100.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --kernel hybrid_ep \
    --torch-cuda-arch-list '9.0' \
    --disable-aggressive-ptx-instrs \
    --warmup 1 \
    --iters 3

  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --kernel hybrid_ep \
    --torch-cuda-arch-list '9.0' \
    --disable-sm90-features \
    --warmup 1 \
    --iters 3
  ```
- Config:
  - pod IDs: `iris-task-5bb63a173954`, `iris-task-a1311a342cb2`
  - shape: `tokens_per_rank=4096`, `hidden=2048`, `local_experts=16`, `topk=2`, `distribution=random`, `world_size=8`
- Result:
  - Both runs rebuild and import `deep_ep` successfully, then fail at the same Hybrid-EP runtime JIT stage.
  - In both cases the failure remains:
    - `error: no instance of overloaded function "cuda::ptx::cp_async_bulk" matches the argument list`
    - `RuntimeError: Failed to compile the code, compile command: /usr/local/cuda/bin/nvcc -std=c++17 -gencode=arch=compute_90,code=sm_90 ...`
- Interpretation:
  - Neither `DISABLE_AGGRESSIVE_PTX_INSTRS=1` nor `DISABLE_SM90_FEATURES=1` changes the failing code path for this H100 / CUDA 12.8 Hybrid-EP JIT.
  - The current Hybrid-EP blocker looks upstream/toolchain-specific rather than something Marin can paper over with the documented environment flags.
- Next action:
  - Record the negative result in the public issue and branch snapshot.
  - Continue using `deep_ep` as the only working torch-side GPU datapath until Hybrid-EP has a different upstream fix or toolchain recommendation.

### 2026-03-14 07:34 - Second DeepEP datapoint lands on a `#3633`-style routed case
- Hypothesis: after the intranode bring-up fix, the same DeepEP path should continue to run under a more stressed routing case drawn from the earlier GPU comparison matrix.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris \
    uv run python .agents/scripts/deepep_krt_smoke.py \
    --run-bench \
    --kernel deep_ep \
    --torch-cuda-arch-list '9.0' \
    --distribution runs \
    --topk 8 \
    --warmup 1 \
    --iters 3
  ```
- Config:
  - pod ID: `iris-task-1fc16e6ee7e2`
  - shape: `tokens_per_rank=4096`, `hidden=2048`, `local_experts=16`, `topk=8`, `distribution=runs`, `world_size=8`
- Result:
  - Second steady-state timing:
    - `RESULT kernel=deep_ep ep=8 distribution=runs topk=8 pass=dispatch_combine time_s=0.001047 tokens_per_s=31285893.37`
- Interpretation:
  - The torch-side DeepEP path is not limited to a single smoke case; it also runs on a routed case closer to the prior GPU comparison matrix.
  - Confidence is still exploratory because the table is small, but the experiment has now moved beyond pure bring-up into actual GPU measurement.
- Next action:
  - Commit and push the harness update and launcher knobs.
  - Update `#3641` with the first working timing table and the Hybrid-EP failure status.
