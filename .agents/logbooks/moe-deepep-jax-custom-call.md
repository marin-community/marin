# MoE DeepEP JAX Custom Call: Research Logbook

## Scope
- Goal: wire a repo-local JAX custom-call benchmark path for DeepEP-backed GPU MoE dispatch/combine and compare it directly against the sealed `#3633` H100x8 benchmark.
- Primary metric(s): steady-state wall time and derived `tokens/s` on the fixed-shape `#3633` H100x8 regime.
- Constraints:
  - Do not restart or reconfigure the CoreWeave Iris cluster without explicit approval.
  - Keep the benchmark directly comparable to `#3633`; avoid drifting to a different shape unless a blocker forces an explicit scoped exception.
  - This thread must measure a JAX custom-call path, not just a JAX-to-Torch bridge.
- Experiment issue: https://github.com/marin-community/marin/issues/3665

## Baseline
- Date: 2026-03-14
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/scripts/bench/bench_deepep_dispatch.py`
  - `lib/levanter/scripts/bench/bench_deepep_torch.py`
  - `levanter.grug.grug_moe`
- Prior sealed baselines:
  - `#3633` seal tag: `moe-gpu-ragged-all-all-h100-seal-20260313`
  - `#3641` seal tag: `moe-deepep-hybrid-ep-seal-20260314`

## Initial Hypotheses
- The torch-side DeepEP / Hybrid-EP wins from `#3641` will not automatically transfer to the JAX path; the integration overhead and custom-call lowering details may materially change the result.
- The cleanest first benchmark is likely a focused JAX custom call around dispatch/combine rather than a full production MoE replacement.
- The right comparison target is the sealed fixed-shape `#3633` GPU harness, not a new bespoke benchmark.

## Stop Criteria
- A repo-local JAX custom-call path runs on CoreWeave H100x8.
- The path can be benchmarked on the fixed-shape `#3633` GPU regime.
- The thread records at least one direct head-to-head comparison against the sealed `#3633` result table, or leaves a precise documented blocker state if the custom call cannot be made runnable.

## Experiment Log
### 2026-03-14 11:00 - Kickoff from sealed torch-side DeepEP experiment
- Hypothesis: sealing the torch-side thread first gives this JAX custom-call experiment a stable parent snapshot and avoids mixing torch-only findings with JAX-native claims.
- Command:
  ```bash
  git worktree add -b research/moe-deepep-jax-custom-call \
    /Users/romain/marin-wt/moe-deepep-jax-custom-call \
    moe-deepep-hybrid-ep-seal-20260314
  ```
- Config:
  - parent seal tag: `moe-deepep-hybrid-ep-seal-20260314`
  - new branch: `research/moe-deepep-jax-custom-call`
- Result:
  - The new research worktree is created from the sealed torch-side DeepEP / Hybrid-EP snapshot.
- Interpretation:
  - The new thread now has both the `#3633` benchmark harness and the torch-side DeepEP research artifacts available in one lineage.
- Next action:
  - Create the new experiment issue and link it back to this logbook.
  - Inspect existing JAX FFI/custom-call integration options and pick the narrowest viable benchmark path.

### 2026-03-14 12:25 - Wire the first JAX custom-call milestone into the #3633 harness
- Hypothesis: the narrowest viable JAX-native DeepEP experiment is to custom-call the raw `deep_ep::layout::get_dispatch_layout` CUDA kernel and splice its metadata into the existing `ragged_a2a` benchmark path.
- Commands:
  ```bash
  python -m py_compile \
    lib/levanter/src/levanter/kernels/deepep/layout_ffi.py \
    lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    .agents/scripts/deepep_jax_krt_bench.py

  uv run ruff check \
    lib/levanter/src/levanter/kernels/deepep/layout_ffi.py \
    lib/levanter/src/levanter/kernels/deepep/__init__.py \
    lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    .agents/scripts/deepep_jax_krt_bench.py

  uv run --package iris python .agents/scripts/deepep_jax_krt_bench.py --help
  ```
- Config:
  - new JAX FFI wrapper: `lib/levanter/src/levanter/kernels/deepep/layout_ffi.py`
  - new custom-call CUDA bridge: `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_layout_ffi.cu`
  - new benchmark kernel: `deepep_layout_ragged_a2a`
  - new CoreWeave launcher: `.agents/scripts/deepep_jax_krt_bench.py`
  - pinned DeepEP main ref for the first cluster run: `567632dd59810d77b3cc05553df953cc0f779799`
- Result:
  - The Python and launcher layers validate locally.
  - The benchmark now has a third kernel choice that routes through a JAX custom call for DeepEP layout metadata.
  - The first raw CoreWeave launcher is ready to run the fixed-shape `#3633` matrix on H100x8 with `current`, `ragged_a2a`, and `deepep_layout_ragged_a2a`.
- Interpretation:
  - The current milestone is honest but narrow: it is a real JAX custom call into DeepEP CUDA code, but it only replaces layout metadata generation inside the existing JAX `ragged_a2a` path rather than replacing the full transport/combine stack.
  - That is still enough to get a clean same-harness head-to-head against `#3633` and tell us whether this custom-call insertion moves the GPU result at all.
- Negative results / fixes:
  - DeepEP `num_tokens_per_rank` counts token reachability per rank, not repeated assignment counts, so they cannot be used directly as the send sizes for this benchmark's repeated-token `ragged_a2a` path.
  - Requesting `num_tokens_per_rdma_rank` on a single-node H100x8 setup is invalid in DeepEP and would trigger the multi-node assertion path; the FFI wrapper now passes `nullptr` there.
- Next action:
  - Commit and push the JAX custom-call snapshot.
  - Run `.agents/scripts/deepep_jax_krt_bench.py` on CoreWeave H100x8.
  - Update `#3665` with the pinned snapshot and the first cluster result.

### 2026-03-14 18:42 - Complete the fixed-shape H100x8 matrix on CoreWeave
- Hypothesis: if the JAX custom call materially helps the GPU path, the new `deepep_layout_ragged_a2a` kernel should separate from vanilla `ragged_a2a` on the sealed `#3633` shape.
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
    uv run --package iris python .agents/scripts/deepep_jax_krt_bench.py \
    --repo-ref moe-deepep-jax-layout-ffi-smokefix-20260314 \
    --task-id deepep-jax-krt-bench-20260314-114210
  ```
- Config:
  - CoreWeave cluster: Iris H100x8 via raw `KubernetesRuntime`
  - task id: `deepep-jax-krt-bench-20260314-114210`
  - pod: `iris-task-9d8747e702c3`
  - shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 shared_expert_dim=2048`
  - kernels: `current`, `ragged_a2a`, `deepep_layout_ragged_a2a`
  - sweep: `distribution in {random, runs}`, `topk in {2, 8}`, `EP in {1, 2, 4, 8}`, `bench_pass=forward_backward`
- Result:
  - Smoke passed before the matrix:
    - `FFI_SMOKE_OK`
    - `FFI_SMOKE_RANK_COUNTS [48, 48, 48, 48, 48, 48, 48, 48]`
    - `FFI_SMOKE_EXPERT_COUNT_SUM 512`
  - Completed matrix (`tokens/s`):

    | distribution | topk | kernel | EP1 | EP2 | EP4 | EP8 |
    | --- | --- | --- | ---: | ---: | ---: | ---: |
    | random | 2 | current | 610333 | 892081 | 1724413 | 2966491 |
    | random | 2 | ragged_a2a | 620045 | 553780 | 1058353 | 2001628 |
    | random | 2 | deepep_layout_ragged_a2a | 617592 | 538626 | 1057965 | 2002620 |
    | random | 8 | current | 151577 | 237428 | 482195 | 956226 |
    | random | 8 | ragged_a2a | 150450 | 138256 | 279590 | 546790 |
    | random | 8 | deepep_layout_ragged_a2a | 151781 | 132724 | 276921 | 539028 |
    | runs | 2 | current | 616690 | 886567 | 1728405 | 2963386 |
    | runs | 2 | ragged_a2a | 614976 | 524339 | 970674 | 1668893 |
    | runs | 2 | deepep_layout_ragged_a2a | 619300 | 527523 | 974219 | 1665751 |
    | runs | 8 | current | 152001 | 235254 | 489235 | 958629 |
    | runs | 8 | ragged_a2a | 151848 | 135279 | 274323 | 526984 |
    | runs | 8 | deepep_layout_ragged_a2a | 130124 | 138148 | 273787 | 530715 |
- Interpretation:
  - The layout-only DeepEP custom call does not improve distributed performance on this workload.
  - For every completed `EP > 1` point, `current` beat `deepep_layout_ragged_a2a`; the gap ranged from about `1.48x` to `1.81x`.
  - Across the distributed cells, `deepep_layout_ragged_a2a` stayed effectively tied with vanilla `ragged_a2a` (roughly `-4%` to `+2%`), so replacing only the layout metadata producer does not move the result.
  - The one notable anomaly was `runs, topk=8, EP=1`, where `deepep_layout_ragged_a2a` dropped to `130124 tok/s`; since that is a single non-distributed point and the distributed cells stayed tied, it is not evidence of a win.
- Negative results / fixes:
  - The first raw CoreWeave submission path exposed a series of launch bugs before the successful run: mounted `/app` cleanup, Git availability in the image, NVSHMEM-disabled DeepEP compilation, CUDA FFI handler registration, and 64-bit JAX operand expectations.
  - None of those fixes changed the final benchmark interpretation: once the smoke passed, the layout-only custom call still matched `ragged_a2a` rather than the stronger torch-side DeepEP / Hybrid-EP result from `#3641`.
- Next action:
  - Treat this layout-only custom call as a completed negative result.
  - If the JAX-native DeepEP story is still worth pursuing, the next step is a deeper custom call around dispatch/combine transport rather than another round of layout-only tuning.
