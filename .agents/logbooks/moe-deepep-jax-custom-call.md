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
