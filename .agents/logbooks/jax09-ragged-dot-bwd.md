# JAX 0.9 Ragged Dot Backward: Research Logbook

## Scope
- Goal: determine whether Marin's JAX 0.9.2 upgrade from PR #5278 unlocks the ragged-dot MoE performance left on the table in PR #4297.
- Primary metric(s): compile-plus-first-step latency and steady-state step latency for Grug MoE ragged-dot forward+backward on H100x8; correctness deltas against the XLA `jax.lax.ragged_dot_general` path.
- Constraints: keep the comparison apples-to-apples on shape, dtype, hardware, mesh, and benchmark harness; treat failed kernels as first-class results.
- GitHub issue: https://github.com/marin-community/marin/issues/5330
- Branch: `research/jax09-ragged-dot-bwd`
- Experiment prefix: `J9RD`
- Key code refs:
  - `lib/haliax/src/haliax/nn/ragged_dot.py`
  - `tmp/pr5278_grug_moe_perf.py`
  - `tmp/j9rd_ragged_dot_probe.py`
  - `tmp/j9rd_grug_moe_compare.py`
  - `tmp/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py`
- Stop criteria: enough evidence to decide whether JAX 0.9.2 enables a faster ragged-dot backward path worth productionizing, including correctness checks and at least one repeated H100x8 MoE benchmark.

## Baseline
- Date: 2026-04-30
- Hardware: CoreWeave Iris CI, H100x8, JAX 0.9.2, jaxlib 0.9.2.
- Fixed shape: `tokens=4096`, `hidden_dim=1024`, `intermediate_dim=2048`, `num_experts=64`, `topk=4`, dtype `bfloat16`, mesh `data=8, expert=1, model=1`.
- Baseline implementation: `RAGGED_DOT_IMPL=xla`.
- Baseline standalone result: median steady-state latency `0.012671966571360826s`; compile-plus-first `8.543s`; job `/romain/j9rd-current-xla-20260430-1729`.
- Current Triton implementation before patch: fails on JAX 0.9.2 because `jax.experimental.pallas.load` and `jax.experimental.pallas.store` no longer exist; job `/romain/j9rd-current-triton-20260430-1729`.

## Initial Hypotheses
- `J9RD-001`: The old forward-only Triton path may need API updates for JAX 0.9.2 before any perf test is meaningful.
- `J9RD-002`: Raw autodiff through `pl.pallas_call` may now work in JAX 0.9.2 and remove the custom VJP.
- `J9RD-003`: If raw autodiff still fails, an explicit custom VJP can still use separate Triton kernels for `dlhs` and `drhs`, following the current Tokamax design.
- `J9RD-004`: Moving backward from XLA to Triton should materially improve MoE training-step latency, because PR #4297 only accelerated forward.

## Experiment Log

### 2026-04-30 17:29 - J9RD-001 Current JAX 0.9.2 Baseline
- Hypothesis: after PR #5278, XLA remains a valid baseline and the old Triton path may break because the Pallas API changed.
- Command:
  ```sh
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-current-xla-20260430-1729 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -e RAGGED_DOT_IMPL xla -- python -u tmp/pr5278_grug_moe_perf.py --label current-xla --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-current-triton-20260430-1729 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -e RAGGED_DOT_IMPL triton -- python -u tmp/pr5278_grug_moe_perf.py --label current-triton --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  ```
- Config: H100x8, Grug MoE fixed shape from baseline.
- Result:
  - `current-xla`: median steady `0.012671966571360826s`, compile-plus-first `8.543s`, first loss `2.941996`, succeeded.
  - `current-triton`: failed with `AttributeError: module 'jax.experimental.pallas' has no attribute 'load'`.
- Interpretation: PR #5278 gives the newer JAX stack, but the in-tree Triton ragged-dot kernel needs JAX 0.9 Pallas load/store updates before it can be measured.
- Next action: port load/store syntax and investigate backward implementation options.

### 2026-04-30 17:34 - J9RD-002 Direct Autodiff Probe
- Hypothesis: JAX 0.9.2 may now support direct autodiff through `pl.pallas_call` for this ragged-dot kernel.
- Command:
  ```sh
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-patched3-direct-probe-20260430-1738 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -- python -u tmp/j9rd_ragged_dot_probe.py --label patched3-direct-probe --tokens 4096 --hidden-dim 1024 --out-dim 2048 --groups 64 --warmup 3 --iters 20 --variants xla triton_custom_vjp triton_raw_direct
  ```
- Config: H100x8, direct ragged dot `lhs[M,K]`, `rhs[G,K,N]`, value+gradient timing.
- Result:
  - `xla`: compile-plus-first `4.5589s`, median steady `0.005851255s`.
  - `triton_custom_vjp`: compile-plus-first `1.1736s`, median steady `0.000655709s`; loss and both gradients match XLA with zero reported max/mean absolute diff in this probe.
  - `triton_raw_direct`: failed in JAX `_pallas_call_jvp_rule` with an `axis_frame` assertion (`env.grid_context is not None`).
- Interpretation: raw direct Pallas autodiff is still not usable for this kernel on JAX 0.9.2. The viable path is an explicit custom VJP where each VJP contraction calls a Triton kernel.
- Next action: implement explicit Triton support for the `dlhs` and `drhs` ragged-dot dimension-number layouts.

### 2026-04-30 17:38 - J9RD-003 Patched Triton Forward+Backward
- Hypothesis: porting the forward kernel to JAX 0.9 Pallas APIs and adding a `drhs` ragged-contracting-dimension kernel will accelerate the full MoE backward pass.
- Code change under test:
  - Updated old `pl.load`/`pl.store` calls to `plgpu.load(ref.at[...])` / `plgpu.store(ref.at[...])`.
  - Added Triton dispatch for `_DLHS_DIM_NUMS` by transposing expert weights.
  - Added Triton dispatch for `_DRHS_DIM_NUMS` with a ragged-contracting-dimension kernel.
  - Kept `jax.custom_vjp`, because `J9RD-002` showed raw direct autodiff still fails.
- Command:
  ```sh
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-patched3-triton-20260430-1738 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -e RAGGED_DOT_IMPL triton -- python -u tmp/pr5278_grug_moe_perf.py --label patched3-triton --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  ```
- Config: H100x8, Grug MoE fixed shape from baseline.
- Result: median steady-state latency `0.006604252615943551s`, compile-plus-first `4.3424s`, first loss `2.972210`, job `/romain/j9rd-patched3-triton-20260430-1738`.
- Interpretation: compared with standalone XLA baseline `0.012671966571360826s`, patched Triton forward+backward reduces median latency by `47.9%` and gives `1.92x` throughput for this microbench.
- Next action: run same-process paired comparisons to reduce drift from different random inputs and compilation context.

### 2026-04-30 17:40 - J9RD-004 Paired Grug MoE Comparisons
- Hypothesis: the patched Triton path remains about 2x faster than XLA when both implementations run in the same process on the same shape and generated arrays.
- Commands:
  ```sh
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-grug-compare-20260430-1740 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -- python -u tmp/j9rd_grug_moe_compare.py --label grug-compare --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-grug-compare-r2-20260430-1742 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -- python -u tmp/j9rd_grug_moe_compare.py --label grug-compare-r2 --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  uv run iris --cluster=coreweave-ci job run --no-wait --job-name j9rd-grug-compare-r3-20260430-1742 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -- python -u tmp/j9rd_grug_moe_compare.py --label grug-compare-r3 --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  ```
- Config: H100x8, same Grug MoE fixed shape from baseline; each run compiles/runs XLA and patched Triton in the same process.
- Result:
  | Run | Iris job | XLA median steady | Triton median steady | XLA compile+first | Triton compile+first |
  | --- | --- | ---: | ---: | ---: | ---: |
  | r1 | `/romain/j9rd-grug-compare-20260430-1740` | `0.01270293747074902s` | `0.00660041393712163s` | `9.165s` | `2.431s` |
  | r2 | `/romain/j9rd-grug-compare-r2-20260430-1742` | `0.012644537724554539s` | `0.00661472394131124s` | `7.864s` | `2.398s` |
  | r3 | `/romain/j9rd-grug-compare-r3-20260430-1742` | `0.012649936135858297s` | `0.006627054652199149s` | `8.095s` | `2.423s` |
- Aggregate: average paired median steady latency `0.012665803777s` for XLA vs `0.006614064177s` for Triton, a `1.915x` speedup (`47.8%` latency reduction, `91.5%` throughput gain). Average compile-plus-first improves from `8.375s` to `2.417s` (`3.46x`).
- Correctness notes:
  - `grad_w_down_diff` and `grad_w_up_gate_diff` are zero in all paired runs.
  - `grad_x_diff` max absolute ranges `0.00030517578125` to `0.00048828125`, with mean absolute about `1.54e-05`; this is small for the bfloat16 path.
  - The paired script's final `loss_diff_abs` uses the last repeated call and is noisy (`0.0129` to `0.0220`), but first-call losses and gradient diffs are the better parity signal for this harness.
- Interpretation: replicated. The newer JAX stack enables a practical, explicit custom-VJP Triton backward path with roughly 1.9x steady-state speedup for this representative MoE microbench. It does not unlock raw direct autodiff through `pallas_call`.
- Next action: productionize before merge: add durable tests/harnesses, add Pallas cost estimates, broaden shape sweep, and decide whether this belongs in Haliax directly or via a Tokamax-style backend module split.

### 2026-04-30 17:44 - Validation
- Command:
  ```sh
  uv run --package marin-haliax --with pytest-timeout pytest -q lib/haliax/tests/test_ragged_dot_dispatch.py lib/haliax/tests/test_moe_linear.py
  ./infra/pre-commit.py --fix lib/haliax/src/haliax/nn/ragged_dot.py
  uv run --package marin-levanter --frozen python -m py_compile tmp/j9rd_grug_moe_compare.py tmp/j9rd_ragged_dot_probe.py tmp/pr5278_grug_moe_perf.py
  ```
- Result:
  - `4 passed, 1 skipped`.
  - Pre-commit passed for `lib/haliax/src/haliax/nn/ragged_dot.py`.
  - Benchmark/probe scripts compile.
- Interpretation: local non-GPU tests still pass; GPU confidence comes from Iris H100 jobs above.
- Next action: report result on issue #5330 and leave branch/worktree available for follow-up productionization.

### 2026-05-01 12:05 - J9RD-005 Stack on PR #5347
- Hypothesis: PR #5347 should be treated as the base stabilization layer, and the backward-performance change should stack cleanly on top of it.
- Context:
  - PR #5347 fixes the JAX 0.9 Pallas memory API regression and validates that patched JAX 0.9 forward Triton performance matches the old JAX 0.8 path.
  - PR #5347 still leaves `_ragged_dot_triton_bwd` on XLA `jax.lax.ragged_dot_general`.
  - Tokamax handles this by routing the VJP through the same ragged-dot abstraction: `dlhs` uses a transposed-RHS default ragged-dot call, while `drhs` uses a separate ragged-contracting-dimension kernel.
- Command:
  ```sh
  git fetch origin pull/5347/head:refs/heads/pr-5347-ragged-dot-api
  git switch -c research/jax09-ragged-dot-bwd-on-5347 pr-5347-ragged-dot-api
  ```
- Result:
  - Branch `research/jax09-ragged-dot-bwd-on-5347` now starts at PR #5347 head `a65b73185`.
  - Follow-up patch keeps PR #5347's forward load/store semantics and adds only backward-layout Triton support plus CPU-level tests for the custom VJP routing.
- Interpretation: this separates concerns cleanly:
  - #5347: restore the existing forward fast path after JAX 0.9.
  - follow-up: validate and ship Triton backward.
- Validation:
  ```sh
  uv run --package marin-haliax --with pytest-timeout pytest -q lib/haliax/tests/test_ragged_dot_dispatch.py lib/haliax/tests/test_moe_linear.py
  ./infra/pre-commit.py --fix lib/haliax/src/haliax/nn/ragged_dot.py lib/haliax/tests/test_ragged_dot_dispatch.py
  uv run iris --cluster=coreweave-ci job run --job-name j9rd-stacked-grug-compare-20260501-1211 --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256G --disk 256G --extra gpu -- python -u tmp/j9rd_grug_moe_compare.py --label stacked-grug-compare --tokens 4096 --hidden-dim 1024 --intermediate-dim 2048 --num-experts 64 --topk 4 --warmup 3 --iters 20
  ```
- Result:
  - Local tests: `7 passed, 1 skipped`.
  - Pre-commit wrapper: passed.
  - H100x8 job `/romain/j9rd-stacked-grug-compare-20260501-1211`: XLA median steady `0.012689603259786963s`, Triton forward+backward median steady `0.006640843348577619s`, `1.911x` speedup / `47.7%` latency reduction.
- Next action: prepare follow-up PR once #5347 lands or is available as the stack base.
