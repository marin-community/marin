# Kernel Sources

Read this file when choosing a kernel to imitate or when checking whether the
repo already has helpers for Pallas API layout, autotuning, cost estimates, or
tests.

## In-Repo Pallas Kernels

- `lib/levanter/src/levanter/kernels/pallas/template_kernel.py`: minimal
  template for API, reference, and Pallas wrapper structure.
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/`: best
  reference for production-style layout, TPU Pallas implementation, fallback
  API, cost estimate plumbing, tuned block-size lookup, and autotune-on-miss
  behavior.
- `lib/levanter/src/levanter/kernels/pallas/mamba3/`: reference/XLA split and
  config layout for sequence kernels.
- `lib/levanter/src/levanter/kernels/pallas/ssd/`: reference/XLA split and
  config layout for structured state-space kernels.
- `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`: local Pallas
  attention reference.

## Shared Helpers

- `lib/levanter/src/levanter/kernels/pallas/autotune_utils.py`: bounded config
  sweeps and result handling.
- `lib/levanter/src/levanter/kernels/pallas/autotune_cache_utils.py`: persistent
  autotune cache helpers.
- `lib/levanter/src/levanter/kernels/pallas/cost_estimate_utils.py`: IO-byte
  augmentation for `pl.CostEstimate`.

## Tests and Harnesses

- `lib/levanter/tests/kernels/test_pallas_template_kernel.py`: template kernel
  test shape.
- `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`:
  production kernel correctness, fallback, and tuning coverage.
- `lib/levanter/tests/kernels/test_pallas_autotune_utils.py`: autotune helper
  behavior.
- `lib/levanter/tests/kernels/test_pallas_mamba3.py`: reference/XLA parity for a
  sequence kernel.

Benchmark and tuning scripts:

- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_xla_streaming.py`
- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_gpu_block_sweep.py`
- `lib/levanter/scripts/bench/bench_tokamax_linear_softmax_ce.py`
- `lib/levanter/scripts/tune/tune_fused_cross_entropy_loss_block_sizes.py`

## External References

- [JAX Pallas Overview](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX Pallas TPU Docs](https://docs.jax.dev/en/latest/pallas/tpu/index.html)
- [JAX Pallas Mosaic GPU Docs](https://docs.jax.dev/en/latest/pallas/gpu/index.html)
- [When XLA Isn't Enough: From Pallas to VLIW](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
- `reference/llo.md`
- `reference/profiling.md`

## Tokamax Notes

Tokamax kernels are useful references for API and kernel structure comparisons.

- Typical install path in this repo's uv environment:
  `.venv/lib/python3.12/site-packages/tokamax/_src/ops`
- Compare numerics/perf on identical shapes/dtypes before drawing conclusions.
- Parse `absl.flags` before accessing Tokamax modules that depend on flags.
- Tokamax Mosaic kernels can OOM VMEM at larger shapes; reduce shape/tile sizes
  for controlled comparisons.
