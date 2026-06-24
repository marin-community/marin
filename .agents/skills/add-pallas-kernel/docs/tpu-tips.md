# TPU Tips

Read this file for TPU Pallas/Mosaic kernels, TPU-specific lowering failures,
scoped VMEM setup, TPU compiler diagnostics, and TPU input-shape constraints.

## Input Shape and Parallel Dimensions

Preserve explicit parallel-dimension semantics on at least one axis, usually
batch, for TPU kernels. Prefer a canonical batched input contract and require
callers to normalize into it before invoking the core kernel.

If Mosaic reports a layout mismatch for a batched integer operand, such as
labels, align the batch block size to the XLA tile size for that TPU generation
or raise a clear pre-lowering error.

Validate TPU alignment constraints before lowering. For example, enforce
required multiples such as 128 in the TPU backend without making reference/XLA
paths unusable.

## Matmul Precision

If Mosaic reports errors like `Expected matmul acc to be 32-bit`:

- Set `preferred_element_type=jnp.float32` in `lax.dot_general` for the kernel
  path, or
- Set `jax_default_matmul_precision=highest` in benchmark scripts.

Prefer explicit kernel-side `preferred_element_type` for deterministic behavior.

## Scoped VMEM Policy

Set `LIBTPU_INIT_ARGS` by TPU generation during microbench/tuning:

- `v5p` / `v5e`: `--xla_tpu_scoped_vmem_limit_kib=50000`
- `v6e`: `--xla_tpu_scoped_vmem_limit_kib=98304`
- `v4`: no special scoped-VMEM override

Record the exact `LIBTPU_INIT_ARGS` value with benchmark and tuning results.

## Compiler Diagnostics and Dumps

Capture compiler diagnostics on serious benchmark/tuning runs:

- HLO dumps via `--xla-dump-dir`.
- Compiler logs via `--compiler-log-path`.
- Exact `XLA_FLAGS`.
- Exact `LIBTPU_INIT_ARGS`.

Useful scripts:

- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
- `lib/levanter/scripts/tune/tune_fused_cross_entropy_loss_block_sizes.py`

For dump-first diagnosis workflow, read
[Performance workflow](performance-workflow.md).
