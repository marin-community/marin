# FP8 ragged_dot notes

## Current status

The implementation now has an opt-in `Fp8RaggedDotOp` for
`haliax.nn.ragged_dot` with delayed per-tensor scaling state, E4M3 forward
operands, E5M2 output-gradient operands in the custom VJP, and non-uniform
`group_sizes` in `bench_fp8_ragged_dot.py`.

The existing bf16 default path remains unchanged. The FP8 path is selected by
passing `fp8_dot=Fp8RaggedDotOp.init(...)`.

This does not yet meet the required >=1.2x non-uniform fwd+bwd speedup. I am
blocked on the fast backend route described below and need a coordinator hint.

## Non-uniform benchmark

`bench_fp8_ragged_dot.py` no longer uses equal-size groups. It generates a
deterministic capacity-limited random distribution:

- distribution: `bounded_normal_sigma_0.18_capacity_1.35`
- target operating point: `E=64`, average `tokens_per_expert=1024`
- generated counts in the latest H100 runs: min `683`, max `1352`, sum `65536`

Command:

```bash
./h100 python bench_fp8_ragged_dot.py --quick --warmups 1 --iters 3 --fp8-implementation triton
```

Measured on the H100:

| GEMM | FP8 implementation | fwd speedup vs bf16 | fwd+bwd speedup vs bf16 | fwd rel-Fro error |
| --- | --- | ---: | ---: | ---: |
| w13 K=2560 N=2560 | Pallas Triton ragged | 0.549x | 0.502x | 0.0424 |
| w2 K=1280 N=2560 | Pallas Triton ragged | 0.602x | 0.531x | 0.0392 |

The previous gamma distribution was much more skewed (min `241`, max `3275`)
and produced similar or worse FP8 speedups, so the miss is not only a
distribution artifact.

## Implemented paths

- `implementation="triton"`: correct Pallas-Triton ragged FP8 path using the
  existing ragged layouts for forward, dlhs, and drhs. Backward casts saved
  E4M3 forward operands to E5M2 because Pallas `pl.dot` rejects mixed FP8
  operands in this kernel.
- `implementation="triton_native"`: a direct `jax_triton` implementation for
  forward, dlhs, and drhs. It is correct on small non-uniform probes but only
  reaches about 0.20x fwd+bwd speedup at the target, so it is not viable.
- `implementation="padded_dense"`: dynamic non-uniform pack -> padded batched
  FP8 GEMM -> scatter. This supports genuine non-uniform `group_sizes`, but the
  pack/scatter and overcompute costs make it slower than the Pallas ragged path
  at the target.

Small H100 correctness probe for `triton_native` and `padded_dense`:

- non-uniform groups `[64, 137, 89, 222]`
- forward output dtype `bfloat16`
- forward relative-Frobenius error about `0.038`
- gradients produced with shapes `(512, 64)` and `(4, 64, 128)`

## Backend/toolchain blocker

The promising route is JAX's installed Mosaic GPU ragged WGMMA kernel:

```python
from jax.experimental.pallas.ops.gpu.ragged_dot_mgpu import ragged_dot
```

It is designed as a Hopper ragged WGMMA implementation and its source explicitly
mentions support for mixed E4M3/E5M2 FP8 operands. However, compiling even a
small FP8 call fails on this H100 image:

```text
ValueError: Only f16 WGMMA supports transposes
```

I reproduced the same error with JAX's installed dense Hopper Mosaic matmul:

```python
from jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu import matmul
```

using E4M3 x E4M3 operands. This indicates the blocker is in the installed
Mosaic GPU FP8 WGMMA lowering path, not only in my ragged wrapper.

I also checked the installed CUTLASS Python package (`cutlass 4.5.2`). Importing
the experimental CuTe path needed for a Python-side grouped FP8 GEMM hits:

```text
NotImplementedError: CuTe Experimental module is only supported on Cuda toolkit 13.1 and above!
```

The H100 environment currently exposes `XLA_FLAGS=--xla_gpu_cuda_data_dir=/app/.xla_cuda`;
I did not change the cluster image or CUDA toolkit.

## Verification run so far

Successful:

```bash
./h100 python -m py_compile bench_fp8_ragged_dot.py \
  lib/haliax/src/haliax/nn/ragged_dot.py \
  lib/haliax/src/haliax/quantization.py
```

Successful direct H100 probes:

- small non-uniform forward/backward for `triton_native`
- small non-uniform forward/backward for `padded_dense`

Blocked:

- non-uniform target performance remains below the required 1.2x.
- Mosaic GPU FP8 WGMMA path rejects FP8 with `Only f16 WGMMA supports transposes`.
- CUTLASS CuTe Python route requires CUDA toolkit 13.1+.

## Request for coordinator hint

I need a hint on the intended fast FP8 ragged backend for this image:

- Is there a known flag, JAX patch, or package version needed to make Mosaic GPU
  FP8 WGMMA accept the ragged/dense Hopper kernels?
- Or should the solution use a different installed custom-call/CUTLASS route
  that avoids the CUDA 13.1 CuTe Python requirement?
