# FP8 ragged_dot notes

## Current result

Implemented an opt-in FP8 `ragged_dot` path with:

- `Fp8RaggedDotOp` delayed-scaling state in `haliax.quantization`
- E4M3 forward operands
- E5M2 output-gradient operands in the custom VJP
- genuine non-uniform `group_sizes` in `bench_fp8_ragged_dot.py`
- no uniform batched-dense headline path

The best fully non-uniform 1024-token operating-point result I measured does
not clear the required 1.2x fwd+bwd speedup. The ceiling is currently blocked
by the weight-gradient contraction.

## 1024 non-uniform H100 measurement

Command:

```bash
FP8_MOSAIC_OUTPUT_FP8=1 \
FP8_MOSAIC_MAX_CONCURRENT_STEPS=4 \
FP8_MOSAIC_GRID_BLOCK_N=4 \
./h100 python bench_fp8_ragged_dot.py --quick --warmups 1 --iters 3 \
  --fp8-implementation mosaic --fp8-block-m 192 --fp8-block-n 128 --fp8-block-k 64
```

Group distribution:

- `bounded_normal_sigma_0.18_capacity_1.35`
- `E=64`, average tokens/expert `1024`
- latest non-aligned distribution before the final aligned wgrad experiment:
  min `683`, max `1352`, sum `65536`

Measured ceiling with Mosaic forward/dgrad and no pad-to-capacity Pallas FP8
wgrad:

| GEMM | fwd speedup | fwd+bwd speedup | fwd rel-Fro error |
| --- | ---: | ---: | ---: |
| w13 K=2560 N=2560 | 0.919x | 0.651x | 0.0424 |
| w2 K=1280 N=2560 | 2.100x | 0.524x | 0.0393 |

I also tried a block-aligned non-uniform distribution with `FP8_MOSAIC_WGRAD=mosaic_nomask`
(min `768`, max `1280`, sum `65536`). That all-Mosaic no-pad path compiled
and ran, but still measured only:

| GEMM | fwd speedup | fwd+bwd speedup | fwd rel-Fro error |
| --- | ---: | ---: | ---: |
| w13 K=2560 N=2560 | 0.755x | 0.643x | 0.0424 |
| w2 K=1280 N=2560 | 0.918x | 0.686x | 0.0393 |

## Backend paths tried

### Working: Mosaic forward and dgrad

The patched pod has `jaxlib 0.10.0.dev0+selfbuilt`. The forward and dgrad use
Hopper Mosaic WGMMA via `_fp8_mosaic_ragged.py`:

- forward: `lhs[M,K] x rhs[G,N,K]` with `transpose_rhs=True` layout
- dgrad: `q_g[M,N] x q_rhs[G,K,N]` with K-contiguous RHS layout
- mixed E5M2 x E4M3 dgrad compiles and runs

### Blocked: generic Mosaic wgrad

I implemented `mosaic_transposed_ragged_dot` for wgrad using
`lhs.T[K,T] x dout[T,N]`, keeping the token contraction dimension contiguous and
avoiding an FP8 WGMMA transpose. The generic version needs first/last block
ragged-boundary masks. That path fails Mosaic layout inference:

```text
ValueError: Layout inference failed to find a solution.
```

The block-aligned no-mask variant compiles but is not fast enough at the 1024
operating point.

### Working but slow: Pallas FP8 wgrad

For arbitrary non-aligned non-uniform groups, wgrad falls back to the existing
genuine-ragged Pallas contracting-dimension kernel. This avoids pad-to-capacity
and keeps an E5M2 operand, but it dominates fwd+bwd time.

### Rejected: padded dense

I also tried dynamic pack -> padded batched FP8 GEMM -> scatter. It is correct
for non-uniform groups but violates the no pad-to-capacity direction for the
headline path and is slower at the target.

## Verification

Successful:

```bash
./h100 python -m py_compile \
  bench_fp8_ragged_dot.py \
  lib/haliax/src/haliax/nn/_fp8_mosaic_ragged.py \
  lib/haliax/src/haliax/nn/ragged_dot.py \
  lib/haliax/src/haliax/quantization.py
```

Successful direct H100 probes:

- small non-uniform Mosaic forward/backward
- small block-aligned no-mask Mosaic wgrad forward/backward
- output gradients use E5M2 in the custom VJP

## Request for next hint

The remaining blocker is a fast arbitrary-ragged FP8 wgrad:

- In-kernel boundary masking for `lhs.T[K,T] x dout[T,N]` causes Mosaic layout
  inference failure.
- Removing the masks only works for block-aligned group sizes and does not clear
  the 1024 fwd+bwd bar.
- Pallas wgrad is genuine ragged but too slow.

I need a hint for the intended arbitrary-ragged FP8 wgrad strategy on this
patched JAX/Mosaic stack.
