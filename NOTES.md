# FP8 ragged_dot notes

## Final result

Implemented an opt-in FP8 `ragged_dot` path with:

- `Fp8RaggedDotOp` delayed-scaling state in `haliax.quantization`
- E4M3 forward operands
- E5M2 output-gradient operands in the custom VJP dgrad path
- genuine non-uniform `group_sizes` in `bench_fp8_ragged_dot.py`
- no uniform batched-dense headline path
- no pad-to-capacity fallback in the measured path

The final checked-in default is:

- forward: Mosaic FP8 ragged WGMMA
- dgrad: Mosaic mixed E5M2 x E4M3 ragged WGMMA
- wgrad: genuine-ragged bf16 Triton fallback

This is correct end-to-end for arbitrary non-uniform groups, but it does not
clear the requested 1.2x fwd+bwd speedup. The measured ceiling is blocked by
the weight-gradient contraction and by the current Mosaic layout-inference wall
for arbitrary-ragged FP8 wgrad.

## 1024 non-uniform H100 measurement

Command:

```bash
FP8_MOSAIC_OUTPUT_FP8=1 \
FP8_MOSAIC_MAX_CONCURRENT_STEPS=4 \
FP8_MOSAIC_GRID_BLOCK_N=4 \
./h100 python bench_fp8_ragged_dot.py --quick --warmups 2 --iters 5 \
  --fp8-implementation mosaic --fp8-block-m 192 --fp8-block-n 128 --fp8-block-k 64
```

Group distribution:

- `bounded_normal_sigma_0.18_capacity_1.35`
- `E=64`, average tokens/expert `1024`
- min `683`, max `1352`, sum `65536`
- arbitrary non-aligned sizes; not uniform batching and not pad-to-capacity

Measured default ceiling with Mosaic FP8 forward/dgrad and bf16 Triton wgrad:

| GEMM | bf16 fwd+bwd | fp8 fwd+bwd | fwd speedup | fwd+bwd speedup | fwd rel-Fro error |
| --- | ---: | ---: | ---: | ---: | ---: |
| w13 K=2560 N=2560 | 5.946 ms | 6.272 ms | 1.022x | 0.948x | 0.0424 |
| w2 K=1280 N=2560 | 3.256 ms | 3.317 ms | 1.012x | 0.982x | 0.0393 |

For comparison, the all-FP8 Pallas wgrad fallback (`FP8_MOSAIC_WGRAD=triton`)
is also correct but slower:

| GEMM | fwd speedup | fwd+bwd speedup | fwd rel-Fro error |
| --- | ---: | ---: | ---: |
| w13 K=2560 N=2560 | 0.825x | 0.646x | 0.0424 |
| w2 K=1280 N=2560 | 0.957x | 0.673x | 0.0393 |

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

I implemented `mosaic_transposed_ragged_dot` for wgrad. The first version used
`lhs.T[K,T] x dout[T,N]`, keeping the token contraction dimension contiguous and
avoiding an FP8 WGMMA transpose. The generic version needs first/last block
ragged-boundary masks and fails Mosaic layout inference:

```text
ValueError: Layout inference failed to find a solution.
```

After the coordinator hint, I tried moving boundary handling out of WGMMA by
packing each expert's token stream to a multiple of the token tile only (not
padding to global capacity), then passing token-contiguous `lhs.T[K,T_pad]` and
`dout.T[N,T_pad]` into a no-mask Mosaic grouped GEMM using
`transpose_ref(rhs_smem)` on the K-contiguous RHS tile. This still fails Mosaic
layout inference on the H100:

```text
ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.
```

I also tried materializing the transposed FP8 operands with an explicit add of a
zero array before the Mosaic call; the same layout-inference failure remains.

The block-aligned no-mask variant without the packing transform compiles but is
not fast enough at the 1024 operating point.

### Working ceiling: bf16 Triton wgrad

For arbitrary non-aligned non-uniform groups, the checked-in default wgrad is
the genuine-ragged bf16 Triton contracting-dimension fallback. It avoids
pad-to-capacity and is faster than the all-FP8 fallback, but it dominates enough
of fwd+bwd that the whole path reaches only `0.948x`/`0.982x` at the 1024
operating point.

The all-FP8 Pallas wgrad fallback remains selectable with
`FP8_MOSAIC_WGRAD=triton`. I fixed its small-shape tile edge by clamping the
contracting-dimension output tile to the actual `K,N` dimensions; direct drhs
fallback parity on `[13, 5, 17, 9]` group sizes is exact vs
`jax.lax.ragged_dot_general`.

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
- direct contracting-dimension fallback parity vs `jax.lax.ragged_dot_general`:
  relative Frobenius `0.0`
- custom-VJP small non-uniform gradient parity with group sizes `[13, 5, 17, 9]`:
  forward rel-Fro `0.0369`, lhs grad rel-Fro `0.0478`, rhs grad rel-Fro `0.0356`
- output gradients use E5M2 in the Mosaic dgrad custom-VJP path

## Documented ceiling

- In-kernel boundary masking for `lhs.T[K,T] x dout[T,N]` causes Mosaic layout
  inference failure.
- Removing the masks only works for block-aligned group sizes and does not clear
  the 1024 fwd+bwd bar.
- Token-tile packing with boundary handling moved to the grid still fails Mosaic
  layout inference on this build.
- `transpose_ref` variants also hit the same Mosaic layout-inference wall.
- The working fallbacks are correct and genuinely ragged, but wgrad remains the
  bottleneck.

## Resume findings

After resuming, I isolated forward tuning first:

- Raw Mosaic FP8 forward with pre-quantized operands is fast enough at the 1024
  operating point:
  - w13 `block_m=192, block_n=128, block_k=128`: `1.294 ms`, `664 TFLOP/s`,
    `1.42x` vs bf16 forward.
  - w2 `block_m=192, block_n=128, block_k=128`: `0.832 ms`, `516 TFLOP/s`,
    `1.26x` vs bf16 forward.
- Full forward is still much slower than raw WGMMA, so the next bottleneck is
  quantize/layout work around the kernel, not the ragged WGMMA body itself.
- `block_k=128` is valid and fast for forward-only, but the current custom-VJP
  fwd+bwd compilation hits a Mosaic shared-memory limit (`requested 262144,
  available 232448`). The backward-safe fwd+bwd tile remains
  `block_m=192, block_n=128, block_k=64`.

The pod had `jaxlib 0.10.0.dev0+selfbuilt`, but the installed Python sources
still contained same-dtype guards in both the Pallas WGMMA wrapper and the Mosaic
lowerer. I added a local `_wgmma` wrapper in `_fp8_mosaic_ragged.py` to bypass
the Python guard for E4M3/E5M2, and patched the pod venv lowerer check in
`/app/.venv/lib/python3.12/site-packages/jax/experimental/mosaic/gpu/wgmma.py`
to allow the E4M3/E5M2 pair. After that, direct mixed dgrad compiled again.

I decoupled backward tile sizes from the forward tile so forward can use
`block_k=128` while dgrad and the bf16 wgrad fallback keep known-safe
`block_k=64`. With `block_m=192, block_n=128, block_k=128`, fwd+bwd compiles
and measures about `0.95x`/`1.00x` for w13/w2 in quick runs. With
`block_m=128, block_n=128, block_k=128`, w13 can reach about `1.09x` in a
short run, but w2 stays below `1.0x`.

The generic masked Mosaic wgrad now compiles after the mixed-lowering patch, but
it still measures around `1.00x`/`0.96x` at `block_m=192, block_n=128,
block_k=64`. The block-padded Mosaic wgrad path was not stable above `1.0x` on
repeat. The official `jax.experimental.pallas.ops.gpu.ragged_dot_mgpu` helper was
competitive for raw same-dtype forward, but slower in the full quantized forward
path, so I did not keep it in the op.

## Second resume findings

Important measurement hygiene correction: `./h100` does not forward local
one-shot environment prefixes. This:

```bash
FP8_MOSAIC_WGRAD=triton ./h100 python ...
```

arrives on the pod without `FP8_MOSAIC_WGRAD`. Use:

```bash
./h100 env FP8_MOSAIC_WGRAD=triton python ...
```

or the benchmark is measuring the default selector.

With correctly propagated env vars, the best working default at the 1024
non-uniform operating point remains the Mosaic FP8 forward + mixed-FP8 dgrad
with bf16 Triton wgrad fallback:

```bash
./h100 python bench_fp8_ragged_dot.py --quick --mode fwd_bwd \
  --warmups 1 --iters 3 --fp8-implementation mosaic \
  --fp8-block-m 192 --fp8-block-n 128 --fp8-block-k 128
```

Measured:

| GEMM | bf16 fwd+bwd | fp8 fwd+bwd | speedup |
| --- | ---: | ---: | ---: |
| w13 K=2560 N=2560 | 5.901 ms | 6.018 ms | 0.981x |
| w2 K=1280 N=2560 | 3.228 ms | 3.211 ms | 1.005x |

Forward-only tuning with real remote envs:

- best measured full-forward config remains
  `block_m=192, block_n=128, block_k=128, max_concurrent_steps=4,
  grid_block_n=8`;
- measured full-forward speedup was about `1.10x` for w13 and `1.06-1.09x`
  for w2, including quantize/dequantize;
- `block_m=256`, `block_n=64`, and `block_n=256` were all slower;
- `block_n=96` triggered a CUDA misaligned-address failure and should not be
  reused without a fresh process and a specific reason.

I added experimental, non-default selectors for the remaining wgrad attempts:

- `FP8_MOSAIC_WGRAD=mosaic_block_pad_pretransposed`: a native Triton kernel
  packs each expert token stream to the wgrad token tile and writes transposed
  FP8 operands, then calls no-mask Mosaic wgrad. This still fails Mosaic layout
  inference in `mosaic_transposed_ragged_dot`, even for small already-materialized
  block-aligned same-dtype FP8 inputs:
  `ValueError: Failed to infer a possible set of layouts.`
- `FP8_MOSAIC_WGRAD=triton`: Pallas-Triton wgrad rejects the required mixed
  E4M3 x E5M2 operands before lowering:
  `TypePromotionError: Input dtypes ('float8_e4m3fn', 'float8_e5m2') have no
  available implicit dtype promotion path.`
- `FP8_MOSAIC_WGRAD=triton_native`: native Triton accepts mixed FP8, but is
  much too slow at the 1024 operating point (`0.47x`/`0.46x` fwd+bwd for
  w13/w2 with `block_m=192, block_n=128, block_k=128`).

The current specific blocker for the >=1.2x target is therefore a fast
arbitrary-ragged mixed-FP8 wgrad. I have a standalone pack+transpose kernel that
materializes token-tile-padded FP8 operands outside Mosaic, but the existing
transposed Mosaic wgrad kernel still fails layout inference when consuming those
operands. The native Triton mixed-FP8 fallback is correct enough to compile, but
far below the needed performance.
