# FP8 CuTe ragged_dot — implementation notes

Branch: `fp8-ragged-cute`.

## Headline

**No forked jaxlib.** The Hopper TMA warp-specialized grouped GEMM runs entirely
through `cutlass.jax.cutlass_call` (NVIDIA CuTe DSL + stock upstream jaxlib).
The standard `nvidia-cutlass-dsl` wheel is the only new dependency.

## TMA through the FFI — what we proved

The FA4-era assumption that "generic memref operands → no TMA" is **wrong for
cutlass_call** specifically. `cutlass_call` operands are built as genuine gmem
pointers (`AddressSpace.gmem`, 256-byte aligned), so the following all work through
the FFI path without any jaxlib patch:

- TMA atoms and tensormap device updates (SMEM mode)
- Warp specialization + setmaxnreg register reallocation
- Persistent group scheduler with barrier pipeline
- TMA store epilogue
- Cluster launch (`cluster=(2,1,1)`, mcast TMA loads)

Verification: job `mcwitt-fp8cute-tma7` — 1130 TFLOP/s at d2560/tpe1024 (cluster
(1,1)), 1230 TFLOP/s at cluster (2,1); max_bit_diff=0 (8 repeated calls); correct
through three successive calls with distinct XLA-reallocated buffers.

### Four load-bearing gotchas

1. **Dynamic-layout dummy initials.** The tiny `initial_a/b/c` descriptor-template
   tensors passed to `cutlass_call` MUST be dynamic-layout (`TensorSpec(static=False)`,
   which overrides `use_static_tensors=True` per-tensor). With static tiny extents CuTe
   canonicalises size-1 Rest modes in the TMA coordinate tensors, collapsing
   tile-coordinate arithmetic for all real tiles beyond the dummy extent → fast garbage
   output (`rel_fro ≈ 1e34`, nondeterministic). The real operand tensors can stay static.

2. **`cute.copy` not raw-NVVM TMA loads.** The vendored `_tma_load_ab_nvvm_no_mcast`
   helper (raw NVVM `CpAsyncBulkTensorGlobalToSharedClusterOp` + `elect_sync`) hits
   `CUDA_ERROR_ILLEGAL_INSTRUCTION` under cutlass 4.5.2 regardless of CUDA graphs or
   register realloc. It was a workaround for an older-toolchain PTXAS bug and is
   obsolete. Use `GROUPED_GEMM_FORCE_CUTE_COPY=1` (or strip the NVVM path — this branch
   strips it from the vendored copy).

3. **`CUTE_DSL_ARCH=sm_90a` is mandatory.** When GPU detection fails the DSL silently
   defaults to `sm_100a` (Blackwell) → a cubin that will not load on H100 with a sticky
   `ILLEGAL_INSTRUCTION`. `compile_options=(cute.GPUArch("sm_90a"),)` does NOT set the
   trace arch — only the env var does. The `ensure_hopper_arch()` helper in
   `_tma_grouped_adapter.py` exports it at call-site construction time.

4. **Bounded modern group scheduler.** The deprecated
   `GroupedGemmTileSchedulerHelper` has an unbounded while-loop that hangs on an
   over-estimate of `total_num_clusters`. This branch uses
   `utils.StaticPersistentGroupTileScheduler` (present in 4.5.2), whose scheduler
   clamps invalid tiles via `is_valid_tile`, so a static upper bound
   `ceil_div(T, tile_m) + E - 1` (times the fixed N cluster-tile count) is safe.

## Mixed-FP8 recipe

| Product | A dtype | B dtype | acc | out |
|---------|---------|---------|-----|-----|
| forward | E4M3 (`lhs`) | E4M3 (`rhs_t`) | F32 | BF16 |
| dgrad   | E5M2 (`grad`) | E4M3 (`rhs`) | F32 | BF16 |
| wgrad   | E4M3 (`lhs_t`) | E5M2 (`grad_t`) | F32 | BF16 |

**Out-scale convention**: the epilogue DIVIDES the F32 accumulator by `out_scale[0]`
before the BF16 cast (haliax dequantize convention: `out = acc / out_scale`).

**Cast-transpose for wgrad.** Hopper 8-bit wgmma only supports the no-transpose layout
(the K dimension must be the contiguous axis of both operands). The wgrad product
contracts the token axis M (which is the innermost axis after transposing both
activations and the output gradient), so both operands are transposed before the
kernel: `lhs_t[K, M]` and `grad_t[N, M]`.

**Fused dual-write weight quantisation.** The fused cast-transpose kernel is
generalised to a batched `[E, M, N]` form (expert axis on `blockIdx.z`): one HBM read
of BF16 `rhs[E, K, N]` emits BOTH FP8 layouts the backward needs — `q_rhs[E, K, N]`
(dgrad, natural) + `q_rhs_t[E, N, K]` (forward, K-contiguous) — replacing three reads
(amax + two quantises). The `_dual_write_q` custom VJP threads delayed-scaling scale +
amax history exactly as `fp8.in_q`.

## 16-token group padding (TMA 16B innermost-coordinate constraint)

For wgrad, the per-group token offset is folded into the TMA *element coordinate*
(not a base-pointer advance), which must be 16B-aligned (= 16 FP8 elements). Groups
whose offset is ≢ 0 (mod 16) fault the TMA hardware. Fix: repack the token axis so
every group starts on a 16-token boundary, zero-filling the <16-token gap after each
group. The host repack and the device prologue agree on the same 16-rounded
exclusive-prefix offsets. Zero pads add exact +0.0 (and sit past the per-group
descriptor extent), so results are bit-identical to unpadded packing. Static padded
width = `round_up_16(T + 16·E)`.

## WgradMode and Fp8RaggedDotBackend

`WgradMode.FP8` (default): wgrad uses the FP8 E4M3×E5M2 CuTe kernel.
`WgradMode.BF16`: wgrad falls back to Triton BF16 ragged_dot (useful as a control
baseline).

`Fp8RaggedDotBackend.CUTE` (default): all three products via the CuTe DSL TMA kernel.
`Fp8RaggedDotBackend.MOSAIC`: Mosaic-GPU Warpgroup path. Requires forked jaxlib for
mixed-wgmma; `init` rejects `fwd_dtype != rev_dtype` (stock-jaxlib same-dtype
constraint, jax#38859). `MOSAIC.__call__` raises (not-vendored extension point).

The default CUTE backend accepts `rev_dtype=jnp.float8_e5m2` for the genuine mixed
E5M2×E4M3 backward (the TE recipe). Pass `rev_dtype=E5M2` explicitly to enable it;
the default `rev_dtype=E4M3` is the uniform backward (matching forward dtype) for
backward compatibility.

## Acceptance results (job `mcwitt-fp8cute-a2f-accept`, d2560, E=64, ODD non-uniform groups)

| tpe | bf16 Triton | single-shot | amort N=2 | amort N=4 | ceiling (amortized steady-state) |
|-----|-------------|-------------|-----------|-----------|----------------------------------|
| 512  | 3.58 ms | 3.96 ms / 0.906× | 1.002× | 1.058× | 3.20 ms / **1.120×** |
| **1024** | **6.15 ms** | **5.60 ms / 1.097×** | 1.173× | **1.215× PASS** | **4.88 ms / 1.260× PASS** |
| 2048 | 11.51 ms | 9.50 ms / 1.212× | 1.262× | 1.288× | 8.75 ms / **1.315×** |

**Gate (≥1.2× amortized e2e, gate config tpe=1024): MET.**

"Amortized steady-state / ceiling" = pre-quantised weights (weights quantised once
per optimiser step, reused per microbatch). "N=4" = grad-accum depth 4. The ceiling
(1.260×) matches the A2b control (1.278×) within pod variance.

N=2 (1.173×) and single-shot (1.097×) fall short of 1.2×. Root cause profiled:

**Component profile (tpe=1024):**
- GEMM TFLOP/s: fwd 720, dgrad 714, wgrad 496
- wgrad is dragged by the 16-token repack: two gathers over `[2560, 65536]`
  token-major operands, ~0.5 ms — the correctness tax of arbitrary group sizes.
  (A2b's 128-aligned wgrad achieved 692 TFLOP/s.)
- Weight dual-write (fused): 0.83 ms vs 2-pass 1.15 ms (−0.32 ms, the M2 win).
- Activation dual-write: 0.39 ms; grad cast-transpose: 0.39 ms.
- Single-shot: ceiling 4.88 + weight marginal 0.73 = 5.60 ms.

### Honest framing

The gate is met in gradient-accumulation ≥4 and weights-reused (steady-state) regimes.
Single-shot and N=2 are below 1.2× due to the odd-group wgrad repack. The identified
follow-up to recover it: fuse the padding into the token-major PRODUCERS (grad
cast-transpose + lhs transpose already write the token-major operands — emit them
pre-padded), removing the two standalone gathers. Not attempted (kernel scope).

**Note on test coverage:** The CPU test suite does not exercise the numeric or
determinism invariants — all such tests are `gpu_only`. An H100 run of
`lib/haliax/tests/test_fp8_ragged.py` is mandatory before trusting a green CPU CI.

## Comparison: CuTe vs Mosaic-GPU

The Mosaic path achieves 1.31× layer fwd+bwd at canonical d2560 but requires the
forked jaxlib for mixed wgmma (Warpgroup kernel). The CuTe path delivers genuine
mixed E5M2×E4M3 without any jaxlib fork (1.260× ceiling / 1.215× at N=4), at the
cost of ~0.5 ms odd-group repack overhead in the wgrad path.
