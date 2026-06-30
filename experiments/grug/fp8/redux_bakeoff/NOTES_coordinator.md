# Coordinator's own fp8 ragged_dot prototype — NOTES

## Approach (simplest complete path)

Mirror the dense direct-quant fp8 op (PR #6660, `fp8_scaled_dot_general` / `quantized_dot`),
but replace the inner `lax.dot_general` with the **existing Pallas-Triton grouped-matmul
kernels** from `haliax.nn.ragged_dot`. The bf16 kernel already:
- accumulates in **fp32** (`acc = jnp.zeros(..., jnp.float32)`),
- casts operands via `jnp.result_type(a, b)` (so same-dtype fp8 operands stay fp8 → Triton
  `pl.dot` lowers to an fp8 MMA on Hopper),
- supports the 3 ragged layouts (fwd / dlhs / drhs) used by the custom VJP.

So the **only structural change** is the *output* dtype: the bf16 helpers set
`out_shape = lhs.dtype`, which for fp8 inputs would truncate the fp32 accumulator back to fp8.
I vendored two tiny variants (`_triton_default_pallas_call_out_dtype`,
`_triton_ragged_contracting_dim_pallas_call_out_dtype`) that take an explicit `out_dtype`
(bf16/fp32) and otherwise reuse the unmodified kernels and block sizes.

Files:
- `lib/haliax/src/haliax/nn/fp8_ragged_dot.py` — the op (`fp8_ragged_dot` + custom-VJP core
  `fp8_scaled_ragged_dot`).
- `bench_fp8_ragged_dot.py` — canonical harness (fp8 vs bf16, fwd / fwd+bwd, TFLOP/s, speedup,
  forward parity check). Has `--candidate module:attr` so the same harness benchmarks the other
  agents' implementations head-to-head.

## Numerics recipe

- Forward: E4M3 × E4M3, fp32 accumulate, dequantize by `lhs_scale * rhs_scale`.
- Backward: output grad → **E5M2** (delayed scaling, amax history). Both backward GEMMs run as
  **genuine mixed fp8** — `dlhs = g(E5M2) @ rhs(E4M3)ᵀ`, `drhs = lhs(E4M3)ᵀ @ g(E5M2)` — the true
  TE recipe (REQUIRED; no re-casting weights/acts to E5M2). To force the MMA to keep the operands
  at their native fp8 dtypes I vendored mixed-capable kernels (`_fp8_ragged_dot_kernel`,
  `_fp8_ragged_contracting_dim_dot_kernel`) that call `pl.dot(a, b)` WITHOUT the bf16 kernel's
  `jnp.result_type` up-promotion (which would collapse a mixed pair to f16 and erase the fp8 win).

Delayed per-tensor scaling state (scale + amax history for lhs/rhs/grad) is threaded through
the custom VJP exactly like `quantized_dot`, ready to be persisted by an
`OverwriteWithGradient` module (the `Fp8RaggedDotOp` eqx module + `MoELinear` wiring is the
obvious next step for full integration; not needed to prove the kernel speedup).

## Validation done (CPU, no GPU)

`cpu_logic_test.py` swaps the Triton call for `ragged_dot_general` (same dim-numbers) to test
the quant/dequant/scaling/VJP math:
- forward rel_err vs f32 = **0.038** (E4M3-typical)
- dlhs / drhs grad rel_err = **0.069 / 0.069** (E5M2 backward)
- all shapes correct; E5M2 grad quantization confirmed (dtype `float8_e5m2`).

## The open risks (resolved only on H100)

1. **Does Pallas-Triton `pl.dot` lower E4M3×E4M3 (forward) to a real Hopper fp8 MMA** and clear
   the ≥20% bar? fp8 halves operand bytes and ~doubles MMA throughput vs bf16, so ≥20% is
   plausible even at the bf16 kernel's small `block_k=32`. Tuning lever if short: raise `block_k`
   (64/128), `block_n`, `num_stages` for the fp8 path.
2. **Does Triton `pl.dot` accept a MIXED E5M2×E4M3 pair at all** (backward, required)? If it
   rejects mixed fp8 or inserts upcasts, the in-marin fallback is the XLA path
   (`jax.lax.ragged_dot_general` with fp8 operands + `preferred_element_type`), which is confirmed
   to accept mixed fp8 on CPU and *may* lower to a cuBLASLt fp8 grouped GEMM on H100. If neither
   in-tree backend yields genuine mixed-fp8 wgmma, the documented escape is a Pallas/Triton (or a
   new in-tree Mosaic-GPU grouped-GEMM) patch — to be written only if the benchmark forces it.

The H100 benchmark therefore sweeps BOTH backends (triton-fp8, xla-fp8) for forward and the mixed
backward, and inspects the lowered HLO/PTX to confirm genuine fp8 MMA (not a bf16/f16 fallback).

## Local-env gotcha for the coordinator

The `.venv` editable-installs haliax from `/home/matt/projects/marin` (which is on the forbidden
research branch). For any run against this worktree, shadow it:
`PYTHONPATH=<worktree>/lib/haliax/src` (and similarly for other libs if needed), or re-`uv sync`
from the worktree. On H100, sync with `--extra gpu` + cuDNN per the project's GPU setup note.
