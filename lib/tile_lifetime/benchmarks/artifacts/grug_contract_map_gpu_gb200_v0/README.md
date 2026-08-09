# Natural Grug Contract+Map typed-FFI replay

This artifact records a one-GB200 execution of a generic Contract+Map region
recovered from the `PRE_SCHEDULER` HLO of an ordinary one-layer Grug training
step. Shuttle replaces the recovered region with one XLA typed-FFI custom call;
the handler executes one generic cuBLAS Contract and a generated source-ordered
scalar Map.

Source revision: `2ed4741e134d89ceff918d6b6b3fbfc929c93254`.

## Result

- The natural baseline and transformed executable both compile and execute.
- The transformed HLO contains exactly one Shuttle custom call and two tuple
  projections.
- The handler executes 35 times: one correctness execution, four warmup pairs,
  and 30 measured pairs.
- The initial baseline/transformed comparison is bitwise exact across 53 result
  leaves, with zero maximum and mean absolute error.
- Counterbalanced medians are 0.552480 ms for baseline and 0.563937 ms for the
  transformed executable, or `1.020737x`.

The recovered boundary is:

```text
inputs:
  mul.73          bf16[8,32]
  reshape.227     bf16[8,32]
  remat2.66       bf16[32,32]
outputs:
  dot_general.226 bf16[8,32]
  mul.963         bf16[8,32]
```

The generated output Map is:

```text
projection_value
round_bf16(bf16_to_f32(input0[index]) * projection_value)
```

## Determinism caveat

Repeated whole-train-step hashes are not bitwise stable for either XLA
baseline or transformed execution: each path produces four unique hashes over
30 repetitions. Their hash families largely overlap, and the initial paired
correctness result is exact. This artifact therefore proves correct region
replacement, not deterministic execution of every operation elsewhere in the
complete Grug step.

## Toolchain repair

JAX 0.11 initially resolved CUDA 13.3 NVVM/NVRTC beside CUDA 13.0 PTXAS, causing
PTX 9.3 to be rejected by a PTX 9.0 assembler. The successful replay pins NVCC,
CRT, NVVM, NVRTC, and NvJitLink coherently at 13.0.88.

The pip CUDA directory contained versioned-only `libcublas.so.13` and
`libcudart.so.13`. Disposable symlinks made the measured run possible. The
corresponding source checkpoint removes that environmental requirement by
linking the resolved absolute shared-library paths and disabling NVCC's
implicit cuDART link.

See `provenance.md` for the exact environment and command,
`final2/summary.json` for every raw timing/hash sample, and `SHA256SUMS` for the
sealed file manifest. Earlier toolchain and benchmark failures are retained to
distinguish environment repair from source correctness.
