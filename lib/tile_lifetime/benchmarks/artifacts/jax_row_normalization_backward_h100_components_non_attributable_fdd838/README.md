# H100 RMS reverse component profile: negative attribution result

This artifact preserves the exact-source H100 replay of Shuttle revision
`fdd8380cf0a37809c5fafa6ab4a82be09cd1f1b1`. The full generated typed-FFI
pipeline remains a valid matched measurement: `0.104100 ms` versus
`0.070481 ms` for natural JAX/XLA (`1.476984x`). It is deterministic within
the capture and agrees with the natural JAX VJP to maximum absolute errors of
`0.0078125` for the input cotangent and `0.00390625` for the feature-scale
cotangent under the declared rounding-reorder policy.

The separately timed component kernels do not attribute that full-pipeline
gap. Their interfaces and algebra differ from the kernels in the full source:

- The full K1 consumes BF16 primal/cotangent values and the FP32 inverse scratch
  produced by K0, computes its correlation directly from those values, and
  stores BF16. The isolated input kernel consumes an externally prepared FP32
  projected cotangent, a BF16 rounded standardized activation, and an external
  inverse, then stores FP32.
- The full K2 consumes BF16 primal/cotangent values plus K0's FP32 inverse and
  stores BF16. The isolated feature kernel consumes FP32 projected cotangent
  plus BF16 standardized activation and stores FP32.

This is observable numerically: isolated outputs differ from the corresponding
full generated outputs by maxima `0.0312643` and `0.512909`. Consequently, the
isolated medians (`0.048492 ms` input, `0.040827 ms` feature) must not be
reported as K1 or K2 times or used to infer a K0 residual. They are useful only
as measurements of two alternate generated Fold programs.

The exact next attribution is one CUDA-profiler-API-delimited Nsight Systems
capture of the unchanged full typed-FFI call. That call retains the real
`ScratchAllocator` allocation and sequential `ShuttleAxisFoldKernel0/1/2`
launches. Splitting the stages into independent typed-FFI handlers would change
scratch ownership, launch/handler overhead, input materialization, and the
producer-consumer path.

`result.json` preserves all 30 counterbalanced raw samples, execution orders,
correctness metrics, and hashes. The three `.cu` files are the exact generated
sources loaded by the measured process. `run_manifest.json` records source,
environment, allocation, and archive identity. Binary build products and the
312 KB raw log are intentionally omitted because they add no unique evidence.
