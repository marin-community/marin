# Generated row-normalization reverse on H100

This artifact records a generated three-stage AxisFoldPipeline for the reverse of
an uncentered row normalization. JAX owns differentiation. Shuttle imports the
natural StableHLO VJP, recovers generic Map and Fold structure, generates CUDA,
and replaces the region with one typed FFI call.

The result is correct and deterministic, but misses the performance target.

## Configuration

- Shuttle source revision: `fdd8380cf0a37809c5fafa6ab4a82be09cd1f1b1`
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- Driver: 595.71.05
- Power limit: 700 W
- JAX and JAXlib: 0.10.1
- CUDA compiler: 13.3.73
- Shape: 2,048 rows by 4,096 features
- Storage: BF16 inputs and outputs, FP32 partial state
- Numerical policy: `allow_rounding_reorder`
- Samples: 30 counterbalanced samples, 10 iterations per sample

The task verified all 90 entries in `source-manifest.sha256` before checking the
GPU or compiling CUDA.

## Correctness

The generated path matches the natural JAX VJP within the declared policy:

- input cotangent maximum absolute error: `0.0078125`
- input cotangent mean absolute error: `1.9742053e-8`
- feature-scale cotangent maximum absolute error: `0.00390625`
- feature-scale cotangent mean absolute error: `9.536743e-7`

Repeated output hashes were stable. The generated handler executed 312 times.
The post-roundtrip audit found one custom call, no copy or transpose adapters,
the expected two roots, and no live internal source instruction.

## Performance

- generated median: `0.102763 ms`
- matched XLA median: `0.067930 ms`
- generated/XLA: `1.512778x`

This does not satisfy the `1.20x` target. The generated source and optimized XLA
HLO are included for the next bounded Fold and launch-decomposition audit.
