# Invalid constant-folded H100 component probe

This raw probe is retained because it exposed a benchmark-boundary bug. The
matched JAX/XLA functions closed over random arrays instead of accepting them
as runtime arguments. XLA therefore compiled constants and copies, while the
opaque generated FFI still executed its Fold kernels.

All performance ratios in `summary.json` are invalid and must not be used. The
optimized HLO under `xla/` makes the failure explicit: the modules have no
parameters and their roots copy constants.

The generated correctness and determinism evidence remains valid. The
corrected runtime-input measurement is in
`../jax_row_normalization_backward_h100_components_corrected_v1`.
