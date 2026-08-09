# S=64 streaming-attention reverse numerical negative result

This artifact preserves a shape-specific negative result from the same H100
replay used for the primary S=2,048 proof. The ordinary JAX reverse entry was
compiled through XLA `PRE_SCHEDULER`, replaced by the generated typed-FFI
handler, and executed. The harness then rejected the output at its BF16
correctness gate:

| Gradient | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| dQ | 0.03125 | 0.000347032 |
| dK | 0.03125 | 0.000619165 |
| dV | 0.0625 | 0.000680739 |

The small-shape configuration was batch 1, sequence 64, 32 query heads, eight
K/V heads, head dimension 128, and 32x32 tiles. The dV maximum exceeds the
0.03125 bound used by the primary benchmark. This is a numerical-policy or
shape-support failure, not a callback/compilation failure, and S=64 must not be
listed as accepted until it receives an explicit shape-appropriate contract or
a more accurate generated schedule.

`source-vjp-stablehlo.mlir.bc` and `original-pre-scheduler-hlo.*` preserve the
natural frontend and the exact GPU module presented to the callback.
`replay.stderr` records the fail-closed exception and exact observed errors.
No timing claim is made from this run.

