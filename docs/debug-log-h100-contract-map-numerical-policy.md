# H100 Contract/Map numerical-policy debug log

## Scope

This source-only investigation follows the sixth H100 evidence launch from
`d164f5406b6073f9ecb695fa0e242875692d3092`. The immutable launch artifact is
sealed separately at
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_sixth_launch_failure_d164f5_v0/`.
No floor was changed before that launch, no timing was accepted, and this work
does not launch another GPU job.

## First failure

The first ordinary-XLA forward output passed the absolute-error ceilings but
failed the raw mean-BF16-ULP ceiling:

- maximum absolute error: `0.001953989267349243`
- mean absolute error: `0.0003520350146573037`
- maximum BF16 ULP distance: `29298`
- mean BF16 ULP distance: `8.55948121645796`
- nonfinite values: `0`
- three repeats: bit-identical

The predeclared limits were maximum ULP `4` and mean ULP `0.25`. The failure
occurred before profiling, timing, or bundle acceptance.

## Findings

The signed BF16 ordering implementation was not the cause. Exhaustively sorting
all 65,280 finite BF16 encodings produces adjacent ordered distances of one,
except the deliberate shared rank for negative and positive zero. Targeted
checks also cover subnormal-to-normal transitions, both signs, maximum finite
values, infinities, and cross-sign distances. The implementation now rejects a
non-BF16 measured operand, shape broadcasting, and NaN payload ordering.

The acceptance domain was wrong. `real_algebra_fp64` evaluates uninterrupted
FP64 algebra, while ordinary JAX rounds the preactivation, mapped hidden value,
forward output, and reverse path through BF16 values. Near zero or a sign
crossing, a small absolute difference can span thousands of representable BF16
values. Raw final-output ULP therefore measures the distance from a deliberately
different arithmetic path, not source-ordered arithmetic parity.

A deterministic CPU JAX 0.10.1 calibration covers every reviewed structural
case, all four outputs, the launch seed, and three hash-derived held-out seeds
per case. All 16 canonical case/output records exceed both former ULP limits.
Only the cubic case exceeds the former global mean-absolute limit. Across all
64 records, the largest observed absolute metrics are:

| Output | Maximum absolute | Mean absolute | Predeclared maximum | Predeclared mean |
| --- | ---: | ---: | ---: | ---: |
| forward | 0.020529985427856445 | 0.002343598986044526 | 0.03125 | 0.00390625 |
| dx | 0.025110244750976562 | 0.002184648998081684 | 0.03125 | 0.00390625 |
| dw0 | 0.02480936050415039 | 0.003111599013209343 | 0.03125 | 0.00390625 |
| dw1 | 0.03213787078857422 | 0.0033505249302834272 | 0.0625 | 0.00390625 |

The predeclared values are the next power-of-two envelopes over that complete
CPU calibration, not values copied from the failed H100 measurement. The exact
records and regeneration command are checked in at
`lib/tile_lifetime/tests/fixtures/h100_contract_map_numerical_calibration.json`
and
`lib/tile_lifetime/benchmarks/h100_contract_map_numerical_calibration.py`.

## Decision

The numerical policy now follows the reference contract:

- Every measured repeat must be BF16 and exactly shape-matched. The zero-
  nonfinite gate covers every repeat and the reference before finite-only
  drift diagnostics can discard a position.
- `source_ordered_fp32` retains hard maximum and mean BF16 ULP gates, absolute
  gates, zero nonfinite values, and bitwise repeatability.
- `real_algebra_fp64` retains per-output maximum and mean absolute gates, zero
  nonfinite values, and bounded absolute repeat drift. BF16 ULP maximum, mean,
  and pairwise values remain required diagnostics but are not acceptance gates.
- Timing remains unreachable until every output passes its applicable absolute,
  nonfinite, and repeatability gates.

The numerical-floor digest and result schema version change with this policy.
This is CPU/source evidence only. A later authorized H100 run must still pass the
new gates before it can emit timing or an accepted 24-record bundle.
