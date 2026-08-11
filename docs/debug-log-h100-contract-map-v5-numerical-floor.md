# Debugging log for H100 contract-map v5 numerical floors

The fifth H100 evidence attempt failed before timing on the first case's
ordinary-XLA forward mean ULP floor. Preserve the immutable floors and make a
future failure retain enough scalar context to diagnose it without exposing
tensor contents.

## Initial status

Job `/dlwh/shuttle-h100-contract-map-evidence-16cb0da-v5` reached the first
case worker for `contract_map_9836cdbed389db24`. The retained exception was:

```text
ValueError: numerical.outputs.forward.mean_ulp_distance exceeds the immutable ordinary_xla numerical floor
```

The task's temporary evidence directory was not exported. The H100 measured
value and the other output metrics are unavailable and cannot be reconstructed
from the controller log.

## Hypothesis 1: the first case conflicts with the reviewed ULP floor

The case is `(rows=43, reduction=104, features=72,
scalar_map=sigmoid_product)`. Its seed is the final eight hexadecimal digits of
the structural case ID, `0xd389db24`.

Ordinary JAX rounds the first contraction, scalar-map output, and final
contraction to BF16. JAX's VJP differentiates through those casts. The
ordinary-XLA floor compares these BF16 results with uninterrupted FP64 real
algebra. Absolute error converts both sides to FP32. ULP distance rounds the
FP64 reference to the actual output dtype, views the two BF16 arrays as ordered
16-bit values, and counts representable BF16 values between them. Mean ULP is
the arithmetic mean across every output element.

The floors were fixed before execution to prevent post-hoc acceptance changes.
Their canonical digest remains
`acbddb2a9c68a2ff7bb91bce6e7a4f354c3098311b222da5c514dd7a41e8f08a`.
The reviewed history explains immutability but contains no retained calibration
or derivation for the ordinary-XLA maximum/mean ULP limits of `4` and `0.25`.
This change does not alter those limits or their digest.

## CPU reproduction

A deterministic CPU-only reproduction used JAX 0.10.1, the exact case seed,
the ordinary-JAX training step, the FP64 reference, and the runner's existing
metric helpers. It produced these forward metrics:

- maximum absolute error: `0.001953989267349243` (limit `0.03125`)
- mean absolute error: `0.0003520350146573037` (limit `0.002`)
- maximum BF16 ULP distance: `29298` (limit `4`)
- mean BF16 ULP distance: `8.55948121645796` (limit `0.25`)
- nonfinite values: `0`
- three repeat identities: equal; all pairwise drift metrics: `0`

These are CPU measurements. They do not recover or estimate the missing H100
values. They show that the same metric contract rejects the first ordinary-JAX
case independently of the generated CUDA backends. The CPU ULP distribution
contained 2,167 exact values among 4,472 outputs, 290 values above four ULPs,
and one sign crossing near zero. That sign crossing had small absolute error
but a large ordered-BF16 distance.

## Changes to make

Parse and validate all scalar output and pairwise-repeat metrics before
reporting a floor violation. The bounded failure includes the reviewed case,
backend, measurement boundary, output role, reference, exact measured scalar,
immutable limit, absolute and ULP summaries, nonfinite count, repeat count,
repeat-identity equality, and maximum pairwise drift scalars. It never includes
tensor values, tensor representations, or repeat hashes.

The pre-timing runner identifies numerical validation as the logical training
step. Final evidence validation uses the record's declared boundary. Case,
backend, boundary, and output names remain closed reviewed values, and tests
cap the complete diagnostic at 1,024 characters.

## Results

The focused behavior tests fail against source commit `16cb0da` because the
pre-timing validator accepts no case or boundary and both validation paths emit
only the rejected field name. They pass after the diagnostic change. The
immutable floor constants and digest are unchanged.

The benchmark and runner behavior suites pass all 218 tests. The package suite
passes 951 tests with one historical snapshot test deselected because its
`SHA256SUMS` references an untracked `*.stdout.log` that is absent from the
exact Git worktree. Running that test alone fails before exercising this change
with `FileNotFoundError` for the missing snapshot member.

## Future work

- [ ] Decide with measured evidence whether uninterrupted FP64 algebra is the
  intended ordinary-XLA ULP reference or whether the fixed floors need a new,
  separately reviewed contract.
- [ ] Assert BF16 dtype, item size, and equal shapes before interpreting output
  storage as ordered 16-bit values.
- [ ] Split actual, reference, and difference nonfinite counts. The current
  aggregate can count an actual NaN both directly and through its difference.
- [ ] Add a CPU preflight for every ordinary-JAX case before reserving another
  H100, once the intended reference contract is reviewed.
