# Debugging log for H100 Contract/Map v7 source-ordered parity

Determine why the first generated SOURCE_ORDERED reverse result failed its
pre-timing numerical gate, preserve the rejected run, and repair only a proven
semantic mismatch.

## Initial status

The single reviewed H100 v7 job used source
`3deae6618d52b3cfe1bf993b40f8e5192928be20` and immutable image digest
`sha256:d011ebdfc00d5b3a423d872b86388126636e4755cc5eb722c078f6c6d2ce598a`.
It passed capsule authentication, import audit, tool preflight, generated
compilation, loaded-shared-library topology validation, and ordinary-XLA
numerics. The first SOURCE_ORDERED `dx` check then reported maximum ULP
distance `29608` against limit `1`, maximum absolute error `0.00390625`, mean
absolute error `0.0003416987310629338`, and three bitwise-identical repeats.
Timing and profiler evidence did not begin. The job had one failed attempt,
zero preemptions, zero failure retries, and no relaunch.

The exact terminal evidence is sealed under
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_seventh_launch_failure_3deae6_v0`.

## Hypothesis 1: BF16 ULP ordering is wrong near zero

The largest distance crosses zero: generated `dx[19,37]` is BF16 `0xb9bd`
(`-0.0003604888916015625`) while the reference is BF16 `0x39eb`
(`0.0004482269287109375`). Their absolute difference is
`0.0008087158203125`. The signed BF16 ordering makes this a large raw distance,
but the output also has 2,505 nonexact elements and 980 elements beyond one
ULP. A near-zero policy effect alone does not explain the broad mismatch, so
the immutable gate remains unchanged.

## Hypothesis 2: host scalar math differs from CUDA scalar math

The host scalar evaluator uses binary64 Python operations and `math.exp` or
`math.tanh`; generated CUDA uses explicit round-to-nearest FP32 add,
subtract, and multiply plus `expf` or `tanhf`. Re-evaluating every reviewed
case with explicit FP32 AST operations produced the same final BF16 values as
the current host evaluator for forward, `dx`, `dw0`, and `dw1`. The failed
first case uses sigmoid rather than tanh. The cubic case has no transcendental
and exhibits the same reverse-only mismatch. Scalar math is therefore not the
cause of this failure.

## Hypothesis 3: reverse fusion removes an authoritative BF16 boundary

The mechanically differentiated program gives the hidden-adjoint Contract a
BF16 result. The materialized CPU reference rounds the Contract accumulator to
BF16 before the pointwise VJP. Generated CUDA fuses that Contract with the Map
but passes the raw FP32 accumulator directly to `generated_phi_vjp`, skipping
the typed result boundary.

Deterministic CPU emulation used the exact reviewed seeds, BF16 inputs,
literal increasing-order FP32 reductions, and the authoritative differentiated
operation plan. Changing only the hidden-adjoint round reproduces the H100
failure exactly for the first `dx`: maximum ULP `29608`, mean ULP
`28.51408765652952`, maximum absolute error `0.00390625`, and mean absolute
error `0.0003416987310629338`.

| Case | Map | Output | Changed BF16 values | ULP > 1 | Maximum ULP | Maximum absolute error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `9836cdbed389db24` | sigmoid product | forward | 0 | 0 | 0 | 0 |
| `9836cdbed389db24` | sigmoid product | dx | 2,505 | 980 | 29,608 | 0.00390625 |
| `9836cdbed389db24` | sigmoid product | dw0 | 4,076 | 1,615 | 29,498 | 0.001953125 |
| `9836cdbed389db24` | sigmoid product | dw1 | 0 | 0 | 0 | 0 |
| `79045ff9bdc7c783` | tanh product | forward | 0 | 0 | 0 | 0 |
| `79045ff9bdc7c783` | tanh product | dx | 11,687 | 4,355 | 29,959 | 0.00390625 |
| `79045ff9bdc7c783` | tanh product | dw0 | 9,485 | 3,565 | 29,937 | 0.00390625 |
| `79045ff9bdc7c783` | tanh product | dw1 | 0 | 0 | 0 | 0 |
| `b4c693e52135022a` | cubic mix | forward | 0 | 0 | 0 | 0 |
| `b4c693e52135022a` | cubic mix | dx | 34,218 | 13,289 | 30,414 | 0.03125 |
| `b4c693e52135022a` | cubic mix | dw0 | 17,437 | 6,714 | 30,427 | 0.03125 |
| `b4c693e52135022a` | cubic mix | dw1 | 0 | 0 | 0 | 0 |
| `eb4a28b4408cfb90` | sigmoid product | forward | 0 | 0 | 0 | 0 |
| `eb4a28b4408cfb90` | sigmoid product | dx | 94,136 | 36,287 | 30,164 | 0.0078125 |
| `eb4a28b4408cfb90` | sigmoid product | dw0 | 33,226 | 12,923 | 30,221 | 0.015625 |
| `eb4a28b4408cfb90` | sigmoid product | dw1 | 0 | 0 | 0 | 0 |

Forward and `dw1` do not consume the faulty intermediate. Both `dx` and
`dw0` do, matching the observed topology. The reference and numerical policy
are not changed.

## Changes to make

Preserve fusion but round the hidden-adjoint FP32 accumulator to BF16 and
convert it back to FP32 before evaluating the pointwise VJP. Apply this typed
boundary to both generated policies. Add generator tests that reject direct
use of the raw accumulator.

When a future numerical floor fails, append only bounded scalar evidence for
the maximum-ULP pair: logical index, canonical BF16 hex and value, sign,
unbiased exponent, class, absolute and ULP error, finite value count, and
mismatch counts for exact, one-ULP, and the immutable per-output absolute
threshold. Retain the existing 1024-character base diagnostic and cap the
combined diagnostic at 2048 characters. Do not serialize arrays or repeat
content hashes.

## Results

The generated reverse now restores the BF16 boundary locally before the fused
Map. Focused Contract/Map generator, runner, and evidence-contract tests pass:
272 tests in 2.70 seconds, including exact regeneration of the checked CPU
calibration fixture. The tests cover both numerical policies, reject the old
raw-accumulator spelling, classify positive and negative zero, subnormal,
normal, infinity, and NaN BF16 scalars, and exercise the full runner failure
path for the exact near-zero sign-crossing diagnostic.

The wider tile-lifetime suite passes 977 tests in 74.81 seconds when excluding
`test_benchmark_snapshot.py`. Five tests in that excluded module pass; its one
remaining test cannot read an ignored historical raw `.stdout.log` fixture
that is absent from this isolated source-capsule worktree. The missing fixture
is unrelated to this change.

No image was built, no GPU was queried, and no job was launched or retried for
this repair.

## Future work

- [ ] Run the reviewed single H100 evidence job only after source review and
  canonical integration.
- [ ] If a residual mismatch remains, compare the exact BF16 `(z,
  hidden_adjoint)` scalar pairs in an isolated CUDA diagnostic before changing
  any numerical policy.
