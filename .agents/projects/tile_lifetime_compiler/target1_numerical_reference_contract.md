# Target 1 numerical reference contract

Status: local reference contract; no scorecard promotion.

The Target 1 BF16 rowwise program now has an independent NumPy closed-form
reference for the `2048x4096` and `7x13` shapes. It covers forward, JAX-owned
VJP backward, and composed forward/backward boundaries. This reference is not
an expert performance oracle and contains no hardware or timing evidence.

The public inputs are deterministic BF16 arrays constructed by NumPy
`linspace` from fixed float32 endpoints. Their dtype, shape, byte digest, and
role are pinned in
`lib/shuttle/mlir/jax_patch/target1-rowwise-bf16-numerical-oracle-v1.json`.
The reference converts those BF16 inputs to binary64, evaluates the forward
formula and its analytic VJP directly, then rounds each public output to BF16.
It does not call the ordinary-JAX program, Shuttle, or an oracle library.

For row `i` and feature `j`, the reference is:

```text
r_i       = 1 / sqrt(sum_j(x_ij²) / features + 1e-5)
y_ij      = BF16(x_ij * r_i * gamma_j)
row_vjp_i = sum_j(dy_ij * x_ij * gamma_j)
dx_ij     = BF16(dy_ij * gamma_j * r_i
                 - x_ij * r_i³ * row_vjp_i / features)
dgamma_j  = BF16(sum_i(dy_ij * x_ij * r_i))
```

Every multiplication, reduction, division, and square root above operates in
binary64. The final cast is the only reference BF16 rounding boundary. This is
the analytic VJP of the source program: JAX promotes BF16 `x`, `gamma`, and
`dy` cotangents to the float32 primal computation, differentiates the final
BF16 cast by converting its cotangent to float32, and casts the resulting
float32 input cotangents back to the BF16 input dtypes. The reference replaces
those internal float32 calculations and reductions with binary64 while
retaining the same public BF16 input and output boundaries.

The local analytic comparison records maximum absolute error, mean absolute
error, relative L-infinity error, and maximum BF16 ULP distance. Relative
L-infinity error divides maximum absolute error by the larger of the
reference L-infinity norm and `2^-7`, one BF16 unit at magnitude one. The
predeclared limits are `2^-6` maximum absolute error, `1e-6` mean absolute
error, `2^-7` relative L-infinity error, and eight BF16 ULPs. The absolute,
relative, and ULP bounds are tied to BF16 resolution. The `1e-6` mean bound is
deliberately shape-and-input-specific: it is a frozen local analytic gate above
the observed maximum mean error of `1.963966822504659e-8`, not a general BF16
floor. The contract records every local ordinary-JAX output digest and all four
observed metrics on the CPU backend with JAX X64 disabled, Python 3.12.11, and
an arm64 host. This local device-class record is explicitly not H100,
GB200/B200, or scorecard evidence. These bounds cover the pinned ordinary-JAX
deviations from the binary64 analytic formula without claiming equivalence
between their reduction orders. Mutation tests pin each limit independently.

The ordinary-JAX bitwise gate has a narrower meaning. The current Shuttle
lowering is identity-shaped, so both `SOURCE_ORDERED` and `FAST` must match the
disabled ordinary-JAX result bitwise. That proves the compiler round trip does
not change this source program; it does not prove ordinary JAX is bitwise equal
to the higher-precision analytic reference. It is not: at `2048x4096`, the
ordinary-JAX result differs from the rounded analytic result by up to one BF16
ULP for `y` and seven BF16 ULPs for `dx`.

No tolerance for a non-identity `FAST` rewrite is declared. Such a rewrite must
revise this contract before execution, including an independently reviewed
error bound and the evaluation rule comparing Shuttle error to the matched
expert oracle or dtype-resolution floor. The scorecard therefore retains
`oracle_not_pinned`, the pending representative-shape coordinate, and all
hardware/performance blockers. A local analytic reference alone cannot promote
an H100 or GB200/B200 cell.
