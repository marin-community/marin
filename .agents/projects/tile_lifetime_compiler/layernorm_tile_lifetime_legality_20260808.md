# LayerNorm Tile-Lifetime Legality

Date: 2026-08-08

## Conclusion

The normalization-materialization trick generalizes from RMSNorm to LayerNorm,
but the delayed form is not the same one-scalar rewrite.

For one row `u`, feature vectors `gamma` and `beta`, row mean `mu`, inverse
standard deviation `r`, and a right-hand weight matrix `W`:

```text
y = ((u - mu) * r) * gamma + beta
z = y @ W
```

There are two legal tile-lifetime placements under a numerical policy that
permits floating-point reassociation.

## Consumer preparation

The producer stores the unnormalized activation and emits partial first and
second moments. A small row reduction computes `mu` and `r`. The next Contract
prepares each register-resident input fragment as:

```text
u_fragment
    -> subtract mu
    -> multiply r
    -> multiply gamma
    -> add beta
    -> convert to the BF16 matrix operand
    -> matrix mainloop
```

This eliminates the materialized normalized activation. It requires two row
statistics rather than RMSNorm's one inverse-RMS scalar, plus feature-vector
loads for both `gamma` and `beta`.

## Delayed output placement

Right contraction distributes to give the real-number identity:

```text
y @ W
  = r * ((u * gamma) @ W)
    - (mu * r) * (gamma @ W)
    + beta @ W
```

Therefore the producer can materialize `u * gamma`, the consumer can contract
it first, and the finalization can apply two row-scaled column corrections.

Unlike RMSNorm, this needs two parameter-dependent vectors:

```text
gamma_w = gamma @ W
beta_w  = beta @ W
```

They are only reusable while `gamma`, `beta`, and `W` remain unchanged. This is
plausible for inference, but may be unattractive during training or when cache
invalidation dominates. It must be a costed candidate, not an unconditional
rewrite.

## Numerical contract

The current prototype emits partial `sum(u)` and `sum(u*u)`, then uses:

```text
mu       = sum(u) / K
variance = sum(u*u) / K - mu*mu
r        = rsqrt(variance + epsilon)
```

This is algebraically correct but does not reproduce a two-pass source
computation of `mean((u - mean(u))^2)` in floating point. It can suffer severe
cancellation for shifted rows. The regression test contains a row for which
the FP32 moment formula returns zero variance while the two-pass formula returns
`0.125`.

Consequently:

- `BITWISE_EXACT` rejects both synthesized placements and materializes;
- the current generic placements require `ALLOW_ROUNDING_REORDER`;
- a genuinely source-ordered statistics policy requires a matching two-pass or
  Welford `Fold`, which is not implemented yet.

The consumer-preparation placement preserves the normalization-to-BF16 operand
boundary after row statistics are available. It does not by itself make the
statistics source ordered.

## Implemented proof

The backend-neutral prototype now includes:

- a LayerNorm semantic node and graph builder;
- erasure into generic `Map`/`Fold` primitives;
- generic recognition of centered affine normalization before a right
  `Contract`;
- producer `partial_sum` and `partial_sum_square` emissions;
- consumer-preparation and delayed-output candidates;
- generated subtract, row-scale, feature-scale, bias, and correction
  attachments;
- materialized fallback under the bitwise policy;
- numerical equivalence and cancellation tests.

The focused compiler tests pass. This establishes the algebra and planning
legality.

The training path now also accepts an ordinary centered JAX normalization,
lets JAX form its VJP, recovers the resulting StableHLO as generic row Folds,
and generates a four-stage typed-FFI pipeline:

```text
row-mean Fold
→ centered-second-moment Fold
→ input-cotangent Fold + Map
→ feature-scale-cotangent Fold
```

The exact HLO replacement is shared with the uncentered RMS-style path. It
contains no LayerNorm dispatch key, and the generated source contains only the
recovered scalar expressions and generic axis-Fold stages. CPU/reference
execution matches the natural JAX VJP within the declared BF16 reassociation
tolerance.

This is an executable-generation proof, not a GPU performance result. The
centered pipeline is currently four separate GPU stages, and the generic
feature-axis Fold remains slower than XLA in the measured uncentered case.
Remaining work is a Welford/two-pass Fold for a source-ordered statistics
policy, a faster generic row-Fold lowering, H100/GB200 measurement of the
centered path, and cache invalidation for the delayed `gamma @ W` and `beta @
W` inference candidate.
