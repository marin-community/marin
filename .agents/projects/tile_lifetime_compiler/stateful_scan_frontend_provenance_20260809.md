# StatefulScan frontend provenance audit

## Verdict

The generic affine recovery and recurrent/chunkwise physical generators do not
dispatch on Gated DeltaNet, Kimi Delta Attention, or another model name. The
repository nevertheless contained two different frontend histories that had
been described too uniformly:

- The early `stateful_scan_generated_h100` recurrent-core checkpoint recovered
  its `TensorExpression` from `delta_rule_update_expression`. Its kernel timing
  remains valid backend evidence, but that artifact is not a clean end-to-end
  natural-frontend proof.
- The later matched `stateful_scan_affine_pipeline_h100` checkpoint did start
  its primary path from ordinary JAX `lax.scan`, exported a structured
  `stablehlo.while`, recovered generic tensor expressions, and then selected
  the generic affine pipeline. Its 1.1354x result is clean at that recorded
  boundary. The subsequent `stateful_scan_affine_pipeline_h100_v0` paired
  capture reported 0.465824 ms versus 0.424304 ms, or 1.097854x, through the
  same natural primary path.

The remaining provenance defect was real but narrower than invalidating the
matched result: the public `compile_gated_delta_scan` and
`compile_kimi_delta_scan` helpers assembled `StatefulScan` and chunk equations
by hand, while three older generated-kernel harnesses and the pipeline mutation
matrix recovered from the reference expression fixture. Those paths could
make a clean backend look like a clean frontend.

## Corrected accepted path

The public compilation path is now:

```text
ordinary JAX recurrence
  -> jax.lax.scan
  -> jax.export StableHLO with one structured while
  -> generic TensorExpression graph
  -> recover_affine_state_update
  -> generic StatefulScan
  -> recurrent and factored-affine chunk candidates
```

The compiler records an artifact SHA-256, verified source kind, and structured
while count. Candidate validation rejects reference-expression provenance. The
accepted H100 harnesses recover through this path and record the source kind
and artifact digest rather than importing `delta_rule_update_expression`.

`delta_rule_update_expression` remains useful for isolated recovery unit tests.
Its module and API now state that it is reference-only. The old hand-authored
named plans remain callable only through explicitly named
`build_*_reference_plan` helpers and are not exported as accepted compilation
entrypoints.

## Mutation evidence

Three changes now originate in the natural JAX recurrence and pass through the
same importer, affine recovery, and two backend families:

1. scalar diagonal decay with a rank-one update;
2. per-key diagonal decay with a rank-two update;
3. scalar `exp(log_decay) * exp(log_decay)` diagonal math with a rank-two
   update.

The three exported artifacts have distinct SHA-256 digests. The altered
diagonal adds an operation to the StableHLO step body, while all three compile
to the same generic recurrent and factored-affine backend families. Scheduling
keys contain only generic scan extent, state rank, primitive shape, affine
transition structure, and numerical policy.

## Acceptance accounting

The old 0.138544 ms recurrent-core number should be cited as physical-generator
evidence, not as natural-frontend acceptance. The later matched affine-pipeline
result can remain accepted because its preserved primary source and serialized
StableHLO establish the natural path. A future GPU replay should use the
current harness so the mutation matrix and public entrypoint share that same
provenance; this change performs no GPU measurement and does not replace the
recorded latency.

The frontend currently uses JAX for differentiation/export ownership and only
recovers the already differentiated or explicit scan program. No Torch object
participates in compiler semantic recovery. Torch remains in standalone GPU
benchmark execution code, not in the accepted frontend contract.
