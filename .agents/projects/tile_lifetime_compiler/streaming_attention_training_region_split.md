# Streaming-attention training region split

## Status

The static compiler boundary is implemented. It is not yet a performance
result.

An ordinary JAX attention differentiated with `jax.vjp` is recovered as generic
QK and PV `Contract`s, score `Map`, `DomainRestriction`, and normalized-
exponential `Fold` state. The generated train-step ownership boundary contains
two typed-FFI calls:

```text
early forward readiness
    Q, K, V
      -> generated streaming Contract/Fold forward
      -> BF16 O, FP32 log_sum_exp

ordinary JAX train-step dataflow
      O -> downstream forward computation -> dO

later reverse readiness
    Q, K, V, O, log_sum_exp, dO
      -> generated streaming Contract/Fold reverse
      -> dQ, dK, dV
```

JAX owns automatic differentiation and the output-cotangent dependency. Shuttle
does not create an AD rule or a call that spans from the forward inputs to the
eventual output cotangent.

## Natural Grug boundary

The preserved post-SPMD Grug HLO has two materialized attention forwards:

- the early train-step forward at `dot.16 -> reduce_max.56 -> reduce_sum.634 ->
  dot.17 -> transpose.47`;
- the checkpoint rematerialization at `dot.31 -> reduce_max.64 ->
  reduce_sum.698 -> dot.32 -> transpose.54`.

The paired plan replaces the early region with `Q/K/V -> O/LSE`, rewires both
the normal forward consumers and the checkpoint-rematerialized output consumers
to the generated `O`, and replaces the later reverse with the saved-state ABI.
The old early forward, rematerialized forward, and reverse closures are all
root-dead after the rewrite. Eleven placement collectives remain outside the
two owned regions.

The value input's singleton KV-head axis and the output's physical
`{1,3,2,0}` layout are handled by generic reshape/copy boundary adapters. The
saved state has the explicit shape `f32[batch, query_heads, query_length]` and a
single producer-consumer link from the early call to the later call.

## Numerical and state policy

- Q, K, V, O, dO, dQ, dK, and dV cross the generated boundary as BF16.
- Online maximum, exponential sum, weighted-value accumulation, output-dot,
  and log-normalizer state are FP32.
- The saved coordinate is `log_sum_exp = row_max + log(row_sum_exp)`.
- The physical family implements the recovered score scale and less-equal
  domain restriction. Unsupported or ambiguous physical graphs fail closed.
- The current rewrite declares BF16 Contract boundaries with FP32 online Fold
  state under the existing `allow_rounding_reorder` policy.

A scale mutation from `0.5` to `0.375` keeps the same typed-FFI ABI and AOT
kernel family while changing the generated semantic fingerprint and kernel
specialization. No workload or model name participates in recovery or
generation.

## Distinction from standalone combined parity

The existing standalone forward-plus-backward measurement executes a single
combined generated boundary. It is useful for measuring the physical kernel
family, but it is not a legal ownership proof for a full train step because the
call receives `dO` at the same boundary as Q/K/V and may recompute the forward
internally.

This split is the legal train-step ownership proof: the forward runs when Q/K/V
are ready, publishes O/LSE, and the reverse runs only after JAX's intervening
program produces dO. No latency is reported for the split yet, so standalone
combined parity must not be relabeled as split-region parity.

## Static evidence

Focused command:

```bash
uv run --frozen --package marin-tile-lifetime --group test pytest -q \
  lib/tile_lifetime/tests/test_streaming_attention_backward.py \
  lib/tile_lifetime/tests/test_stablehlo_streaming_attention_backward.py \
  lib/tile_lifetime/tests/test_jax_streaming_attention_backward_ffi.py \
  lib/tile_lifetime/tests/test_xla_streaming_attention_backward_ffi.py \
  lib/tile_lifetime/tests/test_jax_streaming_attention_forward_ffi.py \
  lib/tile_lifetime/tests/test_xla_streaming_attention_training_regions.py
```

Result: 62 tests passed. The scoped pre-commit checks, including Pyrefly, also
pass after staging the checkpoint files.

## Remaining gates

1. Compile and load the new forward-only typed-FFI DSO in the matched CUDA/JAX
   environment.
2. Execute the rewritten natural Grug HLO and verify O/LSE/dQ/dK/dV numerics,
   deterministic hashes, handler counts, and final optimized-HLO liveness.
3. Measure the split forward and reverse independently and as part of the full
   train step on H100 and GB200.
4. Compare the summed owned boundary against a matched FlashAttention training
   oracle. Keep saved-state traffic and the eliminated checkpoint
   rematerialization inside the accounting.

No GPU was allocated for this static checkpoint.
