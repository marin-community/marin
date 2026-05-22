# Debugging log for token-ragged Megablox GMM

Goal: diagnose the TPU Mosaic compile failure seen in the `token_ragged_a2a`
sanity path and keep the fix scoped to ragged GMM dispatch.

## Initial status

The token-rank `ragged_all_to_all` payload exchange had already passed a TPU toy
check with exact payload agreement. The full end-to-end sanity path still failed
before correctness comparison in the TPU Megablox GMM path.

## Hypothesis 1

The failure might be a raw JAX 0.10 Megablox `gmm` compile issue for bf16 inputs
with f32 accumulation.

## Results

Direct Megablox `gmm` calls compiled on TPU for bf16 and f32 inputs, for both
f32 and input-dtype outputs. A minimal `shard_map` wrapper around
`haliax.nn.ragged_dot(..., implementation="megablox")` also compiled for small
and production-like dimensions.

## Hypothesis 2

The failure is specific to the end-to-end autodiff trace for narrow sanity
dimensions.

## Results

With the benchmark's DeepEP layout call replaced by a pure-JAX equivalent, the
full `token_ragged_a2a` forward path compiled for hidden sizes 8, 128, and 512.
The forward+backward path failed only for the tiny hidden-size case. The failing
lowering used a Megablox/Pallas block shape that is invalid for TPU Mosaic when
one of the matmul dimensions is below the TPU block constraint.

## Changes to make

Add a pre-lowering guard in `haliax.nn.ragged_dot` so automatic TPU dispatch
uses XLA instead of Megablox for narrow shapes. Leave explicit Megablox requests
as early failures with a clear error.

## Results

After the guard, the reduced TPU forward+backward `token_ragged_a2a` sanity
case completed with the narrow GMM path falling back to XLA.
