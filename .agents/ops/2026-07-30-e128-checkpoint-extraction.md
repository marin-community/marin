# Grug E128 extraction: sharded checkpoint restore

Validate physical E128 breakout from a nested E256 checkpoint before launching
the production cooldown.

## Initial status

The E256 weights-only restore and balanced-complements training smokes ran on
one eight-H100 node. The first E128 extraction smoke loaded the source
checkpoint but failed before update 0.

## Hypothesis 1: explicit sharding rejects ordinary expert slices

The first failure was a `ShardingTypeError` on an expert tensor shaped
`[8, 256@expert, 768@data, 384@model]`. NumPy-style slicing could not infer the
target PartitionSpec without a collective.

## Changes to make

Slice each router, QB, and expert tensor with `at[...].get(out_sharding=...)`
using the corresponding physical E128 exemplar tensor's `NamedSharding`.

## Results

The second smoke passed source deserialization and sharded extraction, then
reached the first compiled training call. JAX rejected the model because
dimension-identical attention modules retained the E256 source config as static
pytree metadata.

## Hypothesis 2: extracted arrays need the target static pytree

Rebuild the extracted model on the initialized E128 exemplar tree. Copy every
learned non-expert array from the source, copy sliced MoE arrays, and retain the
target modules' E128 static config. The regression test now requires exact
pytree-structure equality with the target exemplar.

## Results

Repository lint, formatting, Pyrefly, and focused extraction tests pass at
source commit `6a2e0900eb`. The third sharded smoke completed two finite
optimizer updates and wrote its local checkpoint. A live thread capture during
its long startup showed the main thread in TensorStore `deserialize`, not a
collective or compilation hang.

The production E128 cooldown subsequently restored the same checkpoint,
physically extracted experts 0--127 into an E128 target tree, and began
training at approximately 634,000 tokens per second. No production retry was
required.

## Future work

- [ ] Avoid loading unused source optimizer leaves; the current weights-only
      restore reads only params and QB state but still takes several minutes.
