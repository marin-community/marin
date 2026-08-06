# Grug Variant Notes

Use this file for variant-specific guidance and examples.

## `ve` — value embeddings

A second token-indexed embedding table, shared across all layers, blended into each layer's
attention value path under a learnable per-layer gate:

```text
v = (1 - lambda_i) * v + lambda_i * value_embed[token_ids]
```

Queries and keys never see the table, so the attention pattern is the base model's exactly; only
the payload a token delivers when attended to carries the extra signal. The table is read once
per forward pass and handed to every block, so it costs `vocab_size x kv_width` parameters and
one embedding lookup — no added matmul FLOPs. Indexing by raw token id rather than by the hidden
state is the point: the signal never travels the residual stream, so it cannot be degraded by the
layers in between, which is why it is worth the most where the stream is furthest from token space.

The table is sized to the KV width (`num_kv_heads * head_dim`), not `hidden_dim`. Under grouped-
query attention those differ, and `v` carries KV heads.

`value_emb_lambda_init` is the only knob: a float turns the gate on at that init, `None` turns the
whole mechanism off and recovers the base model exactly. The ablation's control arm is therefore
the same code path with the knob set to `None`, which is what makes the two arms comparable.

Diagnostics to watch: `ve/lambda/layer_<i>` against depth is the model's own report of where a
token-identity side channel earns its keep, and `throughput/mfu` should be unchanged against the
control — if it moved, the lookup was not free.

The gates are exempt from weight decay: the launch optimizer decays via an explicit inclusion
list rather than levanter's default mask, which would decay them. A decayed gate is pulled toward
zero every step regardless of gradient, so "dialed away" and "nothing opposed the decay" would be
indistinguishable in the lambda readout.

Do not match parameter counts across the arms. The claim is capacity-per-FLOP, and equalizing
parameters erases the effect being measured; compare loss at equal FLOPs (equal tokens) instead.

If this is ever ported onto Muon, the table belongs in the optimizer's embedding partition, not
with the matrices: a lookup table's per-row gradients are sparse, and Muon's orthogonalized update
assumes the dense statistics of a matmul weight.
