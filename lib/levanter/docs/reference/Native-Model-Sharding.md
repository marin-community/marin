# Native model sharding

Native Llama uses JAX arrays on a mesh with explicit axes. Its placement
follows the configured batch axis, the `embed` parameter mapping for FSDP, and
the `mlp` compute mapping for tensor parallelism. The tensor-parallel axis is
distinct from the batch and FSDP axes. The FSDP axis can also shard the batch.

| Tensor | Axis order | Storage sharding | Compute sharding |
| --- | --- | --- | --- |
| Column projection | `[out, in]` | TP on `out`, FSDP on `in` | TP on `out` |
| Row projection | `[out, in]` | FSDP on `out`, TP on `in` | TP on `in` |
| Residual normalization weight | `[hidden]` | FSDP on `hidden` | Replicated |
| Token embedding | `[vocab, hidden]` | TP on `vocab`, FSDP on `hidden` | Gather directly from storage into the residual layout |

The residual stream uses `[batch, sequence, hidden]` with only the batch axis
sharded. Attention heads and intermediate MLP channels are tensor-parallel.
Logits use `[batch, sequence, vocab]` with batch and vocabulary sharding.
Sequence and layer axes are replicated.

Column and row projections gather the FSDP dimension when they use the
weight. The token lookup instead gathers from storage. This lets the
transpose accumulate repeated token IDs without first replicating the
embedding across FSDP. See [Levanter layers](Layers.md) for the module conventions.
