# Levanter layers

Layers are Equinox modules. Parameter arrays are PyTree leaves. Use static
fields for shape and placement information needed during tracing. A static
`init` method builds the parameters and returns the module. Pass a PRNG key
for random initialization.

Named layers identify dimensions with Haliax axes. Array layers document their
input and weight axes in argument names, shape annotations, and docstrings.
Both `Linear` and `RmsNorm` preserve the leading dimensions of their input.

Array parameters can have different storage and compute layouts. `init`
places parameters with `storage_spec`. The module retains `compute_spec`
and applies it when using those parameters. Both specs follow the weight's
axis order. A bias follows the weight axis that it broadcasts over.

| Layer or operation | Where to look | Axes |
| --- | --- | --- |
| Linear projection | `levanter.layers.linear.Linear`, `haliax.nn.Linear` | Array weights are `[out, in]`. The input's last axis is `in` and the output's last axis is `out`. |
| RMS normalization | `levanter.layers.rms_norm.RmsNorm`, `levanter.layers.normalization` | The array layer normalizes the final input axis. Its optional scale and bias each have shape `[dim]`. |
| Attention | `levanter.layers.attention`, `levanter.grug.attention` | Attends from query positions to key positions. Array kernels use batch, sequence, heads, and head size. |
| Rotary embeddings | `levanter.layers.rotary`, `levanter.layers.rotary_native` | Rotates pairs within a head according to token position. |
| Gated DeltaNet | `levanter.layers.gated_deltanet` | Implements recurrent sequence mixing with gated state updates. |
| Repeated layers | [ArrayStacked](Array-Stacked.md), `haliax.nn.Stacked` | A leading parameter axis indexes layers. The stack applies them with a scan. |

See [Native model sharding](Native-Model-Sharding.md) for examples of storage
and compute specs on a model mesh.
