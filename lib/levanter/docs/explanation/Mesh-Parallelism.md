# Mesh Parallelism

Levanter uses [Haliax](https://github.com/stanford-crfm/haliax) *named axes* and JAX *named device meshes*.
Sharding is controlled by mapping **logical axes** (names like `batch`, `embed`, `heads`) onto **physical mesh axes**
(names like `data`, `model`, `replica`, `replica_dcn`).

Most runs never need custom sharding: the defaults are designed to “just work” for common LLM training setups.

!!! tip "The 2 rules that keep you out of trouble"
    For basic use cases, you typically don’t need to think about mesh parallelism as long as you follow these rules:

    1. Name your model’s main hidden dimension `embed`.
    2. Name your batch axis `batch`.
    3. If using model parallelism, name your MLP and attention head axes `mlp` and `heads`, respectively.

    These names are what the default mappings key off of (see `param_mapping` and `batch_axis_name` below).

## The Big Picture

There are three layers to keep straight:

1. **Logical axes (Haliax):** axis names on arrays and parameters, e.g. `batch`, `position`, `embed`, `heads`, `mlp`.
2. **Physical mesh axes (JAX):** axis names on the device mesh, e.g. `data`, `model`, `replica`, `replica_dcn`.
3. **Axis mappings:** a `ResourceMapping` from logical axis name → physical mesh axis name(s).

Levanter’s `trainer.mesh` is a [`levanter.utils.mesh.MeshConfig`][] that defines:

- The *shape* of the mesh (`axes` and `dcn_axes`).
- The *mappings* used for compute vs parameters (`compute_mapping` vs `param_mapping`).

All of this is implemented in `lib/levanter/src/levanter/utils/mesh.py` (recommended starting point when debugging).

## Default Mesh: `(replica_dcn, replica, data, model)`

By default, Levanter builds a mesh with these axis names:

- `replica_dcn`: **cross-slice replication / data parallel** (DCN). This only matters on multi-slice hardware (e.g. TPU pods or multinode GPU clusters.)
- `replica`: **within-slice replication / data parallel** (ICI). Use this when you want to have multiple groups within an ici slice. Think of it
  like “mini-slices” within a slice.
- `data`: **within-slice data-parallel / FSDP axis** (ICI). This is the “absorber” by default: it expands to use the
  remaining devices in the slice.
- `model`: **tensor/model parallel** (ICI). Use this when you want to shard “wide” compute (e.g. heads/MLP) across devices.

Concretely, the default axis-size specs (see `levanter.utils.mesh`, `DEFAULT_ICI_AXIS_SPEC`, and `DEFAULT_DCN_AXIS_SPEC`)
are:

- `axes` (ICI): `{"data": -1, "replica": 1, "model": 1}`
- `dcn_axes` (DCN): `{"replica_dcn": -1}`

`-1` means “absorb what’s left” (and the defaults are merged, so setting one value doesn’t clear the others).

### Why split `replica*` vs `data`?

Levanter’s defaults treat **batch parallelism** and **parameter sharding** as related-but-separate concerns:

- Batch parallelism can span multiple axes: `replica_dcn × replica × data`.
- FSDP-style parameter/optimizer sharding defaults to **only** the `data` axis.

That lets you scale *global batch size* by increasing replicas without necessarily increasing the number of FSDP shards,
which is often helpful when you’re trading memory, comms, and per-device compute.

## Compute Mapping vs Parameter Mapping

Levanter uses two mappings because the “best” sharding for *stored parameters* is not always the best sharding for
*the arrays you actually compute with during forward/backward*.

If you know about FSDP from PyTorch, the "param mapping" describes the FSDP sharding (we use ZeRO-3 style sharding by default),
while the "compute mapping" describes how the model looks when it’s been “de-FSDP’d” for compute.

### `compute_mapping`: forward/backward (“for compute”)

`compute_mapping` controls how activations (and the *effective* model) are sharded inside the training/eval step.
In practice, you can think of it as the mapping for the model **after it has been “de-FSDP’d” for compute**:

- Parameters may be **stored** sharded (FSDP-style).
- For compute, those parameters are typically **materialized / gathered** as needed so the compute sees a different
  sharding than storage.

In Levanter entry points, this is typically the mapping you pass as `axis_resources` to `haliax.named_jit(...)` for the
forward/backward step, and the mapping used in `hax.axis_mapping(...)` contexts for inference/evaluation.

In [`levanter.utils.mesh.MeshConfig`][] this is exposed as `mesh.resolved_compute_mapping`, which merges:

- Default `shared_mapping` entries (notably `mlp` and `heads`) → `model`
- `shared_mapping` overrides (applies to both compute and params)
- The batch mapping for `batch_axis_name` (defaults to `batch`) → `("replica_dcn", "replica", "data")`
- Any explicit `compute_mapping` overrides

### `param_mapping`: parameters + optimizer state (“for storage”)

`param_mapping` controls how **parameters** are sharded *at rest* and therefore how **optimizer state** is partitioned.
If you’re doing Adam-like optimizers, this is usually the biggest memory consumer after activations.

In Levanter entry points, this is the mapping used when sharding a model for training (and it’s the mapping that governs
checkpoint layout and optimizer-state partitioning).

In [`levanter.utils.mesh.MeshConfig`][] this is exposed as `mesh.resolved_param_mapping`, which merges:

- Default `shared_mapping` entries and `shared_mapping` overrides (so “tensor-parallel” axes shard the same way for params
  and compute)
- Explicit `param_mapping` overrides (defaults include `{"embed": "data"}`)

### The default behavior in one sentence

With the defaults:

- `batch` is partitioned across `replica_dcn × replica × data` for compute.
- `embed` is sharded across `data` **for parameters/optimizer state**, but (unless you override it) *not* sharded in
  `compute_mapping`, so compute effectively operates on the “de-FSDP’d” view.

## Common Config Patterns

### Default (recommended starting point)

You can usually omit `mesh` entirely. If you want to be explicit:

```yaml
trainer:
  mesh:
    axes: {data: -1, replica: 1, model: 1}
    dcn_axes: {replica_dcn: -1}
    compute_mapping:
      batch: [replica_dcn, replica, data]
    param_mapping:
      embed: data
```

### Add tensor parallelism (use the `model` axis)

Increase `model` and let `data: -1` shrink accordingly:

```yaml
trainer:
  mesh:
    axes: {model: 2}  # keeps replica:1 and data:-1 from defaults
```

By default, `shared_mapping` treats `mlp` and `heads` as tensor-parallel logical axes and maps them to `model` for both
compute and parameters. If your model uses different axis names (e.g. `kv_head` / `kv_heads`), add them to
`shared_mapping`.

### Add context (sequence) parallelism

Context parallelism means sharding the *query positions you compute* across a mesh axis, typically called `context`.

To do that in Levanter:

1. Add a `context` axis to the mesh.
2. Map the logical `position` axis to the physical `context` axis.
3. Leave `key_position` **unmapped** in the simple case, because attention typically needs keys/values to be present for
   all query blocks (so you all-gather keys/values across `context`).

Example:

```yaml
trainer:
  mesh:
    axes: {data: -1, replica: 1, context: 4, model: 1}  # better to be explicit when adding a new axis
    shared_mapping:
      position: context
```

Why `key_position` stays unsharded: Levanter’s LM configs define `KeyPos` as `max_Pos.alias("key_position")`, so you’ll
often see attention tensors that have both `position` (queries) and `key_position` (keys/values). Sharding only
`position` is the “all-gather K/V” variant; more advanced context-parallel attention patterns typically require an
explicit communication strategy and are beyond this doc.

## Troubleshooting checklist

- If you changed axis names in a model, ensure they match the mapping keys (especially `batch` and `embed`).
- If you add a new mesh axis, remember only one ICI axis and one DCN axis may use `-1`.
- If you get shape errors, check `trainer.compute_axis_mapping` vs `trainer.parameter_axis_mapping` and confirm which
  one your code path is using (compute step vs parameter loading/optimizer).
