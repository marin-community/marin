# Explicit Sharding in Haliax

JAX recently added [explicit sharding (aka "sharding in types")](https://jax.readthedocs.io/en/latest/notebooks/explicit_sharding.html)
to complement its existing implicit sharding model. In many ways, explicit sharding is a better fit for Haliax's design and goals,
and they both share the same core idea that arrays should carry around their sharding information.

Haliax unifies this core idea with named axes more broadly as a tool for writing reusable, better documented tensor code.

With explicit sharding, JAX array tracers carry shardings, and those shardings propagate at trace time. If you perform
an operation that would result in an ambiguous sharding, JAX will raise an error rather than let XLA guess.

This is a note to explain how Haliax and JAX's explicit sharding interact, specifically what is inferred automatically
and what remains explicit so you can predict behavior when working with meshes and axis mappings.
(Note: when they say "sharding in types", they mean that that tracer values carry sharding information as part of their type, not that Python types are modified.)

We won't fully recapitulate JAX's explicit sharding docs; please refer to them for background:

- [JAX explicit sharding overview](https://jax.readthedocs.io/en/latest/notebooks/explicit_sharding.html)


## Meshes and axis mappings

- Use `jax.set_mesh`/`hax.partitioning.set_mesh` with `AxisType.Explicit` to make sharding part of the type.
- Provide an `axis_mapping` (e.g. `{batch: "data", mlp: "model"}`) to map logical axes to mesh axes. Helpers like `pspec_for_axis` and `hax.shard` use this mapping to build `PartitionSpec`s.
  In general, it's a good idea to specify two axis mappings: one for the parameter sharding (typically something like `{"embed": "data", "mlp": "model"}`) and one for the compute sharding (typically something like `{"batch": "data", "mlp": "model"}`).

Note that you can of course still use `AxisType.Implicit` meshes (which are default) in which case everything behaves as before.

## What Haliax infers

- **Contractions (`dot` / `einsum`):** Haliax infers `out_sharding` from the result axes via `pspec_for_axis` and, if needed, from operand shardings to recover the mesh. This avoids JAX’s “ambiguous output sharding” errors when contracting sharded dimensions.
- **Type propagation:** Shardings on inputs flow through traced code; output sharding is constrained by the inferred `PartitionSpec`.

Really contractions are the main place where Haliax does something beyond JAX's default explicit sharding behavior.

## What remains explicit

- **Elementwise / broadcast ops:** If operand shardings conflict (e.g., two different mesh axes on broadcasted dims), Haliax does *not* auto‑reshard.
  JAX will raise a `ShardingTypeError` (duplicate mesh axis, ambiguous layout). You must decide whether to replicate or reshape the operands.
  If you set a compute/activation context mapping (or pass around mappings), something like `hax.auto_sharded(param)` can help pick a legal sharding.

## Practical tips

- Align contracting dimensions with your `axis_mapping`; contractions then lower without extra annotations.
- For elementwise ops that mix shardings (e.g., biases), explicitly reshard/replicate the smaller operand to a legal spec before the op.
- Inspect `jax.typeof(x).sharding` (explicit axes) and `x.sharding.spec` (concrete) to see layouts when debugging. Sometimes, you may need to use `jax.debug.inspect_array_sharding(x)` to get full details.

## Possible future helper

An opt‑in helper (e.g., `safe_broadcast`) could reshard elementwise operands when a unique, legal output sharding is implied by the mapping, and otherwise error.
For now, elementwise reshards stay user‑driven by design.

