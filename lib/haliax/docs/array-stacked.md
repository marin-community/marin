# ArrayStacked

[haliax.nn.ArrayStacked][] is an array-native stack container for scan-over-layers when you are not using named axes in
the stack itself.

If you are using named axes (`Axis`, `NamedArray`) in your stack dimension, see [Scan and Fold](scan.md#stacked) and
[haliax.nn.Stacked][].

## When to use `ArrayStacked`

Use `ArrayStacked` when:

* You want scan-over-layers behavior (`fold`, `scan`, `fold_via`, `scan_via`) with plain `jax.Array` leaves.
* You want layer parameters stored as arrays with a leading layer dimension.
* You do not want to thread a named `Axis` for the layer stack.
* You want to avoid unrolled Python layer loops that can increase compile time and HBM pressure.

Compared to [haliax.nn.Stacked][]:

* `Stacked` uses a named block axis (`Block: Axis`) and can stack `NamedArray` leaves on that axis.
* `ArrayStacked` uses `num_layers: int` and stacks array leaves on positional axis `0`.

## Compile Time and Memory Notes

`ArrayStacked` keeps the scan-over-layers structure explicit (via `lax.scan`) instead of relying on Python-unrolled layer
loops. In practice, this often:

* Reduces compile time.
* Reduces compile-time/live-range memory pressure (which can help avoid compile OOMs).
* Sometimes improves step-time throughput when the unrolled alternative also compiles.

As always, validate with your model, mesh, and batch/sequence settings.

## Initialization

`ArrayStacked.init(num_layers, Module)` is curried, like `Stacked.init(...)`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax


class Block(eqx.Module):
    weight: jax.Array

    @staticmethod
    def init(weight):
        return Block(weight=weight)

    def __call__(self, carry, layer_scale, mask):
        return carry + self.weight * layer_scale + jnp.sum(mask)


num_layers = 4
width = 8

stack = hax.nn.ArrayStacked.init(num_layers, Block)(
    weight=jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
)
```

The init arguments are sliced per-layer on leading axis `0` when they match `num_layers`.

## Fold / Scan Usage

### `fold` / `fold_via`

`fold` is the carry-only pattern:

```python
carry0 = jnp.zeros((width,))
layer_scale = jnp.array([1.0, 2.0, 3.0, 4.0])
mask = jnp.ones((16, 16))

# in_axes follows vmap-like semantics for non-carry arguments.
# layer_scale is scanned on axis 0; mask is shared across layers.
carry = stack.fold_via(Block.__call__, in_axes=(0, None))(carry0, layer_scale, mask)
```

### `scan` / `scan_via`

`scan` expects block functions returning `(carry, output)`:

```python
def block_scan(layer: Block, carry, x):
    carry = carry + layer.weight * x
    return carry, carry

carry, per_layer = stack.scan_via(block_scan, in_axes=0)(jnp.zeros((width,)), jnp.ones((num_layers, width)))
```

## `in_axes` semantics

`ArrayStacked` uses a vmap-like convention for non-carry inputs:

* `in_axes=None` (default): all extra args/kwargs are shared across layers.
* `in_axes=0`: all extra args are sliced on axis 0.
* `in_axes=(args_axes, kwargs_axes)`:
  `args_axes` is a tuple aligned to positional args;
  `kwargs_axes` is a dict aligned to keyword args.

Any argument with an integer axis must be a `jax.Array` with `shape[axis] == num_layers`.

## Other helpers

* `get_layer(i)`: returns layer `i` as a regular module instance.
* `unstacked()`: returns a tuple of per-layer modules.
* `vmap_via(fn, in_axes=..., out_axes=...)`: applies `fn(layer, ...)` independently to each layer and stacks outputs.

## Checkpointing

`ArrayStacked` accepts `gradient_checkpointing` with the same [haliax.nn.ScanCheckpointPolicy][] / `remat` semantics used
for scan-over-layers in [Scan and Fold](scan.md#gradient-checkpointing--rematerialization).

## API

::: haliax.nn.ArrayStacked
