## Vectorization with `haliax.vmap`

`haliax.vmap` is a [`NamedArray`][haliax.NamedArray] aware wrapper around
[`jax.vmap`][jax.vmap].  Instead of supplying positional axis numbers you pass
the [`Axis`][haliax.Axis] (or axis name) you want to map over.  Any
`NamedArray` containing that axis is mapped in parallel and the axis is
reinserted in the output.  Regular JAX arrays can be mapped as well by
providing a `default` spec or per‑argument overrides.

Unlike vanilla `jax.vmap`, you may supply **one or more axes**.  When multiple
axes are given, the function is vmapped over each axis in turn (innermost first).
If an axis isn't already present in the array you must also specify its size,
either by passing an `Axis` object (`Axis("batch", 4)`) or a mapping such as
`{"batch": 4}` so the new dimension can be inserted.

### Basic Example

```python
import haliax as hax

Batch = hax.Axis("batch", 4)

def double(x):
    return x * 2

x = hax.arange(Batch)
y = hax.vmap(double, Batch)(x)
```

The result `y` has the same `Batch` axis as `x`, and each element was processed
in parallel.  With JAX you would write `jax.vmap(double)(x.array)` and manually
specify `in_axes`, but Haliax handles the axis automatically.

For applying many modules in parallel see
[`Stacked.vmap`](scan.md#apply-blocks-in-parallel-with-vmap) which builds on this
primitive.

::: haliax.vmap
