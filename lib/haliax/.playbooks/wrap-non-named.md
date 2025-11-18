# Wrapping Functions with NamedArray Support

This playbook explains how to convert a regular JAX function that works on unnamed arrays into a Haliax function that accepts `NamedArray` inputs and returns `NamedArray` outputs.

## When is wrapping needed?
Many JAX primitives only operate on regular arrays. To integrate them in Haliax you should provide a thin wrapper that handles axis metadata. Simple elementwise operations and reductions have helper utilities.

## Elemwise Unary
For a unary function that acts elementwise (e.g. `jnp.abs`):

```python
from haliax import wrap_elemwise_unary

def abs(a):
    return wrap_elemwise_unary(jnp.abs, a)
```

This preserves axis order and dtype.

## Elemwise Binary
For binary operations (e.g. `jnp.add`), decorate a function with `wrap_elemwise_binary`:

```python
from haliax import wrap_elemwise_binary

@wrap_elemwise_binary
def add(x1, x2):
    return jnp.add(x1, x2)
```

Broadcasting between `NamedArray`s is handled automatically.

## Reductions
Reductions require choosing axes to eliminate. Use `wrap_reduction_call`:

```python
from haliax import wrap_reduction_call

def sum(a, axis=None):
    return wrap_reduction_call(jnp.sum, a, axis)
```

`axis` can be an `AxisSelector` or tuple. The wrapper returns a `NamedArray` with those axes removed.

## Harder Cases
Some functions need bespoke handling. For example `jnp.unique` returns several arrays and may change shape unpredictably. There is no generic helper, so you will need to manually map between `NamedArray` axes and the outputs. Use the lower level utilities in `haliax.wrap` for broadcasting and axis lookup.

## Axis-aware functions
For JAX functions that accept an ``axis`` or ``axes`` argument, prefer a single
``axis`` parameter of type :class:`haliax.AxisSelection` that also accepts an
ordered mapping of axes to sizes. A mapping dispatches to the corresponding
``n``‑dimensional JAX primitive (e.g. :func:`jax.numpy.fft.fftn`) and allows
resizing the transformed axes by specifying integer lengths or ``Axis``
instances.

## Testing
Add tests to ensure that named and unnamed calls produce the same results and that axis names are preserved or removed correctly.

## Documentation
Once your wrapper works, document it. Add `::: haliax.your_function` (replacing the name) to `docs/api.md` so users can find the new API.
