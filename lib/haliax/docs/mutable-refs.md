# Mutable References

JAX now exposes [`jax.Ref`](https://jax.readthedocs.io/en/latest/notebooks/refs.html), a mutable buffer that
remains compatible with transformations such as `jax.jit`, `jax.grad`, and friends. Haliax mirrors that idea with
[`haliax.NamedRef`][haliax.NamedRef], preserving axis metadata so you can keep using named indexing while wiring state
through pure computations.

This page walks through the common idioms you will encounter when working with named references, including ways to
slice references, compose partial selectors, and freeze or swap out the underlying data when you are done.

## Creating a `NamedRef`

Use [`haliax.new_ref`][haliax.new_ref] to wrap an existing tensor. The helper accepts either a
[`NamedArray`][haliax.NamedArray] or a raw JAX array plus an axis specification.

```python
import haliax as hax
import jax

Batch, Feature = hax.make_axes(Batch=4, Feature=8)
weights = hax.random.normal(jax.random.PRNGKey(0), (Batch, Feature))
weights_ref = hax.new_ref(weights)

# You can still inspect the named shape.
assert weights_ref.axes == weights.axes
assert weights_ref.shape == {"Batch": 4, "Feature": 8}
```

Inside a pure function you can allocate fresh references to stage intermediate results:

```python
@jax.jit
def normalize(x: hax.NamedArray) -> hax.NamedArray:
    ref = hax.new_ref(x)
    ref[{"Batch": slice(None)}] = ref[{"Batch": slice(None)}] - hax.mean(x, axis=Batch)
    return ref[...]
```

References follow JAX's rules: you may not return the ref itself from inside a transformed function, pass the same ref
argument multiple times, or close over a ref that is also passed as a parameter. Those restrictions keep aliasing under
control so transformations can reason about side effects.

## Reading and writing

`NamedRef` behaves like a lightweight view of the underlying buffer. Reading always returns a `NamedArray`, while writing
accepts either bare scalars/JAX arrays or another `NamedArray` with matching axes.

```python
logits_ref = hax.new_ref(hax.zeros((Batch, Feature)))

# Read a slice.
logits = logits_ref[{"Batch": 0}]

# Write back in-place.
logits_ref[{"Batch": 0}] = hax.random.uniform(jax.random.PRNGKey(1), logits.axes)

# Mutations are visible through other aliasing refs.
assert logits_ref[{"Batch": 0}].array is not None
```

If you prefer a functional style, use [`haliax.ref.get`][haliax.ref.get] or [`haliax.swap`][haliax.swap]. The latter
swaps in a new value and returns the previous contents, similar to `dict.setdefault`.

## Slice references

Plumbing state often requires staging a subset of the axes. `NamedRef.slice` lets you pre-apply a named indexer to a
reference and reuse it later without repeating the prefix. We call the result a *slice ref*.

```python
Cache = hax.Axis("layer", 24)
Head = hax.Axis("head", 8)
cache = hax.zeros((Cache, Head))
cache_ref = hax.new_ref(cache)

# Focus on a subset of layers.
window = cache_ref.slice({"layer": slice(4, 8)})

# Indexing the slice ref automatically splices the prefix into the base ref.
window[{"layer": 0, "head": 3}] = 1.0  # updates layer 4, head 3 in the original buffer
# `.value()` reads the staged slice.
window_value = window.value()
```

Slice refs compose. Applying `.slice(...)` to an existing slice ref merges the new selection with the previous one, so
`cache_ref.slice({"layer": slice(1, 4)}).slice({"layer": 0})` updates the second layer of the original cache. Integer and
slice prefixes are supported; advanced indexing still needs to happen at call time (`slice` with `NamedArray` selectors or
lists is not yet folded into the prefix).

If you need to recover the original reference (without any staged prefixes), call [`NamedRef.unsliced`][haliax.NamedRef.unsliced].

The merging logic mirrors `NamedArray.__getitem__`: ellipses expand to the remaining axes, implicit dimensions are
filled in order, and conflicting assignments raise errors. See [Indexing and Slicing](indexing.md) for a refresher.

## Working with transformations

Because `NamedRef` is registered as a PyTree node, you can nest it inside larger pytree structures or pass it through
`jax.tree_map` for bookkeeping. Only the underlying `jax.Ref` is considered a leaf, so axis metadata is treated as static
structure and does not trigger recompilation.

Keep in mind the JAX restrictions on mutable functions:

- Do not return a ref from inside `jax.jit`, `jax.grad`, `jax.vmap`, `jax.lax.scan`, etc.
- Avoid closing over a ref you also pass into the function (JAX disallows those aliasing patterns).
- Use `jax.lax.stop_gradient` if you are plumbing values that should not affect differentiation.

Pure functions that create refs internally continue to compose normally with autodiff and vectorization. If your ref is
only used for bookkeeping (e.g., collecting auxiliary statistics), stop gradients before writing into it:

```python
def collect_metrics(x, stats_ref):
    y = hax.sin(x)
    stats_ref[{"Batch": slice(None)}] += jax.lax.stop_gradient(y)
    return y
```

## Freezing and swapping

When you are done mutating a reference, call [`haliax.freeze`][haliax.freeze] to convert it back into an immutable
`NamedArray`.

```python
final = hax.freeze(cache_ref)
assert isinstance(final, hax.NamedArray)
```

[`haliax.swap`][haliax.swap] provides an atomic-style update: it returns the previous value while storing the new one.
The helper integrates with slice refs, so you can swap just a subset of the buffer.

```python
prev = hax.swap(cache_ref, {"layer": slice(0, 2)}, hax.ones((Cache.resize(2), Head)))
```

See [`tests/test_named_ref.py`](tests/test_named_ref.py) for runnable examples that exercise the API.
