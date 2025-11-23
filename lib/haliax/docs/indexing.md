# Indexing and Slicing

Haliax supports Numpy-style indexing, including so-called [Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing),
though the syntax is necessarily different. Most forms of indexing are supported, except we don't support indexing with
booleans right now. (JAX doesn't support indexing with non-constant bool arrays anyway,
so I don't think it's worth the effort to implement it in Haliax.)

## Basic Indexing

Basic indexing works basically like you would expect: you can use integers or slices to index into an array.
Haliax supports two syntaxes for indexing: one accepts a dict of axis names and indices, and the other accepts
an alternating sequence of axis names and indices. The latter is useful for indexing with a small number of indices.

```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))

a[{"X": 1, "Y": 2, "Z": 3}]  # returns a scalar jnp.ndarray
a[{"X": 1, "Y": 2, "Z": slice(3, 5)}]  # return a NamedArray with axes = Axis("Z", 2)
a[{"X": 1, "Y": slice(2, 4), "Z": slice(3, 5)}]  # return a NamedArray with axes = Axis("Y", 2), Axis("Z", 2)

a["X", 1, "Y", 2, "Z", 3]  # returns a scalar jnp.ndarray
a["X", 1, "Y", 2, "Z", 3:5]  # return a NamedArray with axes = Axis("Z", 2)
a["X", 1, "Y", 2:4, "Z", 3:5]  # return a NamedArray with axes = Axis("Y", 2), Axis("Z", 2)
```

Unfortunately, Python won't let us use `:` slice syntax inside of a dictionary, so we have to use `slice` instead.
This is why we have the second syntax, which is a bit less idiomatic in some ways, but it's more convenient.

Otherwise, the idea is pretty straightforward: any unspecified axes are treated as though indexed with `:` in NumPy,
slices are kept in reduced dimensions, and integers eliminate dimensions. If all dimensions are eliminated, a scalar
JAX ndarray is returned.

The following types are supported for indexing:

* Integers, including scalar JAX arrays
* Slices
* [haliax.dslice][] objects (See [Dynamic Slices](#dynamic-slices) below.)
* Lists of integers
* Named arrays (See [Advanced Indexing](#advanced-indexing) below.)
* 1-D JAX Arrays of integers

1-D JAX Arrays are interpreted as NamedArrays with a single axis with the same name as
the one they are slicing. That is:

```python
import haliax as hax
import jax
import jax.numpy as jnp

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y))

sliced = a["X", jnp.array([1, 2, 3])]

# same as
a.array[jnp.array([1, 2, 3]), :]
```

Note that boolean arrays are not supported, as JAX does not support them in JIT-compiled code. You
can use [haliax.where][] for most of the same functionality, though.

### Shapes in JAX

Before we continue, a note on shapes in JAX. Most JAX code will be used inside `jit`, which means that the sizes of all
arrays must be determined at compile time (i.e. when JAX interprets your functions abstractly). This is a hard
requirement in XLA.

A consequence of this restriction is that certain indexing patterns aren't allowed in `jit`-ed JAX code:

```python
import jax.numpy as jnp
import jax

@jax.jit
def f(x, slice_size: int):
    num_blocks = x.shape[0] // slice_size
    def body(i, m):
        return i + jnp.mean(x[i * slice_size : (i + 1) * slice_size])
    jax.lax.fori_loop(0, num_blocks, lambda i, m: m + body(i, m), 0.0)


f(jnp.arange(10), 2)
# IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax.
# Found slice(Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=2/0)>,
# Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=2/0)>, None). To index a statically sized
# array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized
# arrays within JIT compiled functions).
```

This is a not-uncommon pattern in situations where you want to process a large array in chunks. In Haliax, we provide
two solutions: [haliax.slice][] and dynamic slices ([haliax.dslice][] a.k.a. [haliax.ds][]).

## Dynamic Slices

[haliax.slice][] is a convenience function that wraps [jax.lax.dynamic_slice][] and allows you to slice an array with a
dynamic start and size. This is useful for situations where you need to slice an array in a way that can't be determined
at compile time. For example, the above example can be written as follows:

```python
import jax

import haliax as hax

N = hax.Axis("N", 10)
q = hax.arange(N)

@hax.named_jit
def f(x, slice_size: int):
    num_blocks = N.size // slice_size
    def body(i, m):
        return i + hax.mean(hax.slice(x, {"N": i * slice_size}, {"N": slice_size}))
    jax.lax.fori_loop(0, num_blocks, body, 0.0)
```


In light of the requirement that all array sizes be known at compile time, Haliax provides both a simple [haliax.slice][]
function, as well as [haliax.dslice][], which can be used with `[]`. The simple slice function is just a wrapper
around [jax.lax.dynamic_slice][] and not worth discussing here.

`dslice` is a trick borrowed from the new experimental [jax.experimental.pallas][] module. It's essentially a slice,
except that instead of a start and an end (and maybe a stride), it takes a start and a size. The size must be
statically known, but the start can be dynamic. This allows us to write the above example as follows:

```python
import jax
import haliax as hax

N = hax.Axis("N", 10)
q = hax.arange(N)

@hax.named_jit
def f(x, slice_size: int):
    num_blocks = N.size // slice_size
    def body(i, m):
        return i + hax.mean(x["N", hax.dslice(i * slice_size, slice_size)])
    jax.lax.fori_loop(0, num_blocks, body, 0.0)

f(q, 2)
```

When indexing with ``dslice`` the slice is gathered starting at ``start`` for
``size`` elements.  Reads beyond the end of the array return the ``fill_value``
(0 by default).  When used with ``at`` updates, any writes outside the bounds of
the array are dropped.  These semantics match JAX's scatter/gather behavior.

For convenience/brevity, `dslice` is aliased as `ds`. In addition, we also expose `dblock`, which is a convenience
function for computing the start and size of a slice given a block index and the size of the slice. Thus, the above
example can be written as follows:

```python
import jax
import haliax as hax

N = hax.Axis("N", 10)
q = hax.arange(N)

@hax.named_jit
def f(x, slice_size: int):
    num_blocks = N.size // slice_size
    def body(i, m):
        return i + hax.mean(x["N", hax.ds.block(i, slice_size)])
    jax.lax.fori_loop(0, num_blocks, body, 0.0)

f(q, 2)
```

It's not a huge improvement, but it's a bit more convenient.


## Advanced Indexing

NumPy's [Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) is supported, though we use named arrays for the indices instead of normal arrays.
(Though, as noted above, you can use 1-D JAX arrays of integers as well.)
In NumPy, the indexed arrays must be broadcastable to the same shape. Advanced indexing in Haliax is similar,
except that it follows Haliax's broadcasting rules, meaning that shared names are broadcasted together,
while non-shared names are treated as separate axes and are cross-producted.
In particular, axes with the same name must have the same size.

```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))

I1 = hax.Axis("I1", 5)
I2 = hax.Axis("I2", 5)
I3 = hax.Axis("I3", 5)
ind1 = hax.random.randint(jax.random.PRNGKey(0), (I1,), 0, 10)
ind2 = hax.random.randint(jax.random.PRNGKey(0), (I2, I3), 0, 20)

a[{"X": ind1}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("Y", 20), Axis("Z", 30)

a[{"X": ind1, "Y": ind2}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("I2", 5), Axis("I3", 5), Axis("Z", 30)
a[{"X": ind1, "Y": ind2, "Z": 3}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("I2", 5), Axis("I3", 5)
```

The order of the indices in the dictionary doesn't matter, and you can mix and match basic and advanced indexing.
The actual sequence of axes is a bit complex, both in Haliax and in NumPy. If you need a specific order, it's probably
best to use rearrange.

In keeping with the one-axis-per-name rule, you are allowed to index using axes with a name present in the array,
if it would be eliminated by the indexing operation. For example:

```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

X2 = hax.Axis("X", 5)
Y2 = hax.Axis("Y", 5)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))
ind1 = hax.random.randint(jax.random.PRNGKey(0), (X2,), 0, 10)
ind2 = hax.random.randint(jax.random.PRNGKey(0), (Y2,), 0, 10)

a[{"X": ind1, "Y": ind2}]  # returns a NamedArray with axes = Axis("X", 5), Axis("Y", 5), Axis("Z", 30)

a[{"Y": ind1}]  # error, "X" is not eliminated by the indexing operation

a[{"X": ind2, "Y": ind1}]  # ok, because X and Y are eliminated by the indexing operation
```

## Index Update

JAX is a functional version of NumPy, so it doesn't directly support in-place updates. It does
however [provide an `at` syntax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at)
to express the same logic (and that will typically be optimized to be as efficient as an in-place update). Haliax
provides a similar syntax for updating arrays.

```python
import haliax as hax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

a = hax.zeros((X, Y, Z))

a.at[{"X": 1, "Y": 2, "Z": 3}].set(1.0)  # sets a[1, 2, 3] to 1.0
a.at["X", 1].set(2.0)  # sets a[1, :, :] to 2.0

a.at[{"X": 1, "Y": hax.ds(3, 5), "Z": 3}].add(1.0)  # adds 1.0 to a[1, 3:8, 3]
```

Haliax supports the same `at` functionality as JAX, just with named arrays and additionally dslices. A summary of the
`at` syntax is as follows:

| Alternate Syntax             | Equivalent In-Place Operation |
|------------------------------|-------------------------------|
| `x = x.at[idx].set(y)`       | `x[idx] = y`                  |
| `x = x.at[idx].add(y)`       | `x[idx] += y`                 |
| `x = x.at[idx].multiply(y)`  | `x[idx] *= y`                 |
| `x = x.at[idx].divide(y)`    | `x[idx] /= y`                 |
| `x = x.at[idx].power(y)`     | `x[idx] **= y`                |
| `x = x.at[idx].min(y)`       | `x[idx] = minimum(x[idx], y)` |
| `x = x.at[idx].max(y)`       | `x[idx] = maximum(x[idx], y)` |
| `x = x.at[idx].apply(ufunc)` | `ufunc.at(x, idx)`            |
| `x = x.at[idx].get()`        | `x = x[idx]`                  |

These methods also have options to control out-of-bounds behavior, as well as allowing you
to specify that the indices are sorted or unique. (If they are, XLA can sometimes optimize the
operation more effectively.)

!!! note

    These named arguments are not passed to `at`, but to the next method in the chain.

(This is copied from the JAX documentation:)

* `mode`: One of `"promise_in_bounds"`, `"clip"`, `"drop"`, or `"fill"`. See [JAX's documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.GatherScatterMode.html#jax.lax.GatherScatterMode) for more details.
* `indices_are_sorted`: If `True`, the implementation will assume that the indices passed to `at` are sorted in ascending order, which can lead to more efficient execution on some backends.
* `unique_indices`: If `True`, the implementation will assume that the indices passed to `at` are unique, which can result in more efficient execution on some backends.
* `fill_value`: Only applies to the `get()` method: the fill value to return for out-of-bounds slices when mode is 'fill'. Ignored otherwise. Defaults to NaN for inexact types, the largest negative value for signed types, the largest positive value for unsigned types, and True for booleans.

!!! tip

    It's worth emphasizing that these functions are typically compiled to scatter-add and friends (as appropriate).
    This is the preferred way to do scatter/gather operations in JAX, as well as in Haliax.

## Scatter/Gather

Haliax supports scatter/gather semantics in its indexing operations. When an axis
is indexed by another NamedArray (or a 1-D JAX array), the values of that axis
are gathered according to the index array and the axes of the indexer are
inserted into the result.

```python
import haliax as hax
import jax.numpy as jnp

B, S, V = Axis("batch", 4), Axis("seq", 3), Axis("vocab", 7)
x = hax.arange((B, S, V))
idx = hax.arange((B, S), dtype=jnp.int32) % V.size

out = x["vocab", idx]
```

Here `out` has axes `(B, S)` and its values match `jax.numpy.take_along_axis`
on the underlying ndarray.

For scatter-style updates where each batch writes to a different position, use
[`updated_slice`][haliax.updated_slice]:

```python
Batch = hax.Axis("batch", 2)
Seq = hax.Axis("seq", 5)
New = hax.Axis("seq", 2)

cache = hax.zeros((Batch, Seq), dtype=int)
lengths = hax.named([1, 3], axis=Batch)
kv = hax.named([[1, 2], [3, 4]], axis=(Batch, New))

result = updated_slice(cache, {"seq": lengths}, kv)
```

This inserts `[1, 2]` starting at position `1` in batch `0` and `[3, 4]` starting
at position `3` in batch `1`.
