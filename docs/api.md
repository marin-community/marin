# Main API Reference
In general, the API is designed to be similar to JAX's version of NumPy's API, with the main difference being
that we use names (either strings or [haliax.Axis][] objects) to specify axes instead of integers. This shows up for creating
arrays (see [haliax.zeros][] and [haliax.ones][]) as well as things like reductions (see [haliax.sum][] and
[haliax.mean][]).

## PyTree Helpers

PyTrees are the lingua franca for composing state in JAX ecosystems. Haliax provides drop-in replacements for the
[`jax.tree`][] helpers that are aware of [`NamedArray`][haliax.NamedArray] semantics. They preserve axis metadata across
transformations while interoperating with standard JAX containers, so you can use them anywhere you would have reached
for JAX's versions.

Use these helpers whenever you need to map, flatten, or rebuild PyTrees that might include `NamedArray` instances:

* [`haliax.tree.map`][] mirrors [`jax.tree.map`][] but forwards to Haliax's [`haliax.tree_util.tree_map`][] so axis names remain
  intact.
* [`haliax.tree.scan_aware_map`][] descends into [`haliax.nn.Stacked`][haliax.nn.Stacked] modules so that each layer is
  transformed individually, effectively treating them as if they were unrolled when applying
  [`haliax.tree_util.scan_aware_tree_map`][].
* [`haliax.tree.flatten`][] / [`haliax.tree.unflatten`][] match the familiar flattening API while handling `NamedArray`
  payloads safely.
* [`haliax.tree.leaves`][] and [`haliax.tree.structure`][] provide direct access to the leaves and PyTree structure.

All of these helpers accept the same `is_leaf` hook you might already use with JAX's utilities. They should be the first
tools you reach for when you need deterministic tree transforms that understand named axes.

::: haliax.tree.map
::: haliax.tree.scan_aware_map
::: haliax.tree.flatten
::: haliax.tree.unflatten
::: haliax.tree.leaves
::: haliax.tree.structure

[`jax.tree`]: https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.html
[`jax.tree.map`]: https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html

## Axis Types

If you already speak NumPy or `jax.numpy`, think of Haliax as swapping positional axes (`axis=0`) for named axes
(`axis="batch"`). The type hints in this section describe the different ways those named axes can be provided to the API.
They appear throughout the documentation and in signatures so that you can quickly tell which forms are accepted.

| Name | Accepts | When to use it | Example |
| --- | --- | --- | --- |
| [`Axis`][haliax.Axis] | `Axis(name: str, size: int)` | Define a named dimension with an explicit size | `Batch = Axis("batch", 32)` |
| [`AxisSelector`][haliax.AxisSelector] | `Axis` or `str` | Refer to an existing axis whose size can be inferred from the arrays you pass in | `x.sum(axis="batch")` |
| [`AxisSpec`][haliax.AxisSpec] | `dict[str, int]`, `Axis`, or a sequence of `Axis` objects | Create or reshape arrays when the axis sizes must be provided | `hax.zeros((Batch, Feature))` |
| [`AxisSelection`][haliax.AxisSelection] | `dict[str, int | None]`, `AxisSpec`, or a sequence of `AxisSelector` values | Work with one or more existing axes (reductions, indexing helpers, flattening, …) | `x.sum(axis=("batch", Feature))` |

### Axis

An [`Axis`][haliax.Axis] is the fundamental building block: it is a tiny dataclass that stores a name and a size.  You can
construct one directly or use [`haliax.make_axes`][] to generate several at a time.

```python
import haliax as hax
from haliax import Axis

Batch = Axis("batch", 32)
Feature = Axis("feature", 128)
x = hax.ones((Batch, Feature))
print(Batch.name, Batch.size)
```

Using `Axis` objects keeps array creation explicit and gives reusable handles you can share between different tensors.
Equality compares both the name and size so you get guardrails when wiring pieces together.

### AxisSelector

An [`AxisSelector`][haliax.AxisSelector] accepts either an `Axis` object or just the axis name as a string.  It is used
whenever a function can read the axis size from one of its arguments.  This mirrors how NumPy lets you pass `axis=0` when
reducing an array:

```python
total = x.sum(axis=Batch)       # using the Axis handle
same_total = x.sum(axis="batch")  # using only the name
```

Strings are convenient when you only care about the name, but `Axis` objects still work so you can keep using the handles
you created earlier.  If an axis with that name is missing, Haliax raises a `ValueError`.

### AxisSpec

An [`AxisSpec`][haliax.AxisSpec] is used when Haliax needs full size information to create or reshape an array.  You can
provide a shape dictionary (sometimes called a "shape dict") that maps names to sizes, or a sequence of `Axis` objects:

```python
shape = {"batch": 32, "feature": 128}
y = hax.zeros(shape)                 # using a shape dict
z = hax.zeros((Batch, Feature))      # using the Axis objects directly
```

Both forms describe the same layout.  Python dictionaries preserve insertion order, so the ordering in a shape dict matches
the order that axes appear in the array.  Sequences must contain `Axis` objects (not plain strings) because Haliax cannot
otherwise know the axis sizes.

### AxisSelection

[`AxisSelection`][haliax.AxisSelection] generalizes the previous aliases so you can talk about several axes at once.  It
shows up in reductions, indexing helpers, axis-mapping utilities, and anywhere you might have written `axis=(0, 1)` in
NumPy.  You may supply:

* a sequence mixing `Axis` objects and strings, e.g. `("batch", Feature)` when reducing two axes,
* an `AxisSpec`, which is handy when you already have a tuple of `Axis` objects, or
* a "partial shape dict" where the values are either sizes or `None` to indicate "any size".  Dictionaries are useful when
  you only care about a subset of axes or want to assert a particular size.

```python
# Reduce over two axes using a tuple of selectors.
scalar = x.sum(axis=("batch", Feature))

# Ask for the axes by name and optionally pin sizes.
x.resolve_axis({"batch": None, "feature": None})  # returns {"batch": 32, "feature": 128}

from haliax.axis import selects_axis
assert selects_axis((Batch, "feature"), {"batch": None, "feature": 128})
```

Occasionally, an axis size can be inferred in some circumstances but not others. When this happens we still use
`AxisSelector` (or `AxisSelection` for multiple axes) but document the behavior in the docstring. A `RuntimeError` will be
raised if the size cannot be inferred.

::: haliax.Axis
::: haliax.AxisSelector
::: haliax.AxisSpec
::: haliax.AxisSelection

### Axis Manipulation

::: haliax.make_axes
::: haliax.axis.axis_name
::: haliax.axis.concat_axes
::: haliax.axis.union_axes
::: haliax.axis.intersect_axes
::: haliax.axis.eliminate_axes
::: haliax.axis.without_axes
::: haliax.axis.selects_axis
::: haliax.axis.is_axis_compatible


## Array Creation
::: haliax.named
::: haliax.zeros
::: haliax.ones
::: haliax.full
::: haliax.zeros_like
::: haliax.ones_like
::: haliax.full_like
::: haliax.arange
::: haliax.linspace
::: haliax.logspace
::: haliax.geomspace


### Combining Arrays


::: haliax.concatenate
::: haliax.stack

(We don't include `hstack` or `vstack` because they are subsumed by `stack`.)

## Array Manipulation

### Broadcasting

See also the section on [Broadcasting](broadcasting.md).

::: haliax.broadcast_axis
::: haliax.broadcast_to

### Slicing

See also the section on [Indexing and Slicing](indexing.md).

::: haliax.index
::: haliax.slice
::: haliax.take
::: haliax.updated_slice

#### Dynamic Slicing

::: haliax.dslice
::: haliax.dblock
::: haliax.ds

### Shape Manipulation

::: haliax.flatten
::: haliax.flatten_axes
::: haliax.rearrange
::: haliax.unbind
::: haliax.unflatten_axis
::: haliax.split
::: haliax.ravel


## Operations

[Binary](#binary-operations) and [unary](#unary-operations) operations are all more or less directly from JAX's NumPy API.
The only difference is they operate on named arrays instead.

### Matrix Multiplication

See also the page on [Matrix Multiplication](matmul.md) as well as the [cheat sheet section](cheatsheet.md#matrix-multiplication).

::: haliax.dot
::: haliax.einsum

## Reductions

Reduction operations are things like [haliax.sum][] and [haliax.mean][] that reduce an array along one or more axes.
Except for [haliax.argmin][] and [haliax.argmax][], they all have the form:

```python
def sum(x, axis: Optional[AxisSelection] = None, where: Optional[NamedArray] = None) -> haliax.NamedArray:
    ...
```

with the behavior closely following that of JAX's NumPy API. The `axis` argument can
be a single axis (or axis name), a tuple of axes, or `None` to reduce all axes. The `where` argument is a boolean array
that specifies which elements to include in the reduction. It must be broadcastable to the input array, using
Haliax's [broadcasting rules](broadcasting.md).

The result of a reduction operation is always [haliax.NamedArray][] with the reduced axes removed.
If you reduce all axes, the result is a NamedArray with 0 axes, i.e. a scalar.
You can convert it to a [jax.numpy.ndarray][] with [haliax.NamedArray.scalar][], or just [haliax.NamedArray.array][].

::: haliax.all
::: haliax.amax
::: haliax.amin
::: haliax.any
::: haliax.argmax
::: haliax.argmin
::: haliax.max
::: haliax.mean
::: haliax.min
::: haliax.nanargmax
::: haliax.nanargmin
::: haliax.nanmax
::: haliax.nanmean
::: haliax.nanmin
::: haliax.nanprod
::: haliax.nanstd
::: haliax.nansum
::: haliax.nanvar
::: haliax.prod
::: haliax.ptp
::: haliax.std
::: haliax.sum
::: haliax.var

### Axis-wise Operations
Axis-wise operations are things like [haliax.cumsum][] and [haliax.sort][] that operate on a single axis of an array but
don't reduce it.

::: haliax.cumsum
::: haliax.cumprod
::: haliax.nancumprod
::: haliax.nancumsum
::: haliax.sort
::: haliax.argsort

### Unary Operations

The `A` in these operations means [haliax.NamedArray][], a `Scalar`, or [jax.numpy.ndarray][].
These are all more or less directly from JAX's NumPy API.

::: haliax.abs
::: haliax.absolute
::: haliax.angle
::: haliax.arccos
::: haliax.arccosh
::: haliax.arcsin
::: haliax.arcsinh
::: haliax.arctan
::: haliax.arctanh
::: haliax.around
::: haliax.bitwise_count
::: haliax.bitwise_invert
::: haliax.bitwise_not
::: haliax.cbrt
::: haliax.ceil
::: haliax.conj
::: haliax.conjugate
::: haliax.copy
::: haliax.cos
::: haliax.cosh
::: haliax.deg2rad
::: haliax.degrees
::: haliax.exp
::: haliax.exp2
::: haliax.expm1
::: haliax.fabs
::: haliax.fix
::: haliax.floor
::: haliax.frexp
::: haliax.i0
::: haliax.imag
::: haliax.invert
::: haliax.iscomplex
::: haliax.isfinite
::: haliax.isinf
::: haliax.isnan
::: haliax.isneginf
::: haliax.isposinf
::: haliax.isreal
::: haliax.log
::: haliax.log10
::: haliax.log1p
::: haliax.log2
::: haliax.logical_not
::: haliax.ndim
::: haliax.negative
::: haliax.positive
::: haliax.rad2deg
::: haliax.radians
::: haliax.real
::: haliax.reciprocal
::: haliax.rint
::: haliax.round
::: haliax.rsqrt
::: haliax.sign
::: haliax.signbit
::: haliax.sin
::: haliax.sinc
::: haliax.sinh
::: haliax.square
::: haliax.sqrt
::: haliax.tan
::: haliax.tanh
::: haliax.trunc

### Binary Operations
::: haliax.add
::: haliax.arctan2
::: haliax.bitwise_and
::: haliax.bitwise_left_shift
::: haliax.bitwise_or
::: haliax.bitwise_right_shift
::: haliax.bitwise_xor
::: haliax.divide
::: haliax.divmod
::: haliax.equal
::: haliax.float_power
::: haliax.floor_divide
::: haliax.fmax
::: haliax.fmin
::: haliax.fmod
::: haliax.greater
::: haliax.greater_equal
::: haliax.hypot
::: haliax.left_shift
::: haliax.less
::: haliax.less_equal
::: haliax.logaddexp
::: haliax.logaddexp2
::: haliax.logical_and
::: haliax.logical_or
::: haliax.logical_xor
::: haliax.maximum
::: haliax.minimum
::: haliax.mod
::: haliax.multiply
::: haliax.nextafter
::: haliax.not_equal
::: haliax.power
::: haliax.remainder
::: haliax.right_shift
::: haliax.subtract
::: haliax.true_divide

### Polynomial Operations

::: haliax.poly
::: haliax.polyadd
::: haliax.polysub
::: haliax.polymul
::: haliax.polydiv
::: haliax.polyint
::: haliax.polyder
::: haliax.polyval
::: haliax.polyfit
::: haliax.roots
::: haliax.trim_zeros
::: haliax.vander

### Other Operations

::: haliax.bincount
::: haliax.clip
::: haliax.packbits
::: haliax.unpackbits
::: haliax.isclose
::: haliax.allclose
::: haliax.array_equal
::: haliax.array_equiv
::: haliax.pad
::: haliax.searchsorted
::: haliax.top_k
::: haliax.nonzero
::: haliax.trace
::: haliax.tril
::: haliax.triu
::: haliax.where

### FFT

All FFT helpers accept an ``axis`` argument which may be a single axis, its
name, or an ordered mapping from axes to output sizes.  Passing a mapping
dispatches to the ``n``‑dimensional variants in :mod:`jax.numpy.fft`.

For example::

    import jax.numpy as jnp
    import haliax as hax

    T = hax.Axis("time", 8)
    signal = hax.arange(T, dtype=jnp.float32)

    # operate along a single axis specified by name
    hax.fft(signal, axis="time")

    # resize by passing an Axis object
    hax.fft(signal, axis=hax.Axis("time", 16))

    X, Y = hax.make_axes(X=4, Y=6)
    image = hax.arange((X, Y), dtype=jnp.float32)

    # transform across several axes in order by passing a sequence
    hax.fft(image, axis=("X", "Y"))

    # selectively resize axes by providing a mapping
    hax.fft(image, axis={"X": None, "Y": hax.Axis("Y", 10)})

    # mappings can cover just a subset of axes when only partial resizing is needed
    hax.fft(image, axis={"Y": 10})

::: haliax.fft
::: haliax.ifft
::: haliax.hfft
::: haliax.ihfft
::: haliax.rfft
::: haliax.irfft
::: haliax.fftfreq
::: haliax.rfftfreq
::: haliax.fftshift
::: haliax.ifftshift



## Named Array Reference

Most methods on [haliax.NamedArray][] just call the corresponding `haliax` function with the array as the first argument,
just as with Numpy.
The exceptions are documented here:

::: haliax.NamedArray


## Partitioning API

See also the section on [Partitioning](partitioning.md).

::: haliax.partitioning.axis_mapping
::: haliax.shard
::: haliax.named_jit
::: haliax.fsdp
::: haliax.partitioning.round_axis_for_partitioning
::: haliax.partitioning.physical_axis_name
::: haliax.partitioning.physical_axis_size
::: haliax.partitioning.sharding_for_axis


## Gradient Checkpointing

Haliax mainly just defers to JAX and [equinox.filter_checkpoint][] for gradient checkpointing. However,
we provide a few utilities to make it easier to use.

See also [haliax.nn.ScanCheckpointPolicy][].

::: haliax.tree_checkpoint_name

### Old API

These functions are being deprecated and will be removed in a future release.

::: haliax.shard_with_axis_mapping
::: haliax.auto_sharded
