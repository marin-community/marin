# Haliax Primer

Haliax provides named tensors built on top of JAX.  This primer is written for LLM agents and other downstream libraries and collects the core ideas for quick reference.

## Axes and Named Arrays

Arrays are indexed by `Axis` objects. You can define them explicitly or generate several with `make_axes`.
You may also specify shapes with a **shape dict**, mapping axis names to sizes.

```python
import haliax as hax
from haliax import Axis

Batch = Axis("batch", 4)
Feature = Axis("feature", 8)
# or: Batch, Feature = hax.make_axes(batch=4, feature=8)
# using Axis objects
x = hax.zeros((Batch, Feature))
# or using a shape dict
shape = {"batch": 4, "feature": 8}
x = hax.zeros(shape)
```

Most functions accept either axes or shape dicts interchangeably.

A tensor with named axes is a [`NamedArray`][haliax.NamedArray]. Elementwise operations mirror `jax.numpy` but accept named axes.

### Ways to Describe Axes

If you're used to `axis=0` style code in NumPy or JAX, think of Haliax as swapping those positional indices for names
like `axis="batch"`. The API hints refer to a few helper aliases; this table shows how they map back to familiar
concepts:

| Name | Accepts | Typical use | Example |
| --- | --- | --- | --- |
| [`Axis`][haliax.Axis] | `Axis(name: str, size: int)` | Define a named dimension with a fixed size | `Batch = Axis("batch", 32)` |
| [`AxisSelector`][haliax.AxisSelector] | `Axis` or `str` | Refer to an existing axis when the size can be inferred from the argument | `x.sum(axis="batch")` |
| [`AxisSpec`][haliax.AxisSpec] | `dict[str, int]`, `Axis`, or a sequence of `Axis` objects | Supply complete shape information (array creation, reshaping) | `hax.zeros((Batch, Feature))` |
| [`AxisSelection`][haliax.AxisSelection] | `dict[str, int | None]`, `AxisSpec`, or a sequence of `AxisSelector` values | Work with one or more existing axes (reductions, indexing helpers, flattening, …) | `x.sum(axis=("batch", Feature))` |

The following sections expand on each alias with quick references and NumPy-style parallels.

#### `Axis`: reusable named dimensions

An [`Axis`][haliax.Axis] stores a `name` and a `size`. Create them directly or let
[`haliax.make_axes`][] build a handful at once. Because axes compare by both name and size, they act as reusable handles and
catch many wiring mistakes early.

```python
Batch = Axis("batch", 32)
Feature = Axis("feature", 128)
x = hax.ones((Batch, Feature))
print(Batch.name, Batch.size)
```

#### `AxisSelector`: when the size is already known

Many functions already see the array whose axes you're referencing (e.g. reductions). In those cases you can pass either the
`Axis` object or simply the axis name as a string. Haliax resolves the name against the array, similar to `axis=0` in NumPy.

```python
total = x.sum(axis=Batch)       # use the Axis handle
same_total = x.sum(axis="batch")  # or just the name
```

If you reference an axis name that isn't present, Haliax raises a `ValueError`.

#### `AxisSpec`: describing complete shapes

When Haliax needs explicit sizes—creating arrays, reshaping, broadcasting to a new axis—you provide an [`AxisSpec`][haliax.AxisSpec].
Shape dictionaries keep things close to standard Python, while sequences require actual `Axis` objects so the sizes stay explicit.

```python
shape = {"batch": 32, "feature": 128}
y = hax.zeros(shape)                 # using a shape dict
z = hax.zeros((Batch, Feature))      # or a sequence of Axis objects
```

Python dictionaries preserve insertion order, so the layout in a shape dict matches the order of axes in the resulting array.

#### `AxisSelection`: several axes at once

[`AxisSelection`][haliax.AxisSelection] is the plural form used by reductions, indexing helpers, and flattening utilities. Supply a
tuple mixing `Axis` objects and strings, reuse an existing `AxisSpec`, or pass a partial shape dict where values are either
sizes or `None` for "any size".

```python
scalar = x.sum(axis=("batch", Feature))

# Ask for the axes by name and optionally pin sizes.
x.resolve_axis({"batch": None, "feature": None})  # returns {"batch": 32, "feature": 128}
```

Partial shape dicts shine when you only care about a subset of axes or want assertions about their sizes. If the axis size
cannot be inferred from the provided arguments, Haliax raises a `RuntimeError`.

## Indexing and Broadcasting

Use axis names when slicing. Dictionaries are convenient for several axes:

```python
first = x["batch", 0]
sub = x["batch", 1:3]
# or with a dict
first = x[{"batch": 0}]
sub = x[{"batch": slice(1, 3)}]
```

Axes broadcast by matching names. `broadcast_axis` adds a new axis to an array:

```python
row = hax.arange(Feature)
outer = row.broadcast_axis(Batch) * hax.arange(Batch)
```

See [Indexing and Slicing](indexing.md) and [Broadcasting](broadcasting.md) for details.

## Rearranging Axes

`rearrange` changes axis order and can merge or split axes using einops‑style syntax.  It is useful when interfacing with positional APIs.

```python
# transpose features and batch
x_t = hax.rearrange(x, "batch feature -> feature batch")
```

More examples appear in [Rearrange](rearrange.md).

## Matrix Multiplication

`dot` contracts over named axes while preserving order independence.

```python
Weight = Axis("weight", 8)
w = hax.ones((Feature, Weight))
prod = hax.dot(x, w, axis=Feature)
```

For more complex contractions use [`einsum`][haliax.einsum].  See [Matrix Multiplication](matmul.md).

## Scans and Folds

Use [`scan`][haliax.scan] or [`fold`][haliax.fold] to apply a function along an axis with optional gradient checkpointing.

```python
Time = Axis("time", 10)
sequence = hax.ones((Time, Feature))

def add(prev, cur):
    return prev + cur

result = hax.fold(add, Time)(hax.zeros((Feature,)), sequence)
```

See [Scan and Fold](scan.md) for checkpointing policies and stacked modules.

## Partitioning

Arrays and modules can be distributed across devices by mapping named axes to mesh axes:

```python
with hax.axis_mapping({"batch": "data"}):
    sharded = hax.shard(x)
```

The [Partitioning](partitioning.md) guide explains how to set up device meshes and shard arrays.

## Typing Support

Type annotations use `haliax.haxtyping` which extends `jaxtyping`:

```python
import haliax.haxtyping as ht

def f(t: ht.Float[hax.NamedArray, "batch feature"]):
    ...
```

See [Typing](typing.md) for matching runtime checks and dtype-aware annotations.

---

This primer highlights common patterns.  The [cheatsheet](cheatsheet.md) lists many additional conversions from JAX to Haliax.
