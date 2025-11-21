# Rearrange

## Introduction

Haliax strives to be "order independent": the order of axes should not impact the correctness of the program. However,
when interfacing with non-named APIs (e.g. the JAX Numpy API or Equinox), you have to be able to know exactly what the
order of axes is. In addition, the order of axes can impact performance. To that end, Haliax provides a `rearrange`
function that allows you to specify the order of axes in a tensor.

In addition, it is sometimes necessary to split and merge axes: turning images into patches,
or turning a batch of images into a single image. Without `rearrange`, this is a bit clunky.

`rearrange` comes in two flavors: sequence syntax and einops-style syntax. Sequence
syntax is just for transposing axes, while [einops-style](https://einops.rocks/) syntax is more
powerful and can also split and merge axes.

## Sequence Syntax

The sequence syntax is very simple: you just provide a sequence of axis names, and the tensor
will be rearranged to match that sequence. For example:

```python
import haliax as hax
import jax.random as jrandom

N = hax.Axis("N", 32)
C = hax.Axis("C", 3)
H = hax.Axis("H", 64)
W = hax.Axis("W", 64)

x = hax.random.normal(jrandom.PRNGKey(0), (N, C, H, W))

y = hax.rearrange(x, (N, H, W, C))

# at most one ellipsis is allowed
z = hax.rearrange(x, (N, ..., C))

# you can use strings instead of axis objects
z = hax.rearrange(x, ("N", ..., "C"))
```

As we said before, almost all Haliax operations are order-agnostic, so (this version of) `rearrange` only impacts
performance and allows you to interface with other libraries that need you to specify the order of axes
for an unnamed tensor.

## Einops-style Syntax

[einops](https://einops.rocks/) is a powerful library for manipulating tensor shapes, generalizing
`reshape`, `transpose`, and other shape-manipulation operations. Haliax provides a subset of its functionality
(specifically `rearrange` and not `repeat` or `reduce`, which are less useful in named code). The syntax has been generalized to named
tensors.

If you're used to einops, the syntax should be familiar, with the main differences being specifying names
and the additional "unordered" syntax for selecting dimensions by name.

!!! warning

    This syntax is fairly new. It is pretty well-tested, but it is possible that there are bugs.

### Examples

Examples are probably the best way to get a feel for the syntax:

```python
import haliax as hax
import jax.random as jrandom

N = hax.Axis("N", 32)
C = hax.Axis("C", 3)
H = hax.Axis("H", 64)
W = hax.Axis("W", 64)

x = hax.random.normal(jrandom.PRNGKey(0), (N, C, H, W))

# transpose/permute axes
y = hax.rearrange(x, "N C H W -> N H W C")
# names don't have to match with positional syntax
z = hax.rearrange(x, "num ch h w -> num h w ch")
# ellipsis can be used to specify the rest of the dimensions
z = hax.rearrange(x, "N C ... -> N ... C")

# unordered patterns allow you to match a subset of dimensions by name, rather than using positional matching
# transpose last two dimensions using the unordered syntax
y = hax.rearrange(x, "{H W} -> ... W H")

# don't know the order? use an unordered pattern
y = hax.rearrange(x, "{W C H N} -> N H W C")

# split dims as in einops
y = hax.rearrange(x, "(step microbatch) ... -> step microbatch ...", step=4)
# splitting dims can be done using unordered syntax, similar to positional syntax
y = hax.rearrange(x, "{(N: step microbatch) ...} -> step microbatch ...", step=4)

# merging dims requires a name
x = hax.rearrange(y, "step microbatch ... -> (N: step microbatch) ...")

# you can partially specify the order by using two or more ellipses on the rhs
y = hax.rearrange(x, "{H W} -> ... (F: H W) ...")
y = hax.rearrange(x, "{H W C} -> ... (F: H W) ... C")  # ensures C is the last dimension


# some fancier examples

# split into patches
y = hax.rearrange(x, "N C (patch_h H) (patch_w W) -> N (P: patch_h patch_w) C H W", H=4, W=4)
# unordered version
y = hax.rearrange(x, "{(H: patch_h H) (W: patch_w W) C } -> ... (P: patch_h patch_w) C H W", H=4, W=4)

# split into patches, then merge patches and channels
y = hax.rearrange(x, "N C (patch_h H) (patch_w W) -> N (P: patch_h patch_w) (C: C H W)", H=4, W=4)
# unordered version
y = hax.rearrange(x, "{(H: patch_h H) (W: patch_w W) C } -> ... (P: patch_h patch_w) (C: C H W)", H=4, W=4)
```

### Bindings: Aliasing and Sizing

In the above examples we used keyword arguments to give sizes to split axes, which is the same
as in einops. However, we can also use bindings to alias axes. For example:

```python
# this produces the same result as the previous example
y2 = hax.rearrange(x, "N C (patch_h foo) (patch_w bar) -> N (P: patch_h patch_w) (C: C foo bar)", foo=hax.Axis("H", 4), bar=hax.Axis("W", 4))
assert y.axes == y2.axes
```

This example is a bit contrived, but the point is that this syntax lets us use shorter or different names in the string,
which is occasionally useful.

You can actually pass in a string alias instead of an axis object, and it will be converted to an axis object for you:
For instance, if we wanted "P" to actually be called "patch", but wanted to keep the short syntax, we could do:

```python
y3 = hax.rearrange(x, "N C (nh ph) (nw pw) -> N (P: nh nw) (C: C ph pw)", P="patch", pw=4, ph=4)
```


### Differences from einops

As you may have noticed, there are some differences from einops:

* Merged axes must have a name: `(C: C H W)` instead of `(C H W)`.
* The unordered syntax with `{  }` is new: you select dimensions by name instead of by position.
* As discussed immediately above, you can use bindings to specify axis objects for names as well as sizes.

### Syntax

Semiformally, the syntax is an `lhs -> rhs` pair, where the `lhs` is either ordered or unordered, and the `rhs` is always ordered.
For the `lhs`:

* An *ordered lhs* is a sequence of axis variables (e.g. `x`) or (named or anonymous) split axes (e.g. `(x y)`), separated by spaces or commas, and up to one ellipsis
* An *unordered lhs* is a sequence of axis names (e.g. `x`, where `x` is an axis name in the input array) or named split axes (e.g. `(x: y z)`), surrounded by `{}`, separated by spaces or commas.

* An *axis variable* is an identifier (that need not correspond to an axis name in the input or output.)
* An *axis name* is an identifier that corresponds to an axis name in the input or output.
* An *anonymous split axis* is a parenthesized expression of the form `(ident*)`, e.g. `(N C)`.
* A *named split axis* is a parenthesized expression of the form `(name: ident*)`, where `name` is the name of an axis and the `ident` are axis variable names, e.g. `(N: s mb)`

A note on "axis variable" vs "axis name": the former is an identifier that can refer to any axis and is matched
by **position** in the input, while the latter is an identifier that refers to a specific axis and is matched by **name** in the input
(or used to name an axis in the output).

The `rhs` is always ordered. Its syntax is similar to an ordered `lhs`, except that merged axes must always be named and there may be more than one ellipsis.

* *rhs* is a sequence of axis variable names or named merged axes, separated by spaces or commas, and some number of ellipses.

* *Named merged axes* are parenthesized expressions of the form `(name: ident ident ...)`, where `name` is an axis name and `ident` is an identifier.
The name is used to name the merged axis in the output, and the `ident` are axis variable names that are merged from the input.

Identifiers in the `rhs` must be "bound" by an identifier in the `lhs`, that is, they must appear in the `lhs` as an *axis variable name*.

As in einops: split and merged axes are processed in "C-order": the first dimension is the most significant, and the
last dimension is the least significant.


## Error Handling

`rearrange` attempts to be as helpful as possible when it encounters errors. For example:

```python
y = hax.rearrange(x, "N C H W Z -> N H W C")
# ValueError: Error while parsing:
#    N C H W Z -> N H W C
#            ^
# Too many axes in lhs
```

In general, it will try to give you a helpful error message that points to the problem in the string.


## API Documentation

See [haliax.rearrange][].
