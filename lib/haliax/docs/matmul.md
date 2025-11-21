## Matrix Multiplication

Haliax has two ways to do matrix multiplication (and tensor contractions more generally):
[haliax.dot][] and [haliax.einsum][]. [haliax.dot][] and [haliax.einsum][]
can both express any tensor contraction, though in different situations one or the other may be
more suitable for expressing a particular contraction In general:

- Use [haliax.dot][] when you want to express a simple matrix multiplication over one or a few axes.
- Use [haliax.einsum][] when you want to express a more complex tensor contraction.

See also the API reference for [haliax.dot][] and [haliax.einsum][] and the
[cheat sheet section](cheatsheet.md#matrix-multiplication).

### [`haliax.dot`][haliax.dot]

With [haliax.dot][], you specify the axes to contract over, without needing to write out the
axes you want to keep (though you can if you want):

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))
y = hax.dot(x, w, axis=D)  # shape is (H, W), equivalent to np.einsum("hwd,d->hw", x, w)
```

[haliax.dot][] is at its best when you want to express a simple matrix multiplication over one or a few axes.
Syntactically, [haliax.dot][] is similar to reduction operations like [haliax.sum][] and [haliax.mean][].

The [cheat sheet](cheatsheet.md) has a section on [matrix multiplication](cheatsheet.md#matrix-multiplication)
that gives a few examples. Here are several more:

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)
C = hax.Axis("C", 6)

x = hax.arange((H, W, D, C))
w = hax.arange((D, C))
c = hax.arange((C,))

y = hax.dot(x, c, axis=C) # shape is (H, W, D), equivalent to jnp.dot(x, c)

y = hax.dot(x, w, axis=(D, C))  # shape is (H, W), equivalent to np.einsum("...dc,dc->...", x, w)
y = hax.dot(x, w, axis=(D, C), out_axes=(W, H)) # shape is (W, H) instead of (H, W)
y = hax.dot(x, w, c, axis=(D, C)) # shape is (H, W), equivalent to np.einsum("...dc,dc,c->...", x, w, c)
y = hax.dot(x, c, axis=(H, D, C)) # shape is (W,), equivalent to np.einsum("hwdc,c->w", x, c)
s = hax.dot(x, w, axis=None)  # scalar output, equivalent to np.einsum("hwdc,dc->", x, w)
y = hax.dot(x, w, c, axis=())  # shape is (H, W, D, C), equivalent to np.einsum("hwdc,dc,c->hwdc", x, w, c)
y = hax.dot(x, w, c, axis=(), out_axes=(D, ..., H))  # shape is (D, W, C, H), equivalent to np.einsum("hwdc,dc,c->dwch", x, w, c)
```

### [`haliax.einsum`][haliax.einsum]

[haliax.einsum][] is at its best when you want to express a more complex tensor contraction.
It is similar to [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
or [einops.einsum](https://einops.rocks/api/einsum/) in terms of syntax and behavior,
but extended to work with named axes, including the added flexibility that named axes provide.
Our "flavor" of `einsum` is most similar to `einops.einsum`'s flavor, in that
it supports long names for axes (like `"batch h w, h w channel -> batch channel"`)
rather than the compact notation of `numpy.einsum` (like `"bhwc,hwc->bc"`).

Haliax's version of `einsum` comes in three modes: "ordered", "unordered", and "output axes".
These modes are all accessible through the same function without any flags: the syntax
of the `einsum` string determines which mode is used.

The syntax for Haliax's `einsum` is similar to [`haliax.rearrange`](rearrange.md), which
is in turn similar to [einops.rearrange](https://einops.rocks/api/rearrange/).

#### Ordered Mode

Haliax's `einsum` has an "ordered" mode that is similar to `einops.einsum`'s behavior.
In this mode, the axes in the input arrays are matched to the axes in the `einsum` string in order.
It supports ellipses in the same way as `einops.einsum`. The names in the einsum string need not
match the names of the axes in the input arrays, but the order of the axes must match.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))
y = hax.einsum("h w d, d -> h w", x, w)  # shape is (H, W), equivalent to jnp.einsum("hwd,d->hw", x, w)
y = hax.einsum("... d, d -> ...", x, w)  # same as above
```

The `...` syntax is used to indicate that the axes in the input arrays that are not mentioned in the `einsum` string
should be preserved in the output. This should be the same as `einops.einsum`'s behavior, with the exception
that the names of axes with the same label in the string must have the same names in the input arrays.

(If you notice any differences between Haliax's `einsum`'s ordered syntax and `einops.einsum`, please let us know!)

#### Unordered Mode

In "unordered mode," the axes in the input arrays are matched to the axes in the `einsum` string by name,
using similar rules to [haliax.rearrange][]. Names involved in the operation are specified inside `{}`
on the left hand side of the `->` in the `einsum` string. Axes not specified are implicitly preserved.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))

y = hax.einsum("{H W D} -> H W", x)  # shape is (H, W)
y = hax.einsum("{D} -> ", w)  # shape is (H, W)
y = hax.einsum("{...} -> ", x)  # shape is ()
y = hax.einsum("{H ...} -> H", x)  # shape is (H,)
y = hax.einsum("{H ...} -> ...", x)  # shape is (W, D)
```

This mode is most similar to [haliax.dot][]'s behavior, though it's a bit more expressive.

You can also use axis aliases in the `einsum` string, which can be useful for expressing contractions
in library code or just for shortening the string:

```python
Height = hax.Axis("Height", 3)
Width = hax.Axis("Width", 4)
Depth = hax.Axis("Depth", 5)

x = hax.ones((Height, Width, Depth))
w = hax.ones((Depth,))

y = hax.einsum("{H W D} -> H W", x, H=Height, W=Width, D=Depth)  # shape is (Height, Width)
y = hax.einsum("{D} -> ", w, D=Depth)  # shape is (Height, Width)
```


#### Output Axes Mode

In "output axes" mode, you only specify the axes that should be in the output. All other
axes are implicitly contracted over. This mode is a bit "dangerous" in that it's easy to
accidentally contract over axes you didn't mean to, but it can be useful for expressing
certain contractions concisely.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))

y = hax.einsum("-> H W", x)  # shape is (H, W)
y = hax.einsum("-> D", w)  # shape is (D,)
```

We don't recommend using this mode except in cases when you're sure of the full shape of the input arrays
or you are sure you don't want to let users implicitly batch over any axes.

Output axes mode also supports axis aliases:

```python
Height = hax.Axis("Height", 3)
Width = hax.Axis("Width", 4)
Depth = hax.Axis("Depth", 5)

x = hax.ones((Height, Width, Depth))
w = hax.ones((Depth,))
y = hax.einsum("-> Height Width", x, Height=Height, Width=Width, Depth=Depth)  # shape is (Height, Width)
y = hax.einsum("-> Depth", w, Depth=Depth)  # shape is (Depth,)
```
