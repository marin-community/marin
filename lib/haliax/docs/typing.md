from haliax import NamedArray

# NamedArray Type Annotations

Haliax supports an extension to [`jaxtyping`](https://docs.kidger.site/jaxtyping/)
that allows you to annotate functions and methods that take or return
[`NamedArray`][haliax.core.NamedArray] objects. If you are familiar with
[`jaxtyping`](https://docs.kidger.site/jaxtyping/), the syntax is very similar.
In fact, for non-NamedArrays, it is exactly the same.

```python
from haliax import NamedArray
import haliax.haxtyping as ht

def foo(x: ht.Float[NamedArray, "batch embed ..."]):
    ...
```

At runtime you can verify that a `NamedArray` conforms to a particular
annotation using `matches_axes`:

```python
if not arr.matches_axes(Float[NamedArray, "batch embed ..."]):
    raise ValueError("unexpected axes")
```

## DType-aware annotations

Sometimes it is useful to express both the axes **and** the dtype in the type
annotation.  The :mod:`haliax.typing` module defines symbolic types for all of
JAX's common dtypes that can be indexed just like ``Named``.  In documentation
examples we'll use ``import haliax.typing as ht``:

```python
import haliax.haxtyping as ht

def foo(x: ht.f32[NamedArray, "batch"]):
    ...

def bar(x: ht.i32[NamedArray, "batch"]):
    ...
```

For convenience the module also provides aggregate categories ``Float``,
``Complex``, ``Int`` and ``UInt`` that match any floating point, complex,
signed integer or unsigned integer dtype respectively:

```python
def baz(x: ht.Float[NamedArray, "batch"]):
    ...
```

At runtime ``matches_axes`` also checks the dtype when one is present:

```python
from haliax import Axis, zeros
import haliax.haxtyping as ht

arr = zeros({"batch": 4})
assert arr.matches_axes(ht.f32["batch"])  # dtype and axes both match
```

## FAQ

### Why not use `NamedArray` directly in type annotations?

Using `NamedArray` directly in type annotations doesn't work well with
type checkers like `mypy` or `pyright`. These tools expect types to be
subscripted with other types or forward references (which are strings).
Using `NamedArray` directly would lead to type errors.

### Why not use `jaxtyping` directly?

While `jaxtyping` is a powerful library for type annotations in JAX, it does not
support `NamedArray` objects directly. The `haliax.haxtyping` module extends
`jaxtyping` to include `NamedArray` support, allowing you to annotate functions
and methods that take or return `NamedArray` objects with specific axes and dtypes.

### Why do I have to specify the `NamedArray` type in the annotation?

I hate this, but it's the only way to get type checkers like `mypy` and `pyright` to understand that the type is
a `NamedArray`. Underneath the hood, during type checking, `jaxtyping.Float` (and `haxtyping.Float`) are
essentially type aliases of [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated)
with the `NamedArray` type. There's no other way I could find to get type checkers to understand that the type is a
`NamedArray` or to accept strings like `"batch embed ..."` as valid type annotations.

### How do I use single axes in type annotations with flake or ruff.

Like `jaxtyping`, you need to prepend a space before the axis name to use single axes in type annotations with
flake or ruff. For example, to use a single axis named `batch`, you would write:

```python
def foo(x: ht.Float[NamedArray, " batch"]):
    ...
```

Then suppress F722 in your linter to suppress that error.

See the [jaxtyping documentation](https://docs.kidger.site/jaxtyping/faq/#flake8-or-ruff-are-throwing-an-error) for more
details on the workaround.
