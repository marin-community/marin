# Broadcasting

One area where Haliax's treatment of named axes differs substantially from Numpy-esque positional code is in broadcasting. In traditional positional code, [broadcasting works like this](https://numpy.org/doc/stable/user/basics.broadcasting.html):

```python
import numpy as np

# compute the outer product of two arrays
a = np.arange(5)
b = np.arange(4)

c = a.reshape((-1, 1)) * b.reshape((1, -1))
print(c.shape)
print(c)

# alternatively
c2 = a[:, np.newaxis] * b
```

This prints:
```
(5, 4)
[[ 0  0  0  0]
 [ 0  1  2  3]
 [ 0  2  4  6]
 [ 0  3  6  9]
 [ 0  4  8 12]]
```

To quote the NumPy documentation, for positional arrays, "in order to broadcast, the size of the trailing axes for both arrays in an operation must either be the same size or one of them must be one."

I have found this to be a source of bugs: it is easy to accidentally have an array of size [batch_size, 1] and combine it with an array of size [batch_size], yielding an array of [batch_size, batch_size].

In Haliax, broadcasting is done by matching names. The same operation in Haliax might look like this:

```python
M = hax.Axis("M", 5)
N = hax.Axis("N", 4)

a = hax.arange(M)
b = hax.arange(N)

c = a.broadcast_axis(N) * b
print(c.axes)
print(c.array)
```

```
(Axis(name='N', size=4), Axis(name='M', size=5))
[[ 0  0  0  0  0]
 [ 0  1  2  3  4]
 [ 0  2  4  6  8]
 [ 0  3  6  9 12]]
```

Haliax aims to be "order-independent" as much as possible (while still letting you choose the order for performance or compatibility with positional code).
Its semantics are: "in order to broadcast, identically named Axes of the arrays must have the same size.
In addition, they must share at least one named axis in common, unless one is a scalar."
The second condition is there to avoid bugs: we want to be sure that the arrays have something in common.
To satisfy the second condition, it is not uncommon to use [haliax.broadcast_axis][], like we did above.
This method takes one or more axes and adds them to the array.

## Explicit Broadcasting Functions

* [haliax.broadcast_axis][]
* [haliax.broadcast_to][]


<!--
### Advanced Broadcasting Functions

You probably won't need these, but they're here if you do.

::: haliax.core.broadcast_arrays
::: haliax.core.broadcast_arrays_and_return_axes

-->

## A note on performance

Under the hood, Haliax will automatically broadcast and permute axes so that the underlying positional code produces the correct result.
This is usually not a substantial performance hit because XLA is pretty good about picking optimal shapes,
but if you're doing repeated operations you may want to use [haliax.rearrange][] to change the order of axes.
As an example, in Levanter's GPT-2 implementation, we found using `rearrange` led to a 5% speedup for small models. This
isn't huge, but it's not nothing either.
