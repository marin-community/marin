# Haliax Cheatsheet

This is a cheatsheet for converting common functions and tasks from JAX/NumPy to Haliax.
Please open [an issue on Github](https://github.com/stanford-crfm/haliax/issues/) if you find any errors or omissions.
We're happy to add more examples if you have any that you think would be useful.

## Preamble

Throughout we assume the following:

```python
import jax.numpy as jnp

import haliax as hax

Batch = hax.Axis("batch", 32)
Embed = hax.Axis("embed", 64)
H = hax.Axis("h", 16)
W = hax.Axis("w", 16)
C = hax.Axis("c", 3)


Step = hax.Axis("step", 2)
Mini = hax.Axis("mini", 16)

# for jax
x = jnp.zeros((32, 64))
y = jnp.zeros((32, 64))
z = jnp.zeros((32,))
w = jnp.zeros((64,))
ind = jnp.arange((8,), dtype=jnp.int32)
im = jnp.zeros((32, 16, 16, 3))
w2 = jnp.zeros((3, 64))

# for haliax
x = hax.zeros((Batch, Embed))
y = hax.zeros((Batch, Embed))
z = hax.zeros((Batch,))
w = hax.zeros((Embed,))
ind = hax.arange(hax.Axis("Index", 8), dtype=jnp.int32)
im = hax.zeros((Batch, H, W, C))
w2 = hax.zeros((C, Embed))
```


## Array Creation

| JAX                                            | Haliax                                         |
|------------------------------------------------|------------------------------------------------|
| [`jnp.zeros((32, 64))`][jax.numpy.zeros]       | [`hax.zeros((Batch, Embed))`][haliax.zeros]    |
| [`jnp.ones((32, 64))`][jax.numpy.ones]         | [`hax.ones((Batch, Embed))`][haliax.ones]      |
| [`jnp.zeros_like(x)`][jax.numpy.zeros_like]    | [`hax.zeros_like(x)`][haliax.zeros_like]       |
| [`jnp.ones_like(x)`][jax.numpy.ones_like]      | [`hax.ones_like(x)`][haliax.ones_like]         |
| [`jnp.eye(32)`][jax.numpy.eye]                 | ❌                                              |
| [`jnp.arange(32)`][jax.numpy.arange]           | [`hax.arange(Batch)`][haliax.arange]           |
| [`jnp.linspace(0, 1, 32)`][jax.numpy.linspace] | [`hax.linspace(Batch, 0, 1)`][haliax.linspace] |
| [`jnp.logspace(0, 1, 32)`][jax.numpy.logspace] | [`hax.logspace(Batch, 0, 1)`][haliax.logspace] |
| [`jnp.geomspace(0, 1, 32)`][jax.numpy.geomspace] | [`hax.geomspace(Batch, 0, 1)`][haliax.geomspace] |

### Combining Arrays

| JAX                                                | Haliax                                                   |
|----------------------------------------------------|----------------------------------------------------------|
| [`jnp.concatenate([x, y])`][jax.numpy.concatenate] | [`hax.concatenate("batch", [x, y])`][haliax.concatenate] |
| [`jnp.stack([x, y])`][jax.numpy.stack]             | [`hax.stack("foo", [x, y])`][haliax.stack]               |
| [`jnp.hstack([x, y])`][jax.numpy.hstack]           | [`hax.concatenate("embed", [x, y])`][haliax.concatenate] |
| [`jnp.vstack([x, y])`][jax.numpy.vstack]           | [`hax.concatenate("batch", [x, y])`][haliax.concatenate] |

## Array Manipulation

| JAX                                              | Haliax                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------|
| [`jnp.reshape(x, (2, 16, 64))`][jax.numpy.reshape] | [`hax.unflatten_axis(x, "batch", (Step, Mini)`][haliax.unflatten_axis]  |
| [`jnp.reshape(x, (-1,))`][jax.numpy.reshape]     | [`hax.flatten_axes(x, ("batch", "embed"), "foo")`][haliax.flatten_axes] |
| [`jnp.transpose(x, (1, 0))`][jax.numpy.transpose] | [`hax.rearrange(x, ("embed", "batch"))`][haliax.rearrange]              |

### Shape Manipulation

| JAX                                           | Haliax                                                                              |
|-----------------------------------------------|-------------------------------------------------------------------------------------|
| [`x.transpose((1, 0))`][jax.numpy.transpose]  | [`x.rearrange("embed", "batch")`][haliax.rearrange]                                 |
| [`x.reshape((2, 16, 64))`][jax.numpy.reshape] | [`x.unflatten_axis("batch", (Axis("a", 2), Axis("b", 16)))`][haliax.unflatten_axis] |
| [`x.reshape((-1,))`][jax.numpy.reshape]      | [`x.flatten_axes(("batch", "embed"), "foo")`][haliax.flatten_axes]                  |
| [`jnp.ravel(x)`][jax.numpy.ravel]                | [`hax.ravel(x, "Embed")`][haliax.flatten]                               |
| [`jnp.ravel(x)`][jax.numpy.ravel]                | [`hax.flatten(x, "Embed")`][haliax.flatten]                             |

### Einops-style Rearrange

See also the section on [Rearrange](rearrange.md).

| JAX (with einops)                                                                          | Haliax                                                                   |
|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`einops.rearrange(x, "batch embed -> embed batch")`](https://einops.rocks/api/rearrange/) | [`hax.rearrange(x, ("embed", "batch"))`][haliax.rearrange]               |
| [`einops.rearrange(x, "batch embed -> embed batch")`](https://einops.rocks/api/rearrange/) | [`hax.rearrange(x, "b e -> e b")`][haliax.rearrange]                     |
| [`einops.rearrange(im, "... h w -> ... (h w)")`](https://einops.rocks/api/rearrange/)      | [`hax.flatten_axes(im, ("h", "w"), "hw")`][haliax.flatten_axes]          |
| [`einops.rearrange(im, "... h w c -> ... (h w c)")`](https://einops.rocks/api/rearrange/)  | [`hax.rearrange(im, "{h w c} -> ... (embed: h w c)")`][haliax.rearrange] |
| [`einops.rearrange(x, "b (h w) -> b h w", h=8)`](https://einops.rocks/api/rearrange/)      | [`hax.rearrange(x, "b (h w) -> b h w", h=8)`][haliax.rearrange]          |
| [`einops.rearrange(x, "b (h w) -> b h w", h=8)`](https://einops.rocks/api/rearrange/)      | [`hax.rearrange(x, "{(embed: h w)} -> ... h w", h=8)`][haliax.rearrange] |

### Broadcasting

See also the section on [Broadcasting](broadcasting.md).

| JAX                                                                      | Haliax                                                  |
|--------------------------------------------------------------------------|---------------------------------------------------------|
| [`jnp.broadcast_to(z.reshape(-1, 1), (32, 64))`][jax.numpy.broadcast_to] | [`hax.broadcast_axis(z, Embed)`][haliax.broadcast_axis] |
| Outer product: `z.reshape(-1, 1) * w.reshape(1, -1)`                     | `z * w.broadcast_axis(Batch)`                           |


### Indexing and Slicing

See also the section on [Indexing and Slicing](indexing.md).

| JAX                                                                      | Haliax                                                      |
|--------------------------------------------------------------------------|-------------------------------------------------------------|
| `x[0]`                                                                   | [`x["batch", 0]`][haliax.index]                             |
| `x[:, 0]`                                                                | [`x["embed", 0]`][haliax.index]                             |
| `x[0, 1]`                                                                | [`x["batch", 0, "embed", 1]`][haliax.index]                 |
| `x[0:10]`                                                                | [`x["batch", 0:10]`][haliax.index]                          |
| `x[0:10:2]`                                                              | [`x["batch", 0:10:2]`][haliax.index]                        |
| `x[0, 1:10:2]`                                                           | [`x["batch", 0, "embed", 1:10:2]`][haliax.index]            |
| `x[0, [1, 2, 3]]`                                                        | [`x["batch", 0, "embed", [1, 2, 3]]`][haliax.index]         |
| `x[0, ind]`                                                              | [`x["batch", 0, "embed", ind]`][haliax.index]               |
| `jnp.take_along_axis(x, ind, axis=1)`][jax.numpy.take_along_axis]        | [`hax.take(x, "embed", ind)`][haliax.take]                  |
| [`jax.lax.dynamic_slice_in_dim(x, 4, 10)`][jax.lax.dynamic_slice_in_dim] | [`hax.slice(x, "batch", start=4, length=10)`][haliax.slice] |
| [`jax.lax.dynamic_slice_in_dim(x, 4, 10)`][jax.lax.dynamic_slice_in_dim] | [`x["batch", hax.ds(4, 10)]`][haliax.dslice]                |


## Operations

### Elementwise Operations

Almost all elementwise operations are the same as JAX, except that they work on either [haliax.NamedArray][]
or [jax.numpy.ndarray][] objects.

Any elementwise operation in [jax.nn][] should be in [haliax.nn](nn.md).

### Binary Operations

Similarly, binary operations are the same as JAX, except that they work on  [haliax.NamedArray][] objects.

### Reductions

Reductions are similar to JAX, except that they use an axis name instead of an axis index.

| JAX                                          | Haliax                                            |
|----------------------------------------------|---------------------------------------------------|
| [`jnp.sum(x, axis=0)`][jax.numpy.sum]        | [`hax.sum(x, axis=Batch)`][haliax.sum]            |
| [`jnp.mean(x, axis=(0, 1))`][jax.numpy.mean] | [`hax.mean(x, axis=(Batch, Embed))`][haliax.mean] |
| [`jnp.max(x)`][jax.numpy.max]                | [`hax.max(x)`][haliax.max]                        |
| [`jnp.min(x, where=x > 0)`][jax.numpy.min]   | [`hax.min(x, where=x > 0)`][haliax.min]           |
| [`jnp.argmax(x, axis=0)`][jax.numpy.argmax]  | [`hax.argmax(x, axis=Batch)`][haliax.argmax]      |

### Matrix Multiplication

| JAX                                                            | Haliax                                                             |
|----------------------------------------------------------------|--------------------------------------------------------------------|
| [`jnp.dot(z, x)`][jax.numpy.dot]                               | [`hax.dot(z, x, axis="batch")`][haliax.dot]                        |
| [`jnp.matmul(z, x)`][jax.numpy.matmul]                         | [`hax.dot(z, x, axis="batch")`][haliax.dot]                        |
| [`jnp.dot(w, x.t)`][jax.numpy.dot]                             | [`hax.dot(w, x, axis="embed")`][haliax.dot]                        |
| [`jnp.einsum("ij,j -> i", x, w)`][jax.numpy.einsum]            | [`hax.dot(x, w, axis="embed")`][haliax.dot]                        |
| [`jnp.einsum("i,ij,ij,j -> i", z, x, y, w)`][jax.numpy.einsum] | [`hax.dot(z, x, y, w, axis="embed")`][haliax.dot]                  |
| [`jnp.einsum("ij,j -> ji", x, w)`][jax.numpy.einsum]           | [`hax.dot(x, w, axis=(), out_axes=("embed", "batch")`][haliax.dot] |
| [`jnp.einsum("bhwc,ce -> bhwe", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("b h w c, c e -> b h w e", im, w2)`][haliax.einsum]   |
| [`jnp.einsum("...c,ce -> ...e", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("... c, c e -> ... e", im, w2)`][haliax.einsum]       |
| [`jnp.einsum("bhwc,ce -> bhwe", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("{c embed} -> embed", im, w2)`][haliax.einsum]        |
| [`jnp.einsum("bhwc,ce -> bhwe", im, w2)`][jax.numpy.einsum]    | [`hax.einsum("-> batch h w embed", im, w2)`][haliax.einsum]        |
| [`jnp.einsum("bhwc,ce -> bhwce", im, w2)`][jax.numpy.einsum]   | [`hax.einsum("{...} -> ...", im, w2)`][haliax.einsum]              |
| [`jnp.einsum("bhwc,ce -> ", im, w2)`][jax.numpy.einsum]        | [`hax.einsum("{...} -> ", im, w2)`][haliax.einsum]                 |
| [`jnp.einsum("bhwc,ce -> bhwce", im, w2)`][jax.numpy.einsum]   | [`hax.dot(im, w2, axis=())`][haliax.dot]                           |
| [`jnp.einsum("bhwc,ce -> ", im, w2)`][jax.numpy.einsum]        | [`hax.dot(im, w2, axis=None)`][haliax.dot]                         |
