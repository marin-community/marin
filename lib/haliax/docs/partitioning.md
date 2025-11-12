# Partitioning

Partitioning refers to the process of splitting arrays and computation across multiple devices. Haliax provides a number
of functions for partitioning arrays and computation across multiple devices.


## Tutorial
An introduction to using Haliax's partitioning functions to scale a transformer can be found here: [Distributed Training in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz).

This page is designed to be more of a reference than a tutorial, and we assume you've read the tutorial before reading this page.


## Device Meshes in JAX

See also JAX's tutorial [Distributed Arrays and Automatic Parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
for more details.

One of the main ways JAX provides distributed parallelism is via the [jax.sharding.Mesh][].
A mesh is a logical n-dimensional array of devices. Meshes in JAX are represented as a Numpy ndarray (note: not `jax.numpy`)
of devices and a tuple of axis names. For example, a 2D mesh of 16 devices might look like this:

```python
import jax
import jax.numpy as jnp

from jax.sharding import Mesh

devices = jax.devices()
mesh = Mesh(jnp.array(devices).reshape((-1, 2)), ("data", "model"))
```

![2d Device Mesh showing 16 devices](figures/device_mesh_2d.png)

The mesh above has two axes, `data` and `model`. In JAX's mesh parallelism, arrays are distributed by overlaying axes of
the array on top of the axes of the mesh. For example, if we have a batch of 32 sequences we might do something like this:

```python
from jax.sharding import NamedSharding, PartitionSpec

batch_size = 32
seqlen = 512

batch = jnp.zeros((batch_size, seqlen), dtype=jnp.float32)
batch = jax.device_put(batch, NamedSharding(mesh, PartitionSpec("data", None)))
```

This specifies that the first axis of `batch` should be distributed across the `data` axis of the mesh. The `None` in the
`PartitionSpec` indicates that the second axis of `batch` is not distributed, which means that the data is replicated
so that one copy of the data is partitioned across each row of the mesh.

![Device Mesh showing 16 devices with data partitioned across data axis](figures/device_mesh_2d_batch_partitioned.png)

What's nice about this approach is that jax will automatically schedule computations so that operations are distributed
in the way you would expect: you don't have to explicitly manage communication between devices.

However, JAX sometimes gets confused, and it's not sure how you want your arrays partitioned. In Jax, there's a function
called [jax.lax.with_sharding_constraint][] that lets you explicitly specify the sharding for the outputs of arrays.
You use this function only inside `jit`.

## Haliax Partitioning in a nutshell

As you might imagine, it gets tedious and error-prone to have to specify the partitioning of every array you create. Haliax provides
routines to handle mapping of [haliax.NamedArray][]s automatically.

```python
import haliax as hax

Batch = hax.Axis("batch", 32)
SeqLen = hax.Axis("seqlen", 512)

axis_mapping = {"batch": "data", }

batch = hax.zeros((Batch, SeqLen), dtype=jnp.float32)
batch = hax.shard(batch, axis_mapping)

# we also have "auto_sharded" and support context mappings for axis mappings:
with hax.axis_mapping({"batch": "data"}):
    batch = hax.zeros((Batch, SeqLen), dtype=jnp.float32)
    batch = hax.shard(batch)
```

Unlike in JAX, which has separate APIs for partitioning arrays inside and outside of `jit`, Haliax has a single API:
[haliax.shard][] works inside and outside of `jit`. Haliax automatically
chooses which JAX function to use based on context.


## Axis Mappings

The core data structure we use to represent partitioning is the [haliax.partitioning.ResourceMapping][] which
is just an alias for a `Dict[str, str|Sequence[str]]`. The keys in this dictionary are the names of "logical" Axes in NamedArrays
and the values are the names of axes in the mesh. (In theory you can partition a single Axis across multiple axes in the mesh,
but we don't use this functionality.)

::: haliax.partitioning.ResourceMapping

A context manager can be used to specify an axis mapping for the current thread for the duration of the context:

```python
with hax.axis_mapping({"batch": "data"}):
    batch = hax.zeros((Batch, SeqLen), dtype=jnp.float32)
    batch = hax.auto_sharded(batch)
```

::: haliax.partitioning.axis_mapping

## Partitioning Functions

### Sharding Arrays and PyTrees

These functions are used to shard arrays and PyTrees of arrays, e.g. Modules.
This is the main function you will use to shard arrays:

::: haliax.shard

This function is like `shard` but does not issue a warning if there is no context axis mapping.
It's useful for library code where there may or may not be a context mapping:

::: haliax.auto_sharded

This is an older function that is being deprecated in favor of `shard`. It is functionally equivalent to `shard`:

::: haliax.shard_with_axis_mapping

### `named_jit` and friends

::: haliax.named_jit
::: haliax.fsdp


### Querying the Mesh and Axis Mapping


::: haliax.partitioning.round_axis_for_partitioning
::: haliax.partitioning.physical_axis_name
::: haliax.partitioning.physical_axis_size
::: haliax.partitioning.sharding_for_axis
