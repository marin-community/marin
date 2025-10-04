# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.state.indexing import dslice
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from jaxtyping import ArrayLike, Scalar

import haliax as hax
from haliax import NamedArray


class Histogram(equinox.Module):
    """
    Has enough information to log to tensorboard and wandb
    """

    min: Scalar
    max: Scalar
    num: Scalar | int
    sum: Scalar
    sum_squares: Scalar
    bucket_limits: jax.Array
    bucket_counts: jax.Array

    @staticmethod
    def from_array(array: jax.Array, num_bins: int = 31) -> "Histogram":
        array = array.ravel()
        min = array.min()
        max = array.max()
        num = array.size
        sum = array.sum()
        sum_squares = (array**2).sum()
        counts, edges = jax.numpy.histogram(array, bins=num_bins)
        return Histogram(min, max, num, sum, sum_squares, edges, counts)

    @staticmethod
    def from_named_array(array: hax.NamedArray, num_bins: int = 31) -> "Histogram":
        raw_array = array.array
        min = raw_array.min()
        max = raw_array.max()
        num = array.size
        sum = raw_array.sum()
        sum_squares = (raw_array**2).sum()
        counts, edges = sharded_histogram(array, bins=num_bins)
        return Histogram(min, max, num, sum, sum_squares, edges, counts)

    def to_numpy_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.bucket_counts), np.array(self.bucket_limits)

    @property
    def mean(self) -> Scalar:
        return self.sum / self.num

    @property
    def variance(self) -> Scalar:
        """
        Calculate the variance of the histogram.
        Variance = E[X^2] - (E[X])^2
        where E[X] is the mean and E[X^2] is the mean of squares.
        """
        mean = self.mean
        mean_of_squares = self.sum_squares / self.num
        variance = mean_of_squares - (mean**2)
        return variance


def sharded_histogram(a: NamedArray, bins: int | ArrayLike = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    As [jax.numpy.histogram](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.histogram.html#jax.numpy.histogram),
    except:

    * It preserves sharding
    * It only works with NamedArrays
    * It is more performant on TPUs

    Credit to @aphoh for the original implementation, though that one crashes on TPUs due to some kind of driver bug
    """
    edges = jnp.histogram_bin_edges(a.array, bins=bins)
    return _shardmap_histogram(a, edges), edges


def _single_shard_histogram(a, bin_edges, reduce_mesh):
    """Modified version of jax.numpy.histogram that returns integer counts instead of using the datatype of the input.
    Also avoids searchsorted, which is slow on TPUs.
    Args:
        a (Array): input array
        bin_edges (Array): bins to use for histogram
    Returns:
        Array: counts. has length len(bins) - 1
    """
    a = a.flatten()
    dtype = a.dtype

    a_exp = a[None, :]  # shape: (1, D)
    left_edges = bin_edges[:-1, None]  # shape: (N, 1)
    right_edges = bin_edges[1:, None]  # shape: (N, 1)

    # now bin_idx will be shape (N, D)
    bin_idx = ((a_exp >= left_edges) & (a_exp < right_edges)).astype(dtype)
    counts = bin_idx.sum(axis=1, dtype=dtype)

    # bin_idx = jnp.searchsorted(bin_edges, a, side='right', method='compare_all')
    # bin_idx = jnp.where(a == bin_edges[-1], len(bin_edges) - 1, bin_idx)
    # counts = jnp.zeros(len(bin_edges), a.dtype).at[bin_idx].add(1.0)[1:]

    # pallas histogram
    # counts = histogram_large_a(a, bin_edges)

    if len(reduce_mesh):
        counts = jax.lax.psum(counts, axis_name=reduce_mesh)
    return counts


def _shardmap_histogram(a: NamedArray, bins):
    mesh = hax.partitioning._get_mesh()
    spec = hax.partitioning.pspec_for_axis(a.axes)
    flattened_spec = _flattened_spec(spec)
    shard_h = shard_map(
        functools.partial(_single_shard_histogram, reduce_mesh=flattened_spec),
        mesh=mesh,
        in_specs=(spec, PartitionSpec(None)),
        out_specs=PartitionSpec(
            None,
        ),
        check_rep=False,
    )
    res = shard_h(a.array, bins)

    # the filter misses the last bin, so we need to add it
    if res.size >= 1:
        res = res.at[-1].add(1)
    return res


def _flattened_spec(spec):
    out = []
    for s in spec:
        if isinstance(s, tuple):
            out.extend(s)
        elif s is None:
            pass
        else:
            out.append(s)

    return tuple(out)


TILE_SIZE = 1024  # Can tune based on memory pressure


def histogram_tile_kernel(a_ref, bin_edges_ref, counts_ref):
    @pl.when(pl.program_id(0) == 0)
    def _():
        counts_ref[...] = jnp.zeros_like(counts_ref)

    pid = pl.program_id(0)
    start = pid * TILE_SIZE
    # end = start + TILE_SIZE

    # Load tile of a
    a_tile = a_ref[dslice(start, TILE_SIZE)]
    bin_edges = bin_edges_ref[...]

    # Compute which bin each a_tile[i] belongs to
    # (TILE_SIZE, num_bins)
    in_bin = (a_tile[:, None] >= bin_edges[:-1][None, :]) & (a_tile[:, None] < bin_edges[1:][None, :])

    # Sum over axis 0 → shape: (num_bins,)
    bin_counts = in_bin.sum(axis=0)

    # Accumulate into output counts (safe since grid is sequential)
    counts_ref[...] += bin_counts


def histogram_large_a(a: jax.Array, bin_edges: jax.Array) -> jax.Array:
    num_bins = bin_edges.shape[0] - 1
    padded_len = ((a.shape[0] + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    # Pad a if needed to make all tiles full
    if padded_len > a.shape[0]:
        pad_len = padded_len - a.shape[0]
        a = jnp.pad(a, (0, pad_len), constant_values=jnp.inf)  # inf ensures they don’t fall into any bin

    num_tiles = padded_len // TILE_SIZE

    return pl.pallas_call(
        histogram_tile_kernel,
        out_shape=jax.ShapeDtypeStruct((num_bins,), jnp.int32),
        in_specs=[
            pl.BlockSpec((TILE_SIZE,), lambda i: (i * TILE_SIZE,)),  # Each kernel gets one tile
            pl.BlockSpec(bin_edges.shape, lambda i: (0,)),  # bin_edges shared to all
        ],
        out_specs=pl.BlockSpec((num_bins,), lambda i: (0,)),  # Shared counts across all tiles
        grid=(num_tiles,),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=["arbitrary"]  # Ensure sequential grid (needed for += safety)
        ),
        interpret=True,
    )(a, bin_edges)
