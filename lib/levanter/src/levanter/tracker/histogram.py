# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.state.indexing import dslice
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from jaxtyping import ArrayLike, Scalar

import haliax as hax
from haliax import NamedArray
from haliax.partitioning import shard_map
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


class Histogram(equinox.Module):
    """Bucket payload for a summary statistic."""

    bucket_limits: jax.Array
    bucket_counts: jax.Array

    def to_numpy_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.bucket_counts), np.array(self.bucket_limits)


class SummaryStats(equinox.Module):
    """Summary statistics for an array, optionally including a histogram."""

    min: Scalar
    max: Scalar
    num: Scalar
    nonzero_count: Scalar
    sum: Scalar
    sum_squares: Scalar
    histogram: Histogram | None = None

    @staticmethod
    def from_array(
        array: jax.Array,
        num_bins: int = 31,
        *,
        include_histogram: bool = True,
        min_value: Scalar | None = None,
        max_value: Scalar | None = None,
    ) -> "SummaryStats":
        return SummaryStats.from_sharded_array(
            array,
            num_bins=num_bins,
            include_histogram=include_histogram,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def from_sharded_array(
        array: jax.Array,
        num_bins: int = 31,
        *,
        include_histogram: bool = True,
        min_value: Scalar | None = None,
        max_value: Scalar | None = None,
    ) -> "SummaryStats":
        array = array.ravel()
        min = array.min() if min_value is None else jnp.asarray(min_value, dtype=array.dtype)
        max = array.max() if max_value is None else jnp.asarray(max_value, dtype=array.dtype)
        nonzero_count = jnp.count_nonzero(array)
        num = jnp.asarray(array.size, dtype=nonzero_count.dtype)
        sum = array.sum()
        sum_squares = (array**2).sum()
        histogram = None
        if include_histogram:
            edges = jnp.histogram_bin_edges(jnp.stack([min, max]), bins=num_bins)
            counts = sharded_histogram_array(array, edges)
            histogram = Histogram(edges, counts)
        return SummaryStats(min, max, num, nonzero_count, sum, sum_squares, histogram)

    @staticmethod
    def from_named_array(
        array: hax.NamedArray,
        num_bins: int = 31,
        *,
        include_histogram: bool = True,
    ) -> "SummaryStats":
        raw_array = array.array
        min = raw_array.min()
        max = raw_array.max()
        nonzero_count = jnp.count_nonzero(raw_array)
        num = jnp.asarray(array.size, dtype=nonzero_count.dtype)
        sum = raw_array.sum()
        sum_squares = (raw_array**2).sum()
        histogram = None
        if include_histogram:
            counts, edges = sharded_histogram(array, bins=num_bins)
            histogram = Histogram(edges, counts)
        return SummaryStats(min, max, num, nonzero_count, sum, sum_squares, histogram)

    @property
    def mean(self) -> Scalar:
        return self.sum / self.num

    @property
    def variance(self) -> Scalar:
        """Calculate the variance as E[X^2] - (E[X])^2."""
        return (self.sum_squares / self.num) - (self.mean**2)

    @property
    def rms(self) -> Scalar:
        return jnp.sqrt(self.sum_squares / self.num)

    def to_numpy_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        if self.histogram is None:
            raise ValueError("SummaryStats does not include a histogram")
        return self.histogram.to_numpy_histogram()


def sharded_histogram(a: NamedArray, bins: int | ArrayLike = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Histogram for a NamedArray using its logical axis mapping to pick shard reductions."""
    edges = jnp.histogram_bin_edges(a.array, bins=bins)
    return _shardmap_histogram(a, edges), edges


def sharded_histogram_array(a: jax.Array, bin_edges: ArrayLike) -> jnp.ndarray:
    spec = _array_spec(a)
    flattened_spec = _flattened_spec(spec)

    if len(flattened_spec) == 0:
        return _single_shard_histogram_array(a, bin_edges, ())

    def _wrapped_hist(arr, edges):
        return _single_shard_histogram_array(arr, bin_edges=edges, reduce_mesh=flattened_spec)

    return jax.shard_map(
        _wrapped_hist,
        in_specs=(spec, P(None)),
        out_specs=P(None),
        check_vma=False,
    )(a, bin_edges)


def _single_shard_histogram(a: NamedArray, bin_edges, reduce_mesh):
    """Histogram counts for one NamedArray shard using logical axis mapping."""
    a_flat = a.array.flatten()
    left_edges = bin_edges[:-1, None]
    right_edges = bin_edges[1:, None]
    is_last_bin = (jnp.arange(bin_edges.shape[0] - 1) == (bin_edges.shape[0] - 2))[:, None]

    a_exp = a_flat[None, :]
    in_bin = (a_exp >= left_edges) & ((a_exp < right_edges) | (is_last_bin & (a_exp <= right_edges)))
    counts = in_bin.sum(axis=1, dtype=jnp.int32)

    if len(reduce_mesh):
        counts = jax.lax.psum(counts, axis_name=reduce_mesh)
    return counts


def _single_shard_histogram_array(a: jax.Array, bin_edges, reduce_mesh):
    """Histogram counts for one shard with the last bin inclusive."""
    a = a.flatten()
    num_bins = bin_edges.shape[0] - 1
    orig_len = a.shape[0]
    padded_len = ((orig_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    pad_len = padded_len - orig_len

    if pad_len > 0:
        a = jnp.pad(a, (0, pad_len))

    left_edges = bin_edges[:-1][:, None]
    right_edges = bin_edges[1:][:, None]
    is_last_bin = (jnp.arange(num_bins) == (num_bins - 1))[:, None]

    def _body(tile_index: int, running_counts: jax.Array) -> jax.Array:
        start = tile_index * TILE_SIZE
        tile = jax.lax.dynamic_slice_in_dim(a, start, TILE_SIZE, axis=0)
        valid = (start + jnp.arange(TILE_SIZE)) < orig_len
        tile_exp = tile[None, :]
        in_bin = (tile_exp >= left_edges) & ((tile_exp < right_edges) | (is_last_bin & (tile_exp <= right_edges)))
        return running_counts + (in_bin & valid[None, :]).sum(axis=1, dtype=jnp.int32)

    counts = jax.lax.fori_loop(0, padded_len // TILE_SIZE, _body, jnp.zeros((num_bins,), dtype=jnp.int32))

    if len(reduce_mesh):
        counts = jax.lax.psum(counts, axis_name=reduce_mesh)
    return counts


def _shardmap_histogram(a: NamedArray, bins):
    spec = hax.partitioning.pspec_for_axis(a.axes)
    flattened_spec = _flattened_spec(spec)

    def _wrapped_hist(arr):
        return _single_shard_histogram(arr, bin_edges=bins, reduce_mesh=flattened_spec)

    shard_h = shard_map(_wrapped_hist)
    return shard_h(a)


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


def _array_spec(a: jax.Array) -> P:
    abstract = jax.typeof(a)
    sharding = getattr(abstract, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None:
        return spec

    sharding = getattr(a, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None:
        return spec

    return P(*([None] * a.ndim))


TILE_SIZE = 1024  # Can tune based on memory pressure


def _histogram_cost_reference(a: jax.Array, bin_edges: jax.Array) -> jax.Array:
    # Mirror tile-wise histogram math without `pl.program_id` so estimate_cost can trace it.
    a_tiled = a.reshape((-1, TILE_SIZE))
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    in_bin = (a_tiled[..., None] >= left_edges[None, None, :]) & (a_tiled[..., None] < right_edges[None, None, :])
    return in_bin.sum(axis=(0, 1), dtype=jnp.int32)


def _histogram_cost_estimate(
    a: jax.Array,
    bin_edges: jax.Array,
    *,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(_histogram_cost_reference, a, bin_edges)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )


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
    out_shape = jax.ShapeDtypeStruct((num_bins,), jnp.int32)

    return pl.pallas_call(
        histogram_tile_kernel,
        out_shape=out_shape,
        in_specs=[
            pl.BlockSpec((TILE_SIZE,), lambda i: (i * TILE_SIZE,)),  # Each kernel gets one tile
            pl.BlockSpec(bin_edges.shape, lambda i: (0,)),  # bin_edges shared to all
        ],
        out_specs=pl.BlockSpec((num_bins,), lambda i: (0,)),  # Shared counts across all tiles
        grid=(num_tiles,),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=["arbitrary"]  # Ensure sequential grid (needed for += safety)
        ),
        cost_estimate=_histogram_cost_estimate(
            a,
            bin_edges,
            kernel_inputs_specs=(a, bin_edges),
            kernel_outputs_specs=out_shape,
        ),
        interpret=True,
    )(a, bin_edges)
