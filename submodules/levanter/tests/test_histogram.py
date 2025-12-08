# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import equinox
import jax
from jax.random import PRNGKey
import haliax as hax
from haliax.partitioning import ResourceAxis

import levanter.tracker.histogram
from test_utils import use_test_mesh, skip_if_not_enough_devices


def test_sharded_histogram_simple():

    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with use_test_mesh(), hax.axis_mapping({"batch": ResourceAxis.DATA}):
        a = hax.random.normal(PRNGKey(1), (Batch, Feature))
        a = hax.shard(a)
        hist, bins = levanter.tracker.histogram.sharded_histogram(a, bins=32)

    hist_normal, bins_normal = jax.numpy.histogram(a.array, bins=32)

    assert jax.numpy.allclose(hist, hist_normal)
    assert jax.numpy.allclose(bins, bins_normal)


@skip_if_not_enough_devices(2)
def test_sharded_histogram_tp():
    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with (
        use_test_mesh(tensor_parallelism=2),
        hax.axis_mapping({"batch": ResourceAxis.DATA, "feature": ResourceAxis.MODEL}),
    ):
        a = hax.random.normal(PRNGKey(0), (Batch, Feature)) * 100
        a = hax.shard(a)
        hist, bins = levanter.tracker.histogram.sharded_histogram(a, bins=64)

    jnp_hist, jnp_bins = jax.numpy.histogram(a.array, bins=64)

    assert jax.numpy.allclose(hist, jnp_hist)
    assert jax.numpy.allclose(bins, jnp_bins)


def test_sharded_histogram_with_vmap():
    Layer = hax.Axis("layer", 4)
    Batch = hax.Axis("batch", 16)
    Feature = hax.Axis("feature", 128)

    @equinox.filter_jit
    def jit_vmap_hist(a):
        """
        This function will be JIT compiled and VMapped.
        """
        # Call the sharded histogram function
        hist, bins = hax.vmap(levanter.tracker.histogram.sharded_histogram, Layer)(a, bins=32)
        return hist, bins

    with use_test_mesh(), hax.axis_mapping({"batch": ResourceAxis.DATA}):
        a = hax.random.normal(PRNGKey(1), (Layer, Batch, Feature))
        a = hax.shard(a)
        hist, bins = jit_vmap_hist(a)

    hist_normal, bins_normal = jax.vmap(functools.partial(jax.numpy.histogram, bins=32), in_axes=0)(a.array)

    assert jax.numpy.allclose(hist, hist_normal)
    assert jax.numpy.allclose(bins, bins_normal)
