# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import equinox
import haliax as hax
from haliax.partitioning import ResourceAxis
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from levanter.tracker.histogram import SummaryStats, sharded_histogram_array
from levanter.tracker.json_logger import _to_jsonable
from test_utils import use_test_mesh


def test_sharded_histogram_array_matches_jnp_histogram():
    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with use_test_mesh(), hax.axis_mapping({Batch.name: ResourceAxis.DATA}):
        array = hax.shard(hax.random.normal(PRNGKey(1), (Batch, Feature))).array
        bin_edges = jnp.histogram_bin_edges(array, bins=32)

        counts = sharded_histogram_array(array, bin_edges)
        expected_counts, expected_edges = jnp.histogram(array, bins=bin_edges)

        assert jnp.array_equal(counts, expected_counts)
        assert jnp.allclose(bin_edges, expected_edges)


def test_summary_stats_from_named_array_matches_jnp_histogram():
    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with use_test_mesh(), hax.axis_mapping({Batch.name: ResourceAxis.DATA}):
        array = hax.shard(hax.random.normal(PRNGKey(3), (Batch, Feature)))
        stats = SummaryStats.from_named_array(array)

        assert stats.histogram is not None
        expected_counts, expected_edges = jnp.histogram(array.array, bins=31)

        assert jnp.array_equal(stats.histogram.bucket_counts, expected_counts)
        assert jnp.allclose(stats.histogram.bucket_limits, expected_edges)


def test_sharded_histogram_array_matches_jnp_histogram_under_vmap():
    Layer = hax.Axis("layer", 4)
    Batch = hax.Axis("batch", 16)
    Feature = hax.Axis("feature", 128)

    @equinox.filter_jit
    def vmapped_histogram(array):
        def histogram_for_layer(layer_array):
            bin_edges = jnp.histogram_bin_edges(layer_array, bins=32)
            counts = sharded_histogram_array(layer_array, bin_edges)
            return counts, bin_edges

        return jax.vmap(histogram_for_layer)(array)

    with use_test_mesh(), hax.axis_mapping({Batch.name: ResourceAxis.DATA}):
        array = hax.shard(hax.random.normal(PRNGKey(2), (Layer, Batch, Feature))).array
        counts, bin_edges = vmapped_histogram(array)
        expected_counts, expected_edges = jax.vmap(functools.partial(jnp.histogram, bins=32), in_axes=0)(array)

        assert jnp.array_equal(counts, expected_counts)
        assert jnp.allclose(bin_edges, expected_edges)


def test_histogram_tracks_nonzero_count_and_json_serializes_it():
    histogram = SummaryStats.from_array(jnp.array([0.0, 1.0, -2.0, 0.0], dtype=jnp.float32))

    assert int(histogram.nonzero_count) == 2
    assert histogram.histogram is not None

    serialized = _to_jsonable(histogram)

    assert serialized["nonzero_count"] == 2
    assert serialized["histogram"]["bucket_counts"]


def test_histogram_counts_include_values_on_the_last_bin_edge():
    histogram = SummaryStats.from_array(jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32), num_bins=2)

    assert histogram.histogram is not None
    assert int(histogram.histogram.bucket_counts.sum()) == 3
    assert int(histogram.num) == 3


def test_summary_stats_can_skip_histogram_bins():
    stats = SummaryStats.from_array(jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32), include_histogram=False)

    serialized = _to_jsonable(stats)

    assert stats.histogram is None
    assert serialized["histogram"] is None


def test_summary_stats_num_is_a_jax_scalar_and_unstacks_cleanly():
    stats = SummaryStats.from_array(jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32), include_histogram=False)

    assert jnp.asarray(stats.num).shape == ()
    assert jnp.asarray(stats.nonzero_count).shape == ()

    stacked_stats = jax.vmap(
        lambda row: SummaryStats.from_array(row, include_histogram=False),
    )(jnp.array([[0.0, 1.0, 0.0], [2.0, 3.0, 4.0]], dtype=jnp.float32))

    unstacked = tuple(jax.tree.map(lambda value: value[i], stacked_stats) for i in range(stacked_stats.num.shape[0]))

    assert len(unstacked) == 2
    assert int(unstacked[0].num) == 3
    assert int(unstacked[0].nonzero_count) == 1
    assert int(unstacked[1].num) == 3
    assert int(unstacked[1].nonzero_count) == 3
