# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-layer routing histogram fits W&B's bucket limit by pooling adjacent experts."""

import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.router_metrics import _MAX_HISTOGRAM_BUCKETS, _histogram_from_expert_counts


def test_more_experts_than_buckets_pool_adjacent_counts():
    counts = jnp.arange(1024, dtype=jnp.float32)
    stats = _histogram_from_expert_counts(counts)
    hist_counts, limits = stats.histogram.to_numpy_histogram()
    assert len(hist_counts) == _MAX_HISTOGRAM_BUCKETS
    np.testing.assert_allclose(hist_counts, np.arange(1024).reshape(512, 2).sum(1))
    assert limits[0] == 0 and limits[-1] == 1024
    assert float(stats.num) == float(counts.sum())


def test_small_expert_counts_keep_one_bucket_per_expert():
    hist_counts, limits = _histogram_from_expert_counts(jnp.ones((384,))).histogram.to_numpy_histogram()
    assert len(hist_counts) == 384 and len(limits) == 385
