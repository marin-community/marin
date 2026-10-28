# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integer routing-control checks only; no training or optimizer updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.qb_balancing import bincount_upper_quantile
from experiments.grug.belay_tpu.qb_hist_tpu import bincount_upper_quantile_tpu, factorized_histogram


@pytest.mark.parametrize("bins", [1, 63, 64, 65, 10000])
def test_factorized_counts_include_tail_and_exclude_negative_indices(bins):
    rng = np.random.RandomState(91)
    indices = rng.randint(-1, bins, (131, 8), dtype=np.int32)
    indices[:3] = np.asarray([-1, 0, bins - 1])[:, None]
    expected = np.stack([np.bincount(x[x >= 0], minlength=bins) for x in indices.T])
    actual = jax.jit(functools.partial(factorized_histogram, bins=bins, group_size=8, radix=8))(indices)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bins", [1, 63, 64, 65, 10000])
@pytest.mark.parametrize("scenario", ["random", "equal", "edges"])
def test_two_stage_global_quantile_is_exact(bins, scenario):
    rng = np.random.RandomState(92)
    margins = rng.uniform(-3, 3, (2, 131, 8)).astype(np.float32)
    if scenario == "equal":
        margins.fill(0)
    elif scenario == "edges":
        margins = np.broadcast_to(np.asarray([-3, -1, 0, 1, 3, 0.99, -0.99, 0], np.float32), margins.shape)
    ranks = jnp.asarray([0, 1, 5.25, 131, 261, 262, 263, 7.5], jnp.float32)

    def run(kernel, s):
        lo = jax.lax.pmin(jnp.min(s), "batch")
        hi = jnp.maximum(jax.lax.pmax(jnp.max(s), "batch"), lo + 1e-6)
        return kernel(s, num_experts=8, n_bins=bins, lo=lo, hi=hi, target_rank=ranks, batch_axes=("batch",))

    expected = jax.jit(jax.vmap(functools.partial(run, bincount_upper_quantile), axis_name="batch"))(margins)
    candidate = functools.partial(bincount_upper_quantile_tpu, group_size=8)
    actual = jax.jit(jax.vmap(functools.partial(run, candidate), axis_name="batch"))(margins)
    np.testing.assert_array_equal(actual, expected)
