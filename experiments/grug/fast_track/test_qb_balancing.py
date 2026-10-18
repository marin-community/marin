# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Histogram counts and distributed router thresholds against NumPy oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.qb_balancing import factorized_expert_histogram, local_expert_histogram, qb_beta_hist


@pytest.mark.parametrize("bins", [31, 10000])
@pytest.mark.parametrize("concentrated", [False, True])
def test_grouped_histogram_counts(bins, concentrated):
    rng = np.random.default_rng(0)
    indices = rng.integers(0, bins, (257, 128), dtype=np.int32)
    if concentrated:
        indices = np.where(indices < bins // 2, 0, bins - 1).astype(np.int32)
    expected = np.stack([np.bincount(column, minlength=bins) for column in indices.T])
    for group in (None, 1, 8, 32, 128):
        actual = jax.jit(lambda x, group=group: local_expert_histogram(x, bins, group))(jnp.asarray(indices))
        np.testing.assert_array_equal(np.asarray(actual), expected)
    for group in (16, 64):
        for radix in (64, 128, 256):
            actual = jax.jit(lambda x, group=group, radix=radix: factorized_expert_histogram(x, bins, group, radix))(
                jnp.asarray(indices)
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("local_tokens", [8192,32768])
@pytest.mark.parametrize("constant", [False, True])
def test_qb_thresholds_on_tpu_v4(constant,local_tokens):
    """Exercise the full local histogram shape and cross-chip integer reduction."""
    devices = jax.local_devices()
    if devices[0].platform != "tpu" or devices[0].device_kind != "TPU v4":
        pytest.skip("The tuned histogram path requires TPU v4")
    experts, bins = 512, 10000
    tokens = local_tokens * len(devices)
    rng = np.random.default_rng(3)
    margins = rng.integers(-2048, 2048, (tokens, experts), dtype=np.int32).astype(np.float32)
    if constant:
        margins.fill(0)
    share = np.linspace(1 / experts, 8 / experts, experts, dtype=np.float32)
    lo, hi = margins.min(), margins.max()
    upper = np.maximum(hi, lo + np.float32(1e-6))
    width = (upper - lo) / np.float32(bins)
    indices = np.clip(((margins - lo) / width).astype(np.int32), 0, bins - 1)
    expected = []
    for expert, column in enumerate(indices.T):
        counts = np.bincount(column, minlength=bins)
        cumulative = np.cumsum(counts[::-1])[::-1]
        rank = np.float32(tokens) * share[expert]
        crossing = np.clip(np.count_nonzero(cumulative >= rank) - 1, 0, bins - 1)
        expected.append(lo + crossing * width + width * (cumulative[crossing] - rank) / max(counts[crossing], 1))
    del indices
    mesh = Mesh(np.asarray(devices), ("batch",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        sharded = jax.device_put(margins, NamedSharding(mesh, P("batch", None)))
        actual, actual_lo, actual_hi = jax.jit(
            lambda x: qb_beta_hist(
                x,
                jax.sharding.get_abstract_mesh(),
                target_share=jnp.asarray(share),
                num_experts=experts,
                n_bins=bins,
                batch_axes=("batch",),
            )
        )(sharded)
    # Float32 interpolation of thresholds spanning [-2048, 2047]. The integer
    # count equality has zero tolerance in the separate histogram test.
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=2e-17 if constant else 2e-4)
    np.testing.assert_array_equal(np.asarray(actual_lo), lo)
    np.testing.assert_array_equal(np.asarray(actual_hi), hi)
