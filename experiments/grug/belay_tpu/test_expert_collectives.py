# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check expert ownership and gradient sums with independent NumPy reductions."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.belay_tpu.expert_collectives import all_to_all_reduce_scatter, butterfly_reduce_scatter


@pytest.mark.parametrize("peers", [2, 4, 8])
@pytest.mark.parametrize("reduce", [butterfly_reduce_scatter, all_to_all_reduce_scatter])
def test_expert_reduction_matches_sum_and_destination(peers, reduce):
    if jax.local_device_count() < peers:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=8")
    mesh = Mesh(np.asarray(jax.local_devices()[:peers]), ("expert",))
    rng = np.random.default_rng(294)
    contributions = rng.normal(size=(peers, 24, 8, 128)).astype(np.float32)
    # Distinguish every sender and destination; cancellation checks reduction order.
    contributions[:, 0] *= np.asarray([(-1) ** i for i in range(peers)])[:, None, None]
    expected = contributions.astype(np.float64).sum(axis=0).astype(np.float32)
    flat = contributions.reshape(peers * 24, 8, 128)
    inputs = jax.device_put(flat, NamedSharding(mesh, P("expert", None, None)))
    function = jax.jit(
        jax.shard_map(reduce, mesh=mesh, in_specs=P("expert", None, None), out_specs=P("expert", None, None))
    )
    actual = np.asarray(function(inputs))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
