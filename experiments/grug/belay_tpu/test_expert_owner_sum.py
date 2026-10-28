# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify source-dtype exchange ownership and the full gather VJP."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.belay_tpu.expert_collectives import (
    all_to_all_reduce_scatter,
    gather_expert_weights_with_owner_sum,
)


@pytest.mark.parametrize("peers", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_owner_sum_and_gather_vjp(peers, dtype):
    if jax.local_device_count() < peers:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=16")
    mesh = Mesh(np.asarray(jax.local_devices()[:peers]), ("expert",))
    spec = P("expert", None, None)
    sharding = NamedSharding(mesh, spec)
    rng = np.random.default_rng(433)
    contributions = rng.normal(size=(peers, 48, 8, 128)).astype(dtype)
    # 1024 + 1 - 1024 exposes premature BF16 accumulation for four or more senders.
    contributions[:, 0, 0, 0] = 0
    contributions[0, 0, 0, 0] = 1024
    contributions[1, 0, 0, 0] = 1
    if peers >= 4:
        contributions[2, 0, 0, 0] = -1024
    expected = contributions.astype(np.float64).sum(axis=0).astype(np.float32)
    cotangent = jax.device_put(contributions.reshape(peers * 48, 8, 128), sharding)
    owner_sum = jax.jit(
        jax.shard_map(
            all_to_all_reduce_scatter,
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=False,
        )
    )
    result = owner_sum(cotangent)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-6, atol=2e-6)

    weights = rng.normal(size=(48, 8, 128)).astype(dtype)
    gather = jax.jit(
        jax.shard_map(
            gather_expert_weights_with_owner_sum,
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=False,
        )
    )
    gathered, backward = jax.vjp(gather, jax.device_put(weights, sharding))
    np.testing.assert_array_equal(np.asarray(gathered), np.tile(weights, (peers, 1, 1)))
    actual = backward(cotangent)[0]
    assert actual.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(actual).astype(np.float32),
        expected.astype(dtype).astype(np.float32),
        rtol=2e-6,
        atol=2e-6,
    )
