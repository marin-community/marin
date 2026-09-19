# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep.grugmuon_hero import (
    _newtonschulz_padded_stack_sharded,
    _zeropower_via_newtonschulz_local,
)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple CPU or GPU devices")
def test_muon_expert_stack_preserves_sharding_and_updates():
    devices = np.asarray(jax.devices())
    mesh = Mesh(
        devices.reshape(1, 1, -1, 1), ("replica_dcn", "data", "expert", "model"), axis_types=(AxisType.Explicit,) * 4
    )
    sharding = NamedSharding(mesh, P("expert", None, None))
    values = np.random.default_rng(0).normal(size=(len(devices) * 2, 8, 4)).astype(np.float32)
    reference = jax.jit(jax.vmap(_zeropower_via_newtonschulz_local))(jnp.asarray(values, dtype=jnp.bfloat16))
    with jax.set_mesh(mesh):
        matrices = jax.device_put(jnp.asarray(values, dtype=jnp.bfloat16), sharding)
        actual = jax.jit(lambda x: _newtonschulz_padded_stack_sharded(x, target_sharding=sharding))(matrices)
    assert actual.sharding == sharding
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
