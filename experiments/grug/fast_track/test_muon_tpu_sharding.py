# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-sharded Muon directions agree with per-matrix orthogonalization."""

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon


@pytest.mark.parametrize("coefficient_type,steps", [("quintic", 5), ("aol", 4)])
@pytest.mark.parametrize("matrix_shape", [(64, 128), (128, 64)])
def test_expert_sharded_muon_matches_unsharded(coefficient_type, steps, matrix_shape):
    count = len(jax.devices())
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, count, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    shape = (2, 2 * count, *matrix_shape)
    params = {"expert": jax.random.normal(jax.random.key(0), shape)}
    grads = {"expert": jax.random.normal(jax.random.key(1), shape)}
    transform = _grug_scale_with_muon(momentum=0.95, nesterov=True, steps=steps, coefficient_type=coefficient_type)
    # Match the TPU training flag: bf16 casts must not be elided differently
    # when a tall matrix is transposed before or inside the sharded function.
    update = jax.jit(transform.update, compiler_options={"xla_allow_excess_precision": False})
    expected, _ = update(grads, transform.init(params), params)
    sharding = NamedSharding(mesh, P(None, "expert", None, None))
    params, grads = jax.tree.map(lambda x: jax.device_put(x, sharding), (params, grads))
    with jax.set_mesh(mesh):
        actual, _ = update(grads, transform.init(params), params)
    got, want = np.asarray(actual["expert"], np.float32), np.asarray(expected["expert"], np.float32)
    assert actual["expert"].sharding.spec == params["expert"].sharding.spec
    # Match the existing bf16 Muon bucketing parity tolerance.
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-3 * np.abs(want).max())
