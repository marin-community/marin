# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh
from numpy.testing import assert_allclose

from experiments.grug.moe_pallas_sconv.model import ShortConv


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1]).reshape((1, 1, 1, 1))
    return Mesh(
        devices,
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _short_conv_reference(x: jax.Array, weight: jax.Array, segment_ids: jax.Array) -> jax.Array:
    seq_len = x.shape[1]
    output = jnp.zeros_like(x)
    for lag in range(weight.shape[1]):
        shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :seq_len, :]
        shifted_ids = jnp.pad(segment_ids, ((0, 0), (lag, 0)), constant_values=-1)[:, :seq_len]
        same_segment = (segment_ids >= 0) & (shifted_ids == segment_ids)
        output = output + jnp.where(same_segment[..., None], shifted * weight[:, -1 - lag], 0)
    return output


def test_short_conv_matches_segmented_reference_value_and_gradients():
    x = jnp.arange(2 * 7 * 3, dtype=jnp.float32).reshape(2, 7, 3) / 10
    weight = jnp.array(
        [
            [0.1, -0.2, 0.3, 0.8],
            [-0.4, 0.2, 0.1, 0.9],
            [0.25, 0.15, -0.1, 1.1],
        ],
        dtype=jnp.float32,
    )
    segment_ids = jnp.array(
        [
            [0, 0, 0, 1, 1, 2, 2],
            [3, 3, 4, 4, 4, -1, -1],
        ],
        dtype=jnp.int32,
    )

    def actual_loss(weight, x):
        return jnp.sum(ShortConv(weight=weight, kernel_size=4)(x, segment_ids) ** 2)

    def reference_loss(weight, x):
        return jnp.sum(_short_conv_reference(x, weight, segment_ids) ** 2)

    with set_mesh(_single_device_mesh()):
        actual_value, actual_grads = jax.value_and_grad(actual_loss, argnums=(0, 1))(weight, x)
    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1))(weight, x)

    assert_allclose(np.asarray(actual_value), np.asarray(reference_value), rtol=1e-5, atol=1e-5)
    assert_allclose(np.asarray(actual_grads[0]), np.asarray(reference_grads[0]), rtol=1e-5, atol=1e-5)
    assert_allclose(np.asarray(actual_grads[1]), np.asarray(reference_grads[1]), rtol=1e-5, atol=1e-5)


def test_short_conv_identity_initialization_preserves_valid_tokens():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 4))
    segment_ids = jnp.array([[0, 0, 1, 1, 2, 2], [3, 3, 3, 4, -1, -1]], dtype=jnp.int32)

    with set_mesh(_single_device_mesh()):
        output = ShortConv.init(channels=4, kernel_size=4)(x, segment_ids)

    expected = jnp.where((segment_ids >= 0)[..., None], x, 0)
    assert_allclose(np.asarray(output), np.asarray(expected), rtol=1e-6, atol=1e-6)
