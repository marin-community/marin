# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh

from levanter.grug.loss import (
    chunked_linear_softmax_cross_entropy_loss,
    fused_linear_softmax_cross_entropy_loss,
)


ATOL = 1e-5
RTOL = 1e-5


def _assert_close(actual: jax.Array, expected: jax.Array) -> None:
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"max_abs={difference.max():.3e}, mean_abs={difference.mean():.3e}",
    )


@pytest.mark.parametrize(
    ("chunk_size", "backward_pass_unroll"),
    [
        pytest.param(4, 2, id="ragged-final-chunk"),
        pytest.param(11, 1, id="chunk-larger-than-token-count"),
    ],
)
def test_chunked_cross_entropy_matches_unchunked_value_and_gradients(
    chunk_size: int, backward_pass_unroll: int
) -> None:
    num_devices = len(jax.devices())
    hidden = jax.random.normal(jax.random.key(0), (num_devices, 7, 5), dtype=jnp.float32) * 0.2
    lm_head = jax.random.normal(jax.random.key(1), (5, 13), dtype=jnp.float32) * 0.2
    labels = jax.random.randint(jax.random.key(2), (num_devices, 7), 0, 13, dtype=jnp.int32)
    per_device_weight = jnp.array([1.0, 0.5, 0.0, 1.25, 0.75, 1.0, 0.25], dtype=jnp.float32)
    weight = jnp.broadcast_to(per_device_weight, (num_devices, 7))

    def unchunked_loss(hidden: jax.Array, lm_head: jax.Array, weight: jax.Array) -> jax.Array:
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            lm_head,
            labels,
            weight=weight,
            logsumexp_weight=1e-3,
            implementation="reference",
        )

    def chunked_loss(hidden: jax.Array, lm_head: jax.Array, weight: jax.Array) -> jax.Array:
        return chunked_linear_softmax_cross_entropy_loss(
            hidden,
            lm_head,
            labels,
            weight=weight,
            logsumexp_weight=1e-3,
            chunk_size=chunk_size,
            backward_pass_unroll=backward_pass_unroll,
        )

    mesh = Mesh(
        np.asarray(jax.devices()),
        axis_names=("data",),
        axis_types=(AxisType.Explicit,),
    )
    with jax.set_mesh(mesh):
        expected_loss, expected_grads = jax.value_and_grad(unchunked_loss, argnums=(0, 1, 2))(hidden, lm_head, weight)
        actual_loss, actual_grads = jax.value_and_grad(chunked_loss, argnums=(0, 1, 2))(hidden, lm_head, weight)

    _assert_close(actual_loss, expected_loss)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        _assert_close(actual_grad, expected_grad)
