# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType

from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss


def test_vocab_sharded_loss_matches_replicated_reference():
    if len(jax.devices()) < 2:
        pytest.skip("vocab-sharded loss test needs at least two local devices")

    key = jax.random.PRNGKey(0)
    hidden_key, lm_head_key = jax.random.split(key)
    hidden = jax.random.normal(hidden_key, (2, 3, 4), dtype=jnp.float32)
    lm_head = jax.random.normal(lm_head_key, (4, 8), dtype=jnp.float32)
    labels = jnp.array([[0, 3, 7], [1, 4, 6]], dtype=jnp.int32)
    weight = jnp.array([[1.0, 0.5, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)

    devices = np.array(jax.devices()[:2]).reshape(1, 2)
    mesh = jax.sharding.Mesh(devices, ("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    with jax.set_mesh(mesh):
        sharded = fused_linear_softmax_cross_entropy_loss(
            hidden,
            lm_head,
            labels,
            weight=weight,
            reduction="none",
            implementation="reference",
            vocab_axis="expert",
        )
        replicated = fused_linear_softmax_cross_entropy_loss(
            hidden,
            lm_head,
            labels,
            weight=weight,
            reduction="none",
            implementation="reference",
        )

    assert jnp.allclose(sharded, replicated, atol=1e-5, rtol=1e-5)


def test_vocab_sharded_loss_grad_matches_replicated_reference():
    if len(jax.devices()) < 2:
        pytest.skip("vocab-sharded loss test needs at least two local devices")

    key = jax.random.PRNGKey(1)
    hidden_key, lm_head_key = jax.random.split(key)
    hidden = jax.random.normal(hidden_key, (2, 3, 4), dtype=jnp.float32)
    lm_head = jax.random.normal(lm_head_key, (4, 8), dtype=jnp.float32)
    labels = jnp.array([[0, 3, 7], [1, 4, 6]], dtype=jnp.int32)
    weight = jnp.array([[1.0, 0.5, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)

    devices = np.array(jax.devices()[:2]).reshape(1, 2)
    mesh = jax.sharding.Mesh(devices, ("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    def sharded_loss(hidden_value, lm_head_value):
        return fused_linear_softmax_cross_entropy_loss(
            hidden_value,
            lm_head_value,
            labels,
            weight=weight,
            reduction="mean",
            implementation="reference",
            vocab_axis="expert",
        )

    def replicated_loss(hidden_value, lm_head_value):
        return fused_linear_softmax_cross_entropy_loss(
            hidden_value,
            lm_head_value,
            labels,
            weight=weight,
            reduction="mean",
            implementation="reference",
        )

    with jax.set_mesh(mesh):
        sharded_value, sharded_grads = jax.value_and_grad(sharded_loss, argnums=(0, 1))(hidden, lm_head)
        replicated_value, replicated_grads = jax.value_and_grad(replicated_loss, argnums=(0, 1))(hidden, lm_head)

    assert jnp.allclose(sharded_value, replicated_value, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(sharded_grads[0], replicated_grads[0], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(sharded_grads[1], replicated_grads[1], atol=1e-5, rtol=1e-5)
