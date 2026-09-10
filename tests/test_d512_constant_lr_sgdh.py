# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import optax

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.optimizer import scale_with_grug_sgdh


def test_sgdh_is_stateless_raw_gradient_hyperball_update():
    learning_rate = 0.2
    params = {"matrix": jnp.array([[3.0, 0.0], [0.0, 4.0]])}
    gradients = {"matrix": jnp.array([[1.0, 2.0], [-1.0, 0.5]])}
    transform = scale_with_grug_sgdh(learning_rate)

    state = transform.init(params)
    updates, next_state = transform.update(gradients, state, params)

    param = params["matrix"]
    gradient = gradients["matrix"]
    candidate = param - learning_rate * gradient * jnp.linalg.norm(param) / jnp.linalg.norm(gradient)
    expected_new_param = candidate * jnp.linalg.norm(param) / jnp.linalg.norm(candidate)

    assert isinstance(state, optax.EmptyState)
    assert isinstance(next_state, optax.EmptyState)
    np.testing.assert_allclose(param + updates["matrix"], expected_new_param, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(jnp.linalg.norm(param + updates["matrix"]), jnp.linalg.norm(param), rtol=1e-6)
