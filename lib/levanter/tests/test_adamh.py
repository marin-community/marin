# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax

from levanter.optim.adamh import scale_by_adamh


def _frobenius_norm(param: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.square(param)))


def _per_expert_norm(param: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.square(param), axis=(1, 2)))


def test_scale_by_adamh_reprojects_matrix_to_stored_init_norm() -> None:
    params_init = {"weight": jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4)}
    params_current = {"weight": params_init["weight"] * 1.7}
    grads = {"weight": jnp.linspace(0.1, 1.2, 12, dtype=jnp.float32).reshape(3, 4)}

    optimizer = scale_by_adamh(b1=0.0, b2=0.0, eps=1e-8, learning_rate=0.2)
    state = optimizer.init(params_init)
    updates, _ = optimizer.update(grads, state, params_current)
    new_params = optax.apply_updates(params_current, updates)

    init_norm = _frobenius_norm(params_init["weight"])
    current_norm = _frobenius_norm(params_current["weight"])
    new_norm = _frobenius_norm(new_params["weight"])

    assert not bool(jnp.isclose(current_norm, init_norm))
    assert bool(jnp.allclose(new_norm, init_norm, rtol=1e-5, atol=1e-5))


def test_scale_by_adamh_preserves_per_expert_stored_norms() -> None:
    params_init = {"experts": jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)}
    params_current = {
        "experts": params_init["experts"] * jnp.asarray([[[1.5]], [[0.5]]], dtype=jnp.float32),
    }
    grads = {"experts": jnp.linspace(0.1, 2.4, 24, dtype=jnp.float32).reshape(2, 3, 4)}

    optimizer = scale_by_adamh(b1=0.0, b2=0.0, eps=1e-8, learning_rate=0.15)
    state = optimizer.init(params_init)
    updates, _ = optimizer.update(grads, state, params_current)
    new_params = optax.apply_updates(params_current, updates)

    init_norms = _per_expert_norm(params_init["experts"])
    current_norms = _per_expert_norm(params_current["experts"])
    new_norms = _per_expert_norm(new_params["experts"])

    assert not bool(jnp.allclose(current_norms, init_norms))
    assert bool(jnp.allclose(new_norms, init_norms, rtol=1e-5, atol=1e-5))
