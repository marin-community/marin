# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_norm_preserving_residual.model import norm_preserving_residual


def test_norm_preserving_residual_matches_requested_initial_mixture():
    residual = jnp.array([1.0, 0.0], dtype=jnp.float32)
    hidden = jnp.array([0.0, 1.0], dtype=jnp.float32)

    mixed, beta = norm_preserving_residual(residual, hidden, jnp.asarray(0.0), num_layers=6)

    expected_beta = np.log(2.0)
    expected = np.array(
        [np.sqrt(1.0 - expected_beta / 6), np.sqrt(expected_beta / 6)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-6)
    np.testing.assert_allclose(mixed, expected, rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(jnp.square(mixed)), 1.0, rtol=1e-6)


def test_norm_preserving_residual_caps_beta_before_square_root_domain():
    residual = jnp.array([1.0, 0.0], dtype=jnp.float32)
    hidden = jnp.array([0.0, 1.0], dtype=jnp.float32)

    mixed, beta = norm_preserving_residual(residual, hidden, jnp.asarray(100.0), num_layers=6)

    assert beta < 6
    assert jnp.all(jnp.isfinite(mixed))
    np.testing.assert_allclose(jnp.sum(jnp.square(mixed)), 1.0, rtol=1e-6)
