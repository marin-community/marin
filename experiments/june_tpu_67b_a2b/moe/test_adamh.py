# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamH update regression for zero-norm matrices."""

import jax
import jax.numpy as jnp
import pytest

from experiments.june_tpu_67b_a2b.moe.adamh import scale_by_adamh


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2)])
def test_adamh_zero_matrix_update_stays_finite(shape):
    params = {"matrix": jnp.zeros(shape, dtype=jnp.float32)}
    gradients = {"matrix": jnp.zeros(shape, dtype=jnp.float32)}
    optimizer = scale_by_adamh()

    updates, _ = jax.jit(optimizer.update)(gradients, optimizer.init(params), params)

    assert bool(jnp.all(jnp.isfinite(updates["matrix"])))
    assert bool(jnp.all(updates["matrix"] == 0))
