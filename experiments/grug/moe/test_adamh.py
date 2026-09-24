# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("module", ["experiments.grug.moe.adamh", "experiments.june_tpu_67b_a2b.moe.adamh"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2)])
def test_zero_norm_parameter_stays_finite_during_warmup(module, shape):
    transform = importlib.import_module(module).scale_by_adamh(learning_rate=0.0)
    param = jnp.zeros(shape)

    update, _ = transform.update(jnp.ones(shape), transform.init(param), param)

    assert jnp.all(jnp.isfinite(update))
    assert jnp.array_equal(update, param)
