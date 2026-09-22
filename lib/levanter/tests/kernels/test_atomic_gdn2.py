# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward and full VJP parity against the independent GDN-2 recurrence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.gdn2.candidate.configs import KernelConfig, ScoreLayout
from levanter.kernels.pallas.gdn2.candidate.gdn2_pipeline import gdn2_pallas_forward_trainable
from levanter.kernels.pallas.gdn2.reference import gdn2_reference


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("decay", [0.005, 0.5, 5.0])
@pytest.mark.parametrize("score_layout", list(ScoreLayout))
@pytest.mark.timeout(180)
def test_gdn2_output_state_and_all_input_gradients_match_recurrence(dtype, decay, score_layout):
    shape = (1, 64, 1, 128)
    keys = jax.random.split(jax.random.PRNGKey(17), 9)
    q = jax.random.normal(keys[0], shape) / np.sqrt(128)
    k = jax.random.normal(keys[1], shape)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(keys[2], shape)
    w = jax.nn.sigmoid(jax.random.normal(keys[3], shape))
    b = jax.nn.sigmoid(jax.random.normal(keys[4], shape))
    g = -decay * jax.random.uniform(keys[5], shape, minval=0.5, maxval=1.5)
    h0 = 0.05 * jax.random.normal(keys[6], (1, 1, 128, 128))
    inputs = tuple(x.astype(dtype) for x in (q, k, v, w, b, g)) + (h0,)
    config = KernelConfig(bt=32, bc=16, mb=8, interpret=True, score_layout=score_layout)
    scale = 128**-0.5

    def candidate(q, k, v, w, b, g, h0):
        return gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=h0, config=config)

    def reference(q, k, v, w, b, g, h0):
        return gdn2_reference(q, k, v, w, b, g, scale, h0)

    expected, expected_vjp = jax.vjp(reference, *inputs)
    actual, actual_vjp = jax.vjp(candidate, *inputs)
    cotangents = (jax.random.normal(keys[7], shape), jax.random.normal(keys[8], h0.shape))
    expected_grads = expected_vjp(cotangents)
    actual_grads = actual_vjp(cotangents)
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-2
    names = ("output", "final_state", "dq", "dk", "dv", "dw", "db", "dg", "dh0")
    for name, result, oracle in zip(names, (*actual, *actual_grads), (*expected, *expected_grads), strict=True):
        result, oracle = np.asarray(result, dtype=np.float32), np.asarray(oracle, dtype=np.float32)
        assert np.isfinite(result).all(), name
        error = np.abs(result - oracle)
        np.testing.assert_allclose(
            result,
            oracle,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"{name}: max_abs={error.max():.8g}, mean_abs={error.mean():.8g}",
        )
