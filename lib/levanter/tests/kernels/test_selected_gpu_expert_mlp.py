# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.selected_gpu import selected_expert_mlp


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA selected expert matrix-vector kernel")
def test_selected_experts_match_float32_accumulation_with_bfloat16_projections():
    rng = np.random.default_rng(17)
    hidden, intermediate = 512, 256

    def values(shape):
        return jnp.asarray(rng.normal(0, 0.12, size=shape), jnp.bfloat16)

    x = values((hidden,))
    gate = values((4, hidden, intermediate))
    up = values((4, hidden, intermediate))
    down = values((4, intermediate, hidden))
    run = jax.jit(partial(selected_expert_mlp, activation=jax.nn.silu))

    def dot_bf16(vector, matrix):
        return (vector.astype(jnp.float32) @ matrix.astype(jnp.float32)).astype(jnp.bfloat16)

    for experts in ((-1, -1, -1, -1), (3, -1, 0, -1), (0, 3, 2, 1)):
        weights = jnp.where(jnp.asarray(experts) >= 0, jnp.bfloat16(0.5), jnp.bfloat16(jnp.nan))
        actual = run(x, jnp.asarray(experts, jnp.int32), weights, gate, up, down)
        expected = jnp.zeros_like(x)
        for route, expert in enumerate(experts):
            if expert >= 0:
                h = jax.nn.silu(dot_bf16(x, gate[expert])) * dot_bf16(x, up[expert])
                expected = expected + dot_bf16(h, down[expert]) * weights[route]
        np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=1e-4, rtol=1e-4)
