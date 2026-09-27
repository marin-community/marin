# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.pallas.relu2_mlp import BlockSizes, relu2_mlp
from levanter.kernels.pallas.relu2_mlp.reference import relu2_mlp_reference


def _inputs(e: int, r: int, k: int, n: int, m: int):
    ks = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(ks[0], (e, r, k), jnp.float32)
    w_up = jax.random.normal(ks[1], (e, k, n), jnp.float32) * 0.1
    w_down = jax.random.normal(ks[2], (e, n, m), jnp.float32) * 0.1
    cotangent = jax.random.normal(ks[3], (e, r, m), jnp.float32)
    return x, w_up, w_down, cotangent


@pytest.mark.parametrize("implementation", ["reference", "pallas_interpret"])
@pytest.mark.parametrize("shape", [(2, 100, 128, 192, 64), (3, 64, 64, 64, 128)])
def test_relu2_mlp_matches_reference_value_and_grads(implementation, shape):
    x, w_up, w_down, cotangent = _inputs(*shape)

    def loss(fn):
        return lambda a, b, c: jnp.sum(fn(a, b, c) * cotangent)

    fused = lambda a, b, c: relu2_mlp(a, b, c, implementation=implementation, block_sizes=BlockSizes())  # noqa: E731
    want, want_grads = jax.value_and_grad(loss(relu2_mlp_reference), argnums=(0, 1, 2))(x, w_up, w_down)
    got, got_grads = jax.value_and_grad(loss(fused), argnums=(0, 1, 2))(x, w_up, w_down)

    assert jnp.allclose(got, want, rtol=1e-4)
    for g, w in zip(got_grads, want_grads, strict=True):
        assert float(jnp.max(jnp.abs(g - w))) <= 1e-4 * float(jnp.max(jnp.abs(w)))


def test_relu2_mlp_rejects_misaligned_width():
    x, w_up, w_down, _ = _inputs(1, 64, 128, 96, 64)
    with pytest.raises(ValueError, match="multiple of its block size"):
        relu2_mlp(x, w_up, w_down, implementation="pallas_interpret", block_sizes=BlockSizes(bn=64))
