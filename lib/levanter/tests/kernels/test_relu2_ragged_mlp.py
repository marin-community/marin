# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.pallas.relu2_ragged_mlp import BlockSizes, GmmBlockSizes, TgmmBlockSizes, relu2_ragged_mlp
from levanter.kernels.pallas.relu2_ragged_mlp.reference import relu2_ragged_mlp_reference

# Small tiles so the shapes below cross tile boundaries inside a group and at the end of the buffer.
_BLOCKS = BlockSizes(
    gmm=GmmBlockSizes(bm=16, bn=32, bk=16),
    tgmm=TgmmBlockSizes(bm=16, bn=32, bk=16, splits=3),
)


def _inputs(rows: int, k: int, n: int, m: int, num_groups: int):
    ks = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(ks[0], (rows, k), jnp.float32)
    w_up = jax.random.normal(ks[1], (num_groups, k, n), jnp.float32) * 0.2
    w_down = jax.random.normal(ks[2], (num_groups, n, m), jnp.float32) * 0.2
    cotangent = jax.random.normal(ks[3], (rows, m), jnp.float32)
    return x, w_up, w_down, cotangent


@pytest.mark.parametrize("implementation", ["reference", "pallas_interpret"])
@pytest.mark.parametrize(
    "group_sizes",
    [
        # Empty groups (first, middle, last), a group spanning several tiles, and groups ending mid-tile;
        # the last tile runs past the buffer end.
        pytest.param((0, 5, 0, 37, 1, 7, 0), id="uneven_with_empty"),
        # Rows past sum(group_sizes), as in a capacity-padded receiver buffer.
        pytest.param((9, 0, 20, 3), id="trailing_padding"),
        pytest.param((0, 0, 0), id="all_empty"),
    ],
)
def test_relu2_ragged_mlp_matches_per_group_reference(implementation, group_sizes):
    rows, k, n, m = 57, 32, 64, 32
    x, w_up, w_down, cotangent = _inputs(rows, k, n, m, len(group_sizes))
    sizes = jnp.asarray(group_sizes, jnp.int32)

    def loss(fn):
        return lambda a, b, c: jnp.sum(fn(a, b, c, sizes) * cotangent)

    fused = lambda a, b, c, s: relu2_ragged_mlp(  # noqa: E731
        a, b, c, s, implementation=implementation, block_sizes=_BLOCKS
    )
    want, want_grads = jax.value_and_grad(loss(relu2_ragged_mlp_reference), argnums=(0, 1, 2))(x, w_up, w_down)
    got, got_grads = jax.value_and_grad(loss(fused), argnums=(0, 1, 2))(x, w_up, w_down)
    out = fused(x, w_up, w_down, sizes)

    assert jnp.allclose(out, relu2_ragged_mlp_reference(x, w_up, w_down, sizes), rtol=1e-5, atol=1e-5)
    assert jnp.allclose(got, want, rtol=1e-5, atol=1e-5)
    for name, g, w in zip(["dx", "dw_up", "dw_down"], got_grads, want_grads, strict=True):
        scale = max(float(jnp.max(jnp.abs(w))), 1.0)
        assert float(jnp.max(jnp.abs(g - w))) <= 1e-5 * scale, name
    covered = sum(group_sizes)
    assert not jnp.any(out[covered:]), "rows past sum(group_sizes) must be zero"
    assert not jnp.any(got_grads[0][covered:]), "rows past sum(group_sizes) must get zero gradient"


def test_relu2_ragged_mlp_rejects_misaligned_width():
    x, w_up, w_down, _ = _inputs(16, 32, 48, 32, 2)
    with pytest.raises(ValueError, match="multiple of its block size"):
        relu2_ragged_mlp(
            x, w_up, w_down, jnp.array([8, 8], jnp.int32), implementation="pallas_interpret", block_sizes=_BLOCKS
        )
