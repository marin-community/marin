# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax
import jax.numpy as jnp

import haliax
import haliax.specialized_fns as hfns
from haliax import NamedArray


def test_top_k():
    H, W, D = haliax.make_axes(H=3, W=4, D=5)

    rand = jax.random.uniform(jax.random.PRNGKey(0), (H.size, W.size, D.size))
    n_rand = NamedArray(rand, (H, W, D))

    values, indices = hfns.top_k(n_rand, D, 2)

    assert values.array.shape == (H.size, W.size, 2)
    assert indices.array.shape == (H.size, W.size, 2)
    assert jnp.all(
        jnp.equal(jax.lax.top_k(rand, 2)[0], values.array)
    )  # test that selecting last axis is same as default
    assert jnp.all(
        jnp.equal(jnp.moveaxis(n_rand.take(D, indices).array, 0, -1), values.array)
    )  # test that indexing using indices is same as selected values

    for idx, i in enumerate(n_rand.axes):  # then test selecting all axes
        t = jnp.transpose(rand, (*range(idx), *range(idx + 1, len(n_rand.axes)), idx))
        t = jax.lax.top_k(t, 2)[0]
        t = jnp.moveaxis(t, -1, idx)
        values, indices = hfns.top_k(n_rand, i, 2)
        assert jnp.all(jnp.equal(t, values.array))
        assert jnp.all(jnp.equal(jnp.moveaxis(n_rand.take(i, indices).array, 0, idx), values.array))
