# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from levanter.utils.tree_utils import pack_pytree, unpack_pytree


def test_pack_and_unpack_simple():
    tree = {"a": np.arange(3, dtype=np.float32), "b": np.arange(4, dtype=np.float32).reshape(2, 2)}
    offsets, packed = eqx.filter_jit(pack_pytree)(tree, dtype=jnp.float32)
    rebuilt = eqx.filter_jit(unpack_pytree)(offsets, packed)
    for orig, new in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(rebuilt)):
        np.testing.assert_array_equal(np.asarray(orig, dtype=np.float32), np.array(new))


def test_pack_empty_tree():
    tree = {}
    offsets, packed = pack_pytree(tree, dtype=jnp.float32)
    assert packed.size == 0
    rebuilt = unpack_pytree(offsets, packed)
    assert rebuilt == tree
