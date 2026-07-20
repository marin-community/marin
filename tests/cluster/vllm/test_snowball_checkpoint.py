# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from tests.cluster.vllm.snowball_checkpoint import logical_array_digest


def test_logical_array_digest_is_independent_of_tree_order_and_chunking() -> None:
    first = {
        "z": jnp.arange(24, dtype=jnp.bfloat16).reshape(3, 4, 2),
        "a": jnp.asarray([1.5, -2.0], dtype=jnp.bfloat16),
    }
    reordered = {"a": first["a"], "z": first["z"]}

    expected = logical_array_digest(first, chunk_bytes=8)

    assert logical_array_digest(first, chunk_bytes=64) == expected
    assert logical_array_digest(reordered, chunk_bytes=13) == expected


def test_logical_array_digest_changes_with_names_shapes_dtypes_and_values() -> None:
    original = {"weight": jnp.asarray([[1.0, 2.0]], dtype=jnp.bfloat16)}
    expected = logical_array_digest(original, chunk_bytes=8)

    assert logical_array_digest({"renamed": original["weight"]}, chunk_bytes=8) != expected
    assert logical_array_digest({"weight": original["weight"].reshape(2, 1)}, chunk_bytes=8) != expected
    assert logical_array_digest({"weight": original["weight"].astype(jnp.float32)}, chunk_bytes=8) != expected
    assert logical_array_digest({"weight": original["weight"].at[0, 1].set(3.0)}, chunk_bytes=8) != expected
