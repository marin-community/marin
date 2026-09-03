# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.standalone.naive_ragged_dot import naive_ragged_dot


def test_naive_ragged_dot_applies_each_expert_to_its_contiguous_rows():
    lhs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rhs = jnp.array(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ]
    )
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)

    result = naive_ragged_dot(lhs, rhs, group_sizes)

    expected = jnp.concatenate([lhs[:2] @ rhs[0], lhs[2:] @ rhs[1]], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_naive_ragged_dot_skips_an_empty_expert_without_shifting_later_rows():
    lhs = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rhs = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[99.0, 99.0], [99.0, 99.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ]
    )
    group_sizes = jnp.array([1, 0, 2], dtype=jnp.int32)

    result = naive_ragged_dot(lhs, rhs, group_sizes)

    expected = jnp.concatenate([lhs[:1] @ rhs[0], lhs[1:] @ rhs[2]], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_naive_ragged_dot_rejects_group_sizes_that_do_not_cover_all_rows():
    lhs = jnp.zeros((2, 2), dtype=jnp.float32)
    rhs = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    group_sizes = jnp.array([1, 0], dtype=jnp.int32)

    with pytest.raises(ValueError):
        naive_ragged_dot(lhs, rhs, group_sizes)
