# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from haliax.nn import ragged_dot


def _inputs():
    lhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)
