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


def test_ragged_dot_xla_supports_out_first_rhs_layout():
    lhs, rhs, group_sizes = _inputs()
    rhs_out_first = jnp.swapaxes(rhs, 1, 2)

    baseline = ragged_dot(lhs, rhs, group_sizes, implementation="xla")
    auto_out_first = ragged_dot(lhs, rhs_out_first, group_sizes, implementation="xla")
    out_first = ragged_dot(lhs, rhs_out_first, group_sizes, implementation="xla", rhs_contract_axis=2)

    assert jnp.allclose(baseline, auto_out_first, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(baseline, out_first, rtol=1e-5, atol=1e-5)
