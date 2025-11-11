# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax
import jax.random as jrandom
from jax import numpy as jnp

import haliax as hax
from haliax.nn import MoELinear


def _expected_moe_linear_output(moe: MoELinear, x: hax.NamedArray, group_sizes: hax.NamedArray):
    dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        (
            ((x.axis_indices(moe.In),), (moe.weight.axis_indices(moe.In),)),
            ((), ()),
        ),
        x.axis_indices(hax.axis.without_axes(x.axes, moe.In)),
        (moe.weight.axis_indices(moe.Experts),),
    )
    out_raw = jax.lax.ragged_dot_general(
        lhs=x.array,
        rhs=moe.weight.array,
        group_sizes=group_sizes.array,
        ragged_dot_dimension_numbers=dim_numbers,
    )
    out_axes = hax.replace_axis(x.axes, moe.In, moe.Out)
    out = hax.named(out_raw, out_axes)
    if moe.bias is not None:
        out = out + moe.bias
    return out


def test_moe_linear_matches_ragged_dot_general():
    B, In, Out, E = hax.make_axes(B=3, In=4, Out=5, E=2)
    key = jrandom.PRNGKey(0)
    moe = MoELinear.init(E, In, Out, key=key)

    x = hax.random.normal(jrandom.PRNGKey(1), (B, In))
    group_sizes = hax.named(jnp.array([2, 1], dtype=jnp.int32), (E,))

    actual = moe(x, group_sizes)
    expected = _expected_moe_linear_output(moe, x, group_sizes)

    assert actual.axes == expected.axes
    assert jnp.allclose(actual.array, expected.array, rtol=1e-5, atol=1e-5)


def test_moe_linear_out_first_property():
    E, In, Out = hax.make_axes(E=2, In=4, Out=3)
    moe = MoELinear.init(E, In, Out, key=jrandom.PRNGKey(0), out_first=True)
    assert moe.out_first
    assert moe.weight.axes[:3] == (E, Out, In)

    moe2 = MoELinear.init(E, In, Out, key=jrandom.PRNGKey(1), out_first=False)
    assert not moe2.out_first
    assert moe2.weight.axes[:3] == (E, In, Out)


def test_moe_linear_gmm_matches_ragged_dot_general():
    B, In, Out, E = hax.make_axes(B=3, In=4, Out=5, E=2)
    moe = MoELinear.init(E, In, Out, key=jrandom.PRNGKey(0), use_gmm=True)

    x = hax.random.normal(jrandom.PRNGKey(1), (B, In))
    group_sizes = hax.named(jnp.array([2, 1], dtype=jnp.int32), (E,))

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    with hax.partitioning.set_mesh(mesh):
        actual = hax.named_jit(moe)(x, group_sizes)

    expected = _expected_moe_linear_output(moe, x, group_sizes)

    assert actual.axes == expected.axes
    assert jnp.allclose(actual.array, expected.array, rtol=1e-5, atol=1e-5)
