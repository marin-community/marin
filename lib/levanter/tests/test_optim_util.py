# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from levanter.optim.util import map_flattened_linear_layers


def test_map_flattened_linear_layers_flattens_rest_trees():
    In1, In2, Out1, Out2 = hax.make_axes(In1=2, In2=3, Out1=4, Out2=5)

    class Model(eqx.Module):
        linear: hax.nn.Linear

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    tree1 = Model(linear=hax.nn.Linear.init((In1, In2), (Out1, Out2), key=k1))
    tree2 = Model(linear=hax.nn.Linear.init((In1, In2), (Out1, Out2), key=k2))

    saw_flattened_rest = False

    def combine_linears(linear1: hax.nn.Linear, linear2: hax.nn.Linear):
        nonlocal saw_flattened_rest
        assert linear1.weight.ndim == 2
        assert linear2.weight.ndim == 2
        saw_flattened_rest = True
        return dataclasses.replace(linear1, weight=linear1.weight + linear2.weight)  # type: ignore

    combined = map_flattened_linear_layers(
        combine_linears,
        tree1,
        tree2,
        is_leaf=lambda x: isinstance(x, hax.nn.Linear),
    )

    assert saw_flattened_rest
    expected_weight = tree1.linear.weight + tree2.linear.weight
    assert combined.linear.weight.axes == tree1.linear.weight.axes
    assert jnp.allclose(combined.linear.weight.array, expected_weight.array)
