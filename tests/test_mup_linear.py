# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses

import jax.numpy as jnp
import jax.random as jrandom
import pytest

import haliax as hax
from haliax.nn import Linear


@pytest.mark.parametrize("out_first", [True, False])
def test_mup_linear_init_matches_linear_axes(out_first: bool):
    In = (hax.Axis("I", 3), hax.Axis("J", 2))
    Out = (hax.Axis("O", 5),)
    key_linear, key_mup = jrandom.split(jrandom.PRNGKey(0))

    linear = Linear.init(In, Out, key=key_linear, out_first=out_first)
    mup = MupLinear.init(In, Out, key=key_mup, out_first=out_first)

    assert linear.weight.axes == mup.weight.axes
    if linear.bias is not None:
        assert mup.bias is not None
        assert linear.bias.axes == mup.bias.axes


def test_mup_linear_respects_bias_flag():
    In = hax.Axis("I", 4)
    Out = hax.Axis("O", 3)
    layer = MupLinear.init(In, Out, key=jrandom.PRNGKey(1), use_bias=False)
    assert layer.bias is None


def test_mup_linear_call_matches_linear():
    Batch = hax.Axis("B", 2)
    In = (hax.Axis("I", 3),)
    Out = (hax.Axis("O", 4),)

    weight = hax.ones(hax.concat_axis_specs(Out, In)) * 0.5
    bias = hax.full(Out, 0.25)

    linear = Linear(weight, bias, In, Out)
    mup = MupLinear(weight, bias, In, Out)

    inputs = hax.full(hax.concat_axis_specs(Batch, In), 2.0)

    expected = linear(inputs)
    actual = mup(inputs)

    assert actual.axes == expected.axes
    assert jnp.allclose(actual.array, expected.array)


@pytest.mark.parametrize("out_first", [True, False])
def test_hidden_linear_init_matches_linear_scaling(out_first: bool):
    In = hax.Axis("I", 6)
    Out = hax.Axis("O", 5)
    key = jrandom.PRNGKey(0)

    linear = Linear.init(In, Out, key=key, use_bias=False, out_first=out_first)
    hidden = HiddenLinear.init(In, Out, key=key, use_bias=False, out_first=out_first)

    assert jnp.allclose(hidden.weight.array, linear.weight.array)


def test_output_linear_scales_activation_by_input_size():
    Batch = hax.Axis("B", 2)
    In = hax.Axis("I", 3)
    Out = hax.Axis("O", 2)

    layer = OutputLinear.init(In, Out, key=jrandom.PRNGKey(2), use_bias=False)
    layer = dataclasses.replace(layer, weight=hax.ones(layer.weight.axes))

    inputs = hax.full((Batch, In), 2.0)
    expected = hax.full(hax.concat_axis_specs(Batch, layer.Out), 2.0)

    actual = layer(inputs)

    assert actual.axes == expected.axes
    assert jnp.allclose(actual.array, expected.array)


def test_output_linear_state_dict_scales_weight():
    In = hax.Axis("I", 4)
    Out = hax.Axis("O", 3)

    layer = OutputLinear.init(In, Out, key=jrandom.PRNGKey(3), use_bias=False)
    layer = dataclasses.replace(layer, weight=hax.ones(layer.weight.axes))

    state = layer.to_state_dict()
    assert jnp.allclose(state["weight"], layer.weight.array * layer.mup_active_scale)

    template = OutputLinear.init(In, Out, key=jrandom.PRNGKey(4), use_bias=False)
    restored = template.from_state_dict(state)

    assert jnp.allclose(restored.weight.array, layer.weight.array)


def test_input_linear_behaves_like_base_linear():
    Batch = hax.Axis("B", 2)
    In = hax.Axis("I", 3)
    Out = hax.Axis("O", 4)

    weight = hax.ones((Out, In)) * 0.1
    bias = hax.zeros(Out)

    linear = Linear(weight, bias, In, Out)
    input_linear = InputLinear(weight, bias, In, Out)

    inputs = hax.random.normal(jrandom.PRNGKey(5), (Batch, In))

    expected = linear(inputs)
    actual = input_linear(inputs)

    assert jnp.allclose(actual.array, expected.array)
