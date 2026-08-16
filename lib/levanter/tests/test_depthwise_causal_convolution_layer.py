# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import haliax as hax
from haliax import Axis

from levanter.kernels.pallas.depthwise_causal_convolution import depthwise_causal_convolution
from levanter.layers.depthwise_causal_convolution import DepthwiseCausalConvolution


Batch = Axis("batch", 2)
Pos = Axis("position", 6)
Embed = Axis("embed", 5)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("output_state", [False, True])
def test_depthwise_causal_convolution_layer_matches_raw_op(use_bias: bool, output_state: bool) -> None:
    kernel_size = 4
    key_init, key_x, key_h0 = jax.random.split(jax.random.PRNGKey(0), 3)

    layer = DepthwiseCausalConvolution.init(
        Embed, kernel_size=kernel_size, activation_function=None, use_bias=use_bias, key=key_init
    )

    x = hax.random.normal(key_x, (Batch, Pos, Embed))
    input_state = hax.random.normal(key_h0, (Batch, Embed, layer.State))

    output, output_state_named = layer(x, input_state=input_state, output_state=output_state)

    assert output.axes == (Batch, Pos, Embed)

    output_ref, output_state_ref = depthwise_causal_convolution(
        input=hax.rearrange(x, (Batch, Pos, Embed)).array,
        weight=layer.weight.array,
        bias=None if layer.bias is None else layer.bias.array,
        input_state=hax.rearrange(input_state, (Batch, Embed, layer.State)).array,
        output_state=output_state,
        activation_function=None,
    )

    assert_allclose(np.asarray(hax.rearrange(output, (Batch, Pos, Embed)).array), np.asarray(output_ref))

    if output_state:
        assert output_state_named is not None
        assert output_state_named.axes == (Batch, Embed, layer.State)
        assert_allclose(
            np.asarray(hax.rearrange(output_state_named, (Batch, Embed, layer.State)).array),
            np.asarray(output_state_ref),
        )
    else:
        assert output_state_named is None


def test_depthwise_causal_convolution_layer_without_input_state() -> None:
    kernel_size = 3
    key_init, key_x = jax.random.split(jax.random.PRNGKey(1), 2)

    layer = DepthwiseCausalConvolution.init(
        Embed, kernel_size=kernel_size, activation_function="silu", use_bias=True, key=key_init
    )

    x = hax.random.normal(key_x, (Batch, Pos, Embed))
    output, output_state = layer(x, output_state=True)

    assert output.axes == (Batch, Pos, Embed)
    assert output_state is not None
    assert output_state.axes == (Batch, Embed, layer.State)


def test_depthwise_causal_convolution_layer_grad_runs() -> None:
    kernel_size = 3
    key_init, key_x = jax.random.split(jax.random.PRNGKey(2), 2)

    layer = DepthwiseCausalConvolution.init(
        Embed, kernel_size=kernel_size, activation_function=None, use_bias=True, key=key_init
    )
    x = hax.random.normal(key_x, (Batch, Pos, Embed))

    def loss(layer, x):
        output, _ = layer(x)
        return hax.sum(output).scalar()

    grad_layer, grad_x = jax.grad(loss, argnums=(0, 1))(layer, x)

    assert jnp.all(jnp.isfinite(grad_layer.weight.array))
    assert jnp.all(jnp.isfinite(grad_x.array))
