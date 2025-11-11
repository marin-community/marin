# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax.nn.conv import Conv, ConvTranspose


def test_conv_basic_equiv_to_eqx():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 3)
    Out = hax.Axis("Out", 4)
    hax_conv = Conv.init(("Height", "Width"), In, Out, kernel_size=3, key=key)
    eqx_conv = eqx.nn.Conv(2, 3, 4, kernel_size=3, key=key)

    assert hax_conv.weight.array.shape == eqx_conv.weight.shape
    assert hax_conv.bias.array.shape == eqx_conv.bias.shape[0:1]
    assert jnp.all(hax_conv.weight.array == eqx_conv.weight)

    input = hax.random.normal(jax.random.PRNGKey(1), (In, hax.Axis("Height", 5), hax.Axis("Width", 6)))
    hax_output = hax_conv(input)
    eqx_output = eqx_conv(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test batched
    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx_conv)(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test multibatch
    input = hax.random.normal(
        jax.random.PRNGKey(1),
        (hax.Axis("Batch", 2), hax.Axis("Batch2", 3), In, hax.Axis("Height", 5), hax.Axis("Width", 6)),
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx.filter_vmap(eqx_conv))(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.allclose(hax_output.array, eqx_output)

    input = hax.random.normal(
        jax.random.PRNGKey(1),
        (
            hax.Axis("Batch", 2),
            In,
            hax.Axis("Height", 5),
            hax.Axis("Width", 6),
            hax.Axis("Batch2", 3),
        ),
    )
    hax_output = hax_conv(input).rearrange(("Batch", "Batch2", "Out", "Height", "Width"))
    eqx_output = eqx.filter_vmap(eqx.filter_vmap(eqx_conv))(
        input.rearrange(("Batch", "Batch2", "In", "Height", "Width")).array
    )

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.allclose(hax_output.array, eqx_output)


def test_conv_grouped_equiv_to_eqx():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 4)
    Out = hax.Axis("Out", 6)
    hax_conv = Conv.init(("Height", "Width"), In, Out, kernel_size=3, groups=2, key=key)
    eqx_conv = eqx.nn.Conv(2, 4, 6, kernel_size=3, groups=2, key=key)

    assert hax_conv.weight.array.shape == eqx_conv.weight.shape
    assert hax_conv.bias.array.shape == eqx_conv.bias.shape[0:1]
    assert jnp.all(hax_conv.weight.array == eqx_conv.weight)

    input = hax.random.normal(jax.random.PRNGKey(1), (In, hax.Axis("Height", 5), hax.Axis("Width", 6)))
    eqx_output = eqx_conv(input.array)
    hax_output = hax_conv(input)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test batched
    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx_conv)(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test multibatch
    input = hax.random.normal(
        jax.random.PRNGKey(1),
        (hax.Axis("Batch", 2), hax.Axis("Batch2", 3), In, hax.Axis("Height", 5), hax.Axis("Width", 6)),
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx.filter_vmap(eqx_conv))(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.allclose(hax_output.array, eqx_output)


def test_conv_weird_order():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 3)
    Out = hax.Axis("Out", 4)
    hax_conv = Conv.init(("Height", "Width"), In, Out, kernel_size=3, key=key)
    eqx_conv = eqx.nn.Conv(2, 3, 4, kernel_size=3, key=key)

    assert hax_conv.weight.array.shape == eqx_conv.weight.shape
    assert hax_conv.bias.array.shape == eqx_conv.bias.shape[0:1]
    assert jnp.all(hax_conv.weight.array == eqx_conv.weight)

    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)

    # test weird orders
    input = input.rearrange(("In", "Height", "Width", "Batch"))
    hax_output2 = hax_conv(input).rearrange(("Batch", "Out", "Height", "Width"))

    assert jnp.allclose(hax_output.array, hax_output2.array)


def test_conv_transpose_basic_equiv_to_eqx():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 3)
    Out = hax.Axis("Out", 4)
    hax_conv = ConvTranspose.init(
        ("Height", "Width"), In, Out, kernel_size=3, dilation=2, output_padding=1, stride=2, key=key
    )
    eqx_conv = eqx.nn.ConvTranspose(2, 3, 4, kernel_size=3, dilation=2, output_padding=1, stride=2, key=key)

    assert hax_conv.weight.array.shape == eqx_conv.weight.shape
    assert hax_conv.bias.array.shape == eqx_conv.bias.shape[0:1]
    assert jnp.all(hax_conv.weight.array == eqx_conv.weight)

    input = hax.random.normal(jax.random.PRNGKey(1), (In, hax.Axis("Height", 5), hax.Axis("Width", 6)))
    hax_output = hax_conv(input)
    eqx_output = eqx_conv(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test batched
    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx_conv)(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test multibatch
    input = hax.random.normal(
        jax.random.PRNGKey(1),
        (hax.Axis("Batch", 2), hax.Axis("Batch2", 3), In, hax.Axis("Height", 5), hax.Axis("Width", 6)),
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx.filter_vmap(eqx_conv))(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.allclose(hax_output.array, eqx_output)


def test_weird_orders_conv_transpose():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 3)
    Out = hax.Axis("Out", 4)
    hax_conv = ConvTranspose.init(
        ("Height", "Width"), In, Out, kernel_size=3, dilation=2, output_padding=1, stride=2, key=key
    )

    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)

    # test weird orders
    input = input.rearrange(("In", "Height", "Width", "Batch"))
    hax_output2 = hax_conv(input).rearrange(("Batch", "Out", "Height", "Width"))

    assert jnp.allclose(hax_output.array, hax_output2.array)
