# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import jax.numpy as jnp

import haliax
import haliax as hax
from haliax.nn.pool import max_pool, mean_pool

# Tests largely cribbed from equinox


def test_maxpool1d():
    D = hax.Axis("D", 14)
    x = hax.arange(D)
    output = max_pool((D.resize(2),), x, stride=(3,))
    answer = jnp.array([1, 4, 7, 10, 13], dtype=jnp.int32)

    assert jnp.all(output.array == answer)

    answer = jnp.array([2, 5, 8, 11])
    output = max_pool(D.resize(3), x, stride=(3,), padding=0)
    assert jnp.all(output.array == answer)

    # max_pool = eqx.nn.MaxPool1d(kernel_size=3, stride=3, padding=0, use_ceil=True)
    answer = jnp.array([2, 5, 8, 11, 13])
    output = max_pool(D.resize(3), x, stride=(3,), padding=0, use_ceil=True)
    assert jnp.all(output.array == answer)

    # test batch axes
    B = hax.Axis("B", 2)
    x = x.rearrange("(B D) -> B D", B=B)
    output = max_pool(D.resize(2), x, stride=(3,), padding="VALID")
    answer = jnp.array([[1, 4], [8, 11]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(2), x, stride=(3,), use_ceil=True)
    answer = jnp.array([[1, 4, 6], [8, 11, 13]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(3), x, stride=(3,), padding=0)
    answer = jnp.array([[2, 5], [9, 12]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(3), x, stride=(3,), padding=0, use_ceil=True)
    answer = jnp.array([[2, 5, 6], [9, 12, 13]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(2), x, stride=(3,), padding="SAME")
    answer = jnp.array([[1, 4, 6], [8, 11, 13]])
    assert jnp.all(output.array == answer)


def test_maxpool2d():
    _x = jnp.arange(36).reshape(6, 6)
    x = hax.named(_x, ("H", "W"))

    # max_pool = eqx.nn.MaxPool2d(2, (3, 2))
    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2)), x, stride=(3, 2))
    answer = jnp.array([[7, 9, 11], [25, 27, 29]])

    assert jnp.all(output.array == answer)

    output = max_pool((hax.Axis("H", 3), hax.Axis("W", 3)), x, stride=2, padding=1)
    answer = jnp.array([[7, 9, 11], [19, 21, 23], [31, 33, 35]])

    assert jnp.all(output.array == answer)

    # test batch axes
    B = hax.Axis("B", 2)
    x = haliax.stack(B, [x, x])

    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2)), x, stride=(3, 2))
    answer = jnp.array([[[7, 9, 11], [25, 27, 29]], [[7, 9, 11], [25, 27, 29]]])

    assert jnp.all(output.array == answer)


def test_maxpool3d():
    _x = jnp.arange(64).reshape(4, 4, 4)
    x = hax.named(_x, ("H", "W", "D"))
    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2), hax.Axis("D", 2)), x, stride=(3, 2, 1))

    answer = jnp.array([[[21, 22, 23], [29, 30, 31]]])

    assert jnp.all(output.array == answer)

    answer = jnp.asarray(
        [
            [[37, 39, 39], [45, 47, 47], [45, 47, 47]],
            [[53, 55, 55], [61, 63, 63], [61, 63, 63]],
        ]
    )
    output = max_pool(
        (hax.Axis("H", 3), hax.Axis("W", 3), hax.Axis("D", 3)),
        x,
        stride=2,
        padding=((0, 1), (1, 1), (1, 1)),
        use_ceil=True,
    )
    assert jnp.all(output.array == answer)


def test_mean_pool_1d():
    D = hax.Axis("D", 14)
    x = hax.arange(D)
    output = mean_pool((D.resize(2),), x, stride=(3,))
    answer = jnp.array([0.5, 3.5, 6.5, 9.5, 12.5])

    assert jnp.all(output.array == answer)

    # no pad
    output = mean_pool(D.resize(3), x, stride=(3,), padding=0)
    answer = jnp.array([1, 4, 7, 10])
    assert jnp.all(output.array == answer)

    # pad, no include pad in avg
    output = mean_pool(D.resize(3), x, stride=(3,), padding="SAME", count_include_pad=False)
    answer = jnp.array([1, 4, 7, 10, 12.5])

    assert jnp.all(output.array == answer)

    output = mean_pool(D.resize(3), x, stride=(3,), padding="SAME", count_include_pad=True)
    answer = jnp.array([1, 4, 7, 10, (12 + 13) / 3.0])

    assert jnp.all(output.array == answer)


def test_mean_pool_2d():
    _x = jnp.arange(36).reshape(6, 6)
    x = hax.named(_x, ("H", "W"))

    output = mean_pool((hax.Axis("H", 1), hax.Axis("W", 3)), x, stride=2)
    answer = jnp.array([[1, 3], [13, 15], [25, 27]])

    assert jnp.all(output.array == answer)

    # test batch axes
    B = hax.Axis("B", 2)
    x = haliax.stack(B, [x, x])

    output = mean_pool((hax.Axis("H", 1), hax.Axis("W", 3)), x, stride=2)
    answer = jnp.array([[[1, 3], [13, 15], [25, 27]], [[1, 3], [13, 15], [25, 27]]])

    assert jnp.all(output.array == answer)


def test_mean_pool3d():
    _x = jnp.arange(64).reshape(4, 4, 4)
    x = hax.named(_x, ("H", "W", "D"))
    output = mean_pool((hax.Axis("H", 1), hax.Axis("W", 3), hax.Axis("D", 1)), x, stride=2)

    answer = jnp.array([[[4, 6]], [[36, 38]]])

    assert jnp.all(output.array == answer)


def test_pool_backprop():
    def max_pool_mean(x):
        pooled = max_pool(
            (hax.Axis("H", 2), hax.Axis("W", 2), hax.Axis("D", 2)), x, stride=1, padding=((0, 1), (0, 1), (0, 1))
        )
        return hax.mean(pooled).scalar()

    _x = jnp.arange(64, dtype=jnp.float32).reshape(1, 4, 4, 4)
    x = hax.named(_x, ("B", "H", "W", "D"))
    grad_fn = jax.value_and_grad(max_pool_mean)

    hax_loss, hax_grad = grad_fn(x)

    # compare it to eqx

    eqx_max_pool = eqx.nn.MaxPool3d(2, (1, 1, 1), padding=((0, 1), (0, 1), (0, 1)))

    def eqx_max_pool_mean(x):
        pooled = eqx_max_pool(x)
        return pooled.mean()

    eqx_grad_fn = jax.value_and_grad(eqx_max_pool_mean)
    eqx_loss, eqx_grad = eqx_grad_fn(_x)

    assert jnp.allclose(hax_loss, eqx_loss)
    assert jnp.allclose(hax_grad.array, eqx_grad)
