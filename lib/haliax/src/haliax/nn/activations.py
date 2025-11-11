# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import typing

from jax import nn as jnn
from jax import numpy as jnp

from ..axis import Axis
from ..core import NamedArray
from ..types import Scalar
from ..wrap import wrap_elemwise_unary

A = typing.TypeVar("A", Scalar, NamedArray, jnp.ndarray)


def relu(a: A) -> A:
    return wrap_elemwise_unary(jnn.relu, a)


def relu6(a: A) -> A:
    return wrap_elemwise_unary(jnn.relu6, a)


def sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.sigmoid, a)


def softplus(a: A) -> A:
    return wrap_elemwise_unary(jnn.softplus, a)


def soft_sign(a: A) -> A:
    return wrap_elemwise_unary(jnn.soft_sign, a)


def silu(a: A) -> A:
    return wrap_elemwise_unary(jnn.silu, a)


def swish(a: A) -> A:
    return wrap_elemwise_unary(jnn.swish, a)


def log_sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.log_sigmoid, a)


def leaky_relu(a: A) -> A:
    return wrap_elemwise_unary(jnn.leaky_relu, a)


def hard_sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_sigmoid, a)


def hard_silu(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_silu, a)


def hard_swish(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_swish, a)


def hard_tanh(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_tanh, a)


def elu(a: A) -> A:
    return wrap_elemwise_unary(jnn.elu, a)


def celu(a: A) -> A:
    return wrap_elemwise_unary(jnn.celu, a)


def selu(a: A) -> A:
    return wrap_elemwise_unary(jnn.selu, a)


def gelu(a: A, approximate: bool = True) -> A:
    return wrap_elemwise_unary(jnn.gelu, a, approximate=approximate)


def glu(x: NamedArray, axis: Axis) -> NamedArray:
    axis_index = x.axes.index(axis)
    return NamedArray(jnn.glu(x.array, axis_index), x.axes)


def quick_gelu(x):
    return x * sigmoid(1.702 * x)


def relu_squared(x: A) -> A:
    """ReLU squared activation function. jnp.square(jnp.maximum(0, x))"""

    def _fn(a):
        return jnp.square(jnn.relu(a))

    return typing.cast(A, wrap_elemwise_unary(_fn, x))
