# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import enum
import typing
from functools import partial

import jax
import jax.numpy as jnp

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]
JaxActivationFunction = typing.Callable[[jax.Array], jax.Array]


def _quick_gelu_jax(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(1.702 * x)


def _relu2_jax(x: jax.Array) -> jax.Array:
    return jnp.square(jax.nn.relu(x))


class ActivationFunctionEnum(str, enum.Enum):
    relu = "relu"
    relu2 = "relu2"
    silu = "silu"
    swish = "swish"
    gelu = "gelu"
    gelu_new = "gelu_new"
    quick_gelu = "quick_gelu"
    tanh = "tanh"
    xielu = "xielu"

    def to_fn(self) -> ActivationFunction:
        if self is ActivationFunctionEnum.xielu:
            raise ValueError("xielu is parameterized; use XIELUActivation directly.")
        return TO_FN[self]

    def to_jax_fn(self) -> JaxActivationFunction:
        if self is ActivationFunctionEnum.xielu:
            raise ValueError("xielu is parameterized; use XIELUActivation directly.")
        return TO_JAX_FN[self]


# type: ignore
TO_FN: dict[ActivationFunctionEnum, ActivationFunction] = {
    ActivationFunctionEnum.relu: hnn.relu,
    ActivationFunctionEnum.relu2: hnn.relu_squared,
    ActivationFunctionEnum.silu: hnn.silu,
    ActivationFunctionEnum.swish: hnn.swish,
    ActivationFunctionEnum.gelu: partial(hnn.gelu, approximate=False),
    ActivationFunctionEnum.gelu_new: partial(hnn.gelu, approximate=True),
    ActivationFunctionEnum.quick_gelu: hnn.quick_gelu,
    ActivationFunctionEnum.tanh: hax.tanh,
}


TO_JAX_FN: dict[ActivationFunctionEnum, JaxActivationFunction] = {
    ActivationFunctionEnum.relu: jax.nn.relu,
    ActivationFunctionEnum.relu2: _relu2_jax,
    ActivationFunctionEnum.silu: jax.nn.silu,
    ActivationFunctionEnum.swish: jax.nn.swish,
    ActivationFunctionEnum.gelu: partial(jax.nn.gelu, approximate=False),
    ActivationFunctionEnum.gelu_new: partial(jax.nn.gelu, approximate=True),
    ActivationFunctionEnum.quick_gelu: _quick_gelu_jax,
    ActivationFunctionEnum.tanh: jnp.tanh,
}
