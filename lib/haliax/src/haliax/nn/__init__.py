# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import numbers

import jax.nn as jnn
import jax.numpy as jnp

import haliax
import haliax as hax
import haliax.nn.activations
import haliax.nn.attention as attention
import haliax.nn.normalization

from ..axis import Axis
from ..core import NamedArray
from .activations import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    quick_gelu,
    relu,
    relu6,
    relu_squared,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softplus,
    swish,
)
from .conv import Conv, ConvTranspose
from .dropout import Dropout, dropout
from .embedding import Embedding
from .linear import Linear, MoELinear
from .loss import binary_cross_entropy_loss, cross_entropy_loss, cross_entropy_loss_and_log_normalizers, reduce_loss
from .mlp import MLP
from .normalization import LayerNorm, RmsNorm, log_softmax, logsumexp, softmax, standardize
from .pool import max_pool, mean_pool, min_pool
from .scan import BlockSeq, ScanCheckpointPolicy, Stacked


def one_hot(x: NamedArray | int, class_axis: Axis, *, dtype=None) -> NamedArray:
    """
    Convert an integer to a one-hot vector. This is basically a generalization of [jax.nn.one_hot][]
    for NamedArrays.

    Args:
        x: the integer or NamedArray of integers to convert
        class_axis: the axis to convert to one-hot
        dtype: the dtype of the result. If None, it will default to jax's default (currently float_)
    Returns:
        a NamedArray with the same axes as `x` plus `class_axis`, with 1s in the appropriate places
    """
    if isinstance(x, NamedArray):
        array = jnn.one_hot(x.array, num_classes=class_axis.size, dtype=dtype)
        # Disabling this to prevent a crash in XLA on GPU
        # return hax.auto_sharded(hax.named(array, x.axes + (class_axis,)))
        return hax.named(array, x.axes + (class_axis,))

    x_array = jnp.asarray(x)
    if x_array.ndim != 0:
        raise TypeError("one_hot expects a scalar integer or NamedArray, " f"but got array with shape {x_array.shape}")

    if isinstance(x, numbers.Integral):
        assert class_axis.size > x >= -class_axis.size
    else:
        if not (jnp.issubdtype(x_array.dtype, jnp.integer) or jnp.issubdtype(x_array.dtype, jnp.bool_)):
            raise TypeError(
                "one_hot expects an integer scalar or NamedArray, " f"but got value with dtype {x_array.dtype}"
            )

    one = 1
    if dtype is not None:
        one = dtype(one)

    array = jnp.zeros(class_axis.size, dtype=dtype).at[x_array].set(one)
    return hax.auto_sharded(haliax.named(array, class_axis))


__all__ = [
    "attention",
    "one_hot",
    "binary_cross_entropy_loss",
    "reduce_loss",
    "cross_entropy_loss",
    "cross_entropy_loss_and_log_normalizers",
    "Conv",
    "ConvTranspose",
    "Dropout",
    "dropout",
    "LayerNorm",
    "Linear",
    "MoELinear",
    "Embedding",
    "RmsNorm",
    "Stacked",
    "BlockSeq",
    "MLP",
    "relu",
    "gelu",
    "quick_gelu",
    "glu",
    "relu6",
    "relu_squared",
    "sigmoid",
    "soft_sign",
    "softplus",
    "swish",
    "silu",
    "log_sigmoid",
    "leaky_relu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "logsumexp",
    "softmax",
    "log_softmax",
    "standardize",
    "elu",
    "celu",
    "selu",
    "max_pool",
    "mean_pool",
    "min_pool",
    "ScanCheckpointPolicy",
]
