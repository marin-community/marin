# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
from abc import abstractmethod
from typing import TypeVar

import equinox as eqx
from jax import nn as jnn
from jax import numpy as jnp

import haliax
import haliax as hax

from .._src.state_dict import Mod, ModuleWithStateDictSerialization
from ..axis import AxisSelection, AxisSpec
from ..core import NamedArray
from ..types import Scalar
from ..wrap import unwrap_namedarrays, wrap_axiswise_call, wrap_reduction_call

A = TypeVar("A", Scalar, NamedArray, jnp.ndarray)


class LayerNormBase(ModuleWithStateDictSerialization):
    axis: AxisSpec = eqx.field(static=True)
    weight: NamedArray | None
    bias: NamedArray | None
    eps: float = eqx.field(default=1e-5, static=True)
    dtype: jnp.dtype | None = eqx.field(default=None, static=True)

    @abstractmethod
    def __call__(self, x: NamedArray) -> NamedArray:
        pass

    @classmethod
    def init(
        cls,
        axis: AxisSpec,
        eps: float = 1e-5,
        *,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype: jnp.dtype | None = None,
    ):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None

        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return cls(axis, weight, bias, eps, dtype)

    def flatten_for_export(self: Mod) -> Mod:
        if isinstance(self.axis, hax.Axis):
            return self

        if self.weight is not None:
            weight = self.weight.flatten("__OUT")
        else:
            weight = None

        if self.bias is not None:
            bias = self.bias.flatten("__OUT")
        else:
            bias = None

        return dataclasses.replace(self, weight=weight, bias=bias, axis=hax.flatten_axes(self.axis, "__OUT"))

    def unflatten_from_export(self: Mod, template: Mod) -> Mod:
        if template.axis == self.axis:
            return self

        if self.weight is not None:
            assert isinstance(self.axis, hax.Axis), "Cannot unflatten weight with non-axis axis"
            weight = hax.unflatten_axis(self.weight, self.axis, template.axis)
        else:
            weight = None

        if self.bias is not None:
            assert isinstance(self.axis, hax.Axis), "Cannot unflatten weight with non-axis axis"
            bias = hax.unflatten_axis(self.bias, self.axis, template.axis)

        else:
            bias = None

        return dataclasses.replace(self, weight=weight, bias=bias, axis=template.axis)


class LayerNorm(LayerNormBase):
    r"""
    Normalises the input along the specified axis (or axes), using the mean and variance of the
    input along that axis.
    """

    axis: AxisSpec = eqx.field(static=True)
    weight: NamedArray | None
    bias: NamedArray | None

    eps: float = eqx.field(default=1e-5, static=True)
    dtype: jnp.dtype | None = eqx.field(default=None, static=True)

    def __call__(self, x: NamedArray) -> NamedArray:
        dtype = x.dtype
        mean = x.mean(self.axis)
        var = x.var(self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = (x - mean) * inv
        out = out.astype(dtype)

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias
        return out


class RmsNorm(LayerNormBase):
    r"""
    Implements RMS normalization, which normalizes the input by dividing by the root mean square of the input.
    """

    def __call__(self, x: NamedArray) -> NamedArray:
        in_dtype = x.dtype
        x = x.astype(self.dtype)
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv
        out = out.astype(in_dtype)

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias
        return out


def logsumexp(a: A, axis: AxisSelection | None = None) -> A:
    # TODO: logsumexp indirectly supports where via `b`. we should support it directly
    return wrap_reduction_call(jnn.logsumexp, a, axis=axis, single_axis_only=False, supports_where=False)


# TODO: support where in softmax, etc


def softmax(a: A, axis: AxisSelection | None = None) -> A:
    return wrap_axiswise_call(jnn.softmax, a, axis=axis, single_axis_only=False)


def log_softmax(a: A, axis: AxisSelection | None = None) -> A:
    return wrap_axiswise_call(jnn.log_softmax, a, axis=axis, single_axis_only=False)


def standardize(
    x: NamedArray,
    axis: AxisSpec,
    *,
    mean: NamedArray | None = None,
    variance: NamedArray | None = None,
    epsilon: float = 1e-5,
    where: NamedArray | None = None,
) -> NamedArray:
    """Analogous to [jax.nn.standardize][], but with support for NamedArrays."""
    x, mean, variance, where = haliax.broadcast_arrays(x, mean, variance, where)  # type: ignore
    raw_x, mean, variance, where = unwrap_namedarrays(x, mean, variance, where)
    axis_indices = x.axis_indices(axis)

    plain = jnn.standardize(raw_x, axis_indices, mean=mean, variance=variance, epsilon=epsilon, where=where)
    return NamedArray(plain, x.axes)
