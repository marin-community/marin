# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


"""Wrappers around jax.random functions."""

import jax.random as jrandom

import haliax
from haliax.core import NamedArray, NamedOrNumeric, broadcast_to

from .axis import (
    Axis,
    AxisSelector,
    AxisSpec,
    axis_spec_to_shape_dict,
    axis_spec_to_tuple,
    concat_axes,
    selects_axis,
    to_jax_shape,
)
from .jax_utils import named_call


@named_call
def uniform(
    key, shape: AxisSpec, dtype=float, minval: NamedOrNumeric = 0.0, maxval: NamedOrNumeric = 1.0
) -> NamedArray:
    shape = axis_spec_to_shape_dict(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.uniform(key=key, shape=jax_shape, dtype=dtype, minval=minval, maxval=maxval)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def normal(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.normal(key=key, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def bernoulli(key, shape: AxisSpec, p: NamedOrNumeric):
    shape = axis_spec_to_shape_dict(shape)
    p = broadcast_to(p, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.bernoulli(key=key, p=p, shape=jax_shape)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def randint(key, shape: AxisSpec, minval: NamedOrNumeric, maxval: NamedOrNumeric, dtype=int):
    shape = axis_spec_to_shape_dict(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.randint(key=key, shape=jax_shape, minval=minval, maxval=maxval, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def poisson(key, shape: AxisSpec, lam: NamedOrNumeric, dtype=int):
    shape = axis_spec_to_shape_dict(shape)
    lam = broadcast_to(lam, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.poisson(key=key, lam=lam, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def exponential(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.exponential(key=key, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def gamma(key, shape: AxisSpec, a: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    a = broadcast_to(a, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.gamma(key=key, a=a, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def beta(key, shape: AxisSpec, a: NamedOrNumeric, b: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    a = broadcast_to(a, shape).array
    b = broadcast_to(b, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.beta(key=key, a=a, b=b, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def laplace(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.laplace(key=key, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def cauchy(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.cauchy(key=key, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def logistic(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.logistic(key=key, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def truncated_normal(key, shape: AxisSpec, lower: NamedOrNumeric, upper: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    lower = broadcast_to(lower, shape).array
    upper = broadcast_to(upper, shape).array
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.truncated_normal(key=key, lower=lower, upper=upper, shape=jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def ball(key, shape: AxisSpec, D: Axis, p: float = 2.0, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.ball(key=key, shape=jax_shape, d=D.size, p=p, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, concat_axes(shape, D)))


@named_call
def choice(key, shape: AxisSpec, a: NamedArray, axis: AxisSelector, replace: bool = True, p: NamedArray | None = None):
    """
    Selects random elements from an array along the given axis. If p is provided, the elements are selected
    with probability proportional to their weights and it must be a 1-d array with its only axis being the axis.
    shape and a.axes must not overlap except that axis may be repeated in both.

    :return: Array with shape `shape` + (`a.axes` - `axis`)
    """

    index = a.axis_indices(axis)
    assert index is not None, f"axis {axis} not in a"

    shape = axis_spec_to_shape_dict(shape)
    if p is not None:
        assert p.resolve_axis(axis_spec_to_tuple(axis)) == p.axes, f"p must be 1D with axis {axis} or be None"

    jax_shape = to_jax_shape(shape)
    jax_p = p.array if p is not None else None

    jax_array = jrandom.choice(key, a.array, jax_shape, replace=replace, p=jax_p, axis=index)

    expected_shape = concat_axes(shape, tuple(a.axes[:index] + a.axes[index + 1 :]))

    return haliax.auto_sharded(NamedArray(jax_array, expected_shape))


@named_call
def categorical(key, logits: NamedArray, axis: AxisSelector, shape: AxisSpec | None = None):
    """Sample random values from categorical distributions.

    Args:
      key: a PRNG key used as the random key.
      logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
        so that `softmax(logits, axis)` gives the corresponding probabilities.
      axis: Axis along which logits belong to the same categorical distribution.
      shape: A tuple of axes representing the result shape, or None. if None, the shape is
        `logits.axes - {axis}`. If not None, `logits.axes - {axis}`  must be a subset of shape.
    Returns:
      A random array with int dtype and shape given by ``shape``
    """
    axis = logits.resolve_axis(axis)
    if shape is None:
        shape = tuple(a for a in logits.axes if a != axis)
    else:
        shape = axis_spec_to_tuple(shape)

    # TODO: could alias the axis and rename at end
    if selects_axis(shape, axis):
        raise ValueError(f"axis {axis} cannot be in shape {shape}")

    logits = logits.broadcast_axis(shape)

    index = logits.axis_indices(axis)
    assert index is not None, f"axis {axis} not in logits"

    jax_shape = to_jax_shape(shape)

    jax_array = jrandom.categorical(key, logits.array, axis=index, shape=jax_shape)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def gumbel(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.gumbel(key, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def permutation(key, x: NamedArray, axis: AxisSelector, independent: bool = False):
    axis_index = x.axis_indices(axis)
    jax_array = jrandom.permutation(key, x.array, axis_index, independent=independent)
    return haliax.auto_sharded(NamedArray(jax_array, x.axes))


@named_call
def rademacher(key, shape: AxisSpec, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.rademacher(key, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def t(key, shape: AxisSpec, df: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    df = broadcast_to(df, shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.t(key, df.array, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def weibull_min(key, shape: AxisSpec, scale: NamedOrNumeric, concentration: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    scale = broadcast_to(scale, shape)
    concentration = broadcast_to(concentration, shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.weibull_min(key, scale.array, concentration.array, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def pareto(key, shape: AxisSpec, b: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    b = broadcast_to(b, shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.pareto(key, b.array, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


@named_call
def loggamma(key, shape: AxisSpec, a: NamedOrNumeric, dtype=float):
    shape = axis_spec_to_shape_dict(shape)
    a = broadcast_to(a, shape)
    jax_shape = to_jax_shape(shape)
    jax_array = jrandom.loggamma(key, a.array, jax_shape, dtype=dtype)
    return haliax.auto_sharded(NamedArray(jax_array, shape))


__all__ = [
    "uniform",
    "normal",
    "ball",
    "bernoulli",
    "beta",
    "cauchy",
    "choice",
    "exponential",
    "gamma",
    "gumbel",
    "laplace",
    "logistic",
    "permutation",
    "poisson",
    "rademacher",
    "truncated_normal",
    # "categorical",
    # "dirichlet",
    "loggamma",
    "pareto",
    "t",
    "weibull_min",
]
