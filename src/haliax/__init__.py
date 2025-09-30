# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import typing as t
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike


import haliax.debug as debug
import haliax.nn as nn
import haliax.quantization as quantization
import haliax.random as random
import haliax.state_dict as state_dict
import haliax.tree_util as tree_util
import haliax.util as util
from .field import field

from ._src.dot import dot
from ._src.einsum import einsum
from ._src.rearrange import rearrange
from ._src.scan import ScanCheckpointPolicy
from .axis import (
    Axis,
    AxisSelection,
    AxisSelector,
    AxisSpec,
    axis_name,
    axis_size,
    axis_spec_to_tuple,
    concat_axes,
    dblock,
    ds,
    dslice,
    eliminate_axes,
    make_axes,
    replace_axis,
    resolve_axis,
    selects_axis,
    to_jax_shape,
)
from .core import (
    NamedArray,
    NamedArrayAxes,
    NamedArrayAxesSpec,
    NamedOrNumeric,
    are_shape_checks_enabled,
    broadcast_arrays,
    broadcast_axis,
    broadcast_to,
    enable_shape_checks,
    flatten,
    flatten_axes,
    index,
    named,
    ravel,
    rename,
    roll,
    slice,
    split,
    take,
    unbind,
    unflatten_axis,
    updated_slice,
)
from .haxtyping import Named
from .hof import fold, map, scan, vmap
from .jax_utils import tree_checkpoint_name
from .ops import (
    clip,
    allclose,
    array_equal,
    array_equiv,
    isclose,
    pad_left,
    pad,
    trace,
    tril,
    triu,
    nonzero,
    unique,
    unique_values,
    unique_counts,
    unique_inverse,
    unique_all,
    packbits,
    unpackbits,
    searchsorted,
    bincount,
    where,
)

from .poly import (
    poly,
    polyadd,
    polysub,
    polymul,
    polydiv,
    polyint,
    polyder,
    polyval,
    polyfit,
    roots,
    trim_zeros,
    vander,
)

from .fft import (
    fft,
    fftfreq,
    fftshift,
    hfft,
    ifft,
    ifftshift,
    ihfft,
    irfft,
    rfft,
    rfftfreq,
)
from .partitioning import auto_sharded, axis_mapping, fsdp, named_jit, set_mesh, shard, shard_with_axis_mapping
from .specialized_fns import top_k
from .types import Scalar
from .util import is_named_array
from .wrap import (
    ReductionFunction,
    SimpleReductionFunction,
    wrap_axiswise_call,
    wrap_elemwise_binary,
    wrap_elemwise_unary,
    wrap_reduction_call,
)

T = t.TypeVar("T")
A = t.TypeVar("A", Scalar, NamedArray, jnp.ndarray)


# creation routines
def zeros(shape: AxisSpec, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """Creates a NamedArray with all elements set to 0"""
    if dtype is None:
        dtype = jnp.float32
    return full(shape, 0, dtype)


def ones(shape: AxisSpec, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """Creates a NamedArray with all elements set to 1"""
    if dtype is None:
        dtype = jnp.float32
    return full(shape, 1, dtype)


def full(shape: AxisSpec, fill_value: T, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """Creates a NamedArray with all elements set to `fill_value`"""
    if isinstance(shape, Axis):
        return NamedArray(jnp.full(shape=shape.size, fill_value=fill_value, dtype=dtype), (shape,))
    else:
        x_shape = to_jax_shape(shape)
        return NamedArray(jnp.full(shape=x_shape, fill_value=fill_value, dtype=dtype), shape)


def zeros_like(a: NamedArray, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 0"""
    return NamedArray(jnp.zeros_like(a.array, dtype=dtype), a.axes)


def ones_like(a: NamedArray, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 1"""
    return NamedArray(jnp.ones_like(a.array, dtype=dtype), a.axes)


def full_like(a: NamedArray, fill_value: T, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """Creates a NamedArray with all elements set to `fill_value`"""
    return NamedArray(jnp.full_like(a.array, fill_value, dtype=dtype), a.axes)


def arange(axis: AxisSpec, *, start=0, step=1, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """
    Version of jnp.arange that returns a NamedArray.

    This version differs from jnp.arange (beyond the obvious NamedArray) in two ways:

    1) It can work with a start that is a tracer (i.e. a JAX expression), whereas jax arange is not able to handle
    tracers.
    2) Axis can be more than one axis, in  which case it's equivalent to arange of the product of sizes, followed by
    reshape.

    Examples

    ```python
    X, Y = hax.make_axes(X=3, Y=4)
    # Create a NamedArray along a single axis
    arr = hax.arange(X)  # equivalent to jnp.arange(0, 3, 1)
    # 2D
    arr = hax.arange((X, Y))  # equivalent to jnp.arange(0, 12, 1).reshape(3, 4)
    ```

    """
    size = axis_size(axis)

    arr = jax.lax.iota(dtype=dtype or jnp.result_type(start), size=size) * step + start
    arr = arr.reshape(to_jax_shape(axis))
    return NamedArray(arr, axis_spec_to_tuple(axis))


# TODO: add overrides for arraylike start/stop to linspace, logspace, geomspace
def linspace(
    axis: AxisSelector, *, start: float, stop: float, endpoint: bool = True, dtype: Optional[DTypeLike] = None
) -> NamedArray:
    """
    Version of jnp.linspace that returns a NamedArray.
    If `axis` is a string, the default number of samples (50, per numpy) will be used.
    """
    if isinstance(axis, str):
        axis = Axis(axis, 50)
    return NamedArray(jnp.linspace(start, stop, axis.size, endpoint=endpoint, dtype=dtype), (axis,))


def logspace(
    axis: AxisSelector,
    *,
    start: float,
    stop: float,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    """
    Version of jnp.logspace that returns a NamedArray.
    If `axis` is a string, the default number of samples (50, per numpy) will be used.
    """
    if isinstance(axis, str):
        axis = Axis(axis, 50)
    return NamedArray(jnp.logspace(start, stop, axis.size, endpoint=endpoint, base=base, dtype=dtype), (axis,))


def geomspace(
    axis: AxisSelector, *, start: float, stop: float, endpoint: bool = True, dtype: Optional[DTypeLike] = None
) -> NamedArray:
    """
    Version of jnp.geomspace that returns a NamedArray.
    If `axis` is a string, the default number of samples (50, per numpy) will be used.
    """
    if isinstance(axis, str):
        axis = Axis(axis, 50)
    return NamedArray(jnp.geomspace(start, stop, axis.size, endpoint=endpoint, dtype=dtype), (axis,))


def stack(axis: AxisSelector, arrays: Sequence[NamedArray]) -> NamedArray:
    """Version of [jax.numpy.stack][] that returns a NamedArray"""
    if isinstance(axis, str):
        axis = Axis(axis, len(arrays))
    if len(arrays) == 0:
        return zeros(axis)
    arrays = [a.rearrange(arrays[0].axes) for a in arrays]
    return NamedArray(jnp.stack([a.array for a in arrays], axis=0), (axis,) + arrays[0].axes)


def repeat(
    a: NamedArray, repeats: int | jnp.ndarray, axis: AxisSelector, total_repeat_length: Optional[int] = None
) -> NamedArray:
    """Version of [jax.numpy.repeat][] that returns a NamedArray"""
    index = a.axis_indices(axis)
    if index is None:
        raise ValueError(f"Axis {axis} not found in array {a}")

    return named(
        jnp.repeat(a.array, repeats, axis=index, total_repeat_length=total_repeat_length),
        a.axes[:index] + (axis_name(axis),) + a.axes[index + 1 :],
    )


def tile(a: NamedArray, reps: dict[AxisSelector, int]) -> NamedArray:
    """
    Version of [jax.numpy.tile][] that returns a NamedArray.

    As with the non-named tile, you can add new axes by passing a dict with an axis name as the key
    and the number of reps as the value. The size of the axis (if it exists) will be ignored for new dims.
    That is, the size of the resulting axis will be the number of reps for a new axis, and the size of the
    original axis times the number of reps for an existing axis.
    """
    # we need to convert the reps to a sequence of ints
    new_dims = []
    dim_reps = [1] * len(a.axes)
    for ax, i in reps.items():
        index = a.axis_indices(ax)
        if index is None:
            new_dims.append(Axis(axis_name(ax), i))
        else:
            dim_reps[index] = i

    if len(new_dims) > 0:
        dim_reps = [ax.size for ax in new_dims] + dim_reps

    out_axes = tuple(new_dims) + tuple(ax.name for ax in a.axes)

    return named(jnp.tile(a.array, dim_reps), out_axes)


def concatenate(axis: AxisSelector, arrays: Sequence[NamedArray]) -> NamedArray:
    """Version of [jax.numpy.concatenate][] that returns a NamedArray. The returns array will have the same axis names in the
    same order as the first, with the selected axis extended by the sum of the sizes of the selected axes in the
    concatenated arrays."""
    aname = axis_name(axis)
    total_size: int = _sum(a.resolve_axis(aname).size for a in arrays)  # type: ignore
    if isinstance(axis, str):
        axis = Axis(axis, total_size)
    elif total_size != axis.size:
        raise ValueError(
            f"Cannot concatenate arrays along axis {aname} of size {axis.size} with total size {total_size}"
        )

    if len(arrays) == 0:
        return zeros(axis)

    axis_index = arrays[0].axis_indices(aname)
    if axis_index is None:
        raise ValueError(f"Axis {aname} not found in 0th array {arrays[0]}")

    axes: tuple[AxisSelector, ...] = arrays[0].axes
    # we want to use the axis name for `axis`, because it's not uncommon for those to be different lengths in the arrays
    axes = axes[:axis_index] + (aname,) + axes[axis_index + 1 :]
    arrays = [a.rearrange(axes) for a in arrays]

    new_axes = arrays[0].axes[:axis_index] + (axis,) + arrays[0].axes[axis_index + 1 :]
    return NamedArray(jnp.concatenate([a.array for a in arrays], axis=axis_index), new_axes)


# elementwise unary operations
def abs(a: A) -> A:
    return wrap_elemwise_unary(jnp.abs, a)


def absolute(a: A) -> A:
    return wrap_elemwise_unary(jnp.absolute, a)


def angle(a: A) -> A:
    return wrap_elemwise_unary(jnp.angle, a)


def arccos(a: A) -> A:
    return wrap_elemwise_unary(jnp.arccos, a)


def arccosh(a: A) -> A:
    return wrap_elemwise_unary(jnp.arccosh, a)


def arcsin(a: A) -> A:
    return wrap_elemwise_unary(jnp.arcsin, a)


def arcsinh(a: A) -> A:
    return wrap_elemwise_unary(jnp.arcsinh, a)


def arctan(a: A) -> A:
    return wrap_elemwise_unary(jnp.arctan, a)


def arctanh(a: A) -> A:
    return wrap_elemwise_unary(jnp.arctanh, a)


def around(a: A) -> A:
    return wrap_elemwise_unary(jnp.around, a)


def bitwise_count(a: A) -> A:
    return wrap_elemwise_unary(jnp.bitwise_count, a)


def bitwise_invert(a: A) -> A:
    return wrap_elemwise_unary(jnp.bitwise_invert, a)


def bitwise_not(a: A) -> A:
    return wrap_elemwise_unary(jnp.bitwise_not, a)


def cbrt(a: A) -> A:
    return wrap_elemwise_unary(jnp.cbrt, a)


def ceil(a: A) -> A:
    return wrap_elemwise_unary(jnp.ceil, a)


def conj(a: A) -> A:
    return wrap_elemwise_unary(jnp.conj, a)


def conjugate(a: A) -> A:
    return wrap_elemwise_unary(jnp.conjugate, a)


def copy(a: A) -> A:
    return wrap_elemwise_unary(jnp.copy, a)


def cos(a: A) -> A:
    return wrap_elemwise_unary(jnp.cos, a)


def cosh(a: A) -> A:
    return wrap_elemwise_unary(jnp.cosh, a)


def deg2rad(a: A) -> A:
    return wrap_elemwise_unary(jnp.deg2rad, a)


def degrees(a: A) -> A:
    return wrap_elemwise_unary(jnp.degrees, a)


def exp(a: A) -> A:
    return wrap_elemwise_unary(jnp.exp, a)


def exp2(a: A) -> A:
    return wrap_elemwise_unary(jnp.exp2, a)


def expm1(a: A) -> A:
    return wrap_elemwise_unary(jnp.expm1, a)


def fabs(a: A) -> A:
    return wrap_elemwise_unary(jnp.fabs, a)


def fix(a: A) -> A:
    return wrap_elemwise_unary(jnp.fix, a)


def floor(a: A) -> A:
    return wrap_elemwise_unary(jnp.floor, a)


def frexp(a: A) -> A:
    return wrap_elemwise_unary(jnp.frexp, a)


def i0(a: A) -> A:
    return wrap_elemwise_unary(jnp.i0, a)


def imag(a: A) -> A:
    return wrap_elemwise_unary(jnp.imag, a)


def invert(a: A) -> A:
    return wrap_elemwise_unary(jnp.invert, a)


def iscomplex(a: A) -> A:
    return wrap_elemwise_unary(jnp.iscomplex, a)


def isfinite(a: A) -> A:
    return wrap_elemwise_unary(jnp.isfinite, a)


def isinf(a: A) -> A:
    return wrap_elemwise_unary(jnp.isinf, a)


def isnan(a: A) -> A:
    return wrap_elemwise_unary(jnp.isnan, a)


def isneginf(a: A) -> A:
    return wrap_elemwise_unary(jnp.isneginf, a)


def isposinf(a: A) -> A:
    return wrap_elemwise_unary(jnp.isposinf, a)


def isreal(a: A) -> A:
    return wrap_elemwise_unary(jnp.isreal, a)


def log(a: A) -> A:
    return wrap_elemwise_unary(jnp.log, a)


def log10(a: A) -> A:
    return wrap_elemwise_unary(jnp.log10, a)


def log1p(a: A) -> A:
    return wrap_elemwise_unary(jnp.log1p, a)


def log2(a: A) -> A:
    return wrap_elemwise_unary(jnp.log2, a)


def logical_not(a: A) -> A:
    return wrap_elemwise_unary(jnp.logical_not, a)


def ndim(a: A) -> A:
    return wrap_elemwise_unary(jnp.ndim, a)


def negative(a: A) -> A:
    return wrap_elemwise_unary(jnp.negative, a)


def positive(a: A) -> A:
    return wrap_elemwise_unary(jnp.positive, a)


def rad2deg(a: A) -> A:
    return wrap_elemwise_unary(jnp.rad2deg, a)


def radians(a: A) -> A:
    return wrap_elemwise_unary(jnp.radians, a)


def real(a: A) -> A:
    return wrap_elemwise_unary(jnp.real, a)


def reciprocal(a: A) -> A:
    return wrap_elemwise_unary(jnp.reciprocal, a)


def rint(a: A) -> A:
    return wrap_elemwise_unary(jnp.rint, a)


def round(a: A, decimals: int = 0) -> A:
    return wrap_elemwise_unary(jnp.round, a, decimals=decimals)


def rsqrt(a: A) -> A:
    return wrap_elemwise_unary(jax.lax.rsqrt, a)  # nb this is in lax


def sign(a: A) -> A:
    return wrap_elemwise_unary(jnp.sign, a)


def signbit(a: A) -> A:
    return wrap_elemwise_unary(jnp.signbit, a)


def sin(a: A) -> A:
    return wrap_elemwise_unary(jnp.sin, a)


def sinc(a: A) -> A:
    return wrap_elemwise_unary(jnp.sinc, a)


def sinh(a: A) -> A:
    return wrap_elemwise_unary(jnp.sinh, a)


def square(a: A) -> A:
    return wrap_elemwise_unary(jnp.square, a)


def sqrt(a: A) -> A:
    return wrap_elemwise_unary(jnp.sqrt, a)


def tan(a: A) -> A:
    return wrap_elemwise_unary(jnp.tan, a)


def tanh(a: A) -> A:
    return wrap_elemwise_unary(jnp.tanh, a)


def trunc(a: A) -> A:
    return wrap_elemwise_unary(jnp.trunc, a)


# Reduction functions
def all(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    """
    Named version of [jax.numpy.all](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.all.html#jax.numpy.all).
    """
    return wrap_reduction_call(jnp.all, array, axis, where, single_axis_only=False, supports_where=True)


def amax(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    """
    Aliax for max. See max for details.
    """
    return wrap_reduction_call(jnp.amax, array, axis, where, single_axis_only=False, supports_where=True)


def amin(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    """
    Aliax for min. See min for details.
    """
    return wrap_reduction_call(jnp.amin, array, axis, where, single_axis_only=False, supports_where=True)


def any(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    """True if any elements along a given axis or axes are True. If axis is None, any elements are True."""
    return wrap_reduction_call(jnp.any, array, axis, where, single_axis_only=False, supports_where=True)


def argmax(array: NamedArray, axis: Optional[AxisSelector]) -> NamedArray:
    return wrap_reduction_call(jnp.argmax, array, axis, None, single_axis_only=True, supports_where=False)


def argmin(array: NamedArray, axis: Optional[AxisSelector]) -> NamedArray:
    return wrap_reduction_call(jnp.argmin, array, axis, None, single_axis_only=True, supports_where=False)


def max(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    return wrap_reduction_call(jnp.max, array, axis, where, single_axis_only=False, supports_where=True)


def mean(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(jnp.mean, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype)


def min(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    return wrap_reduction_call(jnp.min, array, axis, where, single_axis_only=False, supports_where=True)


def prod(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(jnp.prod, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype)


def std(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    ddof: int = 0,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.std, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype, ddof=ddof
    )


def ptp(array: NamedArray, axis: Optional[AxisSelection] = None, *, where: Optional[NamedArray] = None) -> NamedArray:
    return wrap_reduction_call(jnp.ptp, array, axis, where, single_axis_only=False, supports_where=True)


def product(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.product, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype
    )


_sum = sum


def sum(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(jnp.sum, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype)


def var(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    ddof: int = 0,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.var, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype, ddof=ddof
    )


def nanargmax(array: NamedArray, axis: Optional[AxisSelector] = None) -> NamedArray:
    return wrap_reduction_call(jnp.nanargmax, array, axis, None, single_axis_only=True, supports_where=False)


def nanargmin(array: NamedArray, axis: Optional[AxisSelector] = None) -> NamedArray:
    return wrap_reduction_call(jnp.nanargmin, array, axis, None, single_axis_only=True, supports_where=False)


def nanmax(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
) -> NamedArray:
    return wrap_reduction_call(jnp.nanmax, array, axis, where, single_axis_only=False, supports_where=True)


def nanmean(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.nanmean, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype
    )


def nanmin(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
) -> NamedArray:
    return wrap_reduction_call(jnp.nanmin, array, axis, where, single_axis_only=False, supports_where=True)


def nanprod(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.nanprod, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype
    )


def nanstd(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    ddof: int = 0,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.nanstd, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype, ddof=ddof
    )


def nansum(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.nansum, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype
    )


def nanvar(
    array: NamedArray,
    axis: Optional[AxisSelection] = None,
    *,
    where: Optional[NamedArray] = None,
    ddof: int = 0,
    dtype: Optional[DTypeLike] = None,
) -> NamedArray:
    return wrap_reduction_call(
        jnp.nanvar, array, axis, where, single_axis_only=False, supports_where=True, dtype=dtype, ddof=ddof
    )


# "Normalization" functions that use an axis but don't change the shape


def cumsum(a: NamedArray, axis: AxisSelector, *, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """
    Named version of [jax.numpy.cumsum](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cumsum.html)
    """
    return wrap_axiswise_call(jnp.cumsum, a, axis, dtype=dtype, single_axis_only=True)


def cumprod(a: NamedArray, axis: AxisSelector, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """
    Named version of [jax.numpy.cumprod](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cumprod.html)
    """
    return wrap_axiswise_call(jnp.cumprod, a, axis, dtype=dtype, single_axis_only=True)


def nancumsum(a: NamedArray, axis: AxisSelector, *, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """
    Named version of [jax.numpy.nancumsum](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nancumsum.html)
    """
    return wrap_axiswise_call(jnp.nancumsum, a, axis, dtype=dtype, single_axis_only=True)


def nancumprod(a: NamedArray, axis: AxisSelector, dtype: Optional[DTypeLike] = None) -> NamedArray:
    """
    Named version of [jax.numpy.nancumprod](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nancumprod.html)
    """
    return wrap_axiswise_call(jnp.nancumprod, a, axis, dtype=dtype, single_axis_only=True)


def sort(a: NamedArray, axis: AxisSelector) -> NamedArray:
    """
    Named version of [jax.numpy.sort](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html)
    """
    return wrap_axiswise_call(jnp.sort, a, axis, single_axis_only=True)


def argsort(a: NamedArray, axis: AxisSelector | None, *, stable: bool = False) -> NamedArray:
    """
    Named version of [jax.numpy.argsort](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.argsort.html).

    If `axis` is None, the returned array will be a 1D array of indices that would sort the flattened array,
    identical to `jax.numpy.argsort(a.array)`.

    Args:
        stable: If ``True``, ensures that the indices of equal elements preserve their relative order.
    """
    return wrap_axiswise_call(jnp.argsort, a, axis, single_axis_only=True, stable=stable)


# elemwise binary ops


# Note that all the heavy lifting is done by the `wrap_elemwise_binary` decorator
@wrap_elemwise_binary
def add(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.add](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html)
    """
    return jnp.add(x1, x2)  # type: ignore


@wrap_elemwise_binary
def arctan2(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.arctan2](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arctan2.html)
    """
    return jnp.arctan2(x1, x2)  # type: ignore


@wrap_elemwise_binary
def bitwise_and(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.bitwise_and](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_and.html)
    """
    return jnp.bitwise_and(x1, x2)  # type: ignore


@wrap_elemwise_binary
def bitwise_left_shift(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.bitwise_left_shift](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_left_shift.html)
    """
    return jnp.bitwise_left_shift(x1, x2)  # type: ignore


@wrap_elemwise_binary
def bitwise_or(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.bitwise_or](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_or.html)
    """
    return jnp.bitwise_or(x1, x2)  # type: ignore


@wrap_elemwise_binary
def bitwise_right_shift(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.bitwise_right_shift](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_right_shift.html)
    """
    return jnp.bitwise_right_shift(x1, x2)  # type: ignore


@wrap_elemwise_binary
def bitwise_xor(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.bitwise_xor](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_xor.html)
    """
    return jnp.bitwise_xor(x1, x2)  # type: ignore


@wrap_elemwise_binary
def divide(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.divide](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.divide.html)
    """
    return jnp.divide(x1, x2)  # type: ignore


@wrap_elemwise_binary
def divmod(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.divmod](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.divmod.html)
    """
    return jnp.divmod(x1, x2)  # type: ignore


@wrap_elemwise_binary
def equal(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.equal](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.equal.html)
    """
    return jnp.equal(x1, x2)  # type: ignore


@wrap_elemwise_binary
def float_power(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.float_power](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.float_power.html)
    """
    return jnp.float_power(x1, x2)  # type: ignore


@wrap_elemwise_binary
def floor_divide(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.floor_divide](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.floor_divide.html)
    """
    return jnp.floor_divide(x1, x2)  # type: ignore


@wrap_elemwise_binary
def fmax(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.fmax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmax.html)
    """
    return jnp.fmax(x1, x2)  # type: ignore


@wrap_elemwise_binary
def fmin(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.fmin](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmin.html)
    """
    return jnp.fmin(x1, x2)  # type: ignore


@wrap_elemwise_binary
def fmod(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.fmod](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmod.html)
    """
    return jnp.fmod(x1, x2)  # type: ignore


@wrap_elemwise_binary
def greater(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.greater](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.greater.html)
    """
    return jnp.greater(x1, x2)  # type: ignore


@wrap_elemwise_binary
def greater_equal(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.greater_equal](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.greater_equal.html)
    """
    return jnp.greater_equal(x1, x2)  # type: ignore


@wrap_elemwise_binary
def hypot(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.hypot](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.hypot.html)
    """
    return jnp.hypot(x1, x2)  # type: ignore


@wrap_elemwise_binary
def left_shift(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.left_shift](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.left_shift.html)
    """
    return jnp.left_shift(x1, x2)  # type: ignore


@wrap_elemwise_binary
def less(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.less](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.less.html)
    """
    return jnp.less(x1, x2)  # type: ignore


@wrap_elemwise_binary
def less_equal(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.less_equal](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.less_equal.html)
    """
    return jnp.less_equal(x1, x2)  # type: ignore


@wrap_elemwise_binary
def logaddexp(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    """
    Named version of [jax.numpy.logaddexp](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logaddexp.html)
    """
    return jnp.logaddexp(x1, x2)  # type: ignore


@wrap_elemwise_binary
def logaddexp2(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.logaddexp2(x1, x2)  # type: ignore


@wrap_elemwise_binary
def logical_and(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.logical_and(x1, x2)  # type: ignore


@wrap_elemwise_binary
def logical_or(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.logical_or(x1, x2)  # type: ignore


@wrap_elemwise_binary
def logical_xor(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.logical_xor(x1, x2)  # type: ignore


@wrap_elemwise_binary
def maximum(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.maximum(x1, x2)  # type: ignore


@wrap_elemwise_binary
def minimum(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.minimum(x1, x2)  # type: ignore


@wrap_elemwise_binary
def mod(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.mod(x1, x2)  # type: ignore


@wrap_elemwise_binary
def multiply(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.multiply(x1, x2)  # type: ignore


@wrap_elemwise_binary
def nextafter(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.nextafter(x1, x2)  # type: ignore


@wrap_elemwise_binary
def not_equal(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.not_equal(x1, x2)  # type: ignore


@wrap_elemwise_binary
def power(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.power(x1, x2)  # type: ignore


@wrap_elemwise_binary
def remainder(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.remainder(x1, x2)  # type: ignore


@wrap_elemwise_binary
def right_shift(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.right_shift(x1, x2)  # type: ignore


@wrap_elemwise_binary
def subtract(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.subtract(x1, x2)  # type: ignore


@wrap_elemwise_binary
def true_divide(x1: NamedOrNumeric, x2: NamedOrNumeric, /) -> NamedOrNumeric:
    return jnp.true_divide(x1, x2)  # type: ignore


# deprecated name
concat_axis_specs = concat_axes

__all__ = [
    "debug",
    "random",
    "tree_util",
    "nn",
    "state_dict",
    "field",
    "Axis",
    "AxisSpec",
    "AxisSelection",
    "AxisSelector",
    "make_axes",
    "axis_name",
    "axis_size",
    "NamedArray",
    "broadcast_to",
    "broadcast_axis",
    "named",
    "dot",
    "roll",
    "split",
    "flatten_axes",
    "slice",
    "updated_slice",
    "ds",
    "dslice",
    "dblock",
    "index",
    "take",
    "unbind",
    "rename",
    "rearrange",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
    "arange",
    "random",
    "abs",
    "absolute",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "around",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copy",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "frexp",
    "i0",
    "imag",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "ndim",
    "negative",
    "positive",
    "rad2deg",
    "radians",
    "real",
    "reciprocal",
    "rint",
    "rsqrt",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "square",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "max",
    "mean",
    "min",
    "nanargmax",
    "nanargmin",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
    "prod",
    "product",
    "ptp",
    "std",
    "sum",
    "var",
    "cumsum",
    "cumprod",
    "nancumprod",
    "nancumsum",
    "sort",
    "scan",
    "fold",
    "map",
    "vmap",
    "trace",
    "nonzero",
    "where",
    "unique",
    "unique_values",
    "unique_counts",
    "unique_inverse",
    "unique_all",
    "packbits",
    "unpackbits",
    "searchsorted",
    "bincount",
    "clip",
    "tril",
    "triu",
    "poly",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyint",
    "polyder",
    "polyval",
    "polyfit",
    "roots",
    "trim_zeros",
    "vander",
    "fft",
    "ifft",
    "hfft",
    "ihfft",
    "rfft",
    "irfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    "add",
    "arctan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "divide",
    "divmod",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "hypot",
    "left_shift",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "right_shift",
    "subtract",
    "true_divide",
    "auto_sharded",
    "axis_mapping",
    "named_jit",
    "fsdp",
    "shard_with_axis_mapping",
    "shard",
    "enable_shape_checks",
    "are_shape_checks_enabled",
    "allclose",
    "array_equal",
    "array_equiv",
    "isclose",
    "pad_left",
    "pad",
    "stack",
    "concatenate",
    "eliminate_axes",
    "resolve_axis",
    "replace_axis",
    "selects_axis",
    "concat_axes",
    "concat_axis_specs",
    "top_k",
    "ravel",
    "flatten",
    "is_named_array",
    "tree_checkpoint_name",
    "ScanCheckpointPolicy",
    "quantization",
    "util",
    "einsum",
    "broadcast_arrays",
    "unflatten_axis",
    "ReductionFunction",
    "SimpleReductionFunction",
    "NamedArrayAxes",
    "NamedArrayAxesSpec",
    "Named",
    "set_mesh",
]
