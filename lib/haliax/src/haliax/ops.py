# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import typing
from typing import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

import haliax

from .axis import Axis, AxisSelector, axis_name
from .core import NamedArray, NamedOrNumeric, broadcast_arrays, broadcast_arrays_and_return_axes, named
from .jax_utils import ensure_scalar, is_scalarish


def trace(array: NamedArray, axis1: AxisSelector, axis2: AxisSelector, offset=0, dtype=None) -> NamedArray:
    """Compute the trace of an array along two named axes."""
    a1_index = array.axis_indices(axis1)
    a2_index = array.axis_indices(axis2)

    if a1_index is None:
        raise ValueError(f"Axis {axis1} not found in array. Available axes: {array.axes}")
    if a2_index is None:
        raise ValueError(f"Axis {axis2} not found in array. Available axes: {array.axes}")

    if a1_index == a2_index:
        raise ValueError(f"Cannot trace along the same axis. Got {axis1} and {axis2}")

    inner = jnp.trace(array.array, offset=offset, axis1=a1_index, axis2=a2_index, dtype=dtype)
    # remove the two indices
    axes = tuple(a for i, a in enumerate(array.axes) if i not in (a1_index, a2_index))
    return NamedArray(inner, axes)


@typing.overload
def where(
    condition: NamedOrNumeric | bool,
    x: NamedOrNumeric,
    y: NamedOrNumeric,
) -> NamedArray: ...


@typing.overload
def where(
    condition: NamedArray,
    *,
    fill_value: int,
    new_axis: Axis,
) -> tuple[NamedArray, ...]: ...


def where(
    condition: NamedOrNumeric | bool,
    x: NamedOrNumeric | None = None,
    y: NamedOrNumeric | None = None,
    fill_value: int | None = None,
    new_axis: Axis | None = None,
) -> NamedArray | tuple[NamedArray, ...]:
    """Like jnp.where, but with named axes."""

    if (x is None) != (y is None):
        raise ValueError("Must either specify both x and y, or neither")

    # one argument form
    if (x is None) and (y is None):
        if not isinstance(condition, NamedArray):
            raise ValueError(f"condition {condition} must be a NamedArray in single argument mode")
        if fill_value is None or new_axis is None:
            raise ValueError("Must specify both fill_value and new_axis")
        return tuple(
            NamedArray(idx, (new_axis,))
            for idx in jnp.where(condition.array, size=new_axis.size, fill_value=fill_value)
        )

    # if x or y is a NamedArray, the other must be as well. wrap as needed for scalars

    if is_scalarish(condition):
        if x is None or y is None:
            raise ValueError("Must specify x and y when condition is a scalar")

        if isinstance(x, NamedArray) and not isinstance(y, NamedArray):
            if not is_scalarish(y):
                raise ValueError("y must be a NamedArray or scalar if x is a NamedArray")
            y = named(y, ())
        elif isinstance(y, NamedArray) and not isinstance(x, NamedArray):
            if not is_scalarish(x):
                raise ValueError("x must be a NamedArray or scalar if y is a NamedArray")
            x = named(x, ())
        x, y = broadcast_arrays(x, y)
        if isinstance(condition, NamedArray):
            condition = ensure_scalar(condition, name="condition")
        return jax.lax.cond(condition, lambda _: x, lambda _: y, None)

    condition, x, y = broadcast_arrays(condition, x, y)  # type: ignore

    assert isinstance(condition, NamedArray)

    def _array_if_named(x):
        if isinstance(x, NamedArray):
            return x.array
        return x

    raw = jnp.where(condition.array, _array_if_named(x), _array_if_named(y))
    return NamedArray(raw, condition.axes)


def nonzero(array: NamedArray, *, size: Axis, fill_value: int = 0) -> tuple[NamedArray, ...]:
    """Like :func:`jax.numpy.nonzero`, but with named axes.

    Args:
        array: The input array to test for nonzero values. Must be a :class:`NamedArray`.
        size: Axis specifying the size of the output axis. This is required because
            JAX requires the size of the result at tracing time.
        fill_value: Value used to fill the output when fewer than ``size`` elements are
            nonzero. Defaults to ``0``.

    Returns:
        A tuple of :class:`NamedArray` objects, one for each axis of ``array``. Each
        returned array has ``size`` as its only axis and contains the indices of the
        nonzero elements along the corresponding input axis.
    """

    if not isinstance(array, NamedArray):
        raise ValueError("array must be a NamedArray")

    return tuple(NamedArray(idx, (size,)) for idx in jnp.nonzero(array.array, size=size.size, fill_value=fill_value))


def clip(array: NamedOrNumeric, a_min: NamedOrNumeric, a_max: NamedOrNumeric) -> NamedArray:
    """Like jnp.clip, but with named axes. This version currently only accepts the three argument form."""
    (array, a_min, a_max), axes = broadcast_arrays_and_return_axes(array, a_min, a_max)
    array = raw_array_or_scalar(array)
    a_min = raw_array_or_scalar(a_min)
    a_max = raw_array_or_scalar(a_max)

    return NamedArray(jnp.clip(array, a_min, a_max), axes)


def tril(array: NamedArray, axis1: Axis, axis2: Axis, k=0) -> NamedArray:
    """Compute the lower triangular part of an array along two named axes."""
    array = array.rearrange((..., axis1, axis2))

    inner = jnp.tril(array.array, k=k)
    return NamedArray(inner, array.axes)


def triu(array: NamedArray, axis1: Axis, axis2: Axis, k=0) -> NamedArray:
    """Compute the upper triangular part of an array along two named axes."""
    array = array.rearrange((..., axis1, axis2))

    inner = jnp.triu(array.array, k=k)
    return NamedArray(inner, array.axes)


def isclose(a: NamedArray, b: NamedArray, rtol=1e-05, atol=1e-08, equal_nan=False) -> NamedArray:
    """Returns a boolean array where two arrays are element-wise equal within a tolerance."""
    a, b = broadcast_arrays(a, b)
    # TODO: numpy supports an array atol and rtol, but we don't yet
    return NamedArray(jnp.isclose(a.array, b.array, rtol=rtol, atol=atol, equal_nan=equal_nan), a.axes)


def allclose(a: NamedArray, b: NamedArray, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    """Returns True if two arrays are element-wise equal within a tolerance."""
    a, b = broadcast_arrays(a, b)
    return bool(jnp.allclose(a.array, b.array, rtol=rtol, atol=atol, equal_nan=equal_nan))


def array_equal(a: NamedArray, b: NamedArray) -> bool:
    """Returns True if two arrays have the same shape and elements."""
    if set(a.axes) != set(b.axes):
        return False
    b = b.rearrange(a.axes)
    return bool(jnp.array_equal(a.array, b.array))


def array_equiv(a: NamedArray, b: NamedArray) -> bool:
    """Returns True if two arrays are shape-consistent and equal."""
    try:
        a, b = broadcast_arrays(a, b)
    except ValueError:
        return False
    return bool(jnp.array_equal(a.array, b.array))


def pad_left(array: NamedArray, axis: Axis, new_axis: Axis, value=0) -> NamedArray:
    """Pad an array along named axes."""
    amount_to_pad_to = new_axis.size - axis.size
    if amount_to_pad_to < 0:
        raise ValueError(f"Cannot pad {axis} to {new_axis}")

    idx = array.axis_indices(axis)

    padding = [(0, 0)] * array.ndim
    if idx is None:
        raise ValueError(f"Axis {axis} not found in array. Available axes: {array.axes}")
    padding[idx] = (amount_to_pad_to, 0)

    padded = jnp.pad(array.array, padding, constant_values=value)
    return NamedArray(padded, array.axes[:idx] + (new_axis,) + array.axes[idx + 1 :])


def pad(
    array: NamedArray,
    pad_width: Mapping[AxisSelector, tuple[int, int]],
    *,
    mode: str = "constant",
    constant_values: NamedOrNumeric = 0,
    **kwargs,
) -> NamedArray:
    """Version of ``jax.numpy.pad`` that works with ``NamedArray``.

    ``pad_width`` should be a mapping from axis (or axis name) to a ``(before, after)``
    tuple specifying how much padding to add on each side of that axis. Any axis
    not present in ``pad_width`` will not be padded.
    """

    padding = []
    new_axes = []
    for ax in array.axes:
        left_right = pad_width.get(ax)
        if left_right is None:
            left_right = pad_width.get(axis_name(ax))  # type: ignore[arg-type]
        if left_right is None:
            left_right = (0, 0)
        left, right = left_right
        padding.append((left, right))
        new_axes.append(ax.resize(ax.size + left + right))

    result = jnp.pad(
        array.array,
        padding,
        mode=mode,
        constant_values=raw_array_or_scalar(constant_values),
        **kwargs,
    )

    return NamedArray(result, tuple(new_axes))


def raw_array_or_scalar(x: NamedOrNumeric):
    if isinstance(x, NamedArray):
        return x.array
    return x


@typing.overload
def unique(
    array: NamedArray, Unique: Axis, *, axis: AxisSelector | None = None, fill_value: ArrayLike | None = None
) -> NamedArray: ...


@typing.overload
def unique(
    array: NamedArray,
    Unique: Axis,
    *,
    return_index: typing.Literal[True],
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


@typing.overload
def unique(
    array: NamedArray,
    Unique: Axis,
    *,
    return_inverse: typing.Literal[True],
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


@typing.overload
def unique(
    array: NamedArray,
    Unique: Axis,
    *,
    return_counts: typing.Literal[True],
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


@typing.overload
def unique(
    array: NamedArray,
    Unique: Axis,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> NamedArray | tuple[NamedArray, ...]: ...


def unique(
    array: NamedArray,
    Unique: Axis,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> NamedArray | tuple[NamedArray, ...]:
    """
    Like jnp.unique, but with named axes.

    Args:
        array: The input array.
        Unique: The name of the axis that will be created to hold the unique values.
        fill_value: The value to use for the fill_value argument of jnp.unique
        axis: The axis along which to find unique values.
        return_index: If True, return the indices of the unique values.
        return_inverse: If True, return the indices of the input array that would reconstruct the unique values.
    """
    size = Unique.size

    is_multireturn = return_index or return_inverse or return_counts

    kwargs = dict(
        size=size,
        fill_value=fill_value,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if axis is not None:
        axis_index = array.axis_indices(axis)
        if axis_index is None:
            raise ValueError(f"Axis {axis} not found in array. Available axes: {array.axes}")
        out = jnp.unique(array.array, axis=axis_index, **kwargs)
    else:
        out = jnp.unique(array.array, **kwargs)

    if is_multireturn:
        unique = out[0]
        next_index = 1
        if return_index:
            index = out[next_index]
            next_index += 1
        if return_inverse:
            inverse = out[next_index]
            next_index += 1
        if return_counts:
            counts = out[next_index]
            next_index += 1
    else:
        unique = out

    ret = []

    if axis is not None:
        out_axes = haliax.axis.replace_axis(array.axes, axis, Unique)
    else:
        out_axes = (Unique,)

    unique_values = haliax.named(unique, out_axes)
    if not is_multireturn:
        return unique_values

    ret.append(unique_values)

    if return_index:
        ret.append(haliax.named(index, Unique))

    if return_inverse:
        if axis is not None:
            assert axis_index is not None
            inverse = haliax.named(inverse, array.axes[axis_index])
        else:
            inverse = haliax.named(inverse, array.axes)
        ret.append(inverse)

    if return_counts:
        ret.append(haliax.named(counts, Unique))

    return tuple(ret)


def unique_values(
    array: NamedArray,
    Unique: Axis,
    *,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> NamedArray:
    """Shortcut for :func:`unique` that returns only unique values."""

    return typing.cast(
        NamedArray,
        unique(
            array,
            Unique,
            axis=axis,
            fill_value=fill_value,
        ),
    )


def unique_counts(
    array: NamedArray,
    Unique: Axis,
    *,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Shortcut for :func:`unique` that also returns counts."""

    values, counts = typing.cast(
        tuple[NamedArray, NamedArray],
        unique(
            array,
            Unique,
            return_counts=True,
            axis=axis,
            fill_value=fill_value,
        ),
    )
    return values, counts


def unique_inverse(
    array: NamedArray,
    Unique: Axis,
    *,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Shortcut for :func:`unique` that also returns inverse indices."""

    values, inverse = typing.cast(
        tuple[NamedArray, NamedArray],
        unique(
            array,
            Unique,
            return_inverse=True,
            axis=axis,
            fill_value=fill_value,
        ),
    )
    return values, inverse


def unique_all(
    array: NamedArray,
    Unique: Axis,
    *,
    axis: AxisSelector | None = None,
    fill_value: ArrayLike | None = None,
) -> tuple[NamedArray, NamedArray, NamedArray, NamedArray]:
    """Shortcut for :func:`unique` returning values, indices, inverse, and counts."""

    values, indices, inverse, counts = typing.cast(
        tuple[NamedArray, NamedArray, NamedArray, NamedArray],
        unique(
            array,
            Unique,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            fill_value=fill_value,
        ),
    )
    return values, indices, inverse, counts


def searchsorted(
    a: NamedArray,
    v: NamedArray | ArrayLike,
    *,
    side: str = "left",
    sorter: NamedArray | ArrayLike | None = None,
    method: str = "scan",
) -> NamedArray:
    """Named version of `jax.numpy.searchsorted`.

    ``a`` and ``sorter`` (if provided) must be one-dimensional.
    The returned array has the same axes as ``v``.
    """

    if a.ndim != 1:
        raise ValueError("searchsorted only supports 1D 'a'")

    if not isinstance(v, NamedArray):
        v = haliax.named(v, ())

    sorter_arr = None
    if sorter is not None:
        sorter_arr = sorter.array if isinstance(sorter, NamedArray) else jnp.asarray(sorter)

    result = jnp.searchsorted(a.array, v.array, side=side, sorter=sorter_arr, method=method)
    return NamedArray(result, v.axes)


def bincount(
    x: NamedArray,
    Counts: Axis,
    *,
    weights: NamedArray | ArrayLike | None = None,
    minlength: int = 0,
) -> NamedArray:
    """Named version of `jax.numpy.bincount`.

    The output axis is specified by ``Counts``.
    """

    if x.ndim != 1:
        raise ValueError("bincount only supports 1D arrays")

    w_array = None
    if weights is not None:
        if isinstance(weights, NamedArray):
            weights = haliax.broadcast_to(weights, x.axes)
            w_array = weights.array
        else:
            w_array = jnp.asarray(weights)

    result = jnp.bincount(x.array, weights=w_array, minlength=minlength, length=Counts.size)
    return NamedArray(result, (Counts,))


def packbits(a: NamedArray, axis: AxisSelector, *, bitorder: str = "big") -> NamedArray:
    """Named version of `jax.numpy.packbits`."""

    axis_index = a.axis_indices(axis)
    if not isinstance(axis_index, int):
        raise ValueError("packbits only supports a single existing axis")

    result = jnp.packbits(a.array, axis=axis_index, bitorder=bitorder)
    old_axis = a.axes[axis_index]
    new_size = (old_axis.size + 7) // 8
    new_axis = old_axis.resize(new_size)
    new_axes = a.axes[:axis_index] + (new_axis,) + a.axes[axis_index + 1 :]
    return NamedArray(result, new_axes)


def unpackbits(
    a: NamedArray,
    axis: AxisSelector,
    *,
    count: int | None = None,
    bitorder: str = "big",
) -> NamedArray:
    """Named version of `jax.numpy.unpackbits`."""

    axis_index = a.axis_indices(axis)
    if not isinstance(axis_index, int):
        raise ValueError("unpackbits only supports a single existing axis")

    result = jnp.unpackbits(a.array, axis=axis_index, count=count, bitorder=bitorder)
    old_axis = a.axes[axis_index]
    new_size = count if count is not None else old_axis.size * 8
    new_axis = old_axis.resize(new_size)
    new_axes = a.axes[:axis_index] + (new_axis,) + a.axes[axis_index + 1 :]
    return NamedArray(result, new_axes)


__all__ = [
    "trace",
    "where",
    "tril",
    "triu",
    "isclose",
    "pad_left",
    "pad",
    "clip",
    "packbits",
    "unpackbits",
    "unique",
    "unique_values",
    "unique_counts",
    "unique_inverse",
    "unique_all",
    "searchsorted",
    "bincount",
]
