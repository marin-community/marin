# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Protocol

import jax

from haliax.core import NamedArray, _broadcast_order, broadcast_to

from .axis import AxisSelection, AxisSelector, axis_spec_to_shape_dict, eliminate_axes
from .jax_utils import ensure_scalar, is_scalarish


def wrap_elemwise_unary(f, a, *args, **kwargs):
    if isinstance(a, NamedArray):
        return NamedArray(f(a.array, *args, **kwargs), a.axes)
    else:
        return f(a, *args, **kwargs)


def wrap_reduction_call(
    fn,
    a,
    axis: AxisSelection | None,
    where: NamedArray | None = None,
    single_axis_only: bool = False,
    supports_where: bool = True,
    **kwargs,
):
    kwargs = dict(kwargs)
    if where is not None and not supports_where:
        raise ValueError(f"where is not supported by {fn.__name__}")

    if kwargs.get("out", None) is not None:
        raise ValueError("out is not supported yet for NamedArray")
    if kwargs.get("keepdims", False):
        raise ValueError("keepdims is not supported for NamedArray")

    def reduce_one_leaf(a):
        nonlocal axis, where
        if isinstance(a, NamedArray):
            if where is not None:
                if not isinstance(where, NamedArray):
                    raise TypeError(f"where must be a NamedArray if a is a NamedArray, but is {where}")
                where = broadcast_to(where, a.axes)
                kwargs["where"] = where.array

            if axis is None:
                result = fn(a.array, axis=None, **kwargs)
                return NamedArray(result, ())
            else:
                axis = axis_spec_to_shape_dict(axis)
                if single_axis_only and len(axis) > 1:
                    raise ValueError(f"{fn.__name__} only supports a single axis")
                indices = a.axis_indices(axis)
                if indices is None or any(x is None for x in indices):
                    raise ValueError(f"axis {axis} is not in {a.axes}")
                new_axes = eliminate_axes(a.axes, axis)

                if single_axis_only:
                    result = fn(a.array, axis=indices[0], **kwargs)
                else:
                    result = fn(a.array, axis=indices, **kwargs)
                return NamedArray(result, tuple(new_axes))
        else:
            if where is not None:
                kwargs["where"] = where
            return fn(a, axis=axis, **kwargs)

    return jax.tree_util.tree_map(reduce_one_leaf, a, is_leaf=lambda x: isinstance(x, NamedArray))


def wrap_axiswise_call(fn, a, axis: AxisSelection | None, *, single_axis_only: bool, **kwargs):
    if isinstance(a, NamedArray):
        if axis is None:
            return fn(a.array, axis=None, **kwargs)
        else:
            axis = axis_spec_to_shape_dict(axis)
            indices = a.axis_indices(axis)
            if any(x is None for x in indices):
                raise ValueError(f"Axis {axis} not in {a.axes}")
            if len(indices) == 1:
                return NamedArray(fn(a.array, axis=indices[0], **kwargs), a.axes)
            elif single_axis_only:
                raise ValueError(f"{fn.__name__} only supports a single axis")
            else:
                return NamedArray(fn(a.array, axis=indices, **kwargs), a.axes)

    else:
        return fn(a, axis=axis, **kwargs)


def wrap_elemwise_binary(op):
    def binop(a, b):
        if isinstance(a, NamedArray) and isinstance(b, NamedArray):
            axes = _broadcast_order(a, b)
            a = broadcast_to(a, axes)
            b = broadcast_to(b, axes)
            return NamedArray(op(a.array, b.array), axes)
        elif isinstance(a, NamedArray):
            # b isn't named.
            if not is_scalarish(a):
                if not is_scalarish(b):
                    raise ValueError(
                        f"Cannot apply {op.__name__} to a NamedArray and a non-scalar {type(b)}. "
                        "Either both must be NamedArrays or b must be a scalar."
                    )
            else:
                if is_scalarish(b):
                    return NamedArray(op(a.array, b), a.axes)
                a = ensure_scalar(a)
                return op(a, b)

            return NamedArray(op(a.array, b), a.axes)
        elif isinstance(b, NamedArray):
            if not is_scalarish(b):
                if not is_scalarish(a):
                    raise ValueError(
                        f"Cannot apply {op.__name__} to a non-scalar {type(a)} and a NamedArray. "
                        "Either both must be NamedArrays or a must be a scalar."
                    )
            else:
                if is_scalarish(a):
                    return NamedArray(op(a, b.array), b.axes)
                b = ensure_scalar(b)
                return op(a, b)

            return NamedArray(op(a, b.array), b.axes)
        else:
            return op(a, b)

    return binop


def unwrap_namedarrays(*a):
    return tuple(x.array if isinstance(x, NamedArray) else x for x in a)


class ReductionFunction(Protocol):
    def __call__(
        self,
        array: NamedArray,
        axis: AxisSelection | None = None,
        where: NamedArray | None = None,
        **kwargs,
    ) -> NamedArray: ...


class SimpleReductionFunction(Protocol):
    def __call__(self, array: NamedArray, axis: AxisSelector | None = None, **kwargs) -> NamedArray: ...


__all__ = [
    "wrap_elemwise_unary",
    "wrap_reduction_call",
    "wrap_axiswise_call",
    "wrap_elemwise_binary",
    "unwrap_namedarrays",
    "ReductionFunction",
    "SimpleReductionFunction",
]
