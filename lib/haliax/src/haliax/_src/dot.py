# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import typing
import warnings

import jax

import haliax
from haliax.axis import (
    Axis,
    AxisSelection,
    PartialAxisSpec,
    axis_name,
    axis_spec_to_shape_dict,
    eliminate_axes,
    rearrange_for_partial_order,
    union_axes,
)
from haliax.core import NamedArray
from haliax.jax_utils import _jittable_dg_einsum
from haliax.types import DTypeLike, PrecisionLike


# deprecated overload
@typing.overload
def dot(
    axis: AxisSelection | None,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_axes: PartialAxisSpec | None = ...,
    dot_general=jax.lax.dot_general,
) -> NamedArray: ...


@typing.overload
def dot(
    *arrays: NamedArray,
    axis: AxisSelection | None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_axes: PartialAxisSpec | None = ...,
    dot_general=jax.lax.dot_general,
) -> NamedArray: ...


def dot(
    *arrays,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_axes: PartialAxisSpec | None = None,
    dot_general=jax.lax.dot_general,
    **kwargs,
) -> NamedArray:
    """Returns the tensor product of two NamedArrays. The axes `axis` are contracted over,
    and any other axes that are shared between the arrays are batched over. Non-contracted Axes in one
    that are not in the other are preserved.

    Note that if `axis` is None, the result will be a scalar, not a NamedArray. The semantics of `axis=None` are
    similar to e.g. how `sum` and other reduction functions work in numpy. If `axis=()`, then the result will be
    an "outer product" of the arrays, i.e. a tensor with shape equal to the concatenation of the shapes of the arrays.

    By default, the order of output axes is determined by the order of the input axes, such that each output axis
    occurs in the same order as it first occurs in the concatenation of the input axes.

    If `out_axes` is provided, the output will be transposed to match the provided axes. `out_axes` may be a partial
    specification of the output axes (using ellipses), in which case the output will be rearranged to be consistent
    with the partial specification. For example, if `out_axes=(..., Height, Width)` and the output axes are
    `(Width, Height, Depth)`, the output will be transposed to `(Depth, Height, Width)`. Multiple ellipses
    are supported, in which case axes will be inserted according to a greedy heuristic that prefers to place
    unconstrained axes as soon as all prior axes in the "natural" order are covered.

    Args:
        *arrays (NamedArray): The arrays to contract.
        axis (AxisSelection): The axes to contract over.
        precision (PrecisionLike, optional): The precision to use. Defaults to None. This argument is passed to `jax.numpy.einsum`,
            which in turn passes it to jax.lax.dot_general.
        preferred_element_type (DTypeLike, optional): The preferred element type of the result. Defaults to None.
            This argument is passed to `jax.numpy.einsum`.
        out_axes (PartialAxisSpec | None, optional): a potentially partial specification of the output axes.
            If provided, the output will be transposed to match the provided axes. Defaults to None.


    Returns:
        NamedArray: The result of the contraction.
    """
    if len(arrays) == 0:
        raise ValueError("Must provide at least one array to dot")

    if "axis" in kwargs:
        axis = kwargs["axis"]
    else:
        axis = arrays[0]
        arrays = arrays[1:]
        if isinstance(axis, NamedArray):
            raise ValueError("Must provide an axis to dot")

        warnings.warn("Axis has been changed to a keyword argument. Please update your code.", DeprecationWarning)

    _ensure_no_mismatched_axes(*arrays)

    # to call dot_general we need two things:
    # list of contractions and list of arrays

    all_axes: tuple[Axis, ...] = ft.reduce(union_axes, (a.axes for a in arrays), ())  # type: ignore
    output_axes: tuple[Axis, ...]
    if axis is None:
        # we want to contract over all the axes
        output_axes = ()
    else:
        output_axes = eliminate_axes(all_axes, axis)

    if out_axes is not None:
        output_axes = rearrange_for_partial_order(out_axes, output_axes)

    array_specs = []

    next_index = 0
    axis_mappings: dict[str, int] = {}

    for a in arrays:
        spec = ""
        for ax in a.axes:
            if ax.name in axis_mappings:
                spec += f"{axis_mappings[ax.name]} "
            else:
                axis_mappings[ax.name] = next_index
                spec += f"{next_index} "
                next_index += 1

        array_specs.append(spec)

    # now compute the output axes:
    output_spec = " ".join(str(axis_mappings[ax.name]) for ax in output_axes)

    # get a name for jax so it's easier to interpret logs
    if axis is None:
        jax_str = f"contract {', '.join(axis_name(ax) for ax in all_axes)} -> <scalar>"
    else:
        axis = axis_spec_to_shape_dict(axis)
        jax_str = f"contract {', '.join(axis)} -> {', '.join(a.name for a in output_axes)}"

    with jax.named_scope(jax_str):
        output = _jittable_dg_einsum(
            ", ".join(array_specs) + "-> " + output_spec,
            *[a.array for a in arrays],
            precision=precision,
            preferred_element_type=preferred_element_type,
            _dot_general=dot_general,
        )

    out = NamedArray(output, output_axes)
    return haliax.auto_sharded(out)


def _ensure_no_mismatched_axes(*arrays: NamedArray):
    """Ensure that all the arrays have no axes with the same name but different sizes"""
    if len(arrays) <= 1:
        return

    known_sizes: dict[str, int] = {}
    for a in arrays:
        for ax in a.axes:
            if ax.name in known_sizes:
                if known_sizes[ax.name] != ax.size:
                    raise ValueError(f"Axis {ax.name} has multiple sizes: {known_sizes[ax.name]} and {ax.size}")
            else:
                known_sizes[ax.name] = ax.size
