# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


# Support for einops-style rearrangement strings, but supporting named axes and unordered matching
import dataclasses
import typing
from types import EllipsisType
from typing import Mapping, Sequence

import jax.lax
import jax.numpy as jnp

from ..axis import Axis, AxisSelector, PartialAxisSpec, axis_name, rearrange_for_partial_order
from ..core import NamedArray
from ..partitioning import auto_sharded
from .parsing import AliasTable, Expression, _resolve_bindings, parse_rearrangement, raise_parse_error


@typing.overload
def rearrange(array: NamedArray, axes: Sequence[AxisSelector | EllipsisType]) -> NamedArray:
    pass


@typing.overload
def rearrange(array: NamedArray, expression: str, **bindings: AxisSelector | int) -> NamedArray:
    pass


def rearrange(array: NamedArray, *args, **kwargs) -> NamedArray:
    """
    Rearrange a tensor according to an einops-style haliax rearrangement string or a sequence of axes.
    See full documentation here: [Rearrange](https://haliax.readthedocs.io/en/latest/rearrange/)

    The sequence form of `rearrange` rearranges an array so that its underlying storage conforms to axes.
    axes may include up to 1 ellipsis, indicating that the remaining axes should be
    permuted in the same order as the array's axes.

    For example, if array has axes (a, b, c, d, e, f) and axes is (e, f, a, ..., c),
    then the output array will have axes (e, f, a, b, c, d).

    The string form of `rearrange` works similarly to einops.rearrange, but also supports named axes and unordered matching.
    The string form of `rearrange` comes in two forms:

    * **Ordered strings** are like einops strings, with the only significant difference that flattened axes are named with
    a colon, e.g. `B H W C -> B (E: H W C)`.
    * **Unordered strings** match axes by name and are marked with the addition of curly braces,
    e.g. `{H W C} -> ... C H W` or `{H W C} -> ... (E: H W C)`

    As with einops, you can provide axis sizes to unflatten axes. For instance, to turn an image patches,
    `hax.rearrange(x, '{ B (H: w1 H) (W: w1 W)} -> (B: B h1 w1) H W ...', H=32, W=32)
    will turn a batch of images into a batch of image patches. Bindings can also be [haliax.Axis][] objects,
    or strings that will be used as the actual name of the resulting axis.

    Examples:
        >>> import haliax as hax
        >>> import jax.random as jrandom
        >>> B, H, W, C = hax.Axis("B", 8), hax.Axis("H", 32), hax.Axis("W", 32), hax.Axis("C", 3)
        >>> x = hax.random.normal( (B, H, W, C))
        >>> # Sequence-based rearrange
        >>> hax.rearrange(x, (C, B, H, W))
        >>> hax.rearrange(x, (C, ...)) # ellipsis means "keep the rest of the axes in the same order"
        >>> # String-based rearrange
        >>> # permute the axes
        >>> hax.rearrange(x, "B H W C -> C B H W")
        >>> # flatten the image (note the assignment of a new name to the flattened axis)
        >>> hax.rearrange(x, "B H W C -> B (E: H W C)")
        >>> # turn the image into patches
        >>> hax.rearrange(x, "{ B (H: h1 H) (W: w1 W) C } -> (B: B h1 w1) (E: H W C) ...", H=2, W=2)
        >>> # names can be longer than one character
        >>> hax.rearrange(x, "{ B (H: h1 H) (W: w1 W) C } -> (B: B h1 w1) (embed: H W C) ...", H=2, W=2)
    """

    if len(args) == 1:
        axes = args[0]
        if isinstance(axes, str):
            return einops_rearrange(array, axes, **kwargs)
        else:
            return axis_spec_rearrange(array, axes)
    elif len(args) > 1:
        raise TypeError("Only one positional argument allowed")

    kwargs = dict(kwargs)
    expression = kwargs.pop("expression", None)
    if expression is not None:
        return einops_rearrange(array, expression, **kwargs)
    else:
        axes = kwargs.pop("axes", None)
        if axes is None:
            raise TypeError("Must specify either axes or expression")
        return axis_spec_rearrange(array, axes)


def axis_spec_rearrange(array: NamedArray, axis_spec: PartialAxisSpec) -> NamedArray:
    if len(axis_spec) == 0 and len(array.axes) != 0:
        raise ValueError("No axes specified")

    # various fast paths
    if len(axis_spec) == 1 and axis_spec[0] is Ellipsis:
        return array

    if axis_spec == array.axes:
        return array

    if axis_spec[-1] is Ellipsis and array.axes[0 : len(axis_spec) - 1] == axis_spec[0 : len(axis_spec) - 1]:
        return array

    if axis_spec[0] is Ellipsis and array.axes[len(axis_spec) - 1 :] == axis_spec[1:]:
        return array

    out_axes = rearrange_for_partial_order(axis_spec, array.axes)

    # now build a permute_spec
    permute_spec = []
    index_of_in = {ax: i for i, ax in enumerate(array.axes)}

    for ax in out_axes:
        permute_spec.append(index_of_in[ax])

    out_axes = tuple(array.axes[i] for i in typing.cast(list[int], permute_spec))
    return NamedArray(jnp.transpose(array.array, permute_spec), out_axes)


def einops_rearrange(array: NamedArray, expression: str, **bindings: AxisSelector | int) -> NamedArray:
    lhs, rhs = parse_rearrangement(expression)

    # the fundamental xla op for rearranging is reshape, which combines both transpose and reshape
    # all rearranges are fundamentally a reshape, followed by a transpose, followed by a reshape
    # as an example of an op that needs both, consider:
    #  '(a b) c d -> (c a) (b d)'
    # where a = 2, b = 3, c = 4, d = 5
    # which is equivalent to:
    # x.reshape((2, 3, 4, 5)).transpose((2, 0, 1, 3)).reshape((4 * 2, 3 * 5))
    plan = _plan_rearrange(expression, lhs, rhs, array, bindings)

    raw_array = array.array

    if plan.intermediate_axes != array.axes:
        raw_array = raw_array.reshape([ax.size for ax in plan.intermediate_axes])
        array = NamedArray(raw_array, plan.intermediate_axes)
        array = auto_sharded(array)
        raw_array = array.array

    final_shape = tuple(ax.size for ax in plan.final_axes)

    if plan.needs_final_reshape:
        finished_array = jax.lax.reshape(raw_array, new_sizes=final_shape, dimensions=plan.transpose)
    elif plan.transpose is not None:
        finished_array = jax.lax.transpose(raw_array, permutation=plan.transpose)
    else:
        finished_array = raw_array

    return auto_sharded(NamedArray(finished_array, plan.final_axes))


@dataclasses.dataclass(frozen=True)
class _Plan:
    intermediate_axes: tuple[Axis, ...]
    transpose: tuple[int, ...] | None
    needs_final_reshape: bool

    final_axes: tuple[Axis, ...]


def _plan_rearrange(
    original_str, lhs: Expression, rhs: Expression, array: NamedArray, input_bindings: Mapping[str, AxisSelector | int]
) -> _Plan:
    aliases = _resolve_bindings(array, input_bindings)
    grouped_new_shapes = _determine_initial_reshape(original_str, lhs, array, aliases)
    intermediate_axes = tuple(ax for split_axes in grouped_new_shapes for ax in split_axes)

    transpose: tuple[int, ...] | None
    transpose, final_axes = _determine_final_transpose_and_reshape(original_str, rhs, aliases, intermediate_axes)

    transposed_intermediate_axes = tuple(intermediate_axes[i] for i in transpose)
    if transposed_intermediate_axes == final_axes:
        needs_final_reshape = False
    else:
        needs_final_reshape = True

    if transpose == tuple(range(len(transpose))):
        transpose = None

    return _Plan(intermediate_axes, transpose, needs_final_reshape, final_axes)


def _determine_final_transpose_and_reshape(
    expression, rhs: Expression, aliases: "AliasTable", intermediate_axes: tuple[Axis, ...]
) -> tuple[tuple[int, ...], tuple[Axis, ...]]:
    # The rhs tells us three things:
    # 1. how to reorder the intermediate axes
    # 2. how to merge the intermediate axes into the final axes
    # 3. where ellipses are in the final axes (where we can put unused intermediate axes)

    # MUTATES `aliases`

    # Our approach is to:
    # 1. Figure out the partial order of the intermediate axes
    # 2. Compute the full final axes (using rearrange_for_partial_order)
    # 3. Figure out the transposition order to get the intermediate axes into the right order
    # return (2) and (3)

    transposed_intermediate_axes_order: list[Axis | EllipsisType] = []

    first_intermediate_for_final: dict[Axis, Axis] = {}  # intermediate -> final

    used_intermediate_axes = set()  # axes must be used exactly once
    has_ellipsis = False

    for cpos, capture in enumerate(rhs.captures):
        if capture == Ellipsis:
            has_ellipsis = True
            transposed_intermediate_axes_order.append(Ellipsis)
            continue

        assert not isinstance(capture, EllipsisType)

        binding = capture.binding
        if binding is None:
            raise_parse_error("All rhs axes must have a name. Use (name: bindings)", expression, capture.char_range)

        sz = 1
        first_axis = None
        for ax_name in capture.axes:
            intermed_axis = aliases.dealias_binding(ax_name)

            if not isinstance(intermed_axis, Axis):
                raise_parse_error(f"Axis {ax_name} is not bound on the lhs", expression, capture.char_range)

            if first_axis is None:
                first_axis = intermed_axis

            if intermed_axis in used_intermediate_axes:
                raise_parse_error(f"Axis {intermed_axis} is used more than once", expression, capture.char_range)
            used_intermediate_axes.add(intermed_axis)
            transposed_intermediate_axes_order.append(intermed_axis)

            sz *= intermed_axis.size
            # now find the position of this axis in the intermediate axes

        # figure out the name of the final axis
        axis = aliases.dealias_binding(binding)
        if axis is None:
            new_axis = Axis(binding, sz)
        else:
            new_axis = Axis(axis_name(axis), sz)

        # TODO: this isn't ideal for the unusual case where we have an empty set of axes to bind to an axis
        # we won't accurately get the order for these unitary axes
        if first_axis is not None:
            first_intermediate_for_final[first_axis] = new_axis

    if not has_ellipsis:
        unused_intermediate_axes = set(intermediate_axes) - used_intermediate_axes
        if len(unused_intermediate_axes) > 0:
            raise ValueError("Not all intermediate axes are used. Use ... to insert unused axes")

    # now we have a partial order of the intermediate axes
    reordered_intermediate_axes = rearrange_for_partial_order(transposed_intermediate_axes_order, intermediate_axes)

    # now we need to figure out the final axes
    final_axes: list[Axis] = []
    transposition_order: list[int] = []

    for intermediate_axis in reordered_intermediate_axes:
        if intermediate_axis not in used_intermediate_axes:
            # this axis is not used, so we keep it from the reordered axes
            final_axes.append(intermediate_axis)
            transposition_order.append(intermediate_axes.index(intermediate_axis))
        else:
            final_axis = first_intermediate_for_final.get(intermediate_axis, None)

            if final_axis is not None:
                final_axes.append(final_axis)

            transposition_order.append(intermediate_axes.index(intermediate_axis))

    return tuple(transposition_order), tuple(final_axes)  # type: ignore


def _determine_initial_reshape(
    expression,
    lhs: Expression,
    array: NamedArray,
    aliases: AliasTable,
) -> list[list[Axis]]:
    # MUTATES `aliases`
    # the lhs all need to be bound to axes in the array, or synthesized as parts of axes.
    # In the lhs, bindings look like either a name, or a name and a list of (new) axes.
    # bindings can either be done by name, or by position, depending on if lhs.is_ordered
    new_shapes: list[list[Axis] | None] = [None] * len(array.axes)
    used_new_names: set[str] = set()  # names can only be used once on a side

    # one subtle difference between the lhs and the rhs is the handling of binding in expressions like (a: b c)
    # in the lhs, this means that a is bound to an axis, and b and c are split out from a
    # in the rhs, this means that b and c must already be bound to axes, and a is a new axis name
    # this makes sense in the same way that pattern matching works

    if lhs.is_ordered:
        # if we start with an ellipsis, we bind from the right
        # if we end with an ellipsis, we bind from the left
        ellipsis_pos = None
        axis_index_for_capture: list[int | None] = [None] * len(lhs.captures)
        covered_axes = set()
        # bind from the left
        axis_pos = 0
        for cpos, capture in enumerate(lhs.captures):
            if capture == Ellipsis:
                if ellipsis_pos is not None:
                    assert False, "should not be here"  # pragma: no cover
                ellipsis_pos = cpos
                break

            assert not isinstance(capture, EllipsisType)

            if axis_pos >= len(array.axes):
                raise_parse_error("Too many axes in lhs", expression, capture.char_range)

            axis_index_for_capture[cpos] = axis_pos
            covered_axes.add(axis_pos)
            axis_pos += 1

        # bind from the right, take care to check second ellipsis
        if ellipsis_pos is not None:
            axis_pos = len(array.axes) - 1
            for cpos in range(len(lhs.captures) - 1, ellipsis_pos, -1):
                capture = lhs.captures[cpos]
                if capture == Ellipsis:
                    raise_parse_error("Only one ellipsis allowed", expression, None)

                assert not isinstance(capture, EllipsisType)

                axis_index_for_capture[cpos] = axis_pos
                if axis_pos in covered_axes:
                    raise_parse_error(
                        f"Axis {array.axes[axis_pos]} is bound more than once",
                        expression,
                        capture.char_range,
                    )
                covered_axes.add(axis_pos)
                axis_pos -= 1
        else:
            # no ellipsis, so we need to check that we covered all axes
            if len(covered_axes) < len(array.axes):
                raise_parse_error(
                    "Not all axes are bound, use ... to skip missing axes", expression, len(expression) - 1
                )  # type: ignore
            elif len(covered_axes) > len(array.axes):
                raise_parse_error("Too many axes are bound", expression, lhs.captures[-1].char_range)  # type: ignore

        # now that we have the bindings, we can figure out the new shapes
        for cpos, capture in enumerate(lhs.captures):
            if capture == Ellipsis:
                continue
            assert not isinstance(capture, EllipsisType)

            axis_index = axis_index_for_capture[cpos]
            if axis_index is None:
                raise_parse_error("Internal error", expression, capture.char_range)

            axis = array.axes[axis_index]
            if new_shapes[axis_index] is not None:
                raise_parse_error(f"Axis {axis} is bound more than once", expression, capture.char_range)

            new_axes = _solve_split_axes(axis, capture, aliases, used_new_names, expression)
            new_shapes[axis_index] = new_axes  # type: ignore

    else:
        # we just need to bind the axes in the lhs to the axes in the array
        for capture in lhs.captures:
            # ellipses are ignored in unordered rearrangements
            if capture == Ellipsis:
                continue

            assert not isinstance(capture, EllipsisType)

            if capture.binding is None:
                raise_parse_error(
                    "Unordered axes must be bound by name, e.g. (a: b) or just a", expression, capture.char_range
                )

            # let's see if we're aliasing it in the bindings
            maybe_alias = aliases.dealias_binding(capture.binding)
            if maybe_alias is None:
                maybe_alias = capture.binding
            else:
                maybe_alias = axis_name(maybe_alias)

            try:
                axis = array.resolve_axis(maybe_alias)
                # aliases.bind_alias(capture.binding, axis, expression, capture.char_range)
            except ValueError:
                raise_parse_error(f"Could not resolve axis {maybe_alias}", expression, capture.char_range)

            try:
                axis_index = array.axes.index(axis)
            except ValueError:
                raise_parse_error(f"Axis {axis} is not in the array", expression, capture.char_range)
            if new_shapes[axis_index] is not None:
                raise_parse_error(f"Axis {axis} is bound more than once", expression, capture.char_range)

            new_axes = _solve_split_axes(axis, capture, aliases, used_new_names, expression)
            new_shapes[axis_index] = new_axes  # type: ignore

    for i in range(len(array.axes)):
        if new_shapes[i] is None:
            new_shapes[i] = [array.axes[i]]

    return new_shapes  # type: ignore


def _solve_split_axes(axis, capture, aliases, used_new_names, expression):
    """
    Given an axis and a capture of the form (a: b c) or (b c) on the lhs, solve for the new axes.
    """
    new_axes: list[Axis | None] = []
    unsolved_axis_index: int | None = None

    # easy case: 1 axis in capture
    if len(capture.axes) == 1:
        new_axis_name = capture.axes[0]
        if new_axis_name in used_new_names:
            raise_parse_error(f"Capture {new_axis_name} is assigned more than once", expression, capture.char_range)
        used_new_names.add(new_axis_name)

        new_axes.append(axis)
        aliases.bind_alias(new_axis_name, axis, expression, capture.char_range)

        return new_axes

    remaining_size = axis.size

    for new_axis_name in capture.axes:
        if new_axis_name in used_new_names:
            raise_parse_error(
                f"Capture {new_axis_name} is assigned more than once in lhs", expression, capture.char_range
            )

        used_new_names.add(new_axis_name)

        new_axis = aliases.dealias_binding(new_axis_name)
        if new_axis is None:
            new_axis = new_axis_name

        if isinstance(new_axis, Axis):
            new_axes.append(new_axis)
            if remaining_size % new_axis.size:
                raise_parse_error(f"Axes do not divide evenly into axis {axis}", expression, capture.char_range)
            remaining_size //= new_axis.size
        else:
            if unsolved_axis_index is not None:
                raise_parse_error(
                    "Sizes for this split axis are ambiguous. You must provide a size as a kwarg.",
                    expression,
                    capture.char_range,
                )
            unsolved_axis_index = len(new_axes)
            new_axes.append(None)

    if unsolved_axis_index is not None:
        # we need to solve for this axis
        unsolved_alias = aliases.dealias_binding(capture.axes[unsolved_axis_index])
        assert not isinstance(unsolved_alias, Axis)
        if unsolved_alias is None:
            unsolved_alias = capture.axes[unsolved_axis_index]
        assert isinstance(unsolved_alias, str)

        new_axis = Axis(unsolved_alias, remaining_size)
        new_axes[unsolved_axis_index] = new_axis
        aliases.bind_alias(capture.axes[unsolved_axis_index], new_axis, expression, capture.char_range)
    return new_axes
