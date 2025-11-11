# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import typing
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Mapping, Sequence, overload

import equinox as eqx

from haliax.util import ensure_tuple

from ._src.util import index_where


@dataclass(frozen=True)
class Axis:
    """Axis is a dataclass that represents an axis of an NamedArray. It has a name and a size."""

    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)

    def resize(self, size) -> "Axis":
        return Axis(self.name, size)

    def __str__(self):
        return f"{self.name}({self.size})"


def make_axes(**kwargs: int) -> tuple[Axis, ...]:
    """
    Convenience function for creating a tuple of Axis objects.

    Example:
    ```
    X, Y = axes(X=10, Y=20)
    ```

    """
    return tuple(Axis(name, size) for name, size in kwargs.items())


AxisSelector = Axis | str
"""AxisSelector is a type that can be used to select a single axis from an array. str or Axis"""
ShapeDict = Mapping[str, int]
"""ShapeDict is a type that can be used to specify the axes of an array, usually for creation or adding a new axis"""
PartialShapeDict = Mapping[str, int | None]
"""Similar to a AxisSelection, in dict form."""

AxisSelection = AxisSelector | Sequence[AxisSelector] | PartialShapeDict
"""AxisSelection is a type that can be used to select multiple axes from an array. str, Axis, or sequence of mixed
str and Axis"""
AxisSpec = Axis | Sequence[Axis] | ShapeDict
"""AxisSpec is a type that can be used to specify the axes of an array, usually for creation or adding a new axis
 whose size can't be determined another way. Axis or sequence of Axis"""


PartialAxisSpec = Sequence[EllipsisType | AxisSelector]
"""Used for rearrange and dot. A tuple of AxisSelectors and Ellipsis. Ellipsis means "any number of axes."
Some functions may require that the Ellipsis is present at most once, while others may allow it to be present
multiple times.
"""

_DUMMY = object()


def selects_axis(selector: AxisSelection, selected: AxisSelection) -> bool:
    """Returns true if the selector has every axis in selected and, if dims are given, that they match"""
    selector_dict = axis_spec_to_shape_dict(selector)

    selected_dict = axis_spec_to_shape_dict(selected)

    if len(selector_dict) < len(selected_dict):
        return False

    for ax, size in selected_dict.items():
        selector_size = selector_dict.get(ax, _DUMMY)
        if selector_size is _DUMMY:
            return False

        if selector_size is None or size is None:
            continue

        if selector_size != size:
            return False

    return True


class _Sentinel: ...


def is_axis_compatible(ax1: AxisSelector, ax2: AxisSelector):
    """
    Returns true if the two axes are compatible, meaning they have the same name and, if both are Axis, the same size
    """
    if isinstance(ax1, str):
        if isinstance(ax2, str):
            return ax1 == ax2
        return ax1 == ax2.name
    if isinstance(ax2, str):
        return ax1.name == ax2
    return ax1.name == ax2.name


@overload
def axis_spec_to_shape_dict(axis_spec: AxisSpec) -> dict[str, int]:  # type: ignore
    ...


@overload
def axis_spec_to_shape_dict(axis_spec: AxisSelection) -> dict[str, int | None]:  # type: ignore
    ...


def axis_spec_to_shape_dict(axis_spec: AxisSelection) -> dict[str, int | None]:  # type: ignore
    if isinstance(axis_spec, Axis):
        return {axis_name(axis_spec): axis_spec.size}

    if isinstance(axis_spec, str):
        return {axis_spec: None}

    if isinstance(axis_spec, Mapping):
        return dict(**axis_spec)

    spec = ensure_tuple(axis_spec)  # type: ignore

    shape_dict: dict[str, int | None] = {}
    for ax in spec:
        if isinstance(ax, Axis):
            shape_dict[ax.name] = ax.size
        elif isinstance(ax, str):
            shape_dict[ax] = None
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return shape_dict


@typing.overload
def axis_spec_to_tuple(axis_spec: ShapeDict) -> tuple[Axis, ...]: ...


@typing.overload
def axis_spec_to_tuple(axis_spec: AxisSpec) -> tuple[Axis, ...]: ...


@typing.overload
def axis_spec_to_tuple(axis_spec: PartialShapeDict) -> tuple[AxisSelector, ...]: ...


@typing.overload
def axis_spec_to_tuple(axis_spec: AxisSelection) -> tuple[AxisSelector, ...]: ...


def axis_spec_to_tuple(axis_spec: AxisSelection) -> tuple[AxisSelector, ...]:
    if isinstance(axis_spec, Mapping):
        return tuple(Axis(name, size) if size is not None else name for name, size in axis_spec.items())

    if isinstance(axis_spec, Axis | str):
        return (axis_spec,)

    if isinstance(axis_spec, Sequence):
        return tuple(axis_spec)

    raise ValueError(f"Invalid axis spec: {axis_spec}")


@overload
def concat_axes(a1: ShapeDict, a2: AxisSpec) -> ShapeDict:
    pass


@overload
def concat_axes(a1: Sequence[Axis], a2: AxisSpec) -> tuple[Axis, ...]:
    pass


@overload
def concat_axes(a1: AxisSpec, a2: AxisSpec) -> AxisSpec:
    pass


@overload
def concat_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    pass


def concat_axes(a1, a2):
    """Concatenates two AxisSpecs. Raises ValueError if any axis is present in both specs"""

    if isinstance(a1, Axis) and isinstance(a2, Axis):
        if axis_name(a1) == axis_name(a2):
            raise ValueError(f"Axis {a1} specified twice")
        return (a1, a2)
    elif isinstance(a1, Mapping):
        out = dict(a1)
        a2 = axis_spec_to_shape_dict(a2)
        for ax, sz in a2.items():
            if ax in a1:
                raise ValueError(f"Axis {ax} specified twice")
            out[ax] = sz

        return out
    elif isinstance(a2, Mapping):
        out = axis_spec_to_shape_dict(a1)
        for ax, sz in a2.items():
            if ax in out:
                raise ValueError(f"Axis {ax} specified twice")
            out[ax] = sz

        return axis_spec_to_tuple(out)
    else:
        a1 = axis_spec_to_tuple(a1)
        a2 = axis_spec_to_tuple(a2)

        a1_names = [axis_name(ax) for ax in a1]
        a2_names = [axis_name(ax) for ax in a2]

        if len(set(a1_names) & set(a2_names)) > 0:
            overlap = [ax for ax in a1_names if ax in a2_names]
            raise ValueError(f"AxisSpecs overlap! {' '.join(str(x) for x in overlap)}")
        return a1 + a2


@typing.overload
def union_axes(a1: ShapeDict, a2: AxisSpec) -> ShapeDict: ...


@typing.overload
def union_axes(a1: AxisSpec, a2: ShapeDict) -> ShapeDict: ...


@typing.overload
def union_axes(a1: AxisSpec, a2: AxisSpec) -> AxisSpec: ...


@typing.overload
def union_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection: ...


def union_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    """
    Similar to concat_axes, but allows axes to be specified multiple times. The resulting AxisSpec will have the
    order of just concatenating each axis spec, but with any duplicate axes removed.

    Raises if any axis is present in both specs with different sizes
    """

    should_return_dict = isinstance(a1, Mapping) or isinstance(a2, Mapping)

    a1_dict = axis_spec_to_shape_dict(a1)
    a2_dict = axis_spec_to_shape_dict(a2)

    for ax, sz in a2_dict.items():
        if ax in a1_dict:
            if sz is not None and a1_dict[ax] is not None and sz != a1_dict[ax]:
                raise ValueError(f"Axis {ax} present in both specs with different sizes")
        else:
            a1_dict[ax] = sz

    if should_return_dict:
        return a1_dict
    else:
        return axis_spec_to_tuple(a1_dict)


@overload
def eliminate_axes(axis_spec: Axis | Sequence[Axis], axes: AxisSelection) -> tuple[Axis, ...]:  # type: ignore
    ...


@overload
def eliminate_axes(axis_spec: ShapeDict, axes: AxisSelection) -> ShapeDict:  # type: ignore
    ...


@overload
def eliminate_axes(axis_spec: AxisSelection, axes: AxisSelection) -> AxisSelection:  # type: ignore
    ...


@overload
def eliminate_axes(axis_spec: PartialShapeDict, axes: AxisSelection) -> PartialShapeDict:  # type: ignore
    ...


def eliminate_axes(axis_spec: AxisSelection, to_remove: AxisSelection) -> AxisSelection:  # type: ignore
    """Returns a new axis spec that is the same as the original, but without any axes in axes.
    Raises if any axis in to_remove is not present in axis_spec"""

    should_return_dict = isinstance(axis_spec, Mapping)

    axis_spec_dict = axis_spec_to_shape_dict(axis_spec)
    to_remove = axis_spec_to_shape_dict(to_remove)
    for ax in to_remove:
        name = axis_name(ax)
        if name not in axis_spec_dict:
            raise ValueError(f"Axis {name} not present in axis spec {axis_spec}")
        _check_size_consistency(axis_spec, to_remove, name, axis_spec_dict[name], to_remove[name])
        del axis_spec_dict[name]

    if should_return_dict:
        return axis_spec_dict
    else:
        return axis_spec_to_tuple(axis_spec_dict)


@typing.overload
def without_axes(axis_spec: ShapeDict, to_remove: AxisSelection, allow_mismatched_sizes=False) -> ShapeDict:  # type: ignore
    ...


@typing.overload
def without_axes(axis_spec: Sequence[Axis], to_remove: AxisSelection, allow_mismatched_sizes=False) -> tuple[Axis, ...]:  # type: ignore
    ...


@typing.overload
def without_axes(axis_spec: AxisSpec, to_remove: AxisSelection, allow_mismatched_sizes=False) -> AxisSpec:  # type: ignore
    ...


@typing.overload
def without_axes(axis_spec: AxisSelection, to_remove: AxisSelection, allow_mismatched_sizes=False) -> AxisSelection:  # type: ignore
    """As eliminate_axes, but does not raise if any axis in to_remove is not present in axis_spec"""


@typing.overload
def without_axes(axis_spec: Sequence[AxisSelector], to_remove: AxisSelection, allow_mismatched_sizes=False) -> tuple[AxisSpec, ...]:  # type: ignore
    ...


@typing.overload
def without_axes(axis_spec: PartialShapeDict, to_remove: AxisSelection, allow_mismatched_sizes=False) -> PartialShapeDict:  # type: ignore
    ...


def without_axes(axis_spec: AxisSelection, to_remove: AxisSelection, allow_mismatched_sizes=False) -> AxisSelection:  # type: ignore
    """
    As eliminate_axes, but does not raise if any axis in to_remove is not present in axis_spec.

    However, this does raise if any axis in to_remove is present in axis_spec with a different size.
    """

    to_remove_dict = axis_spec_to_shape_dict(to_remove)
    was_dict = isinstance(axis_spec, Mapping)

    axis_spec_dict = axis_spec_to_shape_dict(axis_spec)

    for ax, size in to_remove_dict.items():
        if ax in axis_spec_dict:
            if not allow_mismatched_sizes and (
                size is not None and axis_spec_dict[ax] is not None and size != axis_spec_dict[ax]
            ):
                raise ValueError(f"Axis {ax} present in both specs with different sizes: {axis_spec} - {to_remove}")
            del axis_spec_dict[ax]

    if was_dict:
        return axis_spec_dict
    return axis_spec_to_tuple(axis_spec_dict)


@typing.overload
def unsize_axes(axis_spec: PartialShapeDict, to_unsize: AxisSelection) -> PartialShapeDict: ...


@typing.overload
def unsize_axes(axis_spec: AxisSelection, to_unsize: AxisSelection) -> AxisSelection: ...


@typing.overload
def unsize_axes(axis_spec: PartialShapeDict) -> PartialShapeDict: ...


@typing.overload
def unsize_axes(axis_spec: AxisSelection) -> AxisSelection: ...


def unsize_axes(axis_spec: AxisSelection, to_unsize: AxisSelection | None = None) -> AxisSelection:
    """
    This function is used to remove the sizes of axes in an axis spec.
    There are two overloads:
    - If to_unsize is None, then all axes in axis_spec will be unsized
    - If to_unsize is not None, then all axes in to_unsize will be unsized. Raises if any axis in to_unsize is not present in axis_spec
    """

    if to_unsize is None:
        if isinstance(axis_spec, Mapping):
            return {name: None for name in axis_spec}
        elif isinstance(axis_spec, Axis | str):
            return axis_name(axis_spec)
        else:
            return tuple(axis_name(ax) for ax in ensure_tuple(axis_spec))  # type: ignore

    was_dict = isinstance(axis_spec, Mapping)

    to_unsize = axis_spec_to_shape_dict(to_unsize)
    axis_spec_dict: dict[str, int | None] = axis_spec_to_shape_dict(axis_spec)  # type: ignore
    for ax in to_unsize:
        name = axis_name(ax)
        if name not in axis_spec_dict:
            raise ValueError(f"Axis {name} not present in axis spec {axis_spec}")
        axis_spec_dict[name] = None

    if was_dict:
        return axis_spec_dict

    return axis_spec_to_tuple(axis_spec_dict)


@overload
def replace_axis(axis_spec: AxisSpec, old: AxisSelector, new: AxisSpec) -> AxisSpec: ...


@overload
def replace_axis(axis_spec: AxisSelection, old: AxisSelector, new: AxisSelection) -> AxisSelection: ...


def replace_axis(axis_spec: AxisSelection, old: AxisSelector, new: AxisSelection) -> AxisSelection:
    """Returns a new axis spec that is the same as the original, but with any axes in old replaced with new. Raises if old is
    not present in axis_spec"""

    was_dict = isinstance(axis_spec, Mapping)
    axis_spec_dict = axis_spec_to_shape_dict(axis_spec)
    new_dict = axis_spec_to_shape_dict(new)

    found = False
    out = {}
    for ax, size in axis_spec_dict.items():
        if is_axis_compatible(ax, old):
            found = True
            for new_ax, new_size in new_dict.items():
                if new_ax in axis_spec_dict and new_ax != ax:
                    raise ValueError(
                        f"Axis {new_ax} already present in axis spec {axis_spec}. Replacing {ax} with {new_ax} would"
                        " cause a conflict"
                    )
                out[new_ax] = new_size
        else:
            out[ax] = size

    if not found:
        raise ValueError(f"Axis {old} not found in axis spec {axis_spec}")

    if was_dict:
        return out

    return axis_spec_to_tuple(out)


@overload
def intersect_axes(ax1: ShapeDict, ax2: AxisSelection) -> ShapeDict: ...


@overload
def intersect_axes(ax1: tuple[AxisSelector, ...], ax2: AxisSpec) -> tuple[Axis, ...]: ...


@overload
def intersect_axes(ax1: tuple[AxisSelector, ...], ax2: AxisSelection) -> tuple[AxisSelector, ...]:  # type: ignore
    ...


@overload
def intersect_axes(ax1: AxisSpec, ax2: AxisSelection) -> AxisSpec:  # type: ignore
    ...


def intersect_axes(ax1: AxisSelection, ax2: AxisSelection) -> AxisSelection:
    """Returns a tuple of axes that are present in both ax1 and ax2.
    The returned order is the same as ax1.
    """
    ax2_dict = axis_spec_to_shape_dict(ax2)
    out: list[AxisSelector] = []
    was_dict = isinstance(ax1, Mapping)
    ax1_dict = axis_spec_to_shape_dict(ax1)

    for ax, size in ax1_dict.items():
        if ax in ax2_dict:
            sz2 = ax2_dict[ax]
            _check_size_consistency(ax1, ax2, ax, size, sz2)
            if sz2 is not None:
                out.append(Axis(ax, sz2))
            else:
                out.append(ax)

    if was_dict:
        return axis_spec_to_shape_dict(out)
    else:
        return tuple(out)


@overload
def axis_name(ax: AxisSelector) -> str:  # type: ignore
    ...


@overload
def axis_name(ax: Sequence[AxisSelector]) -> tuple[str, ...]:  # type: ignore
    ...


def axis_name(ax: AxisSelection) -> str | tuple[str, ...]:
    """
    Returns the name of the axis. If ax is a string, returns ax. If ax is an Axis, returns ax.name
    """

    def _ax_name(ax: AxisSelector) -> str:
        if isinstance(ax, Axis):
            return ax.name
        else:
            return ax

    if isinstance(ax, (Axis, str)):
        return _ax_name(ax)
    else:
        return tuple(_ax_name(x) for x in ax)


def axis_size(ax: AxisSpec) -> int:
    """
    Returns the size of the axis or the product of the sizes of the axes in the axis spec
    """

    if isinstance(ax, Axis):
        return ax.size
    elif isinstance(ax, Mapping):
        return prod(ax.values())
    else:
        return prod(axis.size for axis in ax)


@typing.overload
def resolve_axis(axis_spec: AxisSpec, axis_selection: AxisSelector) -> Axis: ...


@typing.overload
def resolve_axis(axis_spec: AxisSpec, axis_selection: AxisSelection) -> AxisSpec: ...


def resolve_axis(axis_spec: AxisSpec, axis_selection: AxisSelection) -> AxisSpec:
    """
    Returns the axis or axes in axis_spec that match the name of axis_selection.

    If axis_selection is a str or axis, returns a single Axis. If it is a sequence, returns a sequence of Axes.

    If an axis is present with a different size, raises ValueError.
    """
    as_dict = axis_spec_to_shape_dict(axis_spec)

    if isinstance(axis_selection, str | Axis):
        name = axis_name(axis_selection)
        if name not in as_dict:
            raise ValueError(f"Axis {name} not found in {axis_spec}")

        if isinstance(axis_selection, Axis):
            _check_size_consistency(axis_spec, axis_selection, name, as_dict[name], axis_size(axis_selection))

        return Axis(name, as_dict[name])
    else:
        out = {}
        selection_was_dict = isinstance(axis_selection, Mapping)
        select_dict = axis_spec_to_shape_dict(axis_selection)

        ax: str
        for ax, size in select_dict.items():
            if ax not in as_dict:
                raise ValueError(f"Axis {ax} not found in {axis_spec}")
            _check_size_consistency(axis_spec, axis_selection, ax, as_dict[ax], size)

            out[ax] = as_dict[ax]

        if selection_was_dict:
            return out
        else:
            return axis_spec_to_tuple(out)


class dslice(eqx.Module):
    """Dynamic slice, comprising a (start, length) pair. Also aliased as ``ds``.

    NumPy-style slices like ``a[i:i+16]`` don't work inside :func:`jax.jit`, because
    JAX requires slice bounds to be static. ``dslice`` works around this by
    separating the dynamic ``start`` from the static ``size`` so that you can
    write ``a[dslice(i, 16)]`` or simply ``a[ds(i, 16)]``.

    When used in indexing or ``at`` updates, ``dslice`` behaves like a gather of
    ``size`` elements starting at ``start``. Reads beyond the end of the array are
    filled with a value (0 by default) and writes outside the array bounds are
    dropped, matching JAX's default scatter/gather semantics.

    This class's name is taken from :mod:`jax.experimental.pallas`.
    """

    start: int
    size: int

    def to_slice(self) -> slice:
        return slice(self.start, self.start + self.size)

    def __init__(self, start: int, length: int | Axis):
        """
        As a convenience, if length is an Axis, it will be converted to `length.size`
        Args:
            start:
            length:
        """
        self.start = start
        if isinstance(length, Axis):
            self.size = length.size
        else:
            self.size = length

    @staticmethod
    def block(idx: int, size: int) -> "dslice":
        """
        Returns a dslice that selects a single block of size `size` starting at `idx`
        """
        return dslice(idx * size, size)


ds: typing.TypeAlias = dslice


def dblock(idx: int, size: int) -> dslice:
    """
    Returns a dslice that selects a single block of size `size` starting at `idx`
    """
    return dslice(idx * size, size)


Ax = typing.TypeVar("Ax", AxisSelector, Axis)


def rearrange_for_partial_order(partial_order: PartialAxisSpec, axes: tuple[Ax, ...]) -> tuple[Ax, ...]:
    """Rearrange the axes to fit the provided partial order.
    Uses a greedy algorithm that tries to keep elements in roughly the same order they came in
     (subject to the partial order), but moves them to the earliest slot that is after all prior axes
     in the original order.
     The exact behavior of this function is not guaranteed to be stable, but it should be stable
     for most reasonable use cases. If you really need a specific order, you should provide a full
     order instead of a partial order.
    """

    if partial_order == (Ellipsis,):
        return axes

    spec = axis_spec_to_shape_dict(axes)

    def as_axis(ax_name: str) -> Ax:
        if spec[ax_name] is None:
            return ax_name  # type: ignore
        else:
            return Axis(ax_name, spec[ax_name])  # type: ignore

    if Ellipsis not in partial_order:
        pa: tuple[AxisSelector, ...] = partial_order  # type: ignore
        if set(axis_name(a) for a in pa) != set(spec.keys()) or len(pa) != len(spec.keys()):
            raise ValueError(
                "Partial order must be a permutation of the axes if no ellipsis is provided."
                f" However {pa} is not a permutation of {axes}"
            )

        # reorder axes to match partial order
        return tuple(as_axis(axis_name(name)) for name in pa)

    partial_order_names = [axis_name(s) for s in partial_order if s is not ...]

    uncovered_ordered_elements = set(partial_order_names)

    if len(partial_order_names) != len(uncovered_ordered_elements):
        raise ValueError("Partial order must not contain duplicate elements")

    # replace ... with [], which is where we'll put the remaining axes

    out_order = [[axis_name(a)] if a is not ... else [] for a in partial_order]

    # now we'll fill in the ordered elements
    target_pos = index_where(lambda x: x == [], out_order)

    for ax in axes:
        ax_name = axis_name(ax)
        if ax_name in uncovered_ordered_elements:
            uncovered_ordered_elements.remove(ax_name)
            # already in the right place
            # update target_pos to come after this if possible
            try:
                this_pos = index_where(lambda x: ax_name in x, out_order)
                # find first empty slot after this_pos. prefer not to go backwards
                this_pos = max(this_pos + 1, target_pos)
                target_pos = index_where(lambda x: x == [], out_order, start=this_pos)
            except ValueError:
                # leave it where it is
                pass
        elif ax_name in partial_order_names:
            raise ValueError(f"Axis {ax_name} appears multiple times in the partial order")
        else:
            # this can appear in any ... slot. our heuristic is to put it in the first
            # slot that comes after the most recently seen ordered element
            out_order[target_pos].append(ax_name)

    if len(uncovered_ordered_elements) > 0:
        raise ValueError(f"The following axes are not present in output: {' '.join(uncovered_ordered_elements)}")

    # now we have a list of lists of axis names. we need to flatten it and convert to axes
    return tuple(as_axis(name) for name in sum(out_order, []))


def replace_missing_with_ellipsis(ax1: AxisSelection, ax2: AxisSelection) -> PartialAxisSpec:
    """Returns ax1, except that:

    * any axis not in ax2 is replaced with Ellipsis
    * if ax1 has a str axis where ax2 has an Axis, then it is replaced with the Axis

    Raises if ax1 and ax2 have any axes with the same name but different sizes
    """
    ax2_dict = axis_spec_to_shape_dict(ax2)
    out: list[AxisSelector | EllipsisType] = []
    ax1_dict = axis_spec_to_shape_dict(ax1)

    for ax, size in ax1_dict.items():
        if ax in ax2_dict:
            sz2 = ax2_dict[ax]
            _check_size_consistency(ax1, ax2, ax, size, sz2)
            if sz2 is not None:
                out.append(Axis(ax, sz2))
            else:
                out.append(ax)
        else:
            out.append(Ellipsis)

    return tuple(out)


__all__ = [
    "Axis",
    "AxisSelector",
    "AxisSelection",
    "AxisSpec",
    "PartialAxisSpec",
    "PartialShapeDict",
    "ShapeDict",
    "axis_name",
    "axis_size",
    "concat_axes",
    "union_axes",
    "axis_spec_to_shape_dict",
    "ds",
    "dslice",
    "dblock",
    "eliminate_axes",
    "intersect_axes",
    "is_axis_compatible",
    "replace_axis",
    "resolve_axis",
    "selects_axis",
    "union_axes",
    "without_axes",
    "unsize_axes",
    "rearrange_for_partial_order",
]


def _check_size_consistency(
    spec1: AxisSelection, spec2: AxisSelection, name: str, size1: int | None, size2: int | None
):
    if size1 is not None and size2 is not None and size1 != size2:
        raise ValueError(f"Axis {name} has different sizes in {spec1} and {spec2}: {size1} != {size2}")


def to_jax_shape(shape: AxisSpec):
    if isinstance(shape, Axis):
        return shape.size
    elif isinstance(shape, Sequence):
        return tuple(s.size for s in shape)
    return tuple(shape[a] for a in shape)
