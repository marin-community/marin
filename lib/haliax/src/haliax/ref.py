# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from types import EllipsisType

import jax
import jax.numpy as jnp

from .axis import Axis, AxisSelector, axis_name, axis_spec_to_tuple, dslice as HaliaxDSlice
from .core import (
    NamedArray,
    NamedOrNumeric,
    SliceSpec,
    _compute_new_axes_and_slices_for_index,
    _convert_index_expr_to_dict,
    named,
)
from .jax_utils import is_pallas_dslice

from jax.experimental.pallas import dslice


class _AxisMetadata:
    """Minimal structure supplying just axis metadata for index helpers."""

    def __init__(self, axes: Sequence[Axis]):
        self.axes = tuple(axes)

    def axis_indices(self, axis: AxisSelector) -> int | None:
        name = axis_name(axis)
        for i, ax in enumerate(self.axes):
            if ax.name == name:
                return i
        return None


def _is_trivial_index(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None


def _slice_length(start: int, stop: int, step: int) -> int:
    if step == 0:
        raise ValueError("slice step cannot be zero")
    if step > 0:
        if start >= stop:
            return 0
        return (stop - start + step - 1) // step
    if start <= stop:
        return 0
    step_abs = -step
    return (start - stop - step - 1) // step_abs


def _is_slice_like(value: Any) -> bool:
    return isinstance(value, (slice, range, HaliaxDSlice)) or is_pallas_dslice(value)


def _normalize_slice_value(value: Any, axis: Axis) -> Any:
    if not _is_slice_like(value):
        return value

    if isinstance(value, range):
        # return slice(value.start, value.stop, value.step)
        value = slice(value.start, value.stop, value.step)

    if isinstance(value, slice):
        start, stop, step = value.indices(axis.size)
        length = _slice_length(start, stop, step)
        return dslice(start, length, step)

    if isinstance(value, HaliaxDSlice):
        return _to_pallas_dslice(value)

    if is_pallas_dslice(value):
        return value

    raise ValueError("Should not reach here")


def _dslice_params(value: Any) -> tuple[Any, int, Any] | None:
    """
    For slice like objects, return (start, size, step).
    """
    if isinstance(value, HaliaxDSlice):
        return value.start, value.size, 1
    if is_pallas_dslice(value):
        start = value.start
        size = value.size
        step = value.stride
        return start, size, step

    return None


def _axes_after_prefix(axes: Sequence[Axis], indices: Sequence[Any]) -> tuple[Axis, ...]:
    view_axes: list[Axis] = []
    for axis, sel in zip(axes, indices):
        if isinstance(sel, int):
            continue
        if isinstance(sel, slice):
            start, stop, step = sel.indices(axis.size)
            length = _slice_length(start, stop, step)
            view_axes.append(Axis(axis.name, length))
        else:
            params = _dslice_params(sel)
            if params is not None:
                _start, size, _step = params
                if not isinstance(size, int):
                    size = int(size)
                view_axes.append(Axis(axis.name, size))
                continue
            else:
                raise NotImplementedError(
                    "Slice references currently only support integer or slice prefixes; "
                    f"got {type(sel)} for axis {axis}"
                )
    return tuple(view_axes)


def _combine_index(
    current: slice | int,
    new: Any,
    *,
    base_axis: Axis,
    view_axis: Axis,
) -> Any:
    if isinstance(current, int):
        raise ValueError(f"Axis {base_axis.name} is already fixed by the slice reference")

    if _is_slice_like(current):
        current = _normalize_slice_value(current, axis=base_axis)
    else:
        raise NotImplementedError("Slice references currently only support simple integer/slice prefixes")

    ds_params = _dslice_params(current)
    assert ds_params is not None

    start0, size0, step0 = ds_params
    view_length = size0

    if _is_slice_like(new):
        new = _normalize_slice_value(new, axis=view_axis)

    if _is_slice_like(new):
        n_params = _dslice_params(new)
        assert n_params is not None
        start1, size1, step1 = n_params
        start = start0 + start1 * step0
        step = step0 * step1
        return dslice(start, size1, step)

    if isinstance(new, int):
        if new < -view_length or new >= view_length:
            raise IndexError(f"Index {new} out of bounds for axis {view_axis.name}")
        if new < 0:
            new += view_length
        return start0 + new * step0

    if isinstance(new, NamedArray):
        data = jnp.where(new.array < 0, new.array + view_length, new.array)
        transformed = data * step0 + start0
        return NamedArray(transformed, new.axes)

    if isinstance(new, jnp.ndarray):
        data = jnp.where(new < 0, new + view_length, new)
        return data * step0 + start0

    if isinstance(new, (list, tuple)):
        converted_list: list[int] = []
        for item in new:
            if not isinstance(item, int):
                raise TypeError("Only integer lists/tuples are supported for slice references")
            if item < -view_length or item >= view_length:
                raise IndexError(f"Index {item} out of bounds for axis {view_axis.name}")
            if item < 0:
                item += view_length
            converted_list.append(start0 + item * step0)
        return converted_list

    raise TypeError(
        "Slice references only support integers, slices, dslice, NamedArray, jnp.ndarray, or lists/tuples thereof"
        f"; got {type(new)}"
    )


def _combine_indices(
    axes: Sequence[Axis],
    prefix: Sequence[Any],
    selectors: Mapping[AxisSelector, Any],
) -> tuple[Any, ...]:
    if not selectors:
        return tuple(prefix)

    current = list(prefix)
    view_axes = _axes_after_prefix(axes, prefix)
    view_positions = {ax.name: i for i, ax in enumerate(view_axes)}
    mapping = [i for i, sel in enumerate(prefix) if not isinstance(sel, int)]

    for axis_sel, new_value in selectors.items():
        name = axis_name(axis_sel)
        view_pos = view_positions.get(name)
        if view_pos is None:
            raise ValueError(f"Axis {name} is not available in this slice reference")
        base_pos = mapping[view_pos]
        base_axis = axes[base_pos]
        view_axis = view_axes[view_pos]
        combined = _combine_index(current[base_pos], new_value, base_axis=base_axis, view_axis=view_axis)
        current[base_pos] = combined

    return tuple(current)


def _indices_to_selector(axes: Sequence[Axis], indices: Sequence[Any]) -> dict[AxisSelector, Any]:
    selector: dict[AxisSelector, Any] = {}
    for axis, idx in zip(axes, indices):
        if _is_trivial_index(idx):
            continue
        selector[axis] = idx
    return selector


def _to_pallas_dslice(value):
    if is_pallas_dslice(value):
        return value
    if isinstance(value, HaliaxDSlice):
        return dslice(value.start, value.size)
    raise TypeError("Expected a haliax.dslice or pallas.dslice")


def _is_supported_prefix(idx: Any) -> bool:
    if isinstance(idx, (slice, int, HaliaxDSlice)) or is_pallas_dslice(idx):
        return True
    if isinstance(idx, list) and all(isinstance(it, int) for it in idx):
        return True
    if isinstance(idx, (jnp.ndarray, jax.Array)):
        return idx.ndim <= 1 and jnp.issubdtype(idx.dtype, jnp.integer)
    if isinstance(idx, NamedArray):
        return idx.ndim <= 1 and jnp.issubdtype(idx.array.dtype, jnp.integer)
    if jnp.isscalar(idx):
        dtype = idx.dtype
        if dtype is None:
            dtype = jnp.asarray(idx).dtype
        return jnp.issubdtype(dtype, jnp.integer)
    return False

    # >>> @partial(jax.tree_util.register_dataclass,
    # ...          data_fields=['x', 'y'],
    # ...          meta_fields=['op'])


@functools.partial(
    jax.tree_util.register_dataclass,
    # TODO: prefix should be a data field, but it can't until we make all _prefix members Slice or other arrays
    # data_fields=["_ref", "_prefix"],
    # meta_fields=["_axes"],
    data_fields=["_ref"],
    meta_fields=["_axes", "_prefix"],
)
@dataclass(frozen=True)
class NamedRef:
    """Named wrapper around :class:`jax.Ref` that preserves axis metadata."""

    _ref: jax.Ref
    _axes: tuple[Axis, ...]
    _prefix: tuple[Any, ...]

    def __post_init__(self):
        if len(self._axes) != len(self._prefix):
            raise ValueError("Prefix entries must align with axes")

    @property
    def dtype(self):
        """Return the dtype of the underlying reference."""
        return self._ref.dtype

    @property
    def axes(self) -> tuple[Axis, ...]:
        """Axes visible from this view after applying staged selectors."""
        return _axes_after_prefix(self._axes, self._prefix)

    @property
    def shape(self) -> Mapping[str, int]:
        """Mapping from axis name to size for the current view."""
        return {ax.name: ax.size for ax in self.axes}

    @property
    def named_shape(self) -> Mapping[str, int]:
        return self.shape

    @property
    def ndim(self) -> int:
        """Number of axes in the current view."""
        return len(self.axes)

    def value(self) -> NamedArray:
        """Materialize this reference view as a `NamedArray`."""
        _, axes_spec, index_tuple = self._prepare(Ellipsis)
        result = self._ref[tuple(index_tuple)]
        return named(result, axes_spec)

    def unsliced(self) -> "NamedRef":
        """Return a view of the original reference without staged selectors."""
        full_prefix = tuple(slice(None) for _ in self._axes)
        if self._prefix == full_prefix:
            return self
        return NamedRef(self._ref, self._axes, full_prefix)

    def _prepare(
        self, idx: SliceSpec | EllipsisType | None
    ) -> tuple[tuple[Any, ...], tuple[AxisSelector, ...], list[Any]]:
        """Combine existing prefixes with a new index expression and produce raw JAX indexers."""
        if idx is Ellipsis or idx is None:
            selectors: Mapping[AxisSelector, Any] = {}
        else:
            selectors = _convert_index_expr_to_dict(idx)

        combined = _combine_indices(self._axes, self._prefix, selectors)
        selector_dict = _indices_to_selector(self._axes, combined)
        array_info = _AxisMetadata(self._axes)
        new_axes, ordered = _compute_new_axes_and_slices_for_index(array_info, selector_dict)
        normalized_spec = axis_spec_to_tuple(new_axes)
        index_tuple: list[Any] = []
        for axis, item in zip(self._axes, ordered):
            selector_value = selector_dict.get(axis)
            if selector_value is not None and (
                isinstance(selector_value, HaliaxDSlice) or is_pallas_dslice(selector_value)
            ):
                index_tuple.append(_to_pallas_dslice(selector_value))
            else:
                index_tuple.append(item.array if isinstance(item, NamedArray) else item)
        return combined, normalized_spec, index_tuple

    def __getitem__(self, idx: SliceSpec | EllipsisType = Ellipsis) -> NamedArray:
        """Read from the reference using named indexing semantics."""
        _, axes_spec, index_tuple = self._prepare(idx)
        result = self._ref[tuple(index_tuple)]
        return named(result, axes_spec)

    def __setitem__(self, idx: SliceSpec | EllipsisType, value: NamedOrNumeric) -> None:
        """Write to the reference using named indexing semantics."""
        _, axes_spec, index_tuple = self._prepare(idx)
        if isinstance(value, NamedArray):
            desired = axes_spec
            desired_tuple = axis_spec_to_tuple(desired)
            desired_names = tuple(axis_name(ax) for ax in desired_tuple)
            current_names = tuple(axis_name(ax) for ax in value.axes)
            if set(current_names) != set(desired_names):
                raise ValueError(
                    f"Value axes {current_names} do not match target axes {desired_names}; broadcasting is not yet supported"
                )
            if current_names != desired_names:
                value = value.rearrange(desired_tuple)
            payload = value.array
        else:
            payload = jnp.asarray(value)
        self._ref[tuple(index_tuple)] = payload

    def resolve_axis(self, axis: AxisSelector) -> Axis:
        """Resolve an axis selector to the corresponding axis in the current view."""
        name = axis_name(axis)
        for ax in self.axes:
            if ax.name == name:
                return ax
        raise ValueError(f"Axis {name} is not present in this reference view")

    def slice(self, selector: Mapping[AxisSelector, Any]) -> "NamedRef":
        """Return a new view with the provided selector staged for future operations."""
        normalized = {key: _normalize_slice_value(val, self.resolve_axis(key)) for key, val in selector.items()}
        combined = _combine_indices(self._axes, self._prefix, normalized)
        for idx in combined:
            if not _is_supported_prefix(idx):
                raise TypeError("Slice references only support simple integer/slice prefixes")
        return replace(self, _prefix=combined)

    def unsafe_buffer_pointer(self):  # pragma: no cover
        return self._ref.unsafe_buffer_pointer()


def new_ref(value: NamedArray) -> NamedRef:
    """Construct a `NamedRef` from a `NamedArray`."""
    if isinstance(value, NamedArray):
        base_axes = value.axes
        impl = jax.new_ref(value.array)
    else:
        raise TypeError("new_ref only supports NamedArray inputs")

    prefix = tuple(slice(None) for _ in base_axes)
    return NamedRef(impl, base_axes, prefix)


def freeze(ref: NamedRef) -> NamedArray:
    """Freeze the reference and return its current contents."""
    return ref.value()


def get(ref: NamedRef, idx: SliceSpec | EllipsisType = Ellipsis) -> NamedArray:
    """Functional helper equivalent to `ref[idx]`."""
    return ref[idx]


def swap(ref: NamedRef, idx: SliceSpec | EllipsisType, value: NamedOrNumeric) -> NamedArray:
    """Swap the value at `idx`, returning the previous contents as a `NamedArray`."""
    _, axes_spec, index_tuple = ref._prepare(idx)
    if isinstance(value, NamedArray):
        desired = axes_spec
        desired_tuple = axis_spec_to_tuple(desired)
        desired_names = tuple(axis_name(ax) for ax in desired_tuple)
        current_names = tuple(axis_name(ax) for ax in value.axes)
        if set(current_names) != set(desired_names):
            raise ValueError(
                f"Value axes {current_names} do not match target axes {desired_names}; broadcasting is not yet supported"
            )
        if current_names != desired_names:
            value = value.rearrange(desired_tuple)
        payload = value.array
    else:
        payload = jnp.asarray(value)

    out = jax.ref.swap(ref._ref, tuple(index_tuple), payload, _function_name="haliax.ref.swap")
    return named(out, axes_spec)


__all__ = ["NamedRef", "new_ref", "freeze", "get", "swap"]
