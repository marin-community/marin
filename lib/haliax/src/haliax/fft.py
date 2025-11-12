# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Named wrappers around :mod:`jax.numpy.fft`.

These functions mirror the behaviour of their :mod:`jax.numpy.fft` counterparts
while accepting named axes.  Instead of separate ``fftn``/``fft2`` variants we
provide a single ``fft`` family of functions whose ``axis`` argument controls
which axes are transformed.

The ``axis`` parameter can be one of:

* ``None`` – operate on the last axis.
* ``str`` – name of an existing axis in the input.
* :class:`~haliax.Axis` – specifies both the axis to transform (by name) and the
  desired FFT length.  The output axis is replaced by the provided ``Axis``.
* ``dict`` – mapping from axis selectors (names or ``Axis`` objects) to optional
  sizes.  A value of ``None`` uses the existing axis length.  The mapping order
  determines the order of transforms and dispatches to the ``n``‑dimensional
  variants in :mod:`jax.numpy.fft`.

Example
-------

```python
X, Y = hax.make_axes(X=4, Y=6)
arr = hax.arange((X, Y))

# 1D transform along ``Y``
hax.fft(arr, axis="Y")

# 2D transform across both axes
hax.fft(arr, axis={"X": None, "Y": None})

# Resize the ``Y`` axis before transforming
hax.fft(arr, axis={"Y": Axis("Y", 8)})
```
"""

from __future__ import annotations

from typing import Mapping, MutableSequence, Sequence

import jax.numpy.fft as jfft

from .axis import Axis, AxisSelector, AxisSelection
from .core import NamedArray

AxisSizeLike = int | Axis | None
AxisMapping = Mapping[AxisSelector, AxisSizeLike]


def _single_axis(a: NamedArray, axis: AxisSelector | None):
    if axis is None:
        idx = a.ndim - 1
        ax = a.axes[idx]
        n = None
    elif isinstance(axis, Axis):
        idx = a.axis_indices(axis.name)
        if idx is None:
            raise ValueError(f"Axis {axis} not found in {a.axes}")
        ax = axis
        n = axis.size
    else:
        idx = a.axis_indices(axis)
        if idx is None:
            raise ValueError(f"Axis {axis} not found in {a.axes}")
        ax = a.axes[idx]
        n = None
    return idx, ax, n


def _multi_axis(a: NamedArray, axis: AxisMapping):
    axes_idx: MutableSequence[int] = []
    sizes: MutableSequence[int] = []
    new_axes = list(a.axes)
    for key, val in axis.items():
        idx = a.axis_indices(key)
        if idx is None:
            raise ValueError(f"Axis {key} not found in {a.axes}")
        ax = a.axes[idx]
        if isinstance(val, Axis):
            size = val.size
            new_axes[idx] = val
        elif val is None:
            size = ax.size
        else:
            size = int(val)
            new_axes[idx] = ax.resize(size)
        axes_idx.append(idx)
        sizes.append(size)
    return list(axes_idx), list(sizes), new_axes


def fft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.fft`.

    See module level documentation for the behaviour of the ``axis`` argument.
    """

    if isinstance(axis, Mapping):
        axes_idx, sizes, new_axes = _multi_axis(a, axis)
        out = jfft.fftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, Axis)):
        axes_idx, sizes, new_axes = _multi_axis(a, {ax: None for ax in axis})
        out = jfft.fftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    else:
        idx, new_axis, n = _single_axis(a, axis)  # type: ignore[arg-type]
        out = jfft.fft(a.array, n=n, axis=idx, norm=norm)
        axes = list(a.axes)
        axes[idx] = new_axis
        return NamedArray(out, tuple(axes))


def ifft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.ifft`."""

    if isinstance(axis, Mapping):
        axes_idx, sizes, new_axes = _multi_axis(a, axis)
        out = jfft.ifftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, Axis)):
        axes_idx, sizes, new_axes = _multi_axis(a, {ax: None for ax in axis})
        out = jfft.ifftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    else:
        idx, new_axis, n = _single_axis(a, axis)  # type: ignore[arg-type]
        out = jfft.ifft(a.array, n=n, axis=idx, norm=norm)
        axes = list(a.axes)
        axes[idx] = new_axis
        return NamedArray(out, tuple(axes))


def rfft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.rfft`."""

    if isinstance(axis, Mapping):
        axes_idx, sizes, new_axes = _multi_axis(a, axis)
        out = jfft.rfftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        last_idx = axes_idx[-1]
        last_in = sizes[-1]
        new_axes[last_idx] = new_axes[last_idx].resize(last_in // 2 + 1)
        return NamedArray(out, tuple(new_axes))
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, Axis)):
        axes_idx, sizes, new_axes = _multi_axis(a, {ax: None for ax in axis})
        out = jfft.rfftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        last_idx = axes_idx[-1]
        last_in = sizes[-1]
        new_axes[last_idx] = new_axes[last_idx].resize(last_in // 2 + 1)
        return NamedArray(out, tuple(new_axes))
    else:
        idx, ax, n = _single_axis(a, axis)  # type: ignore[arg-type]
        out = jfft.rfft(a.array, n=n, axis=idx, norm=norm)
        length = n if n is not None else ax.size
        new_axis = ax.resize(length // 2 + 1)
        axes = list(a.axes)
        axes[idx] = new_axis
        return NamedArray(out, tuple(axes))


def _single_axis_irfft(a: NamedArray, axis: AxisSelector | None):
    if axis is None:
        idx = a.ndim - 1
        in_ax = a.axes[idx]
        length = (in_ax.size - 1) * 2
        out_ax = in_ax.resize(length)
        n = None
    elif isinstance(axis, Axis):
        idx = a.axis_indices(axis.name)
        if idx is None:
            raise ValueError(f"Axis {axis} not found in {a.axes}")
        out_ax = axis
        n = axis.size
    else:
        idx = a.axis_indices(axis)
        if idx is None:
            raise ValueError(f"Axis {axis} not found in {a.axes}")
        in_ax = a.axes[idx]
        length = (in_ax.size - 1) * 2
        out_ax = in_ax.resize(length)
        n = None
    return idx, out_ax, n


def _multi_axis_irfft(a: NamedArray, axis: AxisMapping):
    axes_idx: MutableSequence[int] = []
    sizes: MutableSequence[int] = []
    new_axes = list(a.axes)
    items = list(axis.items())
    for i, (key, val) in enumerate(items):
        idx = a.axis_indices(key)
        if idx is None:
            raise ValueError(f"Axis {key} not found in {a.axes}")
        ax = a.axes[idx]
        if isinstance(val, Axis):
            size = val.size
            new_axes[idx] = val
        elif val is None:
            if i == len(items) - 1:
                size = (ax.size - 1) * 2
            else:
                size = ax.size
            new_axes[idx] = ax.resize(size)
        else:
            size = int(val)
            new_axes[idx] = ax.resize(size)
        axes_idx.append(idx)
        sizes.append(size)
    return list(axes_idx), list(sizes), new_axes


def irfft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.irfft`."""

    if isinstance(axis, Mapping):
        axes_idx, sizes, new_axes = _multi_axis_irfft(a, axis)
        out = jfft.irfftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, Axis)):
        axes_idx, sizes, new_axes = _multi_axis_irfft(a, {ax: None for ax in axis})
        out = jfft.irfftn(a.array, s=tuple(sizes), axes=tuple(axes_idx), norm=norm)
        return NamedArray(out, tuple(new_axes))
    else:
        idx, out_ax, n = _single_axis_irfft(a, axis)  # type: ignore[arg-type]
        out = jfft.irfft(a.array, n=n, axis=idx, norm=norm)
        axes = list(a.axes)
        axes[idx] = out_ax
        return NamedArray(out, tuple(axes))


def hfft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.hfft`.

    Only a single axis is supported; passing a dictionary with more than one
    entry will raise an error.
    """

    if isinstance(axis, Mapping):
        if len(axis) != 1:
            raise ValueError("hfft only supports a single axis")
        key, val = next(iter(axis.items()))
        if isinstance(val, Axis):
            axis = val
        elif val is None:
            axis = key
        else:
            name = key.name if isinstance(key, Axis) else key
            axis = Axis(name, int(val))

    idx, out_ax, n = _single_axis_irfft(a, axis)  # type: ignore[arg-type]
    out = jfft.hfft(a.array, n=n, axis=idx, norm=norm)
    axes = list(a.axes)
    axes[idx] = out_ax
    return NamedArray(out, tuple(axes))


def ihfft(
    a: NamedArray,
    axis: AxisSelector | Sequence[AxisSelector] | AxisMapping | None = None,
    norm: str | None = None,
) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.ihfft`.

    Only a single axis is supported; passing a dictionary with more than one
    entry will raise an error.
    """

    if isinstance(axis, Mapping):
        if len(axis) != 1:
            raise ValueError("ihfft only supports a single axis")
        key, val = next(iter(axis.items()))
        if isinstance(val, Axis):
            axis = val
        elif val is None:
            axis = key
        else:
            name = key.name if isinstance(key, Axis) else key
            axis = Axis(name, int(val))

    idx, ax, n = _single_axis(a, axis)  # type: ignore[arg-type]
    out = jfft.ihfft(a.array, n=n, axis=idx, norm=norm)
    length = n if n is not None else ax.size // 2 + 1
    new_axis = ax.resize(length)
    axes = list(a.axes)
    axes[idx] = new_axis
    return NamedArray(out, tuple(axes))


def fftshift(x: NamedArray, axes: AxisSelection | None = None) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.fftshift`."""

    if axes is None:
        out = jfft.fftshift(x.array)
    else:
        idxs = x.axis_indices(axes)
        if isinstance(idxs, tuple):
            if any(i is None for i in idxs):
                raise ValueError(f"Axis {axes} not found in {x.axes}")
        elif idxs is None:
            raise ValueError(f"Axis {axes} not found in {x.axes}")
        out = jfft.fftshift(x.array, axes=idxs)
    return NamedArray(out, x.axes)


def ifftshift(x: NamedArray, axes: AxisSelection | None = None) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.ifftshift`."""

    if axes is None:
        out = jfft.ifftshift(x.array)
    else:
        idxs = x.axis_indices(axes)
        if isinstance(idxs, tuple):
            if any(i is None for i in idxs):
                raise ValueError(f"Axis {axes} not found in {x.axes}")
        elif idxs is None:
            raise ValueError(f"Axis {axes} not found in {x.axes}")
        out = jfft.ifftshift(x.array, axes=idxs)
    return NamedArray(out, x.axes)


def fftfreq(axis: Axis, d: float = 1.0) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.fftfreq`."""

    return NamedArray(jfft.fftfreq(axis.size, d), (axis,))


def rfftfreq(axis: Axis, d: float = 1.0) -> NamedArray:
    """Named version of :func:`jax.numpy.fft.rfftfreq`."""

    new_axis = axis.resize(axis.size // 2 + 1)
    return NamedArray(jfft.rfftfreq(axis.size, d), (new_axis,))


__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
