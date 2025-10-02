# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax
import jax.numpy as jnp

from .axis import Axis, AxisSelector
from .core import NamedArray


def top_k(
    arr: NamedArray, axis: AxisSelector, k: int, new_axis: AxisSelector | None = None
) -> tuple[NamedArray, NamedArray]:
    """
    Select the top k elements along the given axis.
    Args:
        arr (NamedArray): array to select from
        axis (AxisSelector): axis to select from
        k (int): number of elements to select
        new_axis (AxisSelector | None): new axis name, if none, the original axis will be resized to k

    Returns:
        NamedArray: array with the top k elements along the given axis
        NamedArray: array with the top k elements' indices along the given axis
    """
    pos = arr.axis_indices(axis)
    if pos is None:
        raise ValueError(f"Axis {axis} not found in {arr}")
    new_array = jnp.moveaxis(arr.array, pos, -1)  # move axis to the last position
    values, indices = jax.lax.top_k(new_array, k=k)
    values = jnp.moveaxis(values, -1, pos)  # move axis back to its original position
    indices = jnp.moveaxis(indices, -1, pos)

    if new_axis is None:
        axis = arr.resolve_axis(axis)
        new_axis = axis.resize(k)
    elif isinstance(new_axis, str):
        new_axis = Axis(new_axis, k)

    updated_axes = arr.axes[:pos] + (new_axis,) + arr.axes[pos + 1 :]
    return NamedArray(values, updated_axes), NamedArray(indices, updated_axes)
