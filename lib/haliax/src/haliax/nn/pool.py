# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


# Pooling operations, inspired by Flax
from functools import reduce
from typing import Callable, Literal

import jax

import haliax

from ..axis import AxisSpec, axis_spec_to_shape_dict, unsize_axes
from ..core import NamedArray
from ..partitioning import auto_sharded
from ..types import Scalar
from ..util import ensure_tuple

Padding = Literal["SAME", "VALID"] | int | tuple[tuple[int, int], ...]

DEFAULT_PADDING: Literal["VALID"] = "VALID"

# TODO: add dilation?


def pool(
    Window: AxisSpec,
    inputs: NamedArray,
    init: Scalar,
    reduce_fn: Callable[[Scalar, Scalar], Scalar],
    stride: int | tuple[int, ...] | None = None,
    padding: Padding = DEFAULT_PADDING,
    use_ceil: bool = False,
) -> NamedArray:
    """
    General function for pooling. Broadly based on the Flax implementation.

    Pooling functions are implemented using the ReduceWindow XLA op.

    Notes:
        JAX only implements autodiff for a few specific reductions (min, max, sum).

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        init: the initial value for the reduction
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        stride: int, or a sequence of `n` integers, representing the inter-window
          stride (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension, or an integer to pad all dimensions.
        use_ceil: if True, will use ceil instead of floor to compute the output shape

    Returns:
      The output of the reduction for each window slice.
    """
    Window = axis_spec_to_shape_dict(Window)

    reduce_fn = _patch_up_reduce_fn(reduce_fn)

    dims = []
    for ax in inputs.axes:
        if ax.name in Window:
            dims.append(Window[ax.name])
        else:
            dims.append(1)

    if isinstance(stride, int):
        stride = (stride,) * len(Window)

    if stride is not None:
        if len(Window) != len(stride):
            raise ValueError(f"len(Window) ({len(Window)}) != len(stride) ({len(stride)})")
        stride = ensure_tuple(stride)
        stride_map = {w: s for w, s in zip(Window, stride)}
        stride_out = []
        for ax in inputs.axes:
            if ax.name in stride_map:
                stride_out.append(stride_map[ax.name])
            else:
                stride_out.append(1)

        stride = tuple(stride_out)
        del stride_out
    else:
        stride = (1,) * len(dims)
        stride_map = {w: 1 for w in Window}

    if isinstance(padding, int):
        padding = ((padding, padding),) * len(Window)

    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))  # type: ignore
        if len(padding) != len(Window):
            raise ValueError(f"len(padding) ({len(padding)}) != len(Window) ({len(Window)})")
        if not all([len(x) == 2 for x in padding]):
            raise ValueError(f"each entry in padding must be length 2, got {padding}")

        padding_map = {w: p for w, p in zip(Window, padding)}
        if use_ceil:
            window_inputs = {w: ax.size for w, ax in zip(Window, inputs.axes)}
            padding_map = _use_ceil_padding(
                window_inputs=window_inputs,
                window_kernel=(Window),
                window_padding=padding_map,
                window_stride=stride_map,
            )

        padding_out = []
        for ax in inputs.axes:
            if ax.name in padding_map:
                padding_out.append(padding_map[ax.name])
            else:
                padding_out.append((0, 0))

        padding = tuple(padding_out)
    elif padding == "VALID" and use_ceil:
        padding = "SAME"
    else:
        padding = padding.upper()  # type: ignore
        if padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    y = jax.lax.reduce_window(inputs.array, init, reduce_fn, dims, stride, padding)

    out_axes = unsize_axes(inputs.axes, Window)

    return auto_sharded(haliax.named(y, out_axes))


def _patch_up_reduce_fn(reduce_fn):
    if reduce_fn is haliax.max:
        reduce_fn = jax.lax.max
    elif reduce_fn is haliax.min:
        reduce_fn = jax.lax.min
    elif reduce_fn is haliax.sum:
        reduce_fn = jax.lax.add
    elif reduce_fn is haliax.prod:
        reduce_fn = jax.lax.mul

    return reduce_fn


def max_pool(
    Window: AxisSpec,
    inputs: NamedArray,
    stride: int | tuple[int, ...] | None = None,
    padding: Padding = DEFAULT_PADDING,
    use_ceil: bool = False,
) -> NamedArray:
    """
    Max pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        stride: a sequence of `n` integers, representing the inter-window
                stride (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
               of `n` `(low, high)` integer pairs that give the padding to apply before
               and after each spatial dimension.
    Returns:
        The maximum value in each window slice.
    """
    return pool(Window, inputs, -float("inf"), jax.lax.max, stride, padding, use_ceil=use_ceil)


def min_pool(
    Window: AxisSpec,
    inputs: NamedArray,
    stride: int | tuple[int, ...] | None = None,
    padding: Padding = DEFAULT_PADDING,
    use_ceil: bool = False,
) -> NamedArray:
    """
    Min pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        stride: a sequence of `n` integers, representing the inter-window
                stride (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
            of `n` `(low, high)` integer pairs that give the padding to apply before
            and after each spatial dimension.
        use_ceil: if True, will use ceil instead of floor to compute the output shape
    Returns:
        The minimum value in each window slice.
    """
    return pool(Window, inputs, float("inf"), jax.lax.min, stride, padding, use_ceil=use_ceil)


def mean_pool(
    Window: AxisSpec,
    inputs: NamedArray,
    stride: int | tuple[int, ...] | None = None,
    padding: Padding = DEFAULT_PADDING,
    *,
    use_ceil: bool = False,
    count_include_pad: bool = False,
) -> NamedArray:
    """
    Mean pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        stride: a sequence of `n` integers, representing the inter-window stride (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
            of `n` `(low, high)` integer pairs that give the padding to apply before
            and after each spatial dimension.
    Returns:
      The mean value in each window slice.
    """
    tots = pool(Window, inputs, 0, jax.lax.add, stride, padding, use_ceil=use_ceil)
    if count_include_pad:
        Window = axis_spec_to_shape_dict(Window)
        window_size = reduce(lambda x, y: x * y, [s for s in Window.values()])
        return tots / window_size
    else:
        inputs_axes_without_batches = inputs.resolve_axis(unsize_axes(Window))
        ones = haliax.ones(inputs_axes_without_batches)
        tots = tots / pool(Window, ones, 0.0, jax.lax.add, stride, padding)
        return tots


def _use_ceil_padding(
    window_inputs: dict[str, int],
    window_kernel: dict[str, int],
    window_padding: dict[str, tuple[int, int]],
    window_stride: dict[str, int],
):
    # cribbed/adapted from equinox
    new_padding = {}
    for ax in window_inputs.keys():
        input_size = window_inputs[ax]
        kernel_size = window_kernel[ax]
        stride = window_stride[ax]
        left_padding, right_padding = window_padding[ax]
        if (input_size + left_padding + right_padding - kernel_size) % stride == 0:
            new_padding[ax] = (left_padding, right_padding)
        else:
            new_padding[ax] = (left_padding, right_padding + stride)

    return new_padding
