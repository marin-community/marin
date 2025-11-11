# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import string
from functools import cached_property
from typing import Sequence, TypeVar

import equinox as eqx
import jax
import numpy as np
from jaxtyping import PRNGKeyArray

import haliax.partitioning

from ..axis import (
    Axis,
    AxisSelection,
    axis_name,
    axis_spec_to_shape_dict,
    replace_axis,
    selects_axis,
    unsize_axes,
    without_axes,
)
from ..core import NamedArray, named
from ..jax_utils import named_call
from ..random import uniform
from ..util import ensure_tuple

T = TypeVar("T")


class _ConvBase(eqx.Module):
    """
    Base class for Conv and ConvTranspose. Mostly just contains shared code.
    """

    Spatial: tuple[str | Axis, ...] = eqx.field(static=True)
    In: Axis = eqx.field(static=True)
    Out: Axis = eqx.field(static=True)
    weight: NamedArray = eqx.field()
    bias: NamedArray | None = eqx.field()

    def _lhs_dim_spec(self, batch_index, inputs):
        # the dim spec are single letters, for things like NCHW
        lhs_dim_spec = ""
        for i, ax in enumerate(inputs.axes):
            if i == batch_index:
                lhs_dim_spec += "N"
            elif ax.name == self.In.name:
                lhs_dim_spec += "C"
            else:
                index_in_spatial = _index_of_name(self.Spatial, ax.name)
                if index_in_spatial >= 0:
                    lhs_dim_spec += self._spatial_dim_short_names[index_in_spatial]
                else:
                    # shouldn't happen
                    raise ValueError(f"Unexpected axis {ax.name}")
        return lhs_dim_spec

    @cached_property
    def _spatial_dim_short_names(self) -> str:
        banned_letters = "NCIO"
        spec = ""
        for x in self.Spatial:
            name = axis_name(x)
            assert len(name) > 0
            if name[0] not in banned_letters and name[0] not in spec:
                spec += name[0]
            else:
                for x in string.ascii_uppercase:
                    if x not in spec and x not in banned_letters:
                        spec += x
                        break
                else:
                    raise RuntimeError("Too many spatial dimensions")

        return spec

    @cached_property
    def _weight_dim_spec(self) -> str:
        # dim spec in format jax.lax.conv_general_dilated expects
        # it's typically OIWH or similar but in case anyone does something funny with the weight array:
        # "OI" + self._spatial_dim_short_names
        spec = ""
        for ax in self.weight.axes:
            # check name for grouped convolutions
            if ax.name == self.In.name:
                spec += "I"
            elif ax == self.Out:
                spec += "O"
            else:
                index_in_spatial = _index_of_name(self.Spatial, ax.name)
                if index_in_spatial >= 0:
                    spec += self._spatial_dim_short_names[index_in_spatial]
                else:
                    # shouldn't happen
                    raise ValueError(f"Unexpected axis {ax.name}")
        return spec


# Based on Equinox's Conv class
class Conv(_ConvBase):
    """General N-dimensional convolution."""

    kernel_size: tuple[int, ...] = eqx.field(static=True)
    stride: tuple[int, ...] = eqx.field(static=True)
    padding: tuple[tuple[int, int], ...] = eqx.field(static=True)
    dilation: tuple[int, ...] = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    @staticmethod
    def init(
        Spatial: AxisSelection,
        In: Axis,
        Out: Axis,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        """
        Args:
            Spatial: names of spatial dimensions
            In: Axis of input channels
            Out: Axis of output channels
            kernel_size: The size of the convolutional kernel.
            stride: The stride of the convolution. Can be a single number or a tuple
            padding: The amount of padding to apply before and after each spatial
                     dimension.
            dilation: The dilation of the convolution.
            groups: The number of groups to split the input channels into. Each
                    group is convolved separately with its own kernel.
            use_bias: Whether to add a bias after the convolution.
            key: Random key
        """
        Spatial = axis_spec_to_shape_dict(Spatial)
        if len(Spatial) == 0:
            raise ValueError("Spatial must have at least one element")

        kernel_size = _expand_and_check_shape(len(Spatial), kernel_size, "kernel_size")
        stride = _expand_and_check_shape(len(Spatial), stride, "stride")
        dilation = _expand_and_check_shape(len(Spatial), dilation, "dilation")
        padding = _convert_padding_spec(Spatial, padding)

        kernel_spec = tuple(Axis(axis_name(n), s) for n, s in zip(Spatial, kernel_size))
        in_spec = In.resize(In.size // groups)

        weight_key, bias_key = jax.random.split(key, 2)

        limit = 1 / np.sqrt(np.prod(kernel_size) * in_spec.size)

        weight = uniform(weight_key, (Out, in_spec, *kernel_spec), minval=-limit, maxval=limit)
        if use_bias:
            bias = uniform(bias_key, (Out,), minval=-limit, maxval=limit)
        else:
            bias = None

        return Conv(Spatial, In, Out, weight, bias, kernel_size, stride, padding, dilation, groups)

    @named_call
    def __call__(self, inputs, *, key: PRNGKeyArray | None = None):
        """
        Args:
            inputs (NamedArray): Input array
            key (PRNGKeyArray: Not used, compat with other modules

        Returns:
            (NamedArray): Output array, with shape similar to inputs except:
                - `Spatial` dimensions are reduced via the usual convolution formula
                - `In` is replaced with `Out`

        Notes:
            That formula is:
                `out_size = (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1`
        """
        del key

        # check input
        for ax in self.Spatial:
            if not selects_axis(inputs.axes, ax):
                raise ValueError(f"Missing spatial axis {ax} in inputs: {inputs.axes}")

        if not selects_axis(inputs.axes, self.In):
            raise ValueError(f"Missing input axis {self.In} in inputs: {inputs.axes}")

        if selects_axis(inputs.axes, self.Out):
            raise ValueError(f"Output axis {self.Out} already in inputs: {inputs.axes}")

        # this is a bit subtle, but we need to make sure that the input is in the right order
        # and has the right set of dimensions

        # Constraints:
        # * at most 1 batch dimension (we'll vmap as necessary)
        # * at most 1 channel dimension (which we enforce for this module)
        # Spatial dimensions are reduced via the usual convolution formula, so we have to drop to names

        # identify batch dims, which get special treatment
        # jax's conv_general_dilated only supports exactly one batch dimension (not 0), so we vmap over any others.
        # We could choose instead to flatten them, but then we'd definitely lose sharding.
        batch_dims = without_axes(inputs.axes, unsize_axes(self.weight.axes))
        flat_inputs, unflatten = haliax.core.flatten_all_axes_but(
            inputs, "__batch__", batch_dims, reorder_to_front=True
        )
        x = self._do_conv(flat_inputs)
        x = unflatten(x)

        if self.bias is not None:
            x = x + self.bias

        return x

    def _do_conv(self, inputs):
        # _do_conv expects there to be a single __batch__ dimension
        output_axes = _compute_output_axes(inputs, "__batch__", self.In, self.Out)

        batch_index = _index_of_name(inputs.axes, "__batch__")

        lhs_dim_spec = self._lhs_dim_spec(batch_index, inputs)
        rhs_dim_spec = self._weight_dim_spec
        output_dim_spec = lhs_dim_spec
        x = jax.lax.conv_general_dilated(
            lhs=inputs.array,
            rhs=self.weight.array,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
            dimension_numbers=(lhs_dim_spec, rhs_dim_spec, output_dim_spec),
        )

        return named(x, output_axes)


class ConvTranspose(_ConvBase):
    """
    General N-dimensional transposed convolution.

    Based on Equinox's ConvTranspose class
    """

    kernel_size: tuple[int, ...] = eqx.field(static=True)
    stride: tuple[int, ...] = eqx.field(static=True)
    padding: tuple[tuple[int, int], ...] = eqx.field(static=True)
    output_padding: tuple[int, ...] = eqx.field(static=True)
    dilation: tuple[int, ...] = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    @staticmethod
    def init(
        Spatial: AxisSelection,
        In: Axis,
        Out: Axis,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        """

        Args:
            Spatial: Spatial dimensions
            In:  Input channels
            Out:  Output channels
            kernel_size: Kernel size, can be a single number or a tuple
            stride: Stride, can be a single number or a tuple
            padding: Padding, can be a single number or a tuple
            output_padding: Output padding, can be a single number or a tuple
            dilation: Dilation, can be a single number or a tuple
            groups: Number of groups to split the input channels into
            use_bias:  Whether to add on a bias after the convolution
            key:  Random key

        Notes:
            Output padding is the amount of extra padding to add to the output.
            Because the output size is not uniquely determined by the input size for transposed convolutions.
        """
        k_w, k_b = jax.random.split(key, 2)

        Spatial = axis_spec_to_shape_dict(Spatial)

        kernel_size = _expand_and_check_shape(len(Spatial), kernel_size, "kernel_size")
        stride = _expand_and_check_shape(len(Spatial), stride, "stride")
        dilation = _expand_and_check_shape(len(Spatial), dilation, "dilation")
        padding = _convert_padding_spec(Spatial, padding)
        output_padding = _expand_and_check_shape(len(Spatial), output_padding, "output_padding")

        kernel_spec = tuple(Axis(axis_name(n), s) for n, s in zip(Spatial, kernel_size))
        in_spec = In.resize(In.size // groups)

        lim = 1 / np.sqrt(np.prod(kernel_size) * in_spec.size)
        weight = uniform(k_w, (Out, in_spec, *kernel_spec), minval=-lim, maxval=lim)
        if use_bias:
            bias = uniform(k_b, (Out,), minval=-lim, maxval=lim)
        else:
            bias = None

        return ConvTranspose(
            Spatial,
            In,
            Out,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )

    @named_call
    def __call__(self, inputs, *, key: PRNGKeyArray | None = None):
        """
        Args:
            inputs (NamedArray): Input array
            key (PRNGKeyArray: Not used, compat with other modules

        Returns:
            (NamedArray): Output array, with shape similar to inputs except:
                - `Spatial` dimensions are increased via the usual (de)convolution formula
                - `In` is replaced with `Out`

        Notes:
            That formula is:
                `out_size = (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
        """
        del key

        batch_dims = without_axes(inputs.axes, unsize_axes(self.weight.axes))
        flat_inputs, unflatten = haliax.core.flatten_all_axes_but(
            inputs, "__batch__", batch_dims, reorder_to_front=True
        )
        x = self._do_conv(flat_inputs)
        x = unflatten(x)

        if self.bias is not None:
            x = x + self.bias

        output_axes = _compute_output_axes(inputs, batch_dims, self.In, self.Out)

        return x.rearrange(output_axes)

    def _do_conv(self, inputs):
        # cribbed from Equinox's ConvTranspose class
        padding = tuple(
            (d * (k - 1) - p0, d * (k - 1) - p1 + o)
            for k, (p0, p1), o, d in zip(self.kernel_size, self.padding, self.output_padding, self.dilation)
        )

        output_axes = _compute_output_axes(inputs, "__batch__", self.In, self.Out)
        batch_index = _index_of_name(inputs.axes, "__batch__")

        lhs_dim_spec = self._lhs_dim_spec(batch_index, inputs)
        rhs_dim_spec = self._weight_dim_spec
        output_dim_spec = lhs_dim_spec

        x = jax.lax.conv_general_dilated(
            lhs=inputs.array,
            rhs=self.weight.array,
            window_strides=(1,) * len(self.Spatial),
            padding=padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
            dimension_numbers=(lhs_dim_spec, rhs_dim_spec, output_dim_spec),
        )

        return named(x, output_axes)


def _expand_and_check_shape(expected_len: int, spec: T | Sequence[T], name: str) -> tuple[T, ...]:
    spec = ensure_tuple(spec)
    if len(spec) == 1:
        spec = spec * expected_len
    if len(spec) != expected_len:
        raise ValueError(f"Expected {expected_len} elements for {name}, got {len(spec)}")

    return spec


def _convert_padding_spec(Spatial, padding):
    if isinstance(padding, int):
        padding = ((padding, padding),) * len(Spatial)
    elif isinstance(padding, tuple):
        padding = _expand_and_check_shape(len(Spatial), padding, "padding")
        padding_spec: list = []
        for p in padding:
            if isinstance(p, int):
                padding_spec.append((p, p))
            elif isinstance(p, tuple):
                padding_spec.append(p)
            else:
                raise ValueError(f"Invalid padding spec: {padding}")

        padding = tuple(padding_spec)
    else:
        raise ValueError(f"Invalid padding spec: {padding}")

    return padding


def _index_of_name(names: Sequence[str | Axis], name) -> int:
    for i, x in enumerate(names):
        if isinstance(x, Axis):
            x = axis_name(x)
        if x == name:
            return i
    return -1


def _compute_output_axes(inputs, batch_dims, In, Out):
    """
    Does two things:
    1. Replace In with Out
    2. turn spatial dims (non-batch, non-In, non-Out) into raw names b/c they change size in convolutions
    """
    unchanging_dims = [Out, *batch_dims]
    return [ax.name if ax not in unchanging_dims else ax for ax in replace_axis(inputs.axes, In, Out)]
