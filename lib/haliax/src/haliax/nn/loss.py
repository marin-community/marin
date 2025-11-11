# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import typing
import warnings

from jax import numpy as jnp

import haliax as hax
from haliax.axis import AxisSelection, AxisSelector
from haliax.core import NamedArray
from haliax.util import UNSPECIFIED, Unspecified
from haliax.wrap import ReductionFunction


@typing.overload
def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: None = None,
) -> jnp.ndarray | NamedArray: ...


@typing.overload
def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: AxisSelection = ...,
) -> NamedArray: ...


def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: AxisSelection | None = None,
) -> jnp.ndarray | NamedArray:
    loss, _ = cross_entropy_loss_and_log_normalizers(logits, Label, targets)

    # if target_y isn't some kind of floating point, something is wrong, so warn
    if not jnp.issubdtype(targets.dtype, jnp.floating):
        warnings.warn(
            f"target_y has dtype {targets.dtype}, which is not a floating point type. This is probably a mistake."
        )

    loss = maybe_reduce_loss(loss, reduction, reduction_axis, where)

    return loss


@typing.overload
def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: None = None,
) -> jnp.ndarray | NamedArray: ...


@typing.overload
def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: AxisSelection = ...,
) -> NamedArray: ...


def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    where: NamedArray | None = None,
    reduction_axis: AxisSelection | None = None,
) -> jnp.ndarray | NamedArray:
    log_p = hax.nn.log_sigmoid(logits)
    log_not_p = hax.nn.log_sigmoid(-logits)  # == log(1-sigmoid(x))
    targets = targets.astype(logits.dtype)
    loss = -targets * log_p - (1.0 - targets) * log_not_p

    loss = maybe_reduce_loss(loss, reduction, reduction_axis, where)
    return loss


def reduce_loss(
    arr,
    reduction: ReductionFunction | None | Unspecified = UNSPECIFIED,
    reduction_axis: AxisSelection | None = None,
    where: NamedArray | None = None,
):
    """
    Reduce a loss array according to the given reduction and reduction axis.
    If reduction is None, the loss is not reduced.
    If reduction is UNSPECIFIED, the default reduction is used (mean).
    If reduction_axis is None (default), the loss is reduced over all axes.
    """
    return maybe_reduce_loss(arr, reduction, reduction_axis, where)


def maybe_reduce_loss(
    arr,
    reduction: ReductionFunction | None | Unspecified,
    reduction_axis: AxisSelection | None,
    where: NamedArray | None,
):
    if reduction is not None and reduction_axis != ():
        if reduction is UNSPECIFIED:
            reduction = hax.mean
        arr = reduction(arr, where=where, axis=reduction_axis)
    elif where is not None:
        arr = hax.where(where, arr, 0)
    return arr


def cross_entropy_loss_and_log_normalizers(
    pred_y: NamedArray,
    Label: AxisSelector,
    target_y: NamedArray,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute the cross entropy loss and log normalizers for a batch of predictions and targets.

    :param pred_y: a NamedArray with the Label axis (and possibly others for e.g. batch and seq) containing the logits
    :param Label: the Label axis
    :param target_y: a NamedArray with the Label axis (and possibly others) containing the targets

    :return: tuple of two named arrays, with "per position" losses and log normalizers
    """
    log_normalizers = hax.nn.logsumexp(pred_y, Label)
    neg_log_normalized = log_normalizers - pred_y

    loss = hax.dot(target_y, neg_log_normalized, axis=Label)

    return loss, log_normalizers
