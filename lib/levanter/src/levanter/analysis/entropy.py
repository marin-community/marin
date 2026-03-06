# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Functions for computing and visualizing token-level entropy."""

import logging
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

import haliax as hax
import haliax.nn as hnn
from jax import named_call

import levanter.tracker
from levanter.callbacks import StepInfo

from ..data import DataLoader
from ..tracker.histogram import Histogram


B = TypeVar("B")

logger = logging.getLogger(__name__)


@named_call
def entropy_from_logits(logits: hax.NamedArray, axis: hax.AxisSelector) -> hax.NamedArray:
    """
    Computes entropy over the given axis in a numerically stable way using raw logits.
    """
    log_z = hnn.logsumexp(logits, axis=axis)
    probs = hax.exp(logits - log_z)
    entropy = log_z - hax.sum(probs * logits, axis=axis)
    return entropy


@named_call
def top2_gap_from_logits(logits: hax.NamedArray, axis: hax.AxisSelector) -> hax.NamedArray:
    """
    Computes the difference between the top 2 logits along the specified axis.

    Args:
        logits: A NamedArray of logits.
        axis: The axis over which to compute the top-2 gap.

    Returns:
        A NamedArray with the same shape as logits minus `axis`, containing the top-2 gaps.
    """

    # this uses a ton of memory for no particularly good reason. So we do it in two passes:
    # sorted_logits = hax.top_k(logits, axis, 2, "top")[0]
    # top1 = sorted_logits["top", 0]
    # top2 = sorted_logits["top", 1]

    argmax = hax.argmax(logits, axis=axis)
    top1 = hax.take(logits, axis, argmax)
    argmax2 = hax.argmax(hax.where(argmax, -jnp.inf, logits), axis=axis)
    top2 = hax.take(logits, axis, argmax2)
    return top1 - top2


def compute_entropy_histogram(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray | jax.Array],
    test_data,
    max_tokens: int = 1024 * 1024,
    num_bins: int = 64,
) -> Histogram:
    """
    Compute entropy histograms for a given model and dataset.

    Returns:
        Histogram: A Histogram object containing the entropy values.
    """

    entropies_list: list[jnp.ndarray] = []
    total_tokens = 0

    for batch in test_data:
        entropy_vals = _compute_entropy_on_device(logit_fn, model, batch, Vocab)
        entropies_list.append(entropy_vals)
        total_tokens += entropy_vals.size

        if total_tokens >= max_tokens:
            break

    entropies = jnp.concatenate(entropies_list)

    if not entropies.size:
        raise ValueError("No tokens processed")

    return Histogram.from_array(entropies, num_bins=num_bins)


# Top level to avoid recompilation
@eqx.filter_jit
def _compute_entropy_on_device(logit_fn, model, batch: B, Vocab) -> jnp.ndarray:
    with jax.named_scope("logits"):
        logits = logit_fn(model, batch)
    if isinstance(logits, hax.NamedArray):
        entropies = entropy_from_logits(logits, axis=Vocab)
        return entropies.flatten("token").array
    entropies = _entropy_from_logits_array(logits, axis=-1)
    return entropies.reshape(-1)


def cb_compute_entropies(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    batch_axis_resource,
    batch_axis_name: str = "batch",
    num_tokens: int = 10 * 1024 * 1024,
):
    """
    Callback to compute entropy distribution and log it to the tracker.

    Args:
        logit_fn: Function that takes (model, batch) and returns logits
        Vocab (hax.AxisSelector): The vocabulary to use.
        test_data: The test data to use.
        prefix (str | None): The key to log to the tracker. If None, "entropy" is used.
        num_tokens: The number of tokens to use.
        batch_size: The batch size to use.
        batch_axis_resource: Resource for sharding the batch axis.
        batch_axis_name: Name of the batch axis in the dataset examples.

    Returns:
        function: A function that takes a step info and computes and visualizes the log probabilities.
    """
    if prefix is None:
        prefix = "analysis"

    def compute_entropy(step: StepInfo):
        loader_axis_resources = None
        if batch_axis_resource is not None:
            loader_axis_resources = {batch_axis_name: batch_axis_resource}
        data_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            batch_axis_name=batch_axis_name,
            pad_final_batch=False,
            axis_resources=loader_axis_resources,
        )
        model = step.eval_model

        try:
            entropy_hist = compute_entropy_histogram(
                model=model,
                Vocab=Vocab,
                logit_fn=logit_fn,
                test_data=data_loader,
                max_tokens=num_tokens,
            )
        except ValueError as e:
            if "No tokens processed" in str(e):
                logger.warning(f"{prefix} is too small to compute entropy with batch size {batch_size}")
                return
            logger.exception(f"Error computing entropy for {prefix}")
            raise

        levanter.tracker.log({f"{prefix}/entropy": entropy_hist}, step=step.step)

    return compute_entropy


def compute_top2_gap_histogram(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray | jax.Array],
    test_data,
    max_tokens: int = 1024 * 1024,
    num_bins: int = 64,
) -> Histogram:
    gaps = []
    total_tokens = 0
    for batch in test_data:
        gaps.append(_compute_top2_gap_on_device(logit_fn, model, batch, Vocab))
        total_tokens += gaps[-1].size
        if total_tokens >= max_tokens:
            break

    gaps_array = jnp.concatenate(gaps)
    if not gaps_array.size:
        raise ValueError("No tokens processed")
    return Histogram.from_array(gaps_array, num_bins=num_bins)


@eqx.filter_jit
def _compute_top2_gap_on_device(logit_fn, model, batch: B, Vocab) -> jnp.ndarray:
    with jax.named_scope("logits"):
        logits = logit_fn(model, batch)
    if isinstance(logits, hax.NamedArray):
        gaps = top2_gap_from_logits(logits, axis=Vocab)
        return gaps.flatten("token").array
    gaps = _top2_gap_from_logits_array(logits, axis=-1)
    return gaps.reshape(-1)


def _entropy_from_logits_array(logits: jax.Array, axis: int = -1) -> jax.Array:
    log_z = jax.nn.logsumexp(logits, axis=axis)
    probs = jax.nn.softmax(logits, axis=axis)
    return log_z - jnp.sum(probs * logits, axis=axis)


def _top2_gap_from_logits_array(logits: jax.Array, axis: int = -1) -> jax.Array:
    moved = jnp.moveaxis(logits, axis, -1)
    top2_vals, _ = jax.lax.top_k(moved, 2)
    return top2_vals[..., 0] - top2_vals[..., 1]


def cb_compute_top2_gap(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    batch_axis_resource,
    batch_axis_name: str = "batch",
    num_tokens: int = 10 * 1024 * 1024,
):
    if prefix is None:
        prefix = "analysis"

    def compute_top2_gap(step: StepInfo):
        loader_axis_resources = None
        if batch_axis_resource is not None:
            loader_axis_resources = {batch_axis_name: batch_axis_resource}
        data_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            batch_axis_name=batch_axis_name,
            pad_final_batch=False,
            axis_resources=loader_axis_resources,
        )
        model = step.eval_model
        try:
            top2_gap_hist = compute_top2_gap_histogram(
                model=model,
                Vocab=Vocab,
                logit_fn=logit_fn,
                test_data=data_loader,
                max_tokens=num_tokens,
            )
            levanter.tracker.log({f"{prefix}/top2_gap": top2_gap_hist}, step=step.step)
            return top2_gap_hist
        except ValueError as e:
            if "No tokens processed" in str(e):
                logger.warning(f"{prefix} is too small to compute top2_gap with batch size {batch_size}")
                return
            logger.exception(f"Error computing top2_gap for {prefix}")
            raise

    return compute_top2_gap
