# Copyright 2025 The Levanter Authors
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
from haliax.jax_utils import named_call

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
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
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
    entropies = entropy_from_logits(logits, axis=Vocab)
    return entropies.flatten("token").array


def cb_compute_entropies(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
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
        mapping: The resource mapping

    Returns:
        function: A function that takes a step info and computes and visualizes the log probabilities.
    """
    if prefix is None:
        prefix = "analysis"

    def compute_entropy(step: StepInfo):
        data_loader = DataLoader(test_data, batch_size=batch_size, pad_final_batch=False, axis_resources=mapping)
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
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
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
    gaps = top2_gap_from_logits(logits, axis=Vocab)
    return gaps.flatten("token").array


def cb_compute_top2_gap(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
    num_tokens: int = 10 * 1024 * 1024,
):
    if prefix is None:
        prefix = "analysis"

    def compute_top2_gap(step: StepInfo):
        data_loader = DataLoader(test_data, batch_size=batch_size, pad_final_batch=False, axis_resources=mapping)
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
