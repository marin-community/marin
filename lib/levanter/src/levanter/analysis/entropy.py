# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Functions for computing and visualizing token-level entropy."""

import logging
from typing import Callable, TypeVar

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jaxtyping import PyTree

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.data.loader import DataLoader
from levanter.tracker.histogram import SummaryStats

B = TypeVar("B")

logger = logging.getLogger(__name__)

OnDeviceFn = Callable[[Callable, PyTree, B, hax.AxisSelector], jnp.ndarray]


@named_call
def entropy_from_logits(logits: hax.NamedArray, axis: hax.AxisSelector) -> hax.NamedArray:
    """
    Computes entropy over the given axis in a numerically stable way using raw logits.
    """
    log_z = hnn.logsumexp(logits, axis=axis)
    log_probs = logits - log_z
    probs = hax.exp(log_probs)
    return -hax.sum(probs * log_probs, axis=axis)


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

    # hax.top_k(logits, axis, 2, "top") would materialize a sorted copy of the whole axis,
    # so instead we take two argmax passes: find the max, mask out its position, take the max again.
    Axis = logits.resolve_axis(axis)
    argmax = hax.argmax(logits, axis=Axis)
    top1 = hax.take(logits, Axis, argmax)

    # Mask the winning position (the one whose index equals argmax) with -inf so the second pass
    # finds the runner-up. argmax lacks the reduced axis, so broadcast it back over Axis to compare.
    is_top1 = hax.arange(Axis) == argmax.broadcast_axis(Axis)
    argmax2 = hax.argmax(hax.where(is_top1, -jnp.inf, logits), axis=Axis)
    top2 = hax.take(logits, Axis, argmax2)
    return top1 - top2


# Top level to avoid recompilation
@eqx.filter_jit
def _compute_entropy_on_device(logit_fn, model, batch: B, Vocab) -> jnp.ndarray:
    with jax.named_scope("logits"):
        logits = logit_fn(model, batch)
    entropies = entropy_from_logits(logits, axis=Vocab)
    return entropies.flatten("token").array


@eqx.filter_jit
def _compute_top2_gap_on_device(logit_fn, model, batch: B, Vocab) -> jnp.ndarray:
    with jax.named_scope("logits"):
        logits = logit_fn(model, batch)
    gaps = top2_gap_from_logits(logits, axis=Vocab)
    return gaps.flatten("token").array


def _collect_per_token_values(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
    test_data,
    *,
    on_device_fn: OnDeviceFn,
    max_tokens: int,
) -> jnp.ndarray:
    """Run ``on_device_fn`` over batches until ``max_tokens`` and concatenate.

    Raises:
        ValueError: if no tokens were produced.
    """
    chunks: list[jnp.ndarray] = []
    total_tokens = 0
    for batch in test_data:
        values = on_device_fn(logit_fn, model, batch, Vocab)
        chunks.append(values)
        total_tokens += values.size
        if total_tokens >= max_tokens:
            break

    if not chunks:
        raise ValueError("No tokens processed")

    flat = jnp.concatenate(chunks)
    if not flat.size:
        raise ValueError("No tokens processed")
    return flat


def compute_entropy_histogram(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
    test_data,
    max_tokens: int = 1024 * 1024,
    num_bins: int = 64,
) -> SummaryStats:
    """
    Compute entropy histograms for a given model and dataset.

    Returns:
        SummaryStats: A SummaryStats object containing the entropy values.
    """
    entropies = _collect_per_token_values(
        model,
        Vocab,
        logit_fn,
        test_data,
        on_device_fn=_compute_entropy_on_device,
        max_tokens=max_tokens,
    )
    return SummaryStats.from_array(entropies, num_bins=num_bins)


def compute_top2_gap_histogram(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
    test_data,
    max_tokens: int = 1024 * 1024,
    num_bins: int = 64,
) -> SummaryStats:
    gaps = _collect_per_token_values(
        model,
        Vocab,
        logit_fn,
        test_data,
        on_device_fn=_compute_top2_gap_on_device,
        max_tokens=max_tokens,
    )
    return SummaryStats.from_array(gaps, num_bins=num_bins)


def _make_histogram_callback(
    histogram_fn: Callable[..., SummaryStats],
    metric_name: str,
    *,
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
    num_tokens: int,
) -> Callable[[StepInfo], None]:
    if prefix is None:
        prefix = "analysis"

    def callback(step: StepInfo) -> None:
        data_loader = DataLoader(test_data, batch_size=batch_size, pad_final_batch=False, axis_resources=mapping)
        try:
            hist = histogram_fn(
                model=step.eval_model,
                Vocab=Vocab,
                logit_fn=logit_fn,
                test_data=data_loader,
                max_tokens=num_tokens,
            )
        except ValueError as e:
            if "No tokens processed" in str(e):
                logger.warning(f"{prefix} is too small to compute {metric_name} with batch size {batch_size}")
                return
            raise

        levanter.tracker.log({f"{prefix}/{metric_name}": hist}, step=step.step)

    return callback


def cb_compute_entropies(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
    num_tokens: int = 10 * 1024 * 1024,
) -> Callable[[StepInfo], None]:
    """
    Callback to compute entropy distribution and log it to the tracker.

    Args:
        logit_fn: Function that takes (model, batch) and returns logits
        Vocab (hax.AxisSelector): The vocabulary to use.
        test_data: The test data to use.
        prefix (str | None): The key to log to the tracker. If None, "analysis" is used.
        num_tokens: The number of tokens to use.
        batch_size: The batch size to use.
        mapping: The resource mapping

    Returns:
        function: A function that takes a step info and computes and visualizes the log probabilities.
    """
    return _make_histogram_callback(
        compute_entropy_histogram,
        "entropy",
        logit_fn=logit_fn,
        Vocab=Vocab,
        test_data=test_data,
        prefix=prefix,
        batch_size=batch_size,
        mapping=mapping,
        num_tokens=num_tokens,
    )


def cb_compute_top2_gap(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    prefix: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
    num_tokens: int = 10 * 1024 * 1024,
) -> Callable[[StepInfo], None]:
    return _make_histogram_callback(
        compute_top2_gap_histogram,
        "top2_gap",
        logit_fn=logit_fn,
        Vocab=Vocab,
        test_data=test_data,
        prefix=prefix,
        batch_size=batch_size,
        mapping=mapping,
        num_tokens=num_tokens,
    )
