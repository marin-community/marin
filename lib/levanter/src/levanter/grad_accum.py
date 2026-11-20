# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Optional, ParamSpec, TypeVar, cast

import equinox as eqx
import haliax as hax
import haliax.quantization as hq
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.partitioning import ResourceAxis
from haliax.util import is_named_array
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from levanter.metrics import Metric, ReductionType
from levanter.metrics import fold as fold_metric
from levanter.pipeline import reshape_array_into_microbatches
from levanter.utils.jax_utils import zeros_like_tree

Args = ParamSpec("Args")
R = TypeVar("R")


ArrayGradientResult = tuple[tuple[jax.Array, dict[str, Metric]], PyTree[jax.Array]]


def microbatched_arrays(
    fn: Callable[..., ArrayGradientResult], microbatch_size: int
) -> Callable[..., ArrayGradientResult]:
    """Accumulate gradients of a raw-array batch without changing its placement.

    ``fn(batch, **kwargs)`` returns ``((loss, metrics), gradients)``. Every
    nonscalar array in ``batch`` leads with the same example dimension; scalar
    and Python leaves, and keyword arguments other than ``key``, are shared.
    Equal-size microbatch losses and ordinary gradients are averaged. Custom
    gradient state uses its own reduction without averaging. Metrics retain
    their own reductions and the RNG key is split once per microbatch.
    """
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    @functools.wraps(fn)
    def wrapped(batch, **kwargs):
        leaves = [leaf for leaf in jax.tree.leaves(batch) if eqx.is_array(leaf) and leaf.ndim > 0]
        if not leaves:
            raise ValueError("A raw-array batch must contain a nonscalar array")
        batch_size = leaves[0].shape[0]
        if any(leaf.shape[0] != batch_size for leaf in leaves):
            raise ValueError("Every nonscalar batch array must have the same leading example dimension")
        if microbatch_size >= batch_size:
            return fn(batch, **kwargs)
        if batch_size % microbatch_size:
            raise ValueError("Batch size must be divisible by microbatch_size")
        num_microbatches = batch_size // microbatch_size
        mapped, shared = eqx.partition(batch, lambda leaf: eqx.is_array(leaf) and leaf.ndim > 0)
        mapped = jax.tree.map(lambda leaf: reshape_array_into_microbatches(leaf, num_microbatches), mapped)
        keys = kwargs.get("key")
        shared_kwargs = {name: value for name, value in kwargs.items() if name != "key"}
        if keys is not None:
            keys = jax.random.split(keys, num_microbatches)

        def evaluate(microbatch, key):
            arguments = {**shared_kwargs, "key": key} if "key" in kwargs else shared_kwargs
            return fn(eqx.combine(microbatch, shared), **arguments)

        shape = cast(
            ArrayGradientResult,
            eqx.filter_eval_shape(
                evaluate, jax.tree.map(lambda leaf: leaf[0], mapped), None if keys is None else keys[0]
            ),
        )

        def zero(leaf):
            if isinstance(leaf, jax.ShapeDtypeStruct):
                return jnp.zeros(leaf.shape, leaf.dtype, out_sharding=leaf.sharding)
            return leaf

        def identity(metric):
            value = jnp.zeros_like(zero(metric._value))
            if metric.reduction is ReductionType.MAX:
                bound = (
                    False
                    if jnp.issubdtype(value.dtype, jnp.bool_)
                    else (-jnp.inf if jnp.issubdtype(value.dtype, jnp.inexact) else jnp.iinfo(value.dtype).min)
                )
                value = jnp.full_like(value, bound)
            elif metric.reduction is ReductionType.MIN:
                bound = (
                    True
                    if jnp.issubdtype(value.dtype, jnp.bool_)
                    else (jnp.inf if jnp.issubdtype(value.dtype, jnp.inexact) else jnp.iinfo(value.dtype).max)
                )
                value = jnp.full_like(value, bound)
            return Metric(value, jnp.zeros_like(zero(metric._count)), metric.reduction)

        (loss_shape, metric_shapes), gradient_shapes = shape
        initial = (
            (zero(loss_shape), jax.tree.map(identity, metric_shapes, is_leaf=lambda leaf: isinstance(leaf, Metric))),
            jax.tree.map(zero, gradient_shapes),
        )

        def accumulate(accumulator, inputs):
            (loss, metrics), gradients = evaluate(*inputs)
            (acc_loss, acc_metrics), acc_gradients = accumulator
            metrics = jax.tree.map(fold_metric, acc_metrics, metrics, is_leaf=lambda leaf: isinstance(leaf, Metric))
            gradients = hq.accumulate_gradients(acc_gradients, gradients)
            return ((acc_loss + loss, metrics), gradients), None

        ((loss, metrics), gradients), _ = jax.lax.scan(accumulate, initial, (mapped, keys))
        gradients = jax.tree.map(
            lambda leaf: leaf if isinstance(leaf, hq.CustomGradientAccumulation) else leaf / num_microbatches,
            gradients,
            is_leaf=lambda leaf: isinstance(leaf, hq.CustomGradientAccumulation),
        )
        return (loss / num_microbatches, metrics), gradients

    return wrapped


# TODO: should we use a custom_jvp on microbatched?


# cf https://github.com/google-research/t5x/blob/main/t5x/trainer.py#L617
def microbatched(
    fn: Callable[Args, R],
    Batch: Axis,
    microbatch_size: int,
    accum_axis_mapping,
    compute_axis_mapping,
    patch_in_rng_key: Optional[str] = "key",
    accum_dtype: Optional[jnp.dtype] = None,
) -> Callable[Args, R]:
    """
    Wraps a gradient-and-loss function so it runs over microbatches and accumulates the results.

    ``fn`` must return ``((loss, metrics), grads)`` (i.e. the output of
    ``eqx.filter_value_and_grad(loss_fn, has_aux=True)`` where ``loss_fn`` returns
    ``(loss, metrics)``). The batch axis is split into microbatches; the loss and ordinary gradients
    are summed then averaged, custom gradient state uses its own reduction without averaging, and
    ``metrics`` are folded with their own reductions.
    Named arrays with ``Batch`` and plain arrays whose leading dimension is ``Batch.size`` are split;
    other inputs are reused for every microbatch.

    Args:
        fn: the value-and-grad function to wrap.
        Batch: the batch axis.
        microbatch_size: how many examples to process at once (must divide the batch size).
        accum_axis_mapping:  the axis mapping for the accumulator (typically this is the same as the params)
        compute_axis_mapping:  the axis mapping for the computation (typically this is the same as the inputs)
        patch_in_rng_key: if provided, this kwarg will be split, 1 for each accum step. It won't work if the
            PRNGKey is passed in as a positional argument.
        accum_dtype: the dtype of floating point values in the accumulator. If None, this will be inferred from the return type of `fn`.

    Returns:
        a function that splits the batch into microbatches, calls the function on each microbatch, and
        accumulates the results.
    """
    batch_size = Batch.size
    data_axis_size = hax.partitioning.physical_axis_size(Batch, compute_axis_mapping)
    if data_axis_size is None:
        raise ValueError(f"{Batch} axis must be sharded")
    physical_axis_name = hax.partitioning.physical_axis_name(Batch, compute_axis_mapping)
    assert physical_axis_name is not None

    if microbatch_size <= 0:
        raise ValueError(f"Bad value for {microbatch_size=}")

    if microbatch_size >= batch_size:
        return fn

    num_micro_steps = batch_size // microbatch_size

    if num_micro_steps <= 0:
        raise ValueError(
            f"microbatch_size ({microbatch_size}) must be less than batch size ({batch_size}) or divisible into it"
        )

    Microbatch = Batch.resize(microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)
    if num_micro_steps * microbatch_size != batch_size:
        raise ValueError(
            "Batch size must be an integer multiple of microbatch_size. "
            f"Got batch size {batch_size} and microbatch_size {microbatch_size}."
        )

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):

        # first, determine the shape and make accumulator arrays
        r_shape = eqx.filter_eval_shape(fn, *args, **kwargs)
        acc = zeros_like_tree(r_shape, accum_axis_mapping, accum_dtype)

        # then, reshape the inputs from (Batch, ...) to (AccumStep, Microbatch, ...)

        # Special handling for PRNGKey: it comes in as a single key, but we need to split it for each microbatch
        key = kwargs.get(patch_in_rng_key, None)
        if key is not None:
            key = jax.random.split(key, num_micro_steps)
            kwargs = kwargs.copy()
            kwargs.pop(patch_in_rng_key)

        def is_batched(value):
            if isinstance(value, hax.NamedArray):
                return value.has_axis(Batch.name)
            return isinstance(value, jnp.ndarray) and value.ndim > 0 and value.shape[0] == Batch.size

        def is_input_leaf(value):
            return is_named_array(value) or isinstance(value, hq.CustomGradientAccumulation)

        batched_inputs, unbatched_inputs = eqx.partition((args, kwargs), is_batched, is_leaf=is_input_leaf)
        batched_inputs = _reshape_for_microbatch(Batch, Microbatch, AccumStep, batched_inputs, compute_axis_mapping)

        def loop(acc, microbatch_and_key):
            microbatch_inputs, key = microbatch_and_key
            microbatch, microbatch_kwargs = eqx.combine(microbatch_inputs, unbatched_inputs, is_leaf=is_input_leaf)
            with jax.named_scope("compute"):
                microbatch_kwargs = microbatch_kwargs.copy()
                if key is not None:
                    microbatch_kwargs[patch_in_rng_key] = key
                this_r = cast(tuple[tuple[Any, Any], Any], fn(*microbatch, **microbatch_kwargs))

            with jax.named_scope("accum"):
                # Unpack structure: ((loss, metrics_dict), grads)
                (this_loss, this_metrics), this_grads = this_r
                (acc_loss, acc_metrics), acc_grads = acc

                # Accumulate loss (scalar)
                new_loss = acc_loss + this_loss

                new_metrics = jax.tree_util.tree_map(
                    lambda a, b: fold_metric(a, b),
                    acc_metrics,
                    this_metrics,
                    is_leaf=lambda x: isinstance(x, Metric),
                )

                new_grads = hq.accumulate_gradients(acc_grads, this_grads)

                # Repack and shard
                acc = ((new_loss, new_metrics), new_grads)
                acc = hax.shard_with_axis_mapping(acc, accum_axis_mapping)

            return acc

        with jax.named_scope("microbatched"):
            acc = hax.fold(loop, AccumStep)(acc, (batched_inputs, key))

            # Average loss and grads over microbatches. Metrics handle their own reduction internally.
            (loss, metrics), grads = acc
            loss = loss / num_micro_steps
            grads = jax.tree_util.tree_map(
                lambda x: x if isinstance(x, hq.CustomGradientAccumulation) else x / num_micro_steps,
                grads,
                is_leaf=lambda x: isinstance(x, hq.CustomGradientAccumulation),
            )
            acc = ((loss, metrics), grads)

        return cast(R, acc)

    return wrapped_fn


def _reshape_for_microbatch(Batch: Axis, Microbatch: Axis, AccumStep: Axis, inputs, axis_mapping):
    def _reshape(x):
        if isinstance(x, hax.NamedArray):
            if not x.has_axis(Batch.name):
                return x
            with hax.axis_mapping(axis_mapping):
                x = x.unflatten_axis(Batch, (AccumStep, Microbatch))
            return hax.shard(x, axis_mapping)
        elif isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] == Batch.size:
            x = reshape_array_into_microbatches(x, AccumStep.size)
            return with_sharding_constraint(x, PartitionSpec(None, ResourceAxis.DATA, *(None,) * (len(x.shape) - 2)))
        else:
            # assert jnp.isscalar(x)
            return x

    return jax.tree_util.tree_map(_reshape, inputs, is_leaf=is_named_array)
