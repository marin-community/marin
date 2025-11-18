# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import inspect
from functools import wraps

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax.tree_util as htu

from ._src.scan import (
    UnnamedAxisSpec,
    _infer_axis_size_from_tree,
    _is_passive_array,
    _pacify_named_arrays,
    _PassiveNamedArray,
    _prepend_named_batch_axis,
    _zero_if_array_else_none,
    fold,
    map,
    scan,
)
from .axis import Axis, AxisSelection, AxisSelector, axis_spec_to_shape_dict, axis_spec_to_tuple, selects_axis
from .core import NamedArray
from .jax_utils import Static, broadcast_prefix, is_jax_array_like
from .partitioning import physical_axis_name
from .util import is_named_array


def vmap(
    fn,
    axis: AxisSelection,
    *,
    default: PyTree[UnnamedAxisSpec] = _zero_if_array_else_none,
    args: PyTree[UnnamedAxisSpec] = (),
    kwargs: PyTree[UnnamedAxisSpec] = None,
):
    """
    [haliax.NamedArray][]-aware version of [jax.vmap][]. Normal arrays are mapped according to the specs as in
     [equinox.filter_vmap][]

    Because of NamedArrays, vmap is typically less useful than in vanilla JAX, but it is sometimes
    useful for initializing modules that will be scanned over. See [haliax.nn.Stacked][] for an example.

    Args:
        fn (Callable): function to vmap over
        axis (Axis or Sequence[Axis]): axis or axes to vmap over. If a sequence is
            provided, the function will be vmapped over each axis in turn,
            from innermost to outermost.
        default: how to handle (unnamed) arrays by default. Should be either an integer or None, or a callable that takes a PyTree leaf
            and returns an integer or None, or a PyTree prefix of the same. If an integer, the array will be mapped over that axis. If None, the array will not be mapped over.
        args: optional per-argument overrides for how to handle arrays. Should be a PyTree prefix of the same type as default.
        kwargs: optional per-keyword-argument overrides for how to handle arrays. Should be a PyTree prefix of the same type as default.
    """

    if kwargs is None:
        kwargs = {}

    axes = axis_spec_to_shape_dict(axis)
    if len(axes) > 1:
        mapped = fn
        for ax in reversed(axes):
            size = axes.get(ax, None)
            if size is not None:
                ax = Axis(ax, size)  # type: ignore
            mapped = vmap(mapped, ax, default=default, args=args, kwargs=kwargs)
        return mapped
    elif len(axes) == 1:  # type: ignore
        axis = axis_spec_to_tuple(axis)[0]
    else:
        return fn

    signature = inspect.signature(fn)

    # this mirrors equinox's filter_vmap, but it's not really documented there so:
    # we use inspect.signature to align args/kwargs specified in vmap to what actually gets passed in
    # axis_spec_bound_sig's job is to hold that mapping
    signature_default = signature.replace(
        parameters=[
            (
                p
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                else p.replace(default=default)
            )
            for p in signature.parameters.values()
        ]
    )
    axis_spec_bound_sig = signature_default.bind_partial(*args, **kwargs)
    axis_spec_bound_sig.apply_defaults()
    del args, kwargs

    def _index_of_batch_axis(array, default):
        if isinstance(array, NamedArray):
            return array.axis_indices(axis)
        elif callable(default):
            return default(array)
        else:
            return default

    # TODO: tests to exercise this more
    @wraps(fn)
    def wrapped_vmap_fn(*args, **kwargs):
        # TODO: this probably results in a lot of compilation misses. Need to think about it.
        actual_bound = signature.bind(*args, **kwargs)
        actual_bound.apply_defaults()

        # now that we have args, we can figure out what the axis spec is for each arg
        padded_spec_args = axis_spec_bound_sig.args + (default,) * (
            len(actual_bound.args) - len(axis_spec_bound_sig.args)
        )

        padded_spec_kwargs = {
            **axis_spec_bound_sig.kwargs,
            **{k: default for k in actual_bound.kwargs.keys() - axis_spec_bound_sig.kwargs.keys()},
        }

        # want to support padded_spec_args being a tree prefix of the actual args, which this enables
        padded_spec_args = broadcast_prefix(padded_spec_args, actual_bound.args, is_leaf=is_named_array)
        padded_spec_kwargs = broadcast_prefix(padded_spec_kwargs, actual_bound.kwargs)

        arg_axis_specs = htu.tree_map(_index_of_batch_axis, actual_bound.args, padded_spec_args)

        kwarg_axis_specs = htu.tree_map(_index_of_batch_axis, actual_bound.kwargs, padded_spec_kwargs)

        # now we can actually vmap. We used "pacified" versions of NamedArrays that don't check
        # invariants, because intermediates creating during tracing won't have the axes right
        arg_axis_specs = htu.tree_map(_pacify_named_arrays, arg_axis_specs)
        kwarg_axis_specs = htu.tree_map(_pacify_named_arrays, kwarg_axis_specs)

        def wrapped_fn(args, kwargs):
            # the args that come in here are pacified. Their names will still have the batch axis even though the array
            # itself will already have that one removed. We need to turn them back into NamedArrays by removing the axis
            unchilled_args = jax.tree_util.tree_map(_to_unbatched_named_array(axis), args, is_leaf=_is_passive_array)
            unchilled_kwargs = jax.tree_util.tree_map(
                _to_unbatched_named_array(axis), kwargs, is_leaf=_is_passive_array
            )

            out = fn(*unchilled_args, **unchilled_kwargs)

            # now we need to pacify the output, which may include NamedArrays, and add the batch axis back at the end
            chilled = htu.tree_map(_pacify_named_arrays, out)
            arrays, nonarrays = eqx.partition(chilled, is_jax_array_like)
            return arrays, Static(nonarrays)

        spmd_axis_name = physical_axis_name(axis)

        args = htu.tree_map(_pacify_named_arrays, actual_bound.args)
        kwargs = htu.tree_map(_pacify_named_arrays, actual_bound.kwargs)

        result_dynamic, result_static = jax.vmap(
            wrapped_fn,
            in_axes=(arg_axis_specs, kwarg_axis_specs),
            out_axes=0,
            axis_size=axis.size if isinstance(axis, Axis) else None,
            spmd_axis_name=spmd_axis_name,
        )(args, kwargs)

        result = eqx.combine(result_dynamic, result_static.value)

        # if we were passed in a string arg, we need to get its axis size out from some result
        true_axis = _infer_axis_size_from_tree(result, axis)
        if true_axis is None:
            raise ValueError("vmap failed to infer axis size from result")

        result = jax.tree_util.tree_map(_prepend_named_batch_axis(true_axis), result, is_leaf=_is_passive_array)
        return result

    return wrapped_vmap_fn


def _to_unbatched_named_array(axis_to_strip: AxisSelector):
    def to_unbatched_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            if selects_axis(leaf.main_axes, axis_to_strip):
                return leaf.strip_axis(axis_to_strip)
            else:
                return leaf.to_named_array()
        else:
            return leaf

    return to_unbatched_named_array


__all__ = ["scan", "fold", "vmap", "map"]
