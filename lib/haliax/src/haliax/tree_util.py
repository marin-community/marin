# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
import functools

# Removed unused import

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree

import haliax.nn

from .axis import AxisSelector
from .core import NamedArray
from .jax_utils import maybe_rng_split
from .util import is_named_array


def tree_map(fn, tree, *rest, is_leaf=None):
    """
    Version of [jax.tree_util.tree_map][] that automatically treats NamedArrays as leaves.
    """
    old_is_leaf = is_leaf
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, NamedArray)
    else:
        is_leaf = lambda x: old_is_leaf(x) or is_named_array(x)

    return jax.tree.map(fn, tree, *rest, is_leaf=is_leaf)


def scan_aware_tree_map(fn, tree, *rest, is_leaf=None):
    """
    Version of [haliax.tree_util.tree_map][] that is aware of the scan-layer pattern, specifically as implemented
    in hax.nn.Stacked. This function will (implicitly) apply the transform to each layer in each Stacked module
    (using vmap). If there are no Stacked modules in the tree, this function is equivalent to [haliax.tree_util.tree_map][].

    """
    old_is_leaf = is_leaf
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, haliax.nn.Stacked)
    else:
        is_leaf = lambda x: old_is_leaf(x) or isinstance(x, haliax.nn.Stacked)

    mapped_fn = functools.partial(scan_aware_tree_map, fn, is_leaf=is_leaf)

    def rec_fn(x, *rest):
        if isinstance(x, haliax.nn.Stacked):
            new_inner = haliax.vmap(mapped_fn, x.Block)(x.stacked, *[r.stacked for r in rest])
            return dataclasses.replace(x, stacked=new_inner)  # type: ignore
        else:
            return fn(x, *rest)

    return tree_map(rec_fn, tree, *rest, is_leaf=is_leaf)


def tree_flatten(tree, is_leaf=None):
    """
    Version of [jax.tree_util.tree_flatten][] that automatically treats NamedArrays as leaves.
    """
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, NamedArray)
    else:
        is_leaf = lambda x: is_leaf(x) or is_named_array(x)

    return jax.tree_util.tree_flatten(tree, is_leaf=is_leaf)


def tree_unflatten(treedef, leaves):
    """
    Provided for consistency with tree_flatten.
    """
    return jax.tree_util.tree_unflatten(treedef, leaves)


def tree_leaves(tree, is_leaf=None):
    """
    Version of [jax.tree_util.tree_leaves][] that automatically treats NamedArrays as leaves.
    """
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, NamedArray)
    else:
        is_leaf = lambda x: is_leaf(x) or is_named_array(x)

    return jax.tree_util.tree_leaves(tree, is_leaf=is_leaf)


def tree_structure(tree, is_leaf=None):
    """
    Version of [jax.tree_util.tree_structure][] that automatically treats NamedArrays as leaves.
    """
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, NamedArray)
    else:
        is_leaf = lambda x: is_leaf(x) or is_named_array(x)

    return jax.tree_util.tree_structure(tree, is_leaf=is_leaf)


def resize_axis(tree: PyTree[NamedArray], old_axis: AxisSelector, new_size: int, key: PRNGKeyArray | None = None):
    """Resizes the NamedArrays of a PyTree along a given axis. If the array needs to grow and key is not none, then the
    new elements are sampled from a truncated normal distribution with the same mean and standard deviation as the
    existing elements. If the key is none, they're just initialized to the mean. If the array needs to shrink, then it's
    truncated.

    Note: if you have a module that stores a reference to the old axis, then you'll need to update that reference
    manually.

    """
    import haliax.random

    def _resize_one(x, key):
        if not is_named_array(x):
            return x

        assert isinstance(x, NamedArray)

        try:
            current_axis = x.resolve_axis(old_axis)
        except ValueError:
            return x

        if new_size == current_axis.size:
            return x
        elif current_axis.size > new_size:
            return x.slice(current_axis, start=0, length=new_size)
        else:
            num_padding = new_size - current_axis.size

            mean = x.mean(current_axis)
            std = x.std(current_axis)

            # the shape of the padding is the same as the original array, except with the axis size changed
            padding_axes = list(x.axes)
            padding_axes[padding_axes.index(current_axis)] = current_axis.resize(num_padding)

            if key is None:
                padding = mean.broadcast_axis(padding_axes)
            else:
                padding = haliax.random.truncated_normal(key, padding_axes, lower=-2, upper=2) * std + mean

            return haliax.concatenate(current_axis.name, [x, padding])

    leaves, structure = jax.tree_util.tree_flatten(tree, is_leaf=is_named_array)
    keys = maybe_rng_split(key, len(leaves))

    new_leaves = [_resize_one(x, key) for x, key in zip(leaves, keys)]

    return jax.tree_util.tree_unflatten(structure, new_leaves)


# old version of eqx's partition functions
def hashable_partition(pytree, filter_spec):
    dynamic, static = eqx.partition(pytree, filter_spec)
    static_leaves, static_treedef = jtu.tree_flatten(static)
    static_leaves = tuple(static_leaves)
    return dynamic, (static_leaves, static_treedef)


def hashable_combine(dynamic, static):
    static_leaves, static_treedef = static
    static = jtu.tree_unflatten(static_treedef, static_leaves)
    return eqx.combine(dynamic, static)
