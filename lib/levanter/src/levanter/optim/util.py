# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Callable, TypeVar

import equinox as eqx
import jax
import optax.tree_utils
from jaxtyping import PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs
from optax._src.base import init_empty_state

import haliax as hax
from haliax.tree_util import scan_aware_tree_map

import levanter.tracker
from levanter.utils.jax_utils import is_inexact_arrayish


T = TypeVar("T")


def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return eqx.filter_jvp(eqx.filter_grad(f), (x,), (v,))[1]
    # grad_f = eqx.filter_grad(f)
    # _, vjp_fn = eqx.filter_vjp(grad_f, x)
    # return vjp_fn(v)[0]


def tree_gaussian_like(key, tree):
    """
    Samples a tree of gaussian noise with the same structure as `tree`, except for leaves which are not inexact arrays,
    for which it returns None
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    rand_n = lambda x, key: jax.random.normal(key, x.shape) if is_inexact_arrayish(x) else None
    g = jax.tree_util.tree_map(rand_n, leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def log_norm_passthrough(desc: str) -> GradientTransformation:
    """
    Creates a gradient transformation that logs the L2 norm of the updates
    and returns the updates unchanged.
    """

    def init_fn(params):
        return None

    def update_fn(updates, state, params, **extra_args):
        levanter.tracker.jit_log({desc: optax.tree_utils.tree_l2_norm(updates)})
        return updates, None

    return GradientTransformationExtraArgs(init_fn, update_fn)


def scan_aware_clip_by_block_rms(threshold: float) -> GradientTransformation:
    """
    Version of `optax.clip_by_block_rms` that is aware of scan layers
    """

    def update_fn(updates, state, params=None, **extra_args):
        del params

        def _clip_fn(u):
            clip_denom = hax.maximum(1.0, hax.sqrt(hax.mean(u * u)) / threshold)
            return u / clip_denom

        updates = scan_aware_tree_map(_clip_fn, updates)
        return updates, state

    return GradientTransformation(init_empty_state, update_fn)


## utils for muon


def flatten_linear_layers(tree: T) -> T:
    """
    In PyTorch, linear layers are stored as a 2d weight matrix and a 1d bias vector. In Haliax,
    linear layers can have arbitrary dimensions, grouped into input and output axes. This function
    flattens the linear layers in a tree to be compatible with PyTorch-style state dicts.

    :param tree:
    """
    from haliax.nn import Linear

    def _flatten_linear(layer):
        if not isinstance(layer, Linear):
            return layer

        weight = layer.weight
        bias = layer.bias

        # weight and bias can sometimes be None or MaskedNode, so we check for that
        if isinstance(weight, hax.NamedArray) and weight.array is not None:
            out_first = layer._out_first
            weight = weight.flatten_axes(layer.Out, "__OUT__").flatten_axes(layer.In, "__IN__")

            if out_first:
                weight = weight.rearrange((..., "__OUT__", "__IN__"))
            else:
                weight = weight.rearrange((..., "__IN__", "__OUT__"))

            if isinstance(bias, hax.NamedArray):  # bias can be None or some weird sentinel like
                bias = bias.flatten_axes(layer.Out, "__OUT__")

            In = weight.resolve_axis("__IN__")
            Out = weight.resolve_axis("__OUT__")

            return dataclasses.replace(layer, weight=weight, bias=bias, In=In, Out=Out)  # type: ignore
        else:
            return layer

    return jax.tree.map(_flatten_linear, tree, is_leaf=lambda x: isinstance(x, Linear))


def unflatten_linear_layers(template: T, tree_with_flattened_linears: T) -> T:
    """
    Unflattens linear layers in a tree that was flattened with [haliax.state_dict.flatten_linear_layers][].
    Template has the same structure as the tree that was flattened, but with the original (unflattened)
    linear layers.

    Returns:
        The same tree as `tree_with_flattened_linears`, but with the linear layers unflattened to match
        the structure of `template`.
    """

    from haliax.nn import Linear

    def _unflatten_linear(template, flattened):
        assert isinstance(template, Linear) == isinstance(flattened, Linear)

        if not isinstance(template, Linear):
            return flattened

        weight = flattened.weight
        bias = flattened.bias

        if isinstance(weight, hax.NamedArray) and weight.array is not None:
            weight = weight.unflatten_axis("__OUT__", template.Out).unflatten_axis("__IN__", template.In)
            weight = weight.rearrange(template.weight.axes)

        if isinstance(bias, hax.NamedArray) and bias.array is not None:
            bias = bias.unflatten_axis("__OUT__", template.Out)
            assert template.bias is not None, "Flattened bias but template has no bias"
            bias = bias.rearrange(template.bias.axes)

        return dataclasses.replace(template, weight=weight, bias=bias)  # type: ignore

    return jax.tree.map(
        _unflatten_linear, template, tree_with_flattened_linears, is_leaf=lambda x: isinstance(x, Linear)
    )


def map_flattened_linear_layers(
    f: Callable[[hax.nn.Linear], hax.nn.Linear],
    params: PyTree,
    *,
    or_else: Callable | None = None,
    is_leaf: Callable | None = None,
):
    """
    Apply a function to all Linear layers in a PyTree, flattening articulated input/output dims into single dims, then
    unflattening them back into the original structure. This method also takes care of vmapping over scan layers.

    The linear layers will be passed to the function `f` and the result will be used to replace the original linear layer.
    The linear layers passed to `f` will be flattened into 2D (named) arrays, and the result will be unflattened back into the original shape.
    The bias term, if any, will be passed as a 1D named arrays.
    The weight array will not be None, but the bias array may be None.

    Args:
        f: The function to apply to each Linear layer
        params: The PyTree of parameters
        or_else: optional function to apply to non-Linear leaves
        is_leaf: optional function to determine if a node is a leaf. Linears will always be considered leaves.

    Returns:
        The PyTree with the function applied to all Linear layers and the structure preserved otherwise.
        returned linear layers will be unfattened back to their original shape.

    """

    orig_is_leaf = is_leaf

    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, hax.nn.Linear) or x is None
    else:
        is_leaf = lambda x: isinstance(x, hax.nn.Linear) or orig_is_leaf(x) or x is None  # type: ignore

    def map_fn(p):
        if isinstance(p, hax.nn.Linear):
            if p.weight is None:
                return p
            return f(p)
        elif or_else is not None:
            return or_else(p)
        else:
            return p

    # optax uses this MaskedNode stuff that confuses Haliax... Filter it out
    flattened_linear = flatten_linear_layers(params)
    flattened_linear = scan_aware_tree_map(map_fn, flattened_linear, is_leaf=is_leaf)
    # Now we have a flattened tree with linear layers, we can unflatten them back to the original structure
    # params = eqx.combine(masked_nodes, flattened_linear, is_leaf=is_leaf)

    return unflatten_linear_layers(params, flattened_linear)
