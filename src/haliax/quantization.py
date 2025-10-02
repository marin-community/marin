# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


# Support for FP8
# Much of this is lifted from FLAX
# https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py
import dataclasses
import functools
import warnings
from dataclasses import dataclass
from typing import Protocol, TypeVar

import aqt.jax.v2.config as aqt_config
import equinox as eqx
import jax
import jax.random as jrandom
from aqt.jax.v2.aqt_dot_general import DotGeneral
from jax import numpy as jnp
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from jaxtyping import DTypeLike, PyTree

import haliax.nn as hnn
from haliax.state_dict import StateDict
from haliax.types import PrecisionLike

from ._src.fp8 import dot_general_with_precision, in_qdq, out_qdq
from .axis import Axis
from .hof import vmap

T = TypeVar("T")


class OverwriteWithGradient(eqx.Module):
    """
    Sometimes there is state that must be computed in the backward pass which we want to
    persist for subsequent passes. Typically, we see this with quantization, particularly
    FP8. This module is a marker that indicates to [haliax.quantization.apply_updates][] that the
    gradient should be used to overwrite the state rather than added to it.

    Typically this is used in conjunction with [haliax.quantization.partition_for_grad_overwrite][]
    and the types are kinds of DotGeneralOp.
    """

    pass


def partition_for_grad_overwrite(grad: T) -> tuple[T, T]:
    """
    This function is used to partition the state of a module into two parts: one that will be
    overwritten by the gradient and one that will be updated by the gradient. This is used by
    [equinox.apply_updates][] to determine which state should be updated and which should
    be overwritten.
    The usual pattern is something like:

        ```python
        grads = jax.grad(loss_fn)(model)
        overwrites, grads = partition_for_grad_overwrite(grads)
        updates = optimizer.update(grads, params=model)
        model = hax.quant.apply_updates(model, updates, overwrites)
        ```

    """

    def is_overwrite_with_gradient(v):
        return isinstance(v, OverwriteWithGradient)

    x, y = eqx.partition(grad, is_overwrite_with_gradient, is_leaf=is_overwrite_with_gradient)
    return x, y


def apply_updates(tree, updates, overwrites):
    """
    A `jax.tree_util.tree_map`-broadcasted version of
    ```python
    if overwrite is not None:
        return overwrite
    if update is None:
        return model
    else:
        return model + update
    """

    def _apply_update(tree, update, overwrite):
        if overwrite is not None:
            return overwrite

        return eqx.apply_updates(tree, update)

    def is_leaf(x):
        return x is None or isinstance(x, OverwriteWithGradient)

    return jax.tree_util.tree_map(_apply_update, tree, updates, overwrites, is_leaf=is_leaf)


class DotGeneralOp(Protocol):
    """
    This protocol is used to define the signature of the `dot_general` function that is
    passed to the `Linear` module. This is used to allow for custom dot_general functions
    for quantized types.
    """

    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> jnp.ndarray: ...

    @staticmethod
    def default():
        return DefaultDotGeneralOp.init()


class DefaultDotGeneralOp(eqx.Module):
    """
    The default dot_general function that is used by the `Linear` module. This is the
    standard JAX `jax.lax.dot_general` function.

    Notes:
        We could have used `jax.lax.dot_general` directly, but we use this class so that we don't
        unnecessarily have functions as leaves in the module tree.
    """

    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> jnp.ndarray:
        return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision, preferred_element_type)

    # not really necessary, but it's nice to have a singleton
    @staticmethod
    def init():
        if not hasattr(DefaultDotGeneralOp, "_instance"):
            DefaultDotGeneralOp._instance = DefaultDotGeneralOp()

        return DefaultDotGeneralOp._instance


class Fp8DotGeneralOp(OverwriteWithGradient):
    input_scale: jnp.ndarray
    output_grad_scale: jnp.ndarray
    kernel_scale: jnp.ndarray
    input_amax_history: jnp.ndarray
    output_grad_amax_history: jnp.ndarray
    kernel_amax_history: jnp.ndarray
    compute_dtype: DTypeLike | None = eqx.field(static=True)

    @classmethod
    def init(cls, amax_history_length: int = 1024, compute_dtype: DTypeLike = None):
        return cls(
            input_scale=jnp.ones(1, dtype=jnp.float32),
            output_grad_scale=jnp.ones(1, dtype=jnp.float32),
            kernel_scale=jnp.ones(1, dtype=jnp.float32),
            input_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            output_grad_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            kernel_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            compute_dtype=compute_dtype,
        )

    # copied from flax
    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        # Use the `k.dtype` since it aligns with the `dtype` of its layers,
        # namely, the computation data type.
        if self.compute_dtype is None:
            comp_dtype = rhs.dtype
        else:
            comp_dtype = self.compute_dtype
        lhs = jnp.asarray(lhs, comp_dtype)

        x_qdq = in_qdq(comp_dtype, lhs, self.input_scale, self.input_amax_history)
        k_qdq = in_qdq(comp_dtype, rhs, self.kernel_scale, self.kernel_amax_history)
        y_qdq = dot_general_with_precision(x_qdq, k_qdq, dimension_numbers, precision, preferred_element_type)
        y = out_qdq(comp_dtype, y_qdq, self.output_grad_scale, self.output_grad_amax_history)

        return y


class Int8DotGeneralOp(OverwriteWithGradient):

    cfg: DotGeneral

    @classmethod
    def init(cls):
        cfg = aqt_config.config_v3()
        return cls(cfg)

    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type=None,
    ):
        cfg = aqt_config.set_context(self.cfg, jrandom.PRNGKey(42), train_step=None)
        return cfg(lhs, rhs, dimension_numbers, precision, preferred_element_type)

    def to_state_dict(tree: PyTree, prefix: str | None = None) -> StateDict:
        warnings.warn("Ignore all int8 states (if any) for now.")
        return {}


@dataclass(frozen=True)
class QuantizationConfig:
    targets: list[str] | str | None = dataclasses.field(default=None)
    """
    If provided, only modules with names in this list will be quantized. If a single string, will be treated as a regex
    """

    amax_history_length: int = 1024
    compute_dtype: DTypeLike = None

    fp8: bool = False
    int8: bool = False

    def __post_init__(self):
        assert not (self.fp8 and self.int8), "Cannot use FP8 and INT8 quantization at the same time."


def quantize_linear_layers(tree: T, config: QuantizationConfig) -> T:
    """
    Converts a module tree to use FP8/INT8 quantization.
    """
    if config.fp8:
        return _quantize_linear_layers(tree, config, Fp8DotGeneralOp, config.amax_history_length, config.compute_dtype)
    elif config.int8:
        return _quantize_linear_layers(tree, config, Int8DotGeneralOp)
    else:
        warnings.warn("Both fp8 and int8 are set to False. `quantize_linear_layers()` is no-op.")
        return tree


def _quantize_linear_layers(tree: T, config: QuantizationConfig, dot_general_cls, *args, **kwargs) -> T:
    """
    Linear modules that have a name that matches the targets (if provided) will be converted to quantized version.
    (If targets is None, all linear modules will be converted.)

    This essentially goes through and adds corresponding DotGeneralOp to the Linear modules.
    """

    def _is_special_module(module):
        # TODO: add conv?
        return isinstance(module, hnn.Linear) or isinstance(module, hnn.Stacked)

    def _batchify_ctor(ctor, batch_dims):
        # this is gross but it basically just vmaps the ctor over each batch dimension
        return functools.reduce(lambda ctor, batch_axis: vmap(ctor, batch_axis), reversed(batch_dims), ctor)

    # TODO: test scanlayers for dg
    def quantize_module(path_prefix, batch_dims: tuple[Axis, ...], path, module: T) -> T:
        path = path_prefix + path
        if isinstance(module, hnn.Stacked):
            new_inner = jax.tree_util.tree_map_with_path(
                functools.partial(quantize_module, path_prefix + (GetAttrKey("stacked"),), batch_dims + (module.Block,)),  # type: ignore
                module.stacked,
                is_leaf=_is_special_module,
            )
            return dataclasses.replace(module, stacked=new_inner)  # type: ignore
        elif isinstance(module, hnn.Linear):
            if _matches_target(path, config):
                vmapped_dg = _batchify_ctor(dot_general_cls.init, batch_dims)(*args, **kwargs)
                module = dataclasses.replace(module, dot_general=vmapped_dg)  # type: ignore
            return module
        else:
            return module

    return jax.tree_util.tree_map_with_path(
        lambda p, m: quantize_module((), (), p, m), tree, is_leaf=_is_special_module
    )


def _matches_target(key_path, config: QuantizationConfig) -> bool:
    if not key_path:
        key = ""
    else:
        key = _key_path_to_str(key_path[-1:])

    if config.targets is None:
        return True
    if isinstance(config.targets, list):
        return key in config.targets

    import re

    key_path_str = _key_path_to_str(key_path)
    return re.match(config.targets, key_path_str) is not None


def _key_path_to_str(key_path: tuple) -> str:
    out = ""
    for k in key_path:
        match k:
            case SequenceKey(i):  # type: ignore
                out = _join_key(out, str(i))
            case GetAttrKey(name):  # type: ignore
                out = _join_key(out, name)
            case DictKey(key):  # type: ignore
                out = _join_key(out, key)
            case FlattenedIndexKey(i):  # type: ignore
                out = _join_key(out, str(i))
            case _:
                warnings.warn(f"Unsupported key type {k}")
                out = _join_key(out, str(k))
    return out


def _join_key(prefix: str, key: str) -> str:
    if prefix:
        return f"{prefix}.{key}"
    return key
