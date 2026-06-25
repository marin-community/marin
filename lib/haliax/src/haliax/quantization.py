# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

# Support for FP8
# Much of this is lifted from FLAX
# https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py
import dataclasses
import functools
import re
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

from ._src.fp8 import dot_general_with_precision, fp8_scaled_dot_general, in_qdq, out_qdq
from ._src.fp8_ragged import fp8_scaled_ragged_dot, MosaicWgradMode
from .nn.ragged_dot import Implementation
from .axis import Axis
from .core import NamedArray
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

    def is_leaf(v):
        return isinstance(v, (OverwriteWithGradient, NamedArray))

    x, y = eqx.partition(grad, is_overwrite_with_gradient, is_leaf=is_leaf)
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
        if update is None:
            return tree
        return eqx.apply_updates(tree, update)

    def is_leaf(x):
        return x is None or isinstance(x, OverwriteWithGradient) or isinstance(x, NamedArray)

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
        **kwargs,
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
        **kwargs,
    ) -> jnp.ndarray:
        return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision, preferred_element_type, **kwargs)

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
    # Forward-dot precision. None keeps the original DEFAULT forward (which, on the
    # transient operand-QDQ round-trip, XLA strips to a bf16 GEMM); HIGHEST makes the
    # forward re-fuse to a $f8 cuBLASLt matmul. Backward grad dots are always HIGHEST.
    # See logbook GFP8-010/012.
    forward_precision: PrecisionLike = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        amax_history_length: int = 1024,
        compute_dtype: DTypeLike | None = None,
        forward_precision: PrecisionLike = None,
    ):
        return cls(
            input_scale=jnp.ones(1, dtype=jnp.float32),
            output_grad_scale=jnp.ones(1, dtype=jnp.float32),
            kernel_scale=jnp.ones(1, dtype=jnp.float32),
            input_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            output_grad_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            kernel_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            compute_dtype=compute_dtype,
            forward_precision=forward_precision,
        )

    # copied from flax
    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        **kwargs,
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
        # The op controls its own forward precision (self.forward_precision); the
        # caller's `precision` is ignored on the fp8 path, as it always has been.
        y_qdq = dot_general_with_precision(
            x_qdq, k_qdq, dimension_numbers, self.forward_precision, preferred_element_type, **kwargs
        )
        y = out_qdq(comp_dtype, y_qdq, self.output_grad_scale, self.output_grad_amax_history)

        return y


class Fp8DirectDotGeneralOp(OverwriteWithGradient):
    """Direct-quantization FP8 dot — Flax's ``Fp8DirectDotGeneralOp`` (logbook GFP8-014).

    Genuine E4M3 operands flow straight into the forward dot and only the output is
    dequantized (``dq(dot(q(x), q(w)))``); the backward quantizes the output grad to E5M2.
    Unlike :class:`Fp8DotGeneralOp` this does not rely on XLA reconstructing f8 from a
    fake-quant round-trip, so the forward fires ``$f8`` at DEFAULT precision with no
    ``forward_precision`` flip — the path Flax adopted after deprecating the QDQ trick.
    Same per-tensor delayed-scaling state as :class:`Fp8DotGeneralOp`.
    """

    input_scale: jnp.ndarray
    output_grad_scale: jnp.ndarray
    kernel_scale: jnp.ndarray
    input_amax_history: jnp.ndarray
    output_grad_amax_history: jnp.ndarray
    kernel_amax_history: jnp.ndarray
    compute_dtype: DTypeLike | None = eqx.field(static=True)

    @classmethod
    def init(cls, amax_history_length: int = 1024, compute_dtype: DTypeLike | None = None):
        return cls(
            input_scale=jnp.ones(1, dtype=jnp.float32),
            output_grad_scale=jnp.ones(1, dtype=jnp.float32),
            kernel_scale=jnp.ones(1, dtype=jnp.float32),
            input_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            output_grad_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            kernel_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            compute_dtype=compute_dtype,
        )

    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        **kwargs,
    ):
        # Use the kernel dtype as the compute dtype (aligns with the layer dtype), as flax does;
        # the caller's precision/preferred_element_type are ignored on the fp8 path.
        comp_dtype = rhs.dtype if self.compute_dtype is None else self.compute_dtype
        lhs = jnp.asarray(lhs, comp_dtype)
        return fp8_scaled_dot_general(
            lhs,
            rhs,
            dimension_numbers,
            preferred_element_type=comp_dtype,
            lhs_scale=self.input_scale,
            rhs_scale=self.kernel_scale,
            grad_scale=self.output_grad_scale,
            lhs_amax_history=self.input_amax_history,
            rhs_amax_history=self.kernel_amax_history,
            grad_amax_history=self.output_grad_amax_history,
            quantize_compute_type=comp_dtype,
        )


class Fp8RaggedDotOp(OverwriteWithGradient):
    """Direct-quantization FP8 for the grouped (ragged) matmul — the ragged analog of
    :class:`Fp8DirectDotGeneralOp`.

    Carries the same per-tensor delayed-scaling state (input/kernel/output-grad scales and
    amax windows) and dispatches to :func:`haliax._src.fp8_ragged.fp8_scaled_ragged_dot`.
    Unlike the dense ops this is called as ``op(lhs, rhs, group_sizes)`` rather than through
    a ``Linear``'s ``dot_general``, because the MoE expert path invokes ``ragged_dot``
    directly.
    """

    input_scale: jnp.ndarray
    output_grad_scale: jnp.ndarray
    kernel_scale: jnp.ndarray
    input_amax_history: jnp.ndarray
    output_grad_amax_history: jnp.ndarray
    kernel_amax_history: jnp.ndarray
    compute_dtype: DTypeLike | None = eqx.field(static=True)
    implementation: Implementation = eqx.field(static=True, default="auto")
    # Backward output-grad format: E5M2 (Transformer-Engine hybrid, default) or E4M3 (all-E4M3).
    grad_dtype: DTypeLike = eqx.field(static=True, default=jnp.float8_e5m2)
    # Weight-gradient strategy on the mosaic backend: bf16 (default) or the f8 cast-transpose wgrad.
    mosaic_wgrad: MosaicWgradMode = eqx.field(static=True, default=MosaicWgradMode.BF16)

    @classmethod
    def init(
        cls,
        amax_history_length: int = 1024,
        compute_dtype: DTypeLike | None = None,
        implementation: Implementation = "auto",
        grad_dtype: DTypeLike = jnp.float8_e5m2,
        mosaic_wgrad: MosaicWgradMode = MosaicWgradMode.BF16,
    ):
        return cls(
            input_scale=jnp.ones(1, dtype=jnp.float32),
            output_grad_scale=jnp.ones(1, dtype=jnp.float32),
            kernel_scale=jnp.ones(1, dtype=jnp.float32),
            input_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            output_grad_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            kernel_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            compute_dtype=compute_dtype,
            implementation=implementation,
            grad_dtype=grad_dtype,
            mosaic_wgrad=mosaic_wgrad,
        )

    def __call__(self, lhs, rhs, group_sizes):
        comp_dtype = rhs.dtype if self.compute_dtype is None else self.compute_dtype
        lhs = jnp.asarray(lhs, comp_dtype)
        return fp8_scaled_ragged_dot(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=comp_dtype,
            lhs_scale=self.input_scale,
            rhs_scale=self.kernel_scale,
            grad_scale=self.output_grad_scale,
            lhs_amax_history=self.input_amax_history,
            rhs_amax_history=self.kernel_amax_history,
            grad_amax_history=self.output_grad_amax_history,
            quantize_compute_type=comp_dtype,
            grad_dtype=self.grad_dtype,
            implementation=self.implementation,
            mosaic_wgrad=self.mosaic_wgrad,
        )


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
        **kwargs,
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
    compute_dtype: DTypeLike | None = None

    fp8: bool = False
    int8: bool = False

    fp8_forward_precision: PrecisionLike = None
    """Forward-dot precision for FP8. None keeps the legacy DEFAULT forward (bf16 GEMM on H100);
    ``"highest"`` makes the forward re-fuse to a $f8 cuBLASLt matmul. Accepts jax precision aliases."""

    def __post_init__(self):
        assert not (self.fp8 and self.int8), "Cannot use FP8 and INT8 quantization at the same time."


def quantize_linear_layers(tree: T, config: QuantizationConfig) -> T:
    """
    Converts a module tree to use FP8/INT8 quantization.
    """
    if config.fp8:
        return _quantize_linear_layers(
            tree,
            config,
            Fp8DotGeneralOp,
            config.amax_history_length,
            config.compute_dtype,
            config.fp8_forward_precision,
        )
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
