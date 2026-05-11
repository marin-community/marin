# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from typing import Any, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array
from optax import tree_utils as otu

import haliax
from haliax.nn import Linear
from haliax.tree_util import scan_aware_tree_map

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import flatten_linear_layers, label_linear_like_module, unflatten_linear_layers
from levanter.utils.jax_utils import leaf_key_paths


PyTree: TypeAlias = Any
Scalar: TypeAlias = float | jax.Array


@OptimizerConfig.register_subclass("dion")
@dataclass(frozen=True)
class DionConfig(OptimizerConfig):
    """Dion optimizer with AdamW fallback for non-matrix parameters.

    Dion uses amortized power iteration over an error-feedback momentum buffer
    to produce low-rank orthonormal updates for matrix weights.
    """

    learning_rate: float = 0.02
    mu: float = 0.95
    rank_fraction: float = 1.0
    power_iters: int = 1
    epsilon: float = 1e-8

    adam_lr: float = 6e-4
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    def build(self, num_train_steps: int) -> optax.GradientTransformation:
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate: Scalar, adam_lr: Scalar) -> optax.GradientTransformation:
            transformations = {
                "dion": _dion_transform(
                    learning_rate=learning_rate,
                    mu=self.mu,
                    rank_fraction=self.rank_fraction,
                    power_iters=self.power_iters,
                    epsilon=self.epsilon,
                    weight_decay=self.weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                ),
                "adamw": _adamw_fallback_transform(
                    max_grad_norm=self.max_grad_norm,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    adam_lr=adam_lr,
                    weight_decay=self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                ),
            }
            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params: PyTree) -> PyTree:
        """Label Haliax linear weights for Dion and route all other parameters to AdamW."""
        paths = leaf_key_paths(params)

        def mask_fn(param: Any, path: Any) -> Any:
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            if isinstance(param, Linear):
                return label_linear_like_module(param, weight_label="dion", bias_label="adamw")
            return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByDionState(NamedTuple):
    """Optax-compatible state for Dion."""

    momentum: optax.Updates
    right_vectors: PyTree


class _DionLeafResult(NamedTuple):
    """Per-leaf output of the Dion update. Typed to avoid matching arbitrary tuples in tree_map."""

    update: Any
    momentum: Any
    right_vector: Any


def scale_with_dion(
    *,
    mu: float = 0.95,
    rank_fraction: float = 1.0,
    power_iters: int = 1,
    epsilon: float = 1e-8,
) -> optax.GradientTransformation:
    """Scale matrix gradients with Dion's orthonormalized low-rank update."""
    if rank_fraction <= 0:
        raise ValueError("rank_fraction must be positive")
    if power_iters < 1:
        raise ValueError("power_iters must be at least 1")

    power_iters = int(power_iters)

    def init_fn(params: PyTree) -> ScaleByDionState:
        momentum = otu.tree_zeros_like(params)
        flattened_params = flatten_linear_layers(params)
        right_vectors = _init_right_vectors(flattened_params, rank_fraction=rank_fraction, epsilon=epsilon)
        return ScaleByDionState(momentum=momentum, right_vectors=right_vectors)

    def update_fn(updates: optax.Updates, state: ScaleByDionState, params: PyTree | None = None):
        del params
        flattened_updates = flatten_linear_layers(updates)
        flattened_momentum = flatten_linear_layers(state.momentum)

        new_updates, new_momentum, new_right_vectors = _dion_update_tree(
            flattened_updates,
            flattened_momentum,
            state.right_vectors,
            mu=mu,
            power_iters=power_iters,
            epsilon=epsilon,
        )

        updates = unflatten_linear_layers(updates, new_updates)
        momentum = unflatten_linear_layers(state.momentum, new_momentum)
        return updates, ScaleByDionState(momentum=momentum, right_vectors=new_right_vectors)

    return optax.GradientTransformation(init_fn, update_fn)


def _dion_transform(
    *,
    learning_rate: Scalar,
    mu: float,
    rank_fraction: float,
    power_iters: int,
    epsilon: float,
    weight_decay: float,
    build_weight_decay_mask: Callable[[], PyTree],
) -> optax.GradientTransformation:
    components: list[optax.GradientTransformation] = [
        scale_with_dion(mu=mu, rank_fraction=rank_fraction, power_iters=power_iters, epsilon=epsilon)
    ]
    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay, build_weight_decay_mask()))
    components.append(optax.scale(-learning_rate))
    return optax.chain(*components)


def _adamw_fallback_transform(
    *,
    max_grad_norm: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    adam_lr: Scalar,
    weight_decay: float,
    build_weight_decay_mask: Callable[[], PyTree],
) -> optax.GradientTransformation:
    components = []
    if max_grad_norm is not None and max_grad_norm > 0:
        components.append(optax.clip_by_global_norm(max_grad_norm))
    components.append(optax.scale_by_adam(beta1, beta2, epsilon))
    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay, build_weight_decay_mask()))
    components.append(optax.scale(-adam_lr))
    return optax.chain(*components)


def _init_right_vectors(params: PyTree, *, rank_fraction: float, epsilon: float) -> PyTree:
    paths = leaf_key_paths(params, is_leaf=_is_linear_or_masked_node_or_none)
    return haliax.tree_util.tree_map(
        lambda leaf, path: _init_right_vector_leaf(leaf, path, rank_fraction=rank_fraction, epsilon=epsilon),
        params,
        paths,
        is_leaf=_is_linear_or_masked_node_or_none,
    )


def _init_right_vector_leaf(leaf: Any, path: Any, *, rank_fraction: float, epsilon: float) -> Any:
    if _is_masked_node(leaf) or leaf is None:
        return leaf
    if not isinstance(leaf, Linear):
        return None
    weight = leaf.weight
    if not isinstance(weight, haliax.NamedArray) or weight.array is None:
        return None

    array = weight.array
    if array.ndim < 2:
        return None

    Out = weight.resolve_axis("__OUT__")
    In = weight.resolve_axis("__IN__")
    fan_out, fan_in = Out.size, In.size
    rank = _rank_from_shape(fan_out, fan_in, rank_fraction)
    key = jax.random.fold_in(jax.random.PRNGKey(0), _stable_seed(path, array.shape, rank))
    right_vector = jax.random.normal(key, (*array.shape[:-2], fan_in, rank), dtype=array.dtype)
    right_vector = _normalize_columns(right_vector, epsilon)
    Rank = haliax.Axis("__DION_RANK__", rank)
    named_right_vector = haliax.named(right_vector, (*weight.axes[:-2], In, Rank))
    # Wrap V in a Linear shell so the state tree has the same pytree structure as
    # params/updates, which lets scan_aware_tree_map vmap over Stacked layers.
    return dataclasses.replace(leaf, weight=named_right_vector, bias=None)


def _dion_update_tree(
    updates: PyTree,
    momentum: PyTree,
    right_vectors: PyTree,
    *,
    mu: float,
    power_iters: int,
    epsilon: float,
) -> tuple[PyTree, PyTree, PyTree]:
    is_leaf = _is_linear_or_masked_node_or_none

    def update_leaf(update_leaf: Any, momentum_leaf: Any, right_vector_leaf: Any):
        if _is_masked_node(update_leaf) or update_leaf is None:
            return _DionLeafResult(update_leaf, momentum_leaf, right_vector_leaf)
        if not isinstance(update_leaf, Linear):
            return _DionLeafResult(update_leaf, momentum_leaf, right_vector_leaf)
        if not isinstance(momentum_leaf, Linear) or not isinstance(right_vector_leaf, Linear):
            return _DionLeafResult(update_leaf, momentum_leaf, right_vector_leaf)

        update_weight = update_leaf.weight
        momentum_weight = momentum_leaf.weight
        right_vector_weight = right_vector_leaf.weight
        if not (
            isinstance(update_weight, haliax.NamedArray)
            and isinstance(momentum_weight, haliax.NamedArray)
            and isinstance(right_vector_weight, haliax.NamedArray)
        ):
            return update_leaf, momentum_leaf, right_vector_leaf

        Out = update_weight.resolve_axis("__OUT__")
        In = update_weight.resolve_axis("__IN__")
        out_first = update_weight.axes[-2].name == "__OUT__"
        dion_update, new_momentum, new_right_vector = _dion_update_matrix(
            update_weight.array,
            momentum_weight.array,
            right_vector_weight.array,
            fan_out=Out.size,
            fan_in=In.size,
            out_first=out_first,
            mu=mu,
            power_iters=power_iters,
            epsilon=epsilon,
        )

        update_weight = dataclasses.replace(update_weight, array=dion_update)
        momentum_weight = dataclasses.replace(momentum_weight, array=new_momentum)
        right_vector_weight = dataclasses.replace(right_vector_weight, array=new_right_vector)
        return _DionLeafResult(
            dataclasses.replace(update_leaf, weight=update_weight),
            dataclasses.replace(momentum_leaf, weight=momentum_weight),
            dataclasses.replace(right_vector_leaf, weight=right_vector_weight),
        )

    result = scan_aware_tree_map(update_leaf, updates, momentum, right_vectors, is_leaf=is_leaf)
    is_result = lambda x: isinstance(x, _DionLeafResult)
    return (
        jax.tree.map(lambda x: x.update, result, is_leaf=is_result),
        jax.tree.map(lambda x: x.momentum, result, is_leaf=is_result),
        jax.tree.map(lambda x: x.right_vector, result, is_leaf=is_result),
    )


def _dion_update_matrix(
    gradient: Array,
    momentum: Array,
    right_vectors: Array,
    *,
    fan_out: int,
    fan_in: int,
    out_first: bool,
    mu: float,
    power_iters: int,
    epsilon: float,
) -> tuple[Array, Array, Array]:
    _assert_matrix(gradient)
    accumulated = momentum + gradient.astype(momentum.dtype)

    # Ensure M is [Out, In] for the matmuls. V is always [In, r].
    if not out_first:
        accumulated = accumulated.T

    basis = right_vectors.astype(accumulated.dtype)

    left_basis = None
    right_factor = None
    for _ in range(power_iters):
        projected = accumulated @ basis
        left_basis, _ = jnp.linalg.qr(projected)
        right_factor = accumulated.T @ left_basis
        basis = _normalize_columns(right_factor, epsilon)

    assert left_basis is not None
    assert right_factor is not None

    new_momentum_oi = accumulated - (1.0 - mu) * (left_basis @ right_factor.T)
    new_right_vectors = basis.astype(right_vectors.dtype)
    orthonormal_update_oi = left_basis @ new_right_vectors.astype(left_basis.dtype).T

    # Transpose back to original layout if needed
    if not out_first:
        new_momentum_oi = new_momentum_oi.T
        orthonormal_update_oi = orthonormal_update_oi.T

    # Mask to zero when M is degenerate (avoids NaN from QR on a zero matrix).
    # Can't branch inside jit, so multiply by 0/1 instead.
    projected_norm = jnp.linalg.norm(projected)
    nonzero = (projected_norm > epsilon).astype(orthonormal_update_oi.dtype)
    scale = jnp.sqrt(jnp.asarray(fan_out / fan_in, dtype=orthonormal_update_oi.dtype))
    return (scale * orthonormal_update_oi * nonzero).astype(gradient.dtype), new_momentum_oi, new_right_vectors


def _normalize_columns(matrix: Array, epsilon: float) -> Array:
    return matrix / (jnp.linalg.norm(matrix, axis=-2, keepdims=True) + epsilon)


def _rank_from_shape(fan_out: int, fan_in: int, rank_fraction: float) -> int:
    rank = int(rank_fraction * min(fan_out, fan_in))
    return min(max(rank, 1), fan_out, fan_in)


def _stable_seed(path: Any, shape: tuple[int, ...], rank: int) -> int:
    path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
    payload = f"{path_str}|{shape}|{rank}"
    return int(hashlib.sha256(payload.encode()).hexdigest(), 16) % (2**31 - 1)


def _is_masked_node(x: Any) -> bool:
    return isinstance(x, optax.MaskedNode)


def _is_linear_or_masked_node_or_none(x: Any) -> bool:
    return isinstance(x, Linear) or _is_masked_node(x) or x is None


def _assert_matrix(array: Array) -> None:
    if array.ndim != 2:
        raise ValueError(f"Dion expects 2D matrix gradients, got shape {array.shape}")
