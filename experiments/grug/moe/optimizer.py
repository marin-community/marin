# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import (
    DEFAULT_MAX_GROUPED_STACK_SIZE,
    ORTHOGONALIZATION_LAYOUTS,
    STACK_BATCH_SHARDED,
    _grug_scale_with_muon,
)
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe.optimizer_sharding import assert_update_sharding_matches_params, target_named_sharding

Expert3DOptimizer = Literal["muonh", "adamh"]
VALID_EXPERT_3D_OPTIMIZERS: tuple[Expert3DOptimizer, ...] = ("muonh", "adamh")
MayOptimizer = Literal["muonh", "sgd"]
VALID_MAY_OPTIMIZERS: tuple[MayOptimizer, ...] = ("muonh", "sgd")
REPLICA_DCN_AXIS = "replica_dcn"
EXPERT_AXIS = "expert"
MATCH_OPTIMIZER_SHARDING_ENV = "MAY_MATCH_OPTIMIZER_SHARDING"


def _match_optimizer_sharding_enabled() -> bool:
    return os.environ.get(MATCH_OPTIMIZER_SHARDING_ENV, "true").lower() != "false"


def _uses_adamh_baseline_adam_group(path_lower: str) -> bool:
    # Use endswith for attn_gate so attn_gated_norm weights do not match.
    return (
        "token_embed" in path_lower
        or "router_bias" in path_lower
        or path_lower.endswith(".attn_gate")
        or ".router" in path_lower
    )


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    return target_named_sharding(array)


def _expert_momentum_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 3:
        return None

    target_sharding = _target_named_sharding(array)
    if target_sharding is None:
        return None

    mesh = target_sharding.mesh
    if mesh.shape.get(REPLICA_DCN_AXIS, 1) <= 1:
        return None

    spec = target_sharding.spec
    if len(spec) != 3 or spec[0] is None:
        return None

    stack_axis = spec[0] if isinstance(spec[0], tuple) else (spec[0],)
    if EXPERT_AXIS not in stack_axis or REPLICA_DCN_AXIS in stack_axis:
        return None

    sharded_stack_axis = (REPLICA_DCN_AXIS, *stack_axis)
    stack_axis_size = math.prod(int(mesh.shape[name]) for name in sharded_stack_axis if name in mesh.shape)
    if array.shape[0] % stack_axis_size != 0:
        return None

    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharded_stack_axis, *spec[1:]))


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding on updates before applying them."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state
        if not _match_optimizer_sharding_enabled():
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _with_named_update_scope(name: str, transform: optax.GradientTransformation) -> optax.GradientTransformation:
    """Annotate an Optax transform update with a stable profiler region."""

    def init_fn(params):
        return transform.init(params)

    def update_fn(updates, state, params=None):
        with jax.named_scope(name):
            return transform.update(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    if not _match_optimizer_sharding_enabled():
        return updates

    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _match_named_sharding_to_updates(params, updates):
    if not _match_optimizer_sharding_enabled():
        return params

    def match_sharding(param, update):
        if update is None or not hasattr(param, "shape"):
            return param
        target_sharding = _target_named_sharding(update)
        if target_sharding is None:
            return param
        return jax.sharding.reshard(param, target_sharding)

    return jax.tree.map(match_sharding, params, updates, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(
    params,
    direction_updates,
    learning_rate: float,
    *,
    label: str = "scale-invariant hyperball",
):
    with jax.named_scope("muonh/hyperball/match_direction_sharding"):
        direction_updates = _match_named_sharding_to_params(direction_updates, params)
    assert_update_sharding_matches_params(direction_updates, params, f"{label} direction_updates after sharding match")

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            with jax.named_scope("muonh/hyperball/matrix_norms"):
                param_norm = jnp.linalg.norm(param)
                update_norm = jnp.linalg.norm(update)
                step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
                dot = jnp.sum(param * update)
            with jax.named_scope("muonh/hyperball/matrix_projection"):
                new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
                new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
                rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
                return (rescale - 1) * param - rescale * step_scale * update

        axes = tuple(range(1, param.ndim))
        with jax.named_scope("muonh/hyperball/expert_stack_norms"):
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
            step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
            dot = jnp.sum(param * update, axis=axes, keepdims=True)
        with jax.named_scope("muonh/hyperball/expert_stack_projection"):
            new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
            new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
            rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
            return (rescale - 1) * param - rescale * step_scale * update

    with jax.named_scope("muonh/hyperball/project_updates"):
        hyperball_updates = jax.tree.map(
            scale_invariant_update,
            params,
            direction_updates,
            is_leaf=lambda x: x is None,
        )
    assert_update_sharding_matches_params(hyperball_updates, params, f"{label} updates after hyperball")
    return hyperball_updates


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    momentum_sharding_fn=None,
    orthogonalization_layout: str = STACK_BATCH_SHARDED,
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE,
    ns_compute_dtype: str = "input",
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
        momentum_sharding_fn=momentum_sharding_fn,
        orthogonalization_layout=orthogonalization_layout,
        max_grouped_stack_size=max_grouped_stack_size,
        ns_compute_dtype=ns_compute_dtype,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        with jax.named_scope("muonh/direction_update"):
            muon_updates, next_state = muon_transform.update(updates, state, params)
        assert_update_sharding_matches_params(muon_updates, params, "MuonH direction_updates before hyperball")
        with jax.named_scope("muonh/hyperball"):
            muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate, label="MuonH")
        assert_update_sharding_matches_params(muonh_updates, params, "MuonH updates after hyperball")
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate_up,
      mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": _with_named_update_scope("optimizer/group/adamh", adamh_transform()),
                    "adamh_expert": _with_named_update_scope(
                        "optimizer/group/adamh_expert",
                        adamh_expert_transform(),
                    ),
                    "adam": _with_named_update_scope("optimizer/group/adam", adam_transform()),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer for raw Grug MoE arrays.

    - ``muonh``: attention, expert/shared MLP matrices, and GatedNorm matrices
    - ``adamh``: lm head / output projection
    - ``adam``: token embeddings, router leaves, attention gates, and vector norms
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"
    expert_3d_optimizer: Expert3DOptimizer = "muonh"
    orthogonalization_layout: str = STACK_BATCH_SHARDED
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE
    ns_compute_dtype: str = "input"

    def build(self, num_train_steps):
        if self.expert_3d_optimizer not in VALID_EXPERT_3D_OPTIMIZERS:
            valid = ", ".join(VALID_EXPERT_3D_OPTIMIZERS)
            raise ValueError(f"expert_3d_optimizer={self.expert_3d_optimizer!r} must be one of {valid}")
        if self.orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
            valid = ", ".join(ORTHOGONALIZATION_LAYOUTS)
            raise ValueError(f"orthogonalization_layout={self.orthogonalization_layout!r} must be one of {valid}")
        if self.max_grouped_stack_size < 1:
            raise ValueError(f"max_grouped_stack_size={self.max_grouped_stack_size} must be positive")

        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        momentum_sharding_fn=_expert_momentum_sharding,
                        orthogonalization_layout=self.orthogonalization_layout,
                        max_grouped_stack_size=self.max_grouped_stack_size,
                        ns_compute_dtype=self.ns_compute_dtype,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muonh": _with_named_update_scope("optimizer/group/muonh", muonh_transform()),
                    "adamh": _with_named_update_scope("optimizer/group/adamh", adamh_transform_at(learning_rate)),
                    "adam": _with_named_update_scope("optimizer/group/adam", adam_transform_at(adam_lr)),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        if self.expert_3d_optimizer not in VALID_EXPERT_3D_OPTIMIZERS:
            valid = ", ".join(VALID_EXPERT_3D_OPTIMIZERS)
            raise ValueError(f"expert_3d_optimizer={self.expert_3d_optimizer!r} must be one of {valid}")

        paths = leaf_key_paths(params)
        expert_3d_optimizer = self.expert_3d_optimizer

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if "gated_norm" in path_lower:
                return "muonh"
            if ".mlp.expert_mlp.w_" in path_lower and hasattr(param, "ndim") and param.ndim == 3:
                return expert_3d_optimizer
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_sgd_v1")
@dataclass(frozen=True)
class GrugMoeSgdConfig(OptimizerConfig):
    """Stateless SGD for Grug MoE throughput diagnostics."""

    weight_decay: float = 0.0
    max_grad_norm: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale(-learning_rate))
            components.append(_match_named_update_sharding())
            return optax.chain(*components)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)


__all__ = [
    "VALID_EXPERT_3D_OPTIMIZERS",
    "VALID_MAY_OPTIMIZERS",
    "Expert3DOptimizer",
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "GrugMoeSgdConfig",
    "MayOptimizer",
    "scale_with_grug_muonh",
]
