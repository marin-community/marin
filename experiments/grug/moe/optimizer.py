# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.optim import GrugMuonConfig, OptimizerConfig
from levanter.optim.grugmuon import (
    ORTHOGONALIZATION_LAYOUTS,
    STACK_BATCH_SHARDED,
    VMAP_REPLICATED,
    _batch_sharded_stack_target_pspec,
    _match_update_sharding,
    _target_sharding,
    _zeropower_via_newtonschulz_batched_stack_sharded,
    _zeropower_via_newtonschulz_replicated,
)
from levanter.optim.util import CoefficientType
from experiments.grug.moe.adamh import scale_by_adamh
from levanter.utils.jax_utils import leaf_key_paths

_COEFFICIENT_STEPS: dict[CoefficientType, int] = {
    "simple": 1,
    "quintic": 5,
    "polar_express": 8,
    "aol": 4,
}


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.w_gate_up, mlp.w_down, shared.w_*)
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
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
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
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muon")
@dataclass(frozen=True)
class GrugMoeMuonConfig(GrugMuonConfig):
    """Muon for Grug MoE with router, embedding, and scalar paths left on AdamW."""

    def create_mask(self, params, use_kimi_scaling=True):
        del use_kimi_scaling
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "output_proj" in path_lower
                or "router_bias" in path_lower
                or "attn_gate" in path_lower
                or ".router" in path_lower
            ):
                return "adamw"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "muon"
            return "adamw"

        return jax.tree.map(mask_fn, params, paths)


@dataclass(frozen=True)
class MuonPreset:
    learning_rate: float
    adam_lr: float
    momentum: float
    decay: float
    max_grad_norm: float


_MUON_PRESETS: tuple[tuple[int, MuonPreset], ...] = (
    (
        512,
        MuonPreset(
            learning_rate=0.016,
            adam_lr=0.0032,
            momentum=0.95,
            decay=0.8,
            max_grad_norm=1.0,
        ),
    ),
    (
        1024,
        MuonPreset(
            learning_rate=0.008,
            adam_lr=0.0024,
            momentum=0.98,
            decay=1.0,
            max_grad_norm=1.0,
        ),
    ),
    (
        1 << 30,
        MuonPreset(
            learning_rate=0.004,
            adam_lr=0.0012,
            momentum=0.98,
            decay=1.0,
            max_grad_norm=2.0,
        ),
    ),
)


class ScaleByGrugMuonHState(NamedTuple):
    momentum_buffer: optax.Updates


@OptimizerConfig.register_subclass("grug_moe_muonh")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """MuonH for Grug MoE with the AdamH matrix partition swapped to MuonH."""

    adam_lr: float = 6e-4
    momentum: float = 0.9
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-5
    max_grad_norm: float | None = 1.0
    coefficient_type: CoefficientType = "quintic"
    orthogonalization_layout: str = STACK_BATCH_SHARDED

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    _scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        orthogonalization_layout=self.orthogonalization_layout,
                    )
                )
                components.append(_match_update_sharding())
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muonh": muonh_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


def _scale_with_grug_muonh(
    *,
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    learning_rate: float,
    coefficient_type: CoefficientType,
    orthogonalization_layout: str,
) -> optax.GradientTransformation:
    steps = int(steps)
    if orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
        raise ValueError(
            f"Unknown orthogonalization_layout={orthogonalization_layout!r}. "
            f"Expected one of {ORTHOGONALIZATION_LAYOUTS!r}."
        )

    def init_fn(params):
        return ScaleByGrugMuonHState(momentum_buffer=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Grug MuonH requires parameters for the scale-invariant projection")

        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_array(update, param):
            if update is None or param is None or not hasattr(update, "ndim") or update.ndim not in (2, 3):
                return update

            if update.ndim == 2:
                direction = _zeropower_via_newtonschulz_replicated(
                    update,
                    steps=steps,
                    eps=muon_eps,
                    coefficient_type=coefficient_type,
                )
            elif orthogonalization_layout == VMAP_REPLICATED:
                direction = jax.vmap(
                    lambda matrix: _zeropower_via_newtonschulz_replicated(
                        matrix,
                        steps=steps,
                        eps=muon_eps,
                        coefficient_type=coefficient_type,
                    )
                )(update)
            else:
                target_pspec = _batch_sharded_stack_target_pspec(param)
                if target_pspec is None:
                    direction = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps=steps,
                            eps=muon_eps,
                            coefficient_type=coefficient_type,
                        )
                    )(update)
                else:
                    direction = _zeropower_via_newtonschulz_batched_stack_sharded(
                        update,
                        steps=steps,
                        eps=muon_eps,
                        coefficient_type=coefficient_type,
                        target_pspec=target_pspec,
                    )

            target_sharding = _target_sharding(param)
            if target_sharding is not None:
                direction = jax.sharding.reshard(direction, target_sharding)

            return _scale_invariant_muonh_update(param, direction, learning_rate)

        transformed = jax.tree.map(transform_array, updates, params, is_leaf=lambda x: x is None)
        return transformed, ScaleByGrugMuonHState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _scale_invariant_muonh_update(param: jax.Array, direction: jax.Array, learning_rate: float) -> jax.Array:
    if param.ndim == 2:
        param_norm = jnp.linalg.norm(param)
        direction_norm = jnp.maximum(jnp.linalg.norm(direction), 1e-10)
        new_param = param - learning_rate * direction * param_norm / direction_norm
        new_param_norm = jnp.maximum(jnp.linalg.norm(new_param), 1e-10)
        return new_param / new_param_norm * param_norm - param

    axes = tuple(range(1, param.ndim))
    param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
    direction_norm = jnp.sqrt(jnp.sum(jnp.square(direction), axis=axes, keepdims=True))
    safe_direction_norm = jnp.maximum(direction_norm, 1e-10)
    new_param = param - learning_rate * direction * param_norm / safe_direction_norm
    new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
    safe_new_param_norm = jnp.maximum(new_param_norm, 1e-10)
    return new_param / safe_new_param_norm * param_norm - param


def build_grug_moe_muonh_config(
    adamh_config: GrugMoeAdamHConfig,
    *,
    coefficient_type: CoefficientType = "quintic",
    muon_epsilon: float = 1e-5,
) -> GrugMoeMuonHConfig:
    """Convert an AdamH MoE config into a MuonH MoE config without retuning the LR schedule."""
    return GrugMoeMuonHConfig(
        learning_rate=adamh_config.learning_rate,
        weight_decay=0.0,
        min_lr_ratio=adamh_config.min_lr_ratio,
        warmup=adamh_config.warmup,
        decay=adamh_config.decay,
        rewarmup=adamh_config.rewarmup,
        cooldown=adamh_config.cooldown,
        cycle_length=adamh_config.cycle_length,
        cycles=adamh_config.cycles,
        lr_schedule=adamh_config.lr_schedule,
        haps=adamh_config.haps,
        adam_lr=adamh_config.adam_lr,
        momentum=adamh_config.beta1,
        beta1=adamh_config.beta1,
        beta2=adamh_config.beta2,
        epsilon=adamh_config.epsilon,
        muon_epsilon=muon_epsilon,
        max_grad_norm=adamh_config.max_grad_norm,
        coefficient_type=coefficient_type,
        backend_steps=_COEFFICIENT_STEPS[coefficient_type],
        weight_decay_modules=adamh_config.weight_decay_modules,
        default_weight_decay_mask=adamh_config.default_weight_decay_mask,
    )


def build_grug_moe_muon_config(
    *,
    hidden_dim: int,
    coefficient_type: CoefficientType = "aol",
) -> GrugMoeMuonConfig:
    """Return the size-matched Muon preset for the current MoE recipe."""
    preset = next(config for max_hidden_dim, config in _MUON_PRESETS if hidden_dim <= max_hidden_dim)
    return GrugMoeMuonConfig(
        learning_rate=preset.learning_rate,
        adam_lr=preset.adam_lr,
        weight_decay=0.0,
        min_lr_ratio=0.0,
        warmup=0.0,
        momentum=preset.momentum,
        beta1=0.8,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=preset.max_grad_norm,
        lr_schedule="linear",
        decay=preset.decay,
        coefficient_type=coefficient_type,
        backend_steps=_COEFFICIENT_STEPS[coefficient_type],
    )


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonConfig",
    "GrugMoeMuonHConfig",
    "build_grug_moe_muon_config",
    "build_grug_moe_muonh_config",
]
