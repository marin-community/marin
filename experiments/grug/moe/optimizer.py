# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe_hero_ep.optimizer import ExpertNormalization
from experiments.grug.moe_hero_ep.optimizer import scale_with_grug_muonh as scale_with_hero_muonh


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
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


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        # Run the step and the re-projection in float32. With BF16 parameters the
        # parameter-space norms otherwise round to BF16 while the post-step norms
        # stay float32, which biases the Frobenius re-projection at every learning
        # rate, not only at zero.
        param_f32 = param.astype(jnp.float32)
        update_f32 = update.astype(jnp.float32)
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param_f32)
            update_norm = jnp.linalg.norm(update_f32)
            new_param = param_f32 - learning_rate * update_f32 * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
        else:
            axes = tuple(range(1, param.ndim))
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param_f32), axis=axes, keepdims=True))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update_f32), axis=axes, keepdims=True))
            new_param = param_f32 - learning_rate * update_f32 * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        # Ratio form: equal norms give a ratio of exactly 1.0, so a zero raw step
        # yields an exactly zero update without a special case.
        ratio = jnp.maximum(param_norm, 1e-10) / jnp.maximum(new_param_norm, 1e-10)
        return (new_param * ratio - param_f32).astype(param.dtype)

    return jax.tree.map(
        scale_invariant_update,
        params,
        direction_updates,
        is_leaf=lambda x: x is None,
    )


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")
        for leaf in jax.tree.leaves(params):
            if getattr(leaf, "ndim", 0) >= 4:
                raise ValueError(
                    "scale_with_grug_muonh only supports 2-D and 3-D arrays; "
                    "use GrugMoeMuonHConfig or experiments.grug.moe_hero_ep.optimizer.scale_with_grug_muonh "
                    "for 4-D expert banks"
                )

        muon_updates, next_state = muon_transform.update(updates, state, params)
        muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate)
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate,
      mlp.expert_mlp.w_up, mlp.expert_mlp.w_down, shared.w_*)
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
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer: 3 LR groups (muonh / adamh / adam).

    Three LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all GatedNorms.
      Newton-Schulz orthogonalisation + Frobenius hyperball scale-invariant step.
      Four-dimensional expert banks use hero-EP's backend and configured
      ``expert_normalization``; two- and three-dimensional leaves use the
      original generic backend.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate``
      / norm weights.

    ``max_grad_norm`` clips each group separately. ``global_grad_norm`` clips
    the entire tree before grouping; the two settings are mutually exclusive.
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
    use_syrk: bool = False
    expert_normalization: ExpertNormalization = ExpertNormalization.PER_EXPERT
    global_grad_norm: float | None = None

    def __post_init__(self):
        if self.global_grad_norm is not None and self.max_grad_norm is not None:
            raise ValueError("global_grad_norm and per-group max_grad_norm cannot both be set")
        if self.global_grad_norm is not None and self.global_grad_norm <= 0:
            raise ValueError(f"global_grad_norm must be positive, got {self.global_grad_norm}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                common = dict(
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=learning_rate,
                    coefficient_type=self.coefficient_type,
                )
                generic = scale_with_grug_muonh(**common)
                hero = scale_with_hero_muonh(
                    **common, use_syrk=self.use_syrk, expert_normalization=self.expert_normalization
                )

                def update_fn(updates, state, params=None):
                    if params is None:
                        raise ValueError("MuonH requires params for norm-preserving updates")

                    def select(tree, four_dimensional):
                        return jax.tree.map(
                            lambda x: x if (getattr(x, "ndim", None) == 4) == four_dimensional else None,
                            tree,
                            is_leaf=lambda x: x is None,
                        )

                    def apply(transform, four_dimensional):
                        substate = state._replace(momentum_buffer=select(state.momentum_buffer, four_dimensional))
                        return transform.update(
                            select(updates, four_dimensional), substate, select(params, four_dimensional)
                        )

                    generic_updates, generic_state = apply(generic, False)
                    hero_updates, hero_state = apply(hero, True)

                    def merge(a, b):
                        return a if a is not None else b

                    result = jax.tree.map(merge, generic_updates, hero_updates, is_leaf=lambda x: x is None)
                    momentum = jax.tree.map(
                        merge, generic_state.momentum_buffer, hero_state.momentum_buffer, is_leaf=lambda x: x is None
                    )
                    return result, state._replace(momentum_buffer=momentum)

                components.append(optax.GradientTransformation(generic.init, update_fn))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            return optax.multi_transform(transforms, self.create_mask)

        inner = optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )
        if self.global_grad_norm is None:
            return inner
        clip = optax.clip_by_global_norm(self.global_grad_norm)

        def update_fn(updates, state, params=None):
            clipped, _ = clip.update(updates, optax.EmptyState())
            return inner.update(clipped, state, params)

        return optax.GradientTransformation(inner.init, update_fn)

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "router_bias" in path_lower
                or path_lower.endswith(".attn_gate")
                or ".router" in path_lower
            ):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            if path_lower.endswith(".weight"):
                return "adam"
            if hasattr(param, "ndim") and param.ndim in (2, 3, 4):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "scale_with_grug_muonh",
]
