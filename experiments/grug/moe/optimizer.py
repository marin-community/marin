# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import optax

from levanter.optim import OptimizerConfig
from experiments.grug.moe.adamh import normalize_by_global_gradient_rms, scale_by_adamh
from levanter.utils.jax_utils import leaf_key_paths


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

    def _scale_by_adamh(self, learning_rate):
        return scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)

    def _adamh_transform(self, learning_rate, *, include_clip: bool = True):
        components = []
        if include_clip and self.max_grad_norm:
            components.append(optax.clip_by_global_norm(self.max_grad_norm))
        components.append(self._scale_by_adamh(learning_rate))
        return optax.chain(*components)

    def _adam_transform(self, adam_lr, *, include_clip: bool = True):
        components = []
        if include_clip and self.max_grad_norm:
            components.append(optax.clip_by_global_norm(self.max_grad_norm))
        components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
        components.append(optax.scale(-adam_lr))
        return optax.chain(*components)

    def _multi_transform(self, learning_rate, adam_lr, expert_lr, *, include_clip: bool = True):
        return optax.multi_transform(
            {
                "adamh": self._adamh_transform(learning_rate, include_clip=include_clip),
                "adamh_expert": self._adamh_transform(expert_lr, include_clip=include_clip),
                "adam": self._adam_transform(adam_lr, include_clip=include_clip),
            },
            self.create_mask,
        )

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            return self._multi_transform(learning_rate, adam_lr, expert_lr)

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


@OptimizerConfig.register_subclass("grug_moe_adamh_global_grad_norm")
@dataclass(frozen=True)
class GrugMoeAdamHGlobalGradientNormConfig(GrugMoeAdamHConfig):
    """AdamH for Grug MoE with one global gradient RMS normalization."""

    gradient_norm_eps: float = 1e-16

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(normalize_by_global_gradient_rms(eps=self.gradient_norm_eps))
            components.append(self._multi_transform(learning_rate, adam_lr, expert_lr, include_clip=False))
            return optax.chain(*components)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )


__all__ = ["GrugMoeAdamHConfig", "GrugMoeAdamHGlobalGradientNormConfig"]
