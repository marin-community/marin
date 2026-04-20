# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import optax

from levanter.optim import OptimizerConfig
from experiments.grug.moe.adamh import scale_by_adamh
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
    # If set, linear decay stops when LR reaches this fraction of peak, then holds constant.
    # Overrides the normal schedule. 0 = disabled (use normal schedule).
    constant_final_lr_ratio: float = 0.0

    def _constant_final_schedule(self, num_train_steps, peak_lr):
        """Warmup → linear decay → hold constant at constant_final_lr_ratio * peak."""
        warmup_steps = int(self.warmup * num_train_steps) if isinstance(self.warmup, float) else self.warmup
        floor_lr = self.constant_final_lr_ratio * peak_lr
        # Linear decay from peak to floor. Decay ends when it hits the floor.
        # With ratio R, decay covers (1 - R) of the LR range over (total - warmup) steps.
        decay_steps = num_train_steps - warmup_steps
        # How many steps until we hit the floor: decay_steps * (1 - R) / 1 ... no.
        # linear_schedule goes from peak to floor over decay_steps. We want it to
        # hit floor at some step < decay_steps, then hold. The simplest: compute
        # when linear_schedule(peak, 0, decay_steps) hits floor_lr, then hold.
        # peak * (1 - t/decay_steps) = floor_lr → t = decay_steps * (1 - floor_lr/peak)
        # = decay_steps * (1 - R)
        actual_decay_steps = max(1, int(decay_steps * (1.0 - self.constant_final_lr_ratio)))
        warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
        decay = optax.linear_schedule(peak_lr, floor_lr, actual_decay_steps)
        hold = optax.constant_schedule(floor_lr)
        return optax.join_schedules(
            [warmup, decay, hold],
            [warmup_steps, warmup_steps + actual_decay_steps],
        )

    def build(self, num_train_steps):
        if self.constant_final_lr_ratio > 0:
            learning_rate_schedule = self._constant_final_schedule(num_train_steps, self.learning_rate)
            adam_lr_schedule = self._constant_final_schedule(num_train_steps, self.adam_lr)
        else:
            learning_rate_schedule = self.lr_scheduler(num_train_steps)
            adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        if self.constant_final_lr_ratio > 0:
            expert_lr_schedule = self._constant_final_schedule(num_train_steps, expert_lr_val)
        else:
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


__all__ = ["GrugMoeAdamHConfig"]
