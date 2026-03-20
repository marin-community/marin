# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamH optimizer for grug-style raw-array models with AdamH on attention, MLP, and lm-head."""

from dataclasses import dataclass

import jax
import optax

from levanter.optim.adamh import AdamHConfig, scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("grug_attention_mlp_lmhead_adamH")
@dataclass(frozen=True)
class GrugAttentionMlpLmHeadAdamHConfig(AdamHConfig):
    """AdamH on attention, MLP, and output projection matrices; AdamW for router/embed/vector params."""

    def build(self, num_train_steps):
        adamh_lr_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(adamh_learning_rate, adam_lr):
            def adamh_transform():
                return scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_learning_rate)

            def adamw_transform():
                components = [optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)]
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            inner = optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adam": adamw_transform(),
                },
                self.create_mask,
            )

            if self.max_grad_norm:
                return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), inner)
            return inner

        return optax.inject_hyperparams(optimizer)(
            adamh_learning_rate=adamh_lr_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embed" in path_lower or "router" in path_lower or "attn_gate" in path_lower:
                return "adam"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)
