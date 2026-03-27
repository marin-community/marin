# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamH optimizer for grug-style raw-array models.

Standard AdamHConfig uses haliax module introspection to distinguish linear
layers from embeddings/norms.  Grug models are plain pytrees of JAX arrays,
so we classify by ndim and path name instead:

* ndim >= 2 and NOT an embedding  -> AdamH (scale-invariant update)
* everything else (embed, norm, router, bias, scalar) -> AdamW
"""

from dataclasses import dataclass

import jax
import optax

from levanter.optim.adamh import AdamHConfig, scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("grug_adamH")
@dataclass(frozen=True)
class GrugAdamHConfig(AdamHConfig):
    """AdamH for grug raw-array models: AdamH on weight matrices, AdamW on the rest."""

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
            # Embeddings, router weights, and norm scalars use standard Adam.
            if "embed" in path_lower or "router" in path_lower:
                return "adam"
            # Weight matrices (ndim >= 2) get the scale-invariant AdamH update.
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)
