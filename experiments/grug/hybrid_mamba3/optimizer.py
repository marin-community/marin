# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import optax

import haliax

from levanter.optim import AdamConfig, OptimizerConfig
from levanter.optim.skipstep import SkipStepConfig
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("hybrid_split_adam")
@dataclass(frozen=True)
class HybridSplitAdamConfig(AdamConfig):
    """AdamW with separate learning rates for Mamba blocks and transformer blocks."""

    mamba_learning_rate: float = 1e-3

    def build(self, num_train_steps):
        transformer_lr_schedule = self.lr_scheduler(num_train_steps)
        mamba_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.mamba_learning_rate)

        def _optimizer(learning_rate, mamba_learning_rate):
            def adamw_group(group_learning_rate):
                components = []
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon, nesterov=self.nesterov))

                if self.weight_decay > 0:
                    if self.adamc_weight_decay:
                        max_lr = max(self.learning_rate, self.mamba_learning_rate)
                        weight_decay = self.weight_decay * (group_learning_rate / max_lr)
                    else:
                        weight_decay = self.weight_decay
                    components.append(optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()))

                if self.update_rms_clipping is not None:
                    raise ValueError("update_rms_clipping is not supported for HybridSplitAdamConfig")
                if self.clip_update_norm is not None:
                    raise ValueError("clip_update_norm is not supported for HybridSplitAdamConfig")

                components.append(optax.scale(-group_learning_rate))
                return optax.chain(*components)

            transforms = {
                "mamba": adamw_group(mamba_learning_rate),
                "transformer": adamw_group(learning_rate),
            }

            optimizer = optax.multi_transform(transforms, self.create_mask)
            if self.max_grad_norm:
                optimizer = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optimizer)
            return optimizer

        optimizer = optax.inject_hyperparams(_optimizer)(
            learning_rate=transformer_lr_schedule,
            mamba_learning_rate=mamba_lr_schedule,
        )
        if self.skip_bad_steps:
            optimizer = SkipStepConfig.from_bool_int_or_config(self.skip_bad_steps).wrap(optimizer)
        return optimizer

    def create_mask(self, params):
        linear_block_indices = {
            index for index, block in enumerate(params.blocks) if getattr(block, "mixer", None) is not None
        }
        paths = leaf_key_paths(
            params,
            is_leaf=lambda x: eqx.is_array(x) or haliax.is_named_array(x),
        )

        def label_fn(param, path):
            del param
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            for index in linear_block_indices:
                if path_str.startswith(f"blocks.{index}."):
                    return "mamba"
            return "transformer"

        return jax.tree_util.tree_map(
            label_fn,
            params,
            paths,
            is_leaf=lambda x: eqx.is_array(x) or haliax.is_named_array(x),
        )
