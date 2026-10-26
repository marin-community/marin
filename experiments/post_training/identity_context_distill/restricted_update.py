# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restrict optimizer state and updates without changing the Snowball forward."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields, replace

import equinox as eqx
import jax
import optax
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.post_training.identity_context_distill.runtime_recipe import check_precision_runtime_receipt

BASE_RECEIPT_SHA256 = "565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e"
UPDATE_SCHEMA = "restricted_attention_update_v1"
PHYSICAL_NAMES = tuple(
    f"transformer.blocks.{layer}.attn.w_{projection}" for layer in range(18, 24) for projection in ("q", "k", "v", "o")
)
HF_NAMES = tuple(
    f"model.layers.{layer}.self_attn.{projection}_proj.weight"
    for layer in range(18, 24)
    for projection in ("q", "k", "v", "o")
)


def selected_parameter_names(params, trainable_param_patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve full-match patterns and require exactly the reviewed 24-leaf slice."""
    if not trainable_param_patterns:
        raise ValueError("The update slice requires explicit parameter patterns")
    patterns = tuple(re.compile(pattern) for pattern in trainable_param_patterns)
    names = tuple(jax.tree.leaves(leaf_key_paths(params)))
    selected = tuple(name for name in names if any(pattern.fullmatch(name) for pattern in patterns))
    if set(selected) != set(PHYSICAL_NAMES) or len(selected) != len(PHYSICAL_NAMES):
        raise ValueError(f"Update patterns must select exactly the 24 reviewed attention leaves, got {selected}")
    if any(not any(pattern.fullmatch(name) for name in names) for pattern in patterns):
        raise ValueError("An update pattern matched no parameter leaves")
    return tuple(sorted(selected))


@OptimizerConfig.register_subclass("restricted_attention_adam")
@dataclass(frozen=True, kw_only=True)
class RestrictedAdamConfig(AdamConfig):
    """Run Adam and its global-norm clip only on the explicitly selected leaves.

    All gradients are still computed. Frozen leaves have no Adam slots and their
    final updates are None, so apply_updates preserves even signed-zero bits.
    """

    trainable_param_patterns: tuple[str, ...]

    def build(self, num_train_steps):
        adam = super().build(num_train_steps)

        def mask(params):
            selected = frozenset(selected_parameter_names(params, self.trainable_param_patterns))
            return jax.tree.map(lambda name: name in selected, leaf_key_paths(params))

        def labels(params):
            return jax.tree.map(lambda trained: "trained" if trained else "frozen", mask(params))

        partition = optax.multi_transform({"trained": adam, "frozen": optax.set_to_zero()}, labels)

        def update(updates, state, params, **extra_args):
            updates, state = partition.update(updates, state, params=params, **extra_args)
            return eqx.filter(updates, mask(params)), state

        return optax.GradientTransformationExtraArgs(partition.init, update)


def restrict_updates(
    config: TrainLmConfig, *, learning_rate: float, trainable_param_patterns: tuple[str, ...]
) -> TrainLmConfig:
    """Keep the base recipe and replace only its optimizer's update policy."""
    if type(config.optimizer) is not AdamConfig:
        raise ValueError("Restricted attention updates require the reviewed Adam recipe")
    if learning_rate not in (2e-6, 1e-5):
        raise ValueError("Learning rate must be one of the two reviewed restricted-update arms")
    if not trainable_param_patterns:
        raise ValueError("The update slice requires explicit parameter patterns")
    values = {field.name: getattr(config.optimizer, field.name) for field in fields(AdamConfig)}
    optimizer = RestrictedAdamConfig(
        **(values | {"learning_rate": learning_rate}), trainable_param_patterns=trainable_param_patterns
    )
    return replace(config, optimizer=optimizer)


def restricted_runtime_receipt(
    config: TrainLmConfig,
    *,
    base_receipt_bytes: bytes,
    config_bytes: bytes,
    generation_bytes: bytes,
    rounding_bytes: bytes,
) -> dict:
    """Bind a schema-7 update policy to the unchanged, pinned model receipt."""
    if hashlib.sha256(base_receipt_bytes).hexdigest() != BASE_RECEIPT_SHA256:
        raise ValueError("Restricted updates require the reviewed base runtime receipt")
    base = json.loads(base_receipt_bytes)
    optimizer = config.optimizer
    if not isinstance(optimizer, RestrictedAdamConfig):
        raise ValueError("The recipe does not have a restricted attention optimizer")
    if base["schema_version"] != 6 or base["runtime_model_config"] != asdict(config.model):
        raise ValueError("The restricted recipe changed the base model/forward section")
    if base["compute_dtype_overrides"] != config.trainer.compute_dtype_overrides:
        raise ValueError("The restricted recipe changed compute dtype overrides")
    check_precision_runtime_receipt(
        config,
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=rounding_bytes,
    )
    adam = asdict(optimizer)
    patterns = adam.pop("trainable_param_patterns")
    return base | {
        "schema_version": 7,
        "update_policy": {
            "schema": UPDATE_SCHEMA,
            "base_receipt_sha256": BASE_RECEIPT_SHA256,
            "trainable_param_patterns": list(patterns),
            "physical_names": list(PHYSICAL_NAMES),
            "hf_names": list(HF_NAMES),
            "selected_count": len(PHYSICAL_NAMES),
            "adam": adam,
            "clip_scope": "selected_parameter_leaves_only",
            "frozen_optimizer_state": "absent",
            "frozen_updates": "None",
            "frozen_gradients_computed": True,
        },
    }


def base_runtime_receipt(receipt: dict) -> dict:
    """Recover the unchanged schema-6 view for the existing precision checks."""
    if receipt["schema_version"] != 7 or receipt["update_policy"]["schema"] != UPDATE_SCHEMA:
        raise ValueError("Not a restricted attention update receipt")
    return {key: value for key, value in receipt.items() if key != "update_policy"} | {"schema_version": 6}
