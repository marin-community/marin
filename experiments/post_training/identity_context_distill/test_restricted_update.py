# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the selected optimizer state and byte-preserving frozen updates."""

import hashlib
import json
from dataclasses import fields, replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from haliax.quantization import apply_updates
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.optim.config import AdamConfig, LionConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.post_training.identity_context_distill import restricted_update
from experiments.post_training.identity_context_distill.restricted_update import (
    HF_NAMES,
    PHYSICAL_NAMES,
    RestrictedAdamConfig,
    base_runtime_receipt,
    restrict_updates,
    restricted_runtime_receipt,
    selected_parameter_names,
)
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.test_runtime_recipe import inputs

PATTERNS = (r"transformer\.blocks\.(18|19|20|21|22|23)\.attn\.w_(q|k|v|o)",)


@pytest.fixture(scope="module")
def model():
    config = SnowballConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=26,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
    )
    with jax.set_mesh(compact_grug_mesh()):
        return SnowballLMHeadModel.init(hax.Axis("vocab", 16), config, key=jax.random.key(310))


def named_arrays(tree):
    return dict(zip(jax.tree.leaves(leaf_key_paths(tree)), jax.tree.leaves(tree), strict=True))


def restricted_adam(adam):
    return RestrictedAdamConfig(
        **{field.name: getattr(adam, field.name) for field in fields(AdamConfig)}, trainable_param_patterns=PATTERNS
    )


def test_exact_slice_and_rejection_of_wider_or_partial_patterns(model):
    selected = selected_parameter_names(model, PATTERNS)
    expected = {
        f"transformer.blocks.{layer}.attn.w_{projection}"
        for layer in (18, 19, 20, 21, 22, 23)
        for projection in ("q", "k", "v", "o")
    }
    assert set(selected) == expected
    assert set(HF_NAMES) == {
        f"model.layers.{layer}.self_attn.{projection}_proj.weight"
        for layer in (18, 19, 20, 21, 22, 23)
        for projection in ("q", "k", "v", "o")
    }
    assert len(model.to_state_dict()) == 502
    for patterns in (
        ("does_not_exist",),
        (r"transformer\.blocks\.18\.attn\.w_q",),
        (r"transformer\.blocks\..*\.attn\.w_(q|k|v|o)",),
        (*PATTERNS, r"transformer\.token_embed"),
        (*PATTERNS, r"transformer\.blocks\.18\.mlp\..*"),
        (*PATTERNS, "does_not_exist"),
    ):
        with pytest.raises(ValueError, match="pattern"):
            selected_parameter_names(model, patterns)


def test_real_updates_keep_all_frozen_bytes_and_allocate_only_selected_moments(model):
    model = eqx.tree_at(
        lambda value: value.transformer.token_embed, model, -jnp.zeros_like(model.transformer.token_embed)
    )
    before = {name: np.asarray(value).tobytes() for name, value in model.to_state_dict().items()}
    adam = AdamConfig(learning_rate=1e-5, warmup=2, lr_schedule="constant", weight_decay=0, max_grad_norm=0.5)
    optimizer = restricted_adam(adam).build(16)
    state = optimizer.init(model)

    def objective(value):
        return sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(value))

    update = eqx.filter_jit(optimizer.update)
    for _ in range(2):
        gradients = eqx.filter_grad(objective)(model)
        updates, state = update(gradients, state, model)
        assert set(named_arrays(updates)) == set(PHYSICAL_NAMES)
        model = apply_updates(model, updates, jax.tree.map(lambda _: None, model))
    after = {name: np.asarray(value).tobytes() for name, value in model.to_state_dict().items()}
    assert {name for name in before if before[name] != after[name]} == set(HF_NAMES)
    assert np.signbit(np.asarray(model.transformer.token_embed)).all()
    moments = jax.tree.leaves(state, is_leaf=lambda value: isinstance(value, optax.ScaleByAdamState))
    adam_states = [value for value in moments if isinstance(value, optax.ScaleByAdamState)]
    assert len(adam_states) == 1
    assert int(adam_states[0].count) == 2
    assert set(named_arrays(adam_states[0].mu)) == set(PHYSICAL_NAMES)
    assert set(named_arrays(adam_states[0].nu)) == set(PHYSICAL_NAMES)
    assert not jax.tree.leaves(state.inner_states["frozen"])


def test_selected_only_clipping_matches_an_independent_adam_reference(model):
    adam = AdamConfig(learning_rate=1e-5, warmup=2, lr_schedule="constant", weight_decay=0, max_grad_norm=0.5)
    optimizer = restricted_adam(adam).build(16)
    selected = {name: value for name, value in named_arrays(model).items() if name in PHYSICAL_NAMES}
    reference = adam.build(16)
    reference_state = reference.init(selected)
    state = optimizer.init(model)
    paths = leaf_key_paths(model)
    gradients = jax.tree.map(
        lambda value, path: jnp.full_like(value, 0.125 if path in PHYSICAL_NAMES else 1e20), model, paths
    )
    selected_gradients = {name: value for name, value in named_arrays(gradients).items() if name in PHYSICAL_NAMES}
    for _ in range(3):
        updates, state = optimizer.update(gradients, state, model)
        expected, reference_state = reference.update(selected_gradients, reference_state, selected)
        assert isinstance(expected, dict)
        for name, value in named_arrays(updates).items():
            np.testing.assert_array_equal(value, expected[name])


def test_recipe_keeps_model_and_data_and_retains_both_arm_schedules():
    config, _ = inputs()
    for learning_rate in (2e-6, 1e-5):
        restricted = restrict_updates(config, learning_rate=learning_rate, trainable_param_patterns=PATTERNS)
        assert replace(restricted, optimizer=config.optimizer) == config
        schedule = restricted.optimizer.lr_scheduler(16)
        np.testing.assert_allclose(
            [schedule(step) for step in range(16)], [0, learning_rate / 2] + [learning_rate] * 14, rtol=1e-7
        )


def test_receipt_rederives_the_forward_and_binds_each_update_policy(monkeypatch):
    """The only seam is the approved digest of synthetic source/routing metadata."""
    config, args = inputs()
    runtime, base = configure_precision_runtime(config, **args)
    base_bytes = json.dumps(base).encode()
    monkeypatch.setattr(restricted_update, "BASE_RECEIPT_SHA256", hashlib.sha256(base_bytes).hexdigest())
    evidence = {name: args[name] for name in ("config_bytes", "generation_bytes", "rounding_bytes")}
    receipts = []
    for learning_rate in (2e-6, 1e-5):
        restricted = restrict_updates(runtime, learning_rate=learning_rate, trainable_param_patterns=PATTERNS)
        receipt = restricted_runtime_receipt(restricted, base_receipt_bytes=base_bytes, **evidence)
        assert base_runtime_receipt(receipt) == base
        assert receipt["update_policy"]["adam"]["learning_rate"] == learning_rate
        receipts.append(receipt)
        assert isinstance(restricted.model, SnowballConfig)
        changed = replace(restricted, model=replace(restricted.model, fp32_activations=True))
        with pytest.raises(ValueError, match="model/forward"):
            restricted_runtime_receipt(changed, base_receipt_bytes=base_bytes, **evidence)
    receipts[1]["update_policy"]["adam"]["learning_rate"] = 2e-6
    assert receipts[0] == receipts[1]


def test_unreviewed_optimizer_learning_rate_and_receipt_schema_are_refused():
    config, _ = inputs()
    with pytest.raises(ValueError, match="Adam recipe"):
        restrict_updates(replace(config, optimizer=LionConfig()), learning_rate=2e-6, trainable_param_patterns=PATTERNS)
    with pytest.raises(ValueError, match="two reviewed"):
        restrict_updates(config, learning_rate=5e-6, trainable_param_patterns=PATTERNS)
    with pytest.raises(ValueError, match="Not a restricted"):
        base_runtime_receipt({"schema_version": 6})
