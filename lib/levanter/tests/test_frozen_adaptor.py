# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
import safetensors
from haliax.state_dict import to_torch_compatible_state_dict

from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.callbacks import StepInfo
from levanter.checkpoint import CheckpointerConfig, load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, save_hf_checkpoint_callback
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.main.train_lm import (
    TrainLmConfig,
    _load_lm_model_from_configured_source,
    _restore_lm_model_from_partial_checkpoint,
)
from levanter.testing.helpers import use_test_mesh
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import FrozenParameterStorage, TrainerState


class TinyRoutedModel(eqx.Module):
    router_bias: hax.NamedArray
    weight: jax.Array


def tiny_model():
    return TinyRoutedModel(
        hax.named(jnp.full((26, 2), 0.1234567, dtype=jnp.float32), (hax.Axis("layers", 26), hax.Axis("experts", 2))),
        jnp.array([1.0, -2.0]),
    )


def loss(model):
    return jnp.sum(model.router_bias.array**2) + jnp.sum(model.weight**2)


def update(state):
    grads = eqx.filter_grad(loss)(state.model)
    assert np.all(np.asarray(grads.router_bias.array) != 0)
    return state.take_step(grads, key=jax.random.PRNGKey(int(state.step) + 1))[0]


def initial_state(model):
    adaptor = FrozenParameterAdaptorConfig(frozen_field_names=("router_bias",))
    return TrainerState.init(
        optax.adamw(0.1, weight_decay=0.1),
        adaptor.apply(model, key=jax.random.PRNGKey(0)),
        key=jax.random.PRNGKey(0),
        is_trainable=adaptor.trainable_filter(model),
        mp=jmp.get_policy("p=f32,c=bf16"),
        frozen_parameter_storage=FrozenParameterStorage.NATIVE,
    )


def test_frozen_adaptor_updates_other_weights_despite_nonzero_bias_gradient():
    model = tiny_model()
    state = initial_state(model)
    updated = update(state)
    np.testing.assert_array_equal(updated.model.router_bias.array, model.router_bias.array)
    assert updated.model.router_bias.dtype == jnp.float32
    assert not np.array_equal(updated.model.weight, model.weight)
    assert int(updated.step) == 1


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_native_frozen_storage_preserves_input_dtype_while_default_uses_compute(dtype):
    base = tiny_model()
    base = eqx.tree_at(lambda model: model.router_bias, base, base.router_bias.astype(dtype))
    adaptor = FrozenParameterAdaptorConfig(("router_bias",))
    policy = jmp.get_policy("p=f32,c=bf16")
    for storage in FrozenParameterStorage:
        state = TrainerState.init(
            optax.adamw(0.1),
            base,
            key=jax.random.PRNGKey(0),
            mp=policy,
            is_trainable=adaptor.trainable_filter(base),
            frozen_parameter_storage=storage,
        )
        updated = update(state)
        expected = base.router_bias.array
        if storage == FrozenParameterStorage.COMPUTE:
            expected = expected.astype(jnp.bfloat16)
        assert updated.model.router_bias.dtype == expected.dtype
        np.testing.assert_array_equal(updated.model.router_bias.array, expected)
        if dtype == jnp.float32 and storage == FrozenParameterStorage.COMPUTE:
            assert not np.array_equal(updated.model.router_bias.array.astype(jnp.float32), base.router_bias.array)


def test_frozen_adaptor_native_resume_restores_optimizer_and_pinned_base(tmp_path):
    base = tiny_model()
    state = initial_state(base)
    for _ in range(16):
        state = update(state)
    checkpoint = str(tmp_path / "step-15")
    with use_test_mesh():
        save_checkpoint(state.saveable_state, step=15, checkpoint_path=checkpoint)
        assert (tmp_path / "step-15" / "metadata.json").exists()
        template = initial_state(tiny_model())
        restored = load_checkpoint(template.saveable_state, checkpoint)
        assert restored.model.router_bias.array is None
        restored = dataclasses.replace(restored, model=eqx.combine(restored.model, base))
        resumed = restored
        while int(resumed.step) < 64:
            resumed = update(resumed)
    uninterrupted = state
    while int(uninterrupted.step) < 64:
        uninterrupted = update(uninterrupted)
    assert int(restored.step) == 16
    assert int(resumed.step) == 64
    np.testing.assert_array_equal(resumed.model.router_bias.array, base.router_bias.array)
    np.testing.assert_allclose(resumed.model.weight, uninterrupted.model.weight, rtol=1e-6, atol=1e-6)
    for resumed_leaf, expected_leaf in zip(
        jax.tree.leaves(resumed.opt_state), jax.tree.leaves(uninterrupted.opt_state)
    ):
        np.testing.assert_array_equal(resumed_leaf, expected_leaf)


def test_frozen_snowball_biases_survive_update_and_mixed_dtype_hf_export(tmp_path, local_gpt2_tokenizer_path):
    config = SnowballConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=26,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        tokenizer=local_gpt2_tokenizer_path,
    )
    overrides = {f"model.layers.{layer}.mlp.router.bias": "float32" for layer in range(26)}
    adaptor = FrozenParameterAdaptorConfig(("router_bias",), hf_save_dtype_overrides=overrides)
    with jax.set_mesh(compact_grug_mesh()):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 16), config, key=jax.random.PRNGKey(0))
        model = eqx.tree_at(
            lambda value: tuple(block.mlp.router_bias for block in value.transformer.blocks),
            model,
            tuple(jnp.array([0.1234567, -0.2345678], jnp.float32) for _ in range(26)),
        )
        original = to_torch_compatible_state_dict(model)
        converter = dataclasses.replace(config.hf_checkpoint_converter(), reference_checkpoint=None)
        source_path = str(tmp_path / "base")
        converter.save_pretrained(
            model,
            source_path,
            save_tokenizer=False,
            save_reference_code=False,
            dtype=jnp.bfloat16,
            dtype_overrides=overrides,
        )
        source_converter = dataclasses.replace(converter, reference_checkpoint=RepoRef(source_path))
        trainer_config = TrainerConfig(
            id="native-frozen-storage",
            mp=jmp.get_policy("p=f32,c=bf16"),
            frozen_parameter_storage=FrozenParameterStorage.NATIVE,
            tracker=(),
            require_accelerator=False,
            load_checkpoint=False,
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "native")),
        )
        lm_config = TrainLmConfig(model=config, adapter=adaptor, trainer=trainer_config, initialize_from_hf=True)
        trainer = Trainer(trainer_config, optax.adamw(0.1, weight_decay=0.1), loss, add_default_hooks=False)
        source_kwargs = dict(
            config=lm_config,
            converter=source_converter,
            Vocab=hax.Axis("vocab", 16),
            model_key=jax.random.PRNGKey(0),
            adapter_key=jax.random.PRNGKey(1),
            parameter_axis_mapping={},
            trainer=trainer,
        )
        loaded = _load_lm_model_from_configured_source(**source_kwargs)
        compute_config = dataclasses.replace(trainer_config, frozen_parameter_storage=FrozenParameterStorage.COMPUTE)
        compute_trainer = Trainer(compute_config, optax.adamw(0.1), loss, add_default_hooks=False)
        rounded = _load_lm_model_from_configured_source(
            **{
                **source_kwargs,
                "config": dataclasses.replace(lm_config, trainer=compute_config),
                "trainer": compute_trainer,
            }
        )
        for block, original_block in zip(rounded.transformer.blocks, model.transformer.blocks):
            expected_rounded = original_block.mlp.router_bias.astype(jnp.bfloat16).astype(jnp.float32)
            np.testing.assert_array_equal(block.mlp.router_bias, expected_rounded)
            assert not np.array_equal(block.mlp.router_bias, original_block.mlp.router_bias)
        state = trainer.initial_state(
            jax.random.PRNGKey(0),
            model=loaded,
            is_trainable=adaptor.trainable_filter(loaded),
        )
        grads = eqx.filter_grad(lambda value: sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(value)))(state.model)
        for block in grads.transformer.blocks:
            assert np.all(np.asarray(block.mlp.router_bias) != 0)
        state = state.take_step(grads, key=jax.random.PRNGKey(1))[0]
        state = state.take_step(grads, key=jax.random.PRNGKey(2))[0]
        checkpoint = str(tmp_path / "native" / "step-1")
        save_checkpoint(state.saveable_state, step=1, checkpoint_path=checkpoint)
        assert (tmp_path / "native" / "step-1" / "metadata.json").exists()
        restored = load_checkpoint(state.saveable_state, checkpoint)
        reloaded_base = _load_lm_model_from_configured_source(**source_kwargs)
        restored = dataclasses.replace(
            restored,
            model=_restore_lm_model_from_partial_checkpoint(
                restored.model, reloaded_base, adaptor.trainable_filter(reloaded_base)
            ),
        )
        restored = restored.take_step(grads, key=jax.random.PRNGKey(3))[0]
        uninterrupted = state.take_step(grads, key=jax.random.PRNGKey(3))[0]
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(restored.opt_state), jax.tree.leaves(uninterrupted.opt_state)
        ):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)
        updated = to_torch_compatible_state_dict(restored.model)
        callback = save_hf_checkpoint_callback(
            str(tmp_path),
            converter,
            save_dtype=jnp.bfloat16,
            dtype_overrides=adaptor.hf_save_dtype_overrides,
            save_tokenizer=False,
            save_reference_code=False,
        )
        callback(StepInfo(state=restored, loss=0.0, step_duration=0.0))
    actual = {}
    for filename in (tmp_path / "step-2").glob("*.safetensors"):
        with safetensors.safe_open(filename, framework="jax", device="cpu") as shard:
            actual.update({name: shard.get_tensor(name) for name in shard.keys()})
    assert len(actual) == len(original) == 502
    assert set(actual) == set(original)
    for name, value in actual.items():
        if name in overrides:
            assert value.dtype == jnp.float32
            np.testing.assert_array_equal(value, original[name])
        else:
            assert value.dtype == jnp.bfloat16
            np.testing.assert_array_equal(value, updated[name].astype(jnp.bfloat16))
    assert not np.array_equal(updated["model.embed_tokens.weight"], original["model.embed_tokens.weight"])
