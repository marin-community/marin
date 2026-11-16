# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import types

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

from levanter.adaptor import AdaptorExportConfig, FrozenParameterAdaptorConfig
from levanter.callbacks import StepInfo
from levanter.checkpoint import CheckpointerConfig, load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, save_hf_checkpoint_callback
from levanter.field_dtype_overrides import with_field_dtype_overrides
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.main.model_init import ModelInitContext, load_model_from_source
from levanter.main.train_dpo import _install_separate_reference_export_hooks
from levanter.dpo import DpoModel
from levanter.main.train_lm import (
    TrainLmConfig,
    _load_lm_model_from_configured_source,
    _restore_lm_model_from_partial_checkpoint,
)
from levanter.testing.helpers import use_test_mesh
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState


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
        mp=with_field_dtype_overrides(jmp.get_policy("p=f32,c=bf16"), {"router_bias": jnp.dtype("float32")}),
    )


def test_frozen_adaptor_updates_other_weights_despite_nonzero_bias_gradient():
    model = tiny_model()
    state = initial_state(model)
    updated = update(state)
    np.testing.assert_array_equal(updated.model.router_bias.array, model.router_bias.array)
    assert updated.model.router_bias.dtype == jnp.float32
    assert not np.array_equal(updated.model.weight, model.weight)
    assert int(updated.step) == 1


def test_frozen_adaptor_partial_resume_restores_optimizer_and_pinned_base(tmp_path):
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


@pytest.fixture
def fp32_snowball(tmp_path, local_gpt2_tokenizer_path):
    """A 26-layer Snowball saved as a bf16 HF source with fp32 router biases, loaded with them kept in fp32."""
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
        reference_checkpoint=None,
        tokenizer=local_gpt2_tokenizer_path,
    )
    fp32_keys = {f"model.layers.{layer}.mlp.router.bias" for layer in range(26)}
    adaptor = FrozenParameterAdaptorConfig(("router_bias",))
    with jax.set_mesh(compact_grug_mesh()):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 16), config, key=jax.random.PRNGKey(0))
        model = eqx.tree_at(
            lambda value: tuple(block.mlp.router_bias for block in value.transformer.blocks),
            model,
            tuple(jnp.array([0.1234567, -0.2345678], jnp.float32) for _ in range(26)),
        )
        converter = config.hf_checkpoint_converter()
        source_path = str(tmp_path / "base")
        converter.save_pretrained(
            model,
            source_path,
            save_tokenizer=False,
            save_reference_code=False,
            dtype=jnp.bfloat16,
            field_dtype_overrides={"router_bias": jnp.dtype("float32")},
        )
        source_converter = converter.replaced(reference_checkpoint=RepoRef(source_path))
        trainer_config = TrainerConfig(
            id="fp32-router-bias",
            mp=jmp.get_policy("p=f32,c=bf16"),
            field_dtype_overrides={"router_bias": jnp.dtype("float32")},
            tracker=(),
            require_accelerator=False,
            load_checkpoint=False,
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "ckpts")),
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
        yield types.SimpleNamespace(
            model=model,
            original=to_torch_compatible_state_dict(model),
            fp32_keys=fp32_keys,
            adaptor=adaptor,
            converter=converter,
            trainer_config=trainer_config,
            lm_config=lm_config,
            trainer=trainer,
            source_kwargs=source_kwargs,
            tmp_path=tmp_path,
        )


def stepped(env):
    """State after two updates with fp32 router biases, and the gradients used for them."""
    loaded = _load_lm_model_from_configured_source(**env.source_kwargs)
    state = env.trainer.initial_state(
        jax.random.PRNGKey(0),
        model=loaded,
        is_trainable=env.adaptor.trainable_filter(loaded),
    )
    grads = eqx.filter_grad(lambda value: sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(value)))(state.model)
    state = state.take_step(grads, key=jax.random.PRNGKey(1))[0]
    state = state.take_step(grads, key=jax.random.PRNGKey(2))[0]
    return state, grads


def resumed(env, state, grads):
    """Save at step 1, restore with the reloaded pinned base, and take one more update."""
    checkpoint = str(env.tmp_path / "ckpts" / "step-1")
    save_checkpoint(state.saveable_state, step=1, checkpoint_path=checkpoint)
    assert (env.tmp_path / "ckpts" / "step-1" / "metadata.json").exists()
    restored = load_checkpoint(state.saveable_state, checkpoint)
    reloaded_base = _load_lm_model_from_configured_source(**env.source_kwargs)
    restored = dataclasses.replace(
        restored,
        model=_restore_lm_model_from_partial_checkpoint(
            restored.model, reloaded_base, env.adaptor.trainable_filter(reloaded_base)
        ),
    )
    return restored.take_step(grads, key=jax.random.PRNGKey(3))[0]


def load_through_dpo_source(env, mp):
    """Load the bf16 Snowball HF source through the DPO load function."""
    return load_model_from_source(
        context=ModelInitContext(
            model=env.lm_config.model,
            converter=env.source_kwargs["converter"],
            use_hf_model_config=False,
        ),
        Vocab=hax.Axis("vocab", 16),
        model_key=jax.random.PRNGKey(0),
        parameter_axis_mapping={},
        mp=mp,
        hf_ref=True,
    )


def assert_bf16_export_with_fp32_biases(env, export_dir, expected):
    """The saved router biases keep their original float32 bits; every other tensor is ``expected`` in bf16."""
    actual = {}
    for filename in export_dir.glob("*.safetensors"):
        with safetensors.safe_open(filename, framework="jax", device="cpu") as shard:
            actual.update({name: shard.get_tensor(name) for name in shard.keys()})
    assert set(actual) == set(env.original)
    for name, value in actual.items():
        if name in env.fp32_keys:
            assert value.dtype == jnp.float32
            np.testing.assert_array_equal(value, env.original[name])
            assert not np.array_equal(value, env.original[name].astype(jnp.bfloat16).astype(jnp.float32))
        else:
            assert value.dtype == jnp.bfloat16
            np.testing.assert_array_equal(value, expected[name].astype(jnp.bfloat16))


def test_without_field_dtype_overrides_frozen_biases_round_to_compute_dtype(fp32_snowball):
    env = fp32_snowball
    compute_config = dataclasses.replace(env.trainer_config, field_dtype_overrides={})
    compute_trainer = Trainer(compute_config, optax.adamw(0.1), loss, add_default_hooks=False)
    rounded = _load_lm_model_from_configured_source(
        **{
            **env.source_kwargs,
            "config": dataclasses.replace(env.lm_config, trainer=compute_config),
            "trainer": compute_trainer,
        }
    )
    for block, original_block in zip(rounded.transformer.blocks, env.model.transformer.blocks):
        expected_rounded = original_block.mlp.router_bias.astype(jnp.bfloat16).astype(jnp.float32)
        np.testing.assert_array_equal(block.mlp.router_bias, expected_rounded)
        assert not np.array_equal(block.mlp.router_bias, original_block.mlp.router_bias)


@pytest.mark.parametrize("overridden", [True, False])
def test_load_model_from_source_honours_field_dtype_overrides(fp32_snowball, overridden):
    """The DPO load path keeps overridden biases at their stored float32 bits and rounds them otherwise."""
    env = fp32_snowball
    mp = env.trainer.mp if overridden else env.trainer_config.mp
    loaded = load_through_dpo_source(env, mp)
    for block, original_block in zip(loaded.transformer.blocks, env.model.transformer.blocks):
        original = original_block.mlp.router_bias
        expected = original if overridden else original.astype(jnp.bfloat16).astype(jnp.float32)
        assert block.mlp.router_bias.dtype == jnp.float32
        np.testing.assert_array_equal(block.mlp.router_bias, expected)
        assert np.array_equal(block.mlp.router_bias, original) == overridden


def test_dpo_load_and_export_hook_keep_overridden_biases(fp32_snowball):
    """Load through the DPO load function, then export through the real DPO HF hook and read the saved tensors."""
    env = fp32_snowball
    loaded = load_through_dpo_source(env, env.trainer.mp)
    hooks = []
    trainer = types.SimpleNamespace(
        run_id="dpo-export",
        config=env.trainer_config,
        add_hook=lambda hook, *, every: hooks.append(hook),
    )
    export_path = env.tmp_path / "dpo-export"
    _install_separate_reference_export_hooks(
        trainer=trainer,
        converter=env.converter,
        export=AdaptorExportConfig(hf_save_path=str(export_path), hf_save_steps=1, hf_save_dtype="bfloat16"),
    )
    [hook] = hooks
    with jax.set_mesh(compact_grug_mesh()):
        hook(types.SimpleNamespace(step=1, next_step=2, eval_model=DpoModel(policy=loaded, reference=env.model)))
    [step_dir] = export_path.glob("**/step-1")
    assert_bf16_export_with_fp32_biases(env, step_dir, to_torch_compatible_state_dict(loaded))


def test_fp32_frozen_biases_receive_gradients_but_stay_fixed(fp32_snowball):
    env = fp32_snowball
    state, grads = stepped(env)
    for block in grads.transformer.blocks:
        assert np.all(np.asarray(block.mlp.router_bias) != 0)
    for block, original_block in zip(state.model.transformer.blocks, env.model.transformer.blocks):
        np.testing.assert_array_equal(block.mlp.router_bias, original_block.mlp.router_bias)


def test_mixed_dtype_hf_export_keeps_fp32_frozen_biases(fp32_snowball):
    env = fp32_snowball
    state, grads = stepped(env)
    restored = resumed(env, state, grads)
    updated = to_torch_compatible_state_dict(restored.model)
    callback = save_hf_checkpoint_callback(
        str(env.tmp_path),
        env.converter,
        save_dtype=jnp.bfloat16,
        field_dtype_overrides=env.trainer_config.field_dtype_overrides,
        save_tokenizer=False,
        save_reference_code=False,
    )
    with jax.set_mesh(compact_grug_mesh()):
        callback(StepInfo(state=restored, loss=0.0, step_duration=0.0))
    assert_bf16_export_with_fp32_biases(env, env.tmp_path / "step-2", updated)
    assert not np.array_equal(updated["model.embed_tokens.weight"], env.original["model.embed_tokens.weight"])


def test_frozen_adaptor_export_hook_carries_field_dtype_overrides(tmp_path):
    adaptor = FrozenParameterAdaptorConfig(("router_bias",))
    saves = []
    converter = types.SimpleNamespace(save_pretrained=lambda model, path, **kwargs: saves.append((model, kwargs)))
    hooks = []
    trainer = types.SimpleNamespace(
        run_id="run",
        config=types.SimpleNamespace(checkpointer=None, field_dtype_overrides={"router_bias": jnp.dtype("float32")}),
        add_hook=lambda hook, *, every: hooks.append((hook, every)),
    )
    adaptor.install_export_hooks(
        trainer=trainer,
        converter=converter,
        tokenizer=None,
        export=AdaptorExportConfig(hf_save_path=str(tmp_path), hf_save_steps=4, hf_save_dtype="bfloat16"),
    )
    [(hook, every)] = hooks
    assert every == 4
    model = tiny_model()
    hook(types.SimpleNamespace(step=4, eval_model=model))
    [(saved, kwargs)] = saves
    assert saved is model
    assert kwargs["dtype"] == jnp.bfloat16
    assert kwargs["field_dtype_overrides"] == {"router_bias": jnp.dtype("float32")}
