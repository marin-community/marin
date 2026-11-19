# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
import yaml
from draccus.utils import DecodingError, ParsingError
from jax.sharding import AxisType, Mesh
from levanter.checkpoint import save_checkpoint
from levanter.data.text.examples import GrugLmExample
from safetensors.flax import load_file

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.post_training.long_loop.sft import SMOKE_CONFIG, fresh_sft_config
from experiments.post_training.long_loop.sft_runtime import train as sft_train
from experiments.post_training.long_loop.sft_runtime.export_writer import ExportConfig, export_weights
from experiments.post_training.long_loop.sft_runtime.model import grugmoe_inference_state_dict
from experiments.post_training.long_loop.sft_runtime.train import (
    _make_train_step,
    assert_fp32_train_state,
    initial_state,
    initialize_snowball_sft,
)


def _tiny_model_and_mesh(config):
    model = replace(
        config.model,
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    return model, mesh


def test_fresh_sft_loads_base_without_resume_manifest_lookup(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    recipe["trainer"]["trainer"]["load_checkpoint"] = True
    config = fresh_sft_config(
        output=str(tmp_path / "fresh"), initializer=str(tmp_path / "base"), recipe=recipe, run_id="cpu-init"
    )
    model = replace(
        config.model,
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    trainer = replace(config.trainer, special_token_ids=(30, 31), anchor_token_ids=(2, 3))
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    with jax.set_mesh(mesh):
        kwargs = dict(optimizer=optax.adam(0.001), mp=jmp.get_policy("float32"), ema_beta=None)
        base = initial_state(model, key=jax.random.key(1), **kwargs)
        fresh = initial_state(model, key=jax.random.key(2), **kwargs)
        legacy = eqx.tree_at(
            lambda p: p.stacked_blocks.stacked.shared, base.params, base.params.stacked_blocks.stacked.shared[0]
        )
        save_checkpoint(
            {"params": legacy, "pending_qb_betas": base.pending_qb_betas},
            0,
            config.trainer.trainer.initialize_from,
            is_temporary=False,
        )

        def forbidden_manifest_lookup(candidate):
            raise AssertionError(f"fresh start attempted resume manifest lookup: {candidate}")

        restored = restore_grug_state_from_checkpoint(
            fresh,
            checkpoint_search_paths=[str(tmp_path / "fresh/checkpoints")],
            load_checkpoint_setting=config.trainer.trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=False,
            template_for_candidate=forbidden_manifest_lookup,
        )
        actual = initialize_snowball_sft(restored, config.trainer.trainer.initialize_from, mesh, trainer)
        np.testing.assert_array_equal(actual.params.token_embed, base.params.token_embed)
        np.testing.assert_array_equal(actual.params.output_proj[:, 30], base.params.output_proj[:, 2])
        for a, b in zip(jax.tree.leaves(actual.opt_state), jax.tree.leaves(fresh.opt_state), strict=True):
            np.testing.assert_array_equal(a, b)
        assert int(actual.step) == 0

        # The serialized names must agree with schema 2 for each layer, including
        # the second pipeline stage; otherwise Bridge silently skips the experts.
        destination = tmp_path / "hf"
        export_weights(
            ExportConfig(
                "in-memory-initialized-base", "test", model, str(destination), "a" * 40, max_position_embeddings=262144
            ),
            lambda checkpoint, metadata_digest, model_config, export_mesh: actual.params,
            weights_kind="test_frozen_router",
        )
        assert json.loads((destination / "config.json").read_text())["grugmoe_artifact_schema_version"] == 2
        assert json.loads((destination / "config.json").read_text())["max_position_embeddings"] == 262144
        assert json.loads((destination / "config.json").read_text())["max_seq_len"] == 262144
        index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
        tensors = {}
        for shard in set(index.values()):
            tensors.update(load_file(str(destination / shard)))
        native = grugmoe_inference_state_dict(actual.params)
        for layer in range(model.num_layers):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                bank = f"model.layers.{layer}.mlp.experts.{projection}.weight"
                assert bank not in index
                for expert in range(model.num_experts):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    np.testing.assert_array_equal(tensors[name], native[bank][expert].astype("bfloat16"))


def test_remote_sft_timing_destination_survives_config_decode():
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    config = fresh_sft_config(
        output="s3://marin-us-east-02a/run/",
        initializer="s3://marin-us-east-02a/base",
        recipe=recipe,
        run_id="timing-regression",
    )
    # TrainerConfig.log_dir is a local Path: the old implementation collapsed s3:// to s3:/.
    assert config.native_timing_root == "s3://marin-us-east-02a/run/timing/native-sft"


def test_fresh_sft_rejects_bf16_param_policy_and_removed_master_param_switch(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    config = fresh_sft_config(
        output=str(tmp_path / "ok"),
        initializer=str(tmp_path / "base"),
        recipe=recipe,
        run_id="fp32-policy",
    )
    assert jnp.dtype(config.trainer.trainer.mp.param_dtype) == jnp.float32
    assert jnp.dtype(config.trainer.trainer.mp.compute_dtype) == jnp.bfloat16

    bf16_recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    bf16_recipe["trainer"]["trainer"]["mp"] = "compute=bfloat16,params=bfloat16,output=bfloat16"
    with pytest.raises(ParsingError) as exc_info:
        fresh_sft_config(
            output=str(tmp_path / "bf16"),
            initializer=str(tmp_path / "base"),
            recipe=bf16_recipe,
            run_id="bf16-rejected",
        )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "float32 parameters" in str(exc_info.value.__cause__)

    legacy_switch = yaml.safe_load(SMOKE_CONFIG.read_text())
    legacy_switch["trainer"]["master_param_mode"] = "device"
    with pytest.raises(DecodingError):
        fresh_sft_config(
            output=str(tmp_path / "switch"),
            initializer=str(tmp_path / "base"),
            recipe=legacy_switch,
            run_id="switch-removed",
        )


def test_fresh_sft_upcasts_bf16_hf_start_to_fp32_and_preserves_small_updates(tmp_path, caplog):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    config = fresh_sft_config(
        output=str(tmp_path / "fresh"),
        initializer=str(tmp_path / "base"),
        recipe=recipe,
        run_id="bf16-start-upcast",
    )
    model, mesh = _tiny_model_and_mesh(config)
    trainer = replace(config.trainer, special_token_ids=(30, 31), anchor_token_ids=(2, 3))
    mp = jmp.get_policy("compute=bfloat16,params=float32,output=bfloat16")
    optimizer = optax.adam(5e-6)

    with jax.set_mesh(mesh):
        base = initial_state(model, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
        fresh = initial_state(model, optimizer=optimizer, mp=mp, key=jax.random.key(2), ema_beta=None)
        bf16_params = jax.tree.map(
            lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
            base.params,
        )
        legacy = eqx.tree_at(
            lambda p: p.stacked_blocks.stacked.shared,
            bf16_params,
            bf16_params.stacked_blocks.stacked.shared[0],
        )
        save_checkpoint(
            {"params": legacy, "pending_qb_betas": base.pending_qb_betas.astype(jnp.bfloat16)},
            0,
            config.trainer.trainer.initialize_from,
            is_temporary=False,
        )

        with caplog.at_level(logging.INFO, logger=sft_train.__name__):
            actual = initialize_snowball_sft(fresh, config.trainer.trainer.initialize_from, mesh, trainer)

        assert_fp32_train_state(actual)
        assert any("bf16" in record.message.lower() and "float32" in record.message.lower() for record in caplog.records)
        np.testing.assert_array_equal(actual.params.token_embed, bf16_params.token_embed.astype(jnp.float32))
        np.testing.assert_array_equal(
            actual.params.output_proj[:, 30], bf16_params.output_proj[:, 2].astype(jnp.float32)
        )

        # At LR 5e-6 under BF16 compute, FP32 stored weights must preserve sub-BF16-ULP updates.
        before_embed = np.asarray(actual.params.token_embed).copy()
        tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
        batch = GrugLmExample(tokens, jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0))
        step_fn = _make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
        updated, _, _ = step_fn(actual, batch)
        assert_fp32_train_state(updated)
        assert not np.array_equal(np.asarray(updated.params.token_embed), before_embed)


def test_fp32_train_state_assertion_rejects_bf16_resumed_params_and_moments(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    config = fresh_sft_config(
        output=str(tmp_path / "resume"),
        initializer=str(tmp_path / "base"),
        recipe=recipe,
        run_id="bf16-resume",
    )
    model, mesh = _tiny_model_and_mesh(config)
    mp = jmp.get_policy("compute=bfloat16,params=float32,output=bfloat16")

    with jax.set_mesh(mesh):
        fp32_state = initial_state(model, optimizer=optax.adam(1e-3), mp=mp, key=jax.random.key(3), ema_beta=None)
        bf16_state = replace(
            fp32_state,
            step=jnp.asarray(1, dtype=jnp.int32),
            params=jax.tree.map(
                lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
                fp32_state.params,
            ),
            opt_state=jax.tree.map(
                lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
                fp32_state.opt_state,
            ),
        )
        checkpoint_root = tmp_path / "bf16-ckpt"
        save_checkpoint(bf16_state, 1, str(checkpoint_root / "step-1"), is_temporary=False)
        restored = restore_grug_state_from_checkpoint(
            fp32_state,
            checkpoint_search_paths=[str(checkpoint_root)],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
        )
        with pytest.raises(ValueError, match="non-FP32 leaves"):
            assert_fp32_train_state(restored)

        with pytest.raises(ValueError, match="non-FP32 leaves"):
            initial_state(
                model,
                optimizer=optax.adam(1e-3, mu_dtype=jnp.bfloat16),
                mp=mp,
                key=jax.random.key(4),
                ema_beta=None,
            )
