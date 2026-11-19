# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Deletion Tuning V1 entry and its trainer-side wiring.

The production recipe is a pure overlay on the Exp3 production recipe; the
Deletion spec swaps the optimizer and the pod dispatch; the DT trainer scopes
optimizer state to the gate logits; the pod runner installs the seam around
marin's standard runner; and the HF config reload preserves the Deletion block.
"""

from functools import partial
from types import SimpleNamespace

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.models.snowball_deletion import DeletionTuningConfig, DeletionTuningScope
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

import experiments.deletion_tuning.levanter_train as levanter_train_module
import experiments.deletion_tuning.v1_sft as v1_sft_module
from experiments.deletion_tuning.levanter_train import (
    DeletionHFModel,
    DeletionTrainer,
    DeletionTrainSettings,
    deletion_settings_from_config,
    run_deletion_train_lm,
)
from experiments.deletion_tuning.v1_sft import (
    V1_H100_HOUR_LIMIT,
    V1_UPDATES,
    deletion_sft_spec,
    estimated_h100_hours,
    v1_deletion_config,
    v1_optimizer,
    v1_recipe,
)
from experiments.sft.launcher import HFModel, SFTSpec


@pytest.fixture(scope="module")
def mesh():
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)


def _tiny_snowball(deletion_tuning: DeletionTuningConfig | None = None) -> SnowballConfig:
    return SnowballConfig(
        deletion_tuning=deletion_tuning,
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
        moe_implementation="scatter",
    )


def _init_model(config: SnowballConfig, mesh) -> SnowballLMHeadModel:
    with jax.set_mesh(mesh):
        vocab = Axis("vocab", config.vocab_size)
        return SnowballLMHeadModel.init(vocab, config, key=jax.random.PRNGKey(0))


def _install_logits(model: SnowballLMHeadModel) -> SnowballLMHeadModel:
    rng = np.random.default_rng(3)
    logits = []
    for block in model.transformer.blocks:
        layer_logits = rng.normal(0.05, 0.02, size=block.mlp.expert_mlp.w_gate.shape).astype(np.float32)
        layer_logits[0] -= 0.2
        logits.append(layer_logits)
    spec = jax.typeof(model.transformer.blocks[0].mlp.expert_mlp.w_gate).sharding.spec

    def where(m):
        return tuple(block.mlp.gate_logits for block in m.transformer.blocks)

    updated = eqx.tree_at(
        where,
        model,
        tuple(jax.sharding.reshard(jnp.asarray(array), spec) for array in logits),
        is_leaf=lambda x: x is None,
    )
    updated = eqx.tree_at(lambda m: m.transformer.deletion_alpha, updated, jnp.asarray(1e2), is_leaf=lambda x: x is None)
    updated = eqx.tree_at(lambda m: m.transformer.deletion_hard, updated, jnp.asarray(0.0), is_leaf=lambda x: x is None)
    return updated


def _production_recipe() -> dict:
    """The Exp3 production recipe shape V1 overlays (only the fields V1 reads)."""
    model = draccus.encode(_tiny_snowball())
    model.pop("type", None)
    return {
        "format": "sft_spec",
        "profile": "snowball_exp3",
        "batch_size": 256,
        "seq_len": 262144,
        "stop_after_steps": 746,
        "task_timeout_seconds": 20 * 3600,
        "target_positions": 746 * 256 * 262144,
        "stage_guard": {"h100_hour_limit": 5120},
        "initial_eval_match": {"reference_path": "s3://ref"},
        "trainer": {
            "num_train_steps": 746,
            "training_budget": {"duration": "20h", "finalization_reserve": "40m", "update_reserve": "10m"},
        },
        "model": model,
    }


def test_v1_recipe_overlays_the_fixed_decisions():
    recipe = v1_recipe(_production_recipe())
    assert recipe["stop_after_steps"] == V1_UPDATES == 300
    assert recipe["trainer"]["num_train_steps"] == 300
    assert recipe["trainer"]["train_batch_size"] == 64
    assert recipe["batch_size"] == 64
    assert recipe["task_timeout_seconds"] == 12 * 3600
    assert recipe["trainer"]["training_budget"]["duration"] == "12h"
    assert recipe["trainer"]["progress_watchdog"]["startup_timeout"] == "2h"
    assert recipe["trainer"]["mesh"]["dcn_axes"]["context"] == 8
    assert recipe["trainer"]["mesh"]["dcn_axes"]["replica_dcn"] == 1
    assert recipe["trainer"]["mesh"]["axes"]["expert"] == 8
    assert recipe["resources"]["replicas"] == 8
    # the token budget equals the original 75 x 256 plan exactly
    assert recipe["target_positions"] == 300 * 64 * 262144 == 75 * 256 * 262144
    assert recipe["stage_guard"]["h100_hour_limit"] == V1_H100_HOUR_LIMIT == 768
    assert recipe["initial_eval_match"] is None
    model = v1_sft_module.draccus.decode(SnowballConfig, recipe["model"])
    assert model.deletion_tuning == v1_deletion_config()
    assert model.deletion_tuning.scope == DeletionTuningScope.ROUTED_GATE
    assert model.deletion_tuning.calibration_sequences == 64
    # the Exp3 controls carry over untouched
    assert recipe["seq_len"] == 262144


def test_v1_recipe_rejects_a_foreign_recipe():
    recipe = _production_recipe()
    recipe["profile"] = "other"
    with pytest.raises(ValueError, match="Exp3 production"):
        v1_recipe(recipe)


def test_v1_optimizer_and_estimate():
    optimizer = v1_optimizer()
    assert optimizer.learning_rate == 5e-4
    assert optimizer.weight_decay == 0.0
    assert optimizer.lr_schedule == "constant"
    assert optimizer.warmup == 0
    # the point estimate sits below the guard with headroom for the 75 updates
    hours = estimated_h100_hours()
    assert 9.0 < hours < 11.5, hours
    assert V1_H100_HOUR_LIMIT == 64 * 12


def test_deletion_sft_spec_swaps_optimizer_and_dispatch(monkeypatch):
    base_model = HFModel(model_ref="s3://base", tokenizer_path="s3://tok", model_type="snowball")
    spec = SFTSpec(
        name="deletion-tuning/v1-sft",
        version="2026.11.18.1",
        model=base_model,
        chat_template="unused",
        datasets=(),
        optimizer=AdamConfig(learning_rate=3e-4),
        num_train_steps=300,
    )
    monkeypatch.setattr(v1_sft_module, "exp3_sft_spec", lambda name, version, recipe: spec)

    recipe = v1_recipe(_production_recipe())
    built = deletion_sft_spec("deletion-tuning/v1-sft", "2026.11.18.1", recipe)

    assert isinstance(built.optimizer, AdamConfig)
    assert built.optimizer.learning_rate == 5e-4 and built.optimizer.weight_decay == 0.0
    assert built.optimizer_num_train_steps is None
    assert built.prepared_data is None  # the stub carries none; the block override guards on it
    assert isinstance(built.model, DeletionHFModel)
    assert built.model.model_ref == base_model.model_ref
    assert built.model.task_timeout_seconds == base_model.task_timeout_seconds


def test_deletion_hf_model_run_targets_the_dt_main():
    model = DeletionHFModel(model_ref="s3://base", model_type="snowball", task_timeout_seconds=123)
    run = model.run
    assert isinstance(run, partial)
    assert run.keywords["run_fn"] is levanter_train_module.run_deletion_train_lm
    assert run.keywords["task_timeout_seconds"] == 123


def test_deletion_settings_derive_from_the_model_config():
    config = SimpleNamespace(
        model=_tiny_snowball(v1_deletion_config()),
        trainer=SimpleNamespace(num_train_steps=300),
    )
    settings = deletion_settings_from_config(config)
    assert settings == DeletionTrainSettings(
        total_steps=300,
        alpha_start=1e2,
        alpha_end=1e3,
        eta=1 / 2048,
        lam=1e-5,
        lr=5e-4,
        calibration_sequences=64,
        seed=0,
    )


def test_run_deletion_train_lm_installs_and_restores_the_seam(monkeypatch):
    original_trainer = levanter_train_module.train_lm_module.Trainer
    original_loader = levanter_train_module.train_lm_module._load_lm_model_from_configured_source
    seen = {}

    def probe(config):
        patched_trainer = levanter_train_module.train_lm_module.Trainer
        seen["trainer"] = patched_trainer
        seen["settings"] = patched_trainer.deletion_settings
        seen["loader_patched"] = (
            levanter_train_module.train_lm_module._load_lm_model_from_configured_source is not original_loader
        )
        seen["delegated_config"] = config

    monkeypatch.setattr(levanter_train_module, "run_levanter_train_lm", probe)

    model_config = _tiny_snowball(v1_deletion_config())
    pod_config = SimpleNamespace(
        train_config=SimpleNamespace(model=model_config, trainer=SimpleNamespace(num_train_steps=300))
    )
    run_deletion_train_lm(pod_config)

    assert issubclass(seen["trainer"], DeletionTrainer)
    assert seen["settings"] == deletion_settings_from_config(pod_config.train_config)
    assert seen["loader_patched"]
    assert seen["delegated_config"] is pod_config
    # the seam restores the standard globals afterwards
    assert levanter_train_module.train_lm_module.Trainer is original_trainer
    assert levanter_train_module.train_lm_module._load_lm_model_from_configured_source is original_loader


def test_deletion_trainer_initial_state_scopes_optimizer_to_logits(mesh):
    settings = DeletionTrainSettings(
        total_steps=2,
        alpha_start=1e2,
        alpha_end=1e3,
        eta=1 / 2048,
        lam=1e-5,
        lr=5e-4,
        calibration_sequences=1,
        seed=0,
    )
    trainer_cls = type("DeletionTrainerPatched", (DeletionTrainer,), {"deletion_settings": settings})
    trainer_config = TrainerConfig(
        id="dt-v1-entry-test",
        train_batch_size=2,
        num_train_steps=2,
        seed=0,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        mesh=_tiny_mesh(),
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        tracker=WandbConfig(mode="disabled"),
    )
    trainer = trainer_cls(trainer_config, v1_optimizer().build(2), lambda model, *batch: jnp.asarray(0.0))
    with jax.set_mesh(mesh):
        model = _install_logits(_init_model(_tiny_snowball(v1_deletion_config()), mesh))
        state = trainer.initial_state(jax.random.PRNGKey(0), model=model)
        opt_leaves = [leaf for leaf in jax.tree.leaves(state.opt_state) if eqx.is_array(leaf)]
        # the reference unfiltered state covers every parameter
        unfiltered = type(state).init(
            trainer.optimizer, model, key=jax.random.PRNGKey(0), mp=trainer.mp, is_trainable=True
        )
        unfiltered_leaves = [leaf for leaf in jax.tree.leaves(unfiltered.opt_state) if eqx.is_array(leaf)]

    logit_count = len([block for block in model.transformer.blocks if block.mlp.gate_logits is not None])
    assert logit_count == 2
    assert 0 < len(opt_leaves) < len(unfiltered_leaves)
    # two logit leaves cover the whole optimizer state (moments + step counters)
    assert len(opt_leaves) <= 4 * logit_count + 2


def test_apply_deletion_init_matches_the_threshold_rule(mesh):
    """The init inserts per-layer logits whose sign split follows the Wanda threshold."""
    settings = DeletionTrainSettings(
        total_steps=2,
        alpha_start=1e2,
        alpha_end=1e3,
        eta=1 / 64,
        lam=1e-5,
        lr=5e-4,
        calibration_sequences=1,
        seed=0,
    )
    with jax.set_mesh(mesh):
        model = _init_model(_tiny_snowball(), mesh)
        rng = np.random.default_rng(11)
        routed = rng.uniform(0.5, 2.0, size=(2, 4, 8)).astype(np.float32)
        shared = rng.uniform(0.5, 2.0, size=(2, 8)).astype(np.float32)
        w0 = np.asarray(model.transformer.blocks[0].mlp.expert_mlp.w_gate)
        scores0 = np.abs(w0) * routed[0][..., None]
        threshold = float(np.median(scores0))  # inside the score spread |W| * ||x||
        initialized = levanter_train_module.apply_deletion_init(
            model, settings, routed, shared, threshold=threshold, num_layers=2
        )
        for block in initialized.transformer.blocks:
            assert block.mlp.gate_logits is not None
            assert block.mlp.gate_logits.shape == block.mlp.expert_mlp.w_gate.shape
        assert float(initialized.transformer.deletion_alpha) == 1e2
        assert float(initialized.transformer.deletion_hard) == 0.0
        logits0 = np.asarray(initialized.transformer.blocks[0].mlp.gate_logits)
        delete = scores0 <= threshold
        assert delete.any() and not delete.all()
        assert logits0[delete].mean() < 0.0 < logits0[~delete].mean()


def test_preserve_hf_runtime_settings_keeps_deletion_tuning():
    requested = _tiny_snowball(v1_deletion_config())
    reloaded = SnowballConfig.from_hf_config(requested.to_hf_config(requested.vocab_size))
    assert reloaded.deletion_tuning is None, "the reload must drop it so the preserve step is load-bearing"
    preserved = preserve_hf_runtime_settings(reloaded, requested)
    assert preserved.deletion_tuning == v1_deletion_config()
    assert preserved.attention_implementation == "reference"
    assert preserved.moe_implementation == "scatter"
    # absent stays absent
    plain = preserve_hf_runtime_settings(reloaded, _tiny_snowball())
    assert plain.deletion_tuning is None


def _tiny_mesh() -> MeshConfig:
    return MeshConfig(
        axes={"data": 1, "expert": 1, "model": 1, "replica": 1},
        dcn_axes={"context": 1, "replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    )
