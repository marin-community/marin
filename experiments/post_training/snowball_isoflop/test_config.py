# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from levanter.main.train_lm import TrainLmConfig, preserve_hf_runtime_settings
from levanter.models.lm_model import LmExample
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig, SnowballLMHeadModel
from levanter.models.snowball_sft import SnowballSftConfig, snowball_loss_with_dispatch_counts, snowball_update_transform
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.mesh import MeshConfig

from experiments.post_training.snowball_isoflop.config import (
    ANCHOR_IDS,
    SPECIAL_IDS,
    Mask,
    build_pod,
    calibration_cells,
    data_config,
    evaluation_data_config,
    model_receipt,
    primary_cells,
    recipe,
    training_flops_per_position,
)


def test_native_muonh_update_changes_weights_and_keeps_router_bias(tmp_path):
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
    )
    native = build_pod(calibration_cells()[0]).train_config
    optimizer = optax.chain(
        native.optimizer.build(native.optimizer_num_train_steps),
        snowball_update_transform(SnowballSftConfig((30, 31), (2, 3), math.sqrt(32), True)),
    )
    trainer_config = TrainerConfig(
        id="native-muonh-update",
        train_batch_size=1,
        per_device_parallelism=1,
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1, "expert": 1}),
        use_explicit_mesh_axes=True,
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    Batch, Pos = hax.Axis("batch", 1), hax.Axis("position", 8)
    example = LmExample(
        hax.named(jnp.arange(8, dtype=jnp.int32)[None, :], (Batch, Pos)),
        hax.named(jnp.asarray([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=jnp.float32), (Batch, Pos)),
    )

    def loss(model, example, *, key=None):
        return snowball_loss_with_dispatch_counts(model, example, z_loss_weight=1e-4, reduction=hax.mean)

    with Trainer(trainer_config, optimizer, loss, add_default_hooks=False) as trainer:
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.key(0))
        measured, metrics = loss(model, example)
        expected = model.compute_next_token_loss(example, logsumexp_weight=1e-4)
        np.testing.assert_allclose(measured.scalar(), expected.scalar(), rtol=1e-5)
        assert float(metrics["moe/sender_dropped"]) == 0
        assert float(metrics["moe/receiver_dropped"]) == 0
        before = {k: np.array(v) for k, v in model.to_state_dict().items()}
        state = TrainerState.init(optimizer, model, key=jax.random.key(1), mp=trainer_config.mp)
        info = trainer.train_step(state, example)
    assert math.isfinite(info.loss)
    after = info.state.model.to_state_dict()
    assert not np.array_equal(before["lm_head.weight"], after["lm_head.weight"])
    np.testing.assert_array_equal(before["model.layers.0.mlp.router.bias"], after["model.layers.0.mlp.router.bias"])


@pytest.mark.parametrize("cell", calibration_cells(), ids=lambda c: c.name)
def test_native_sft_lowering_reproduces_explicit_recipe_and_survives_hf_reload(cell):
    pod = build_pod(cell)
    config = pod.train_config
    assert isinstance(config, TrainLmConfig)
    e1 = recipe()
    assert config.optimizer.learning_rate == 5e-5
    assert config.optimizer.adam_lr == 5e-5 * 3 / 13
    assert config.optimizer.beta1 == 0.9062
    assert config.optimizer.beta2 == 0.992027944069944
    assert config.optimizer.max_grad_norm is None
    assert config.global_gradient_clip == 1.0
    assert config.optimizer.warmup == e1["optimizer"]["warmup"]
    assert config.optimizer.decay == e1["optimizer"]["decay"]
    assert np.dtype(config.trainer.mp.param_dtype).name == "float32"
    assert np.dtype(config.trainer.mp.compute_dtype).name == "bfloat16"
    assert config.trainer.train_batch_size == 8
    assert config.trainer.per_device_parallelism == 1
    assert config.trainer.offload_opt_state
    assert (config.trainer.loader_fetch_batch_size, config.trainer.loader_max_buffered_batches) == (4, 512)
    assert config.trainer.gc_interval == 1
    assert config.trainer.gradient_accumulation_dtype == "float32"
    assert config.optimizer_num_train_steps == 4750
    assert config.trainer.num_train_steps == cell.stop_step
    assert config.z_loss_weight == 1e-4
    assert config.token_weighted_accumulation
    assert config.snowball_sft.special_token_ids == SPECIAL_IDS
    assert config.snowball_sft.anchor_token_ids == ANCHOR_IDS
    assert config.snowball_sft.special_token_lr_multiplier == math.sqrt(32)
    assert config.snowball_sft.freeze_router_bias
    assert config.trainer.checkpointer.keep_last_temporary_checkpoints is None
    assert not config.trainer.checkpointer.delete_old_temp_checkpoints
    receipt = model_receipt(cell.size)
    reloaded = preserve_hf_runtime_settings(
        SnowballConfig.from_hf_config(GrugMoeHfConfig(**receipt["config"])), config.model
    )
    assert reloaded.hidden_dim == cell.size
    assert reloaded.max_seq_len == 262144
    assert reloaded.qk_mult == 1.75
    assert reloaded.attention_implementation == "gpu_fa4_cute"
    assert reloaded.moe_implementation == "sonic"
    np.testing.assert_allclose(config.optimizer.lr_scheduler(config.optimizer_num_train_steps)(19), 5e-5)


def test_both_masks_keep_every_cache_and_pack_boundary_including_concat_children():
    full = draccus.encode(data_config(Mask.FULL))
    assistant = draccus.encode(data_config(Mask.ASSISTANT))
    count = 0
    for name, component in assistant["components"].items():
        reference = full["components"][name]
        leaves = component.get("children", {"self": component})
        reference_leaves = reference.get("children", {"self": reference})
        for key, leaf in leaves.items():
            original = reference_leaves[key]
            if "format" in leaf and "sft" in (leaf.get("tags") or ()):
                assert leaf["format"]["type"] == "serialized_chat"
                leaf["format"] = original["format"]
                count += 1
            assert leaf == original
    assert count == 216
    assert assistant == full


def test_validation_selection_preserves_reserved_sources_and_packing():
    full = data_config(Mask.ASSISTANT)
    selected = evaluation_data_config()
    assert selected.num_validation_sequences == recipe()["data"]["num_validation_sequences"]
    assert set(selected.components) == set(selected.num_validation_sequences)
    for name, component in selected.components.items():
        assert draccus.encode(component) == draccus.encode(full.components[name])


def test_saved_prefixes_cover_all_approved_budget_endpoints_without_entering_decay():
    cells = primary_cells()
    assert len(cells) == 18
    assert sum(len(c.checkpoint_steps) for c in cells) == 30
    for cell in cells:
        assert cell.stop_step < 4275
        assert cell.stop_step == cell.checkpoint_steps[-1]
        for step in cell.checkpoint_steps:
            compute = step * 8 * 262144 * training_flops_per_position(cell.size)
            target = min((3e18, 1e19, 3e19), key=lambda budget: abs(budget - compute))
            assert 0.99 * target <= compute <= target
