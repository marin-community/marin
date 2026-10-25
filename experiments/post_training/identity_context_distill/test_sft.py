# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.adaptor import AdaptorExportConfig
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.mesh import MeshConfig

from experiments.post_training.identity_context_distill.sft import (
    BASE_MODEL,
    check_export_metadata,
    check_restored_progress,
    check_runtime_model,
    check_runtime_recipe,
    check_source_metadata_receipt,
    identity_sft_config,
    source_export_overrides,
    source_metadata_receipt,
    source_model_config,
)


def test_export_preserves_source_fields_through_converter(tmp_path):
    source = source_arguments()["source_config"]
    source.update(
        global_every=4,
        disable_long_rope=True,
        disable_pko=True,
        qk_mult_long_scale=1.0,
        router_z_loss_coef=0.0,
        num_local_experts=256,
        intermediate_size=1280,
        custom_metadata={"nested": [1, 2]},
    )
    overrides = source_export_overrides(source)
    converter = HFCheckpointConverter(SnowballConfig, HfConfigClass=GrugMoeHfConfig)
    converter = converter.with_config_overrides(overrides)
    model = SimpleNamespace(config=source_model_config(source), Vocab=SimpleNamespace(size=128256))
    written = converter._build_hf_config_dict(model)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(written))
    check_export_metadata(source, json.loads(path.read_text()))
    assert written["qk_mult"] == 1.75
    assert written["max_position_embeddings"] == 262144
    assert "max_seq_len" not in written
    overrides["custom_metadata"]["nested"].append(3)
    assert source["custom_metadata"] == {"nested": [1, 2]}


@pytest.mark.parametrize(
    "field", ["global_every", "disable_pko", "router_z_loss_coef", "qk_mult", "max_position_embeddings"]
)
def test_export_rejects_missing_or_changed_source_fields(field):
    source = source_arguments()["source_config"]
    source.update(global_every=4, disable_pko=True, router_z_loss_coef=0.0)
    exported = source_export_overrides(source)
    missing = dict(exported)
    del missing[field]
    with pytest.raises(ValueError, match="missing source field"):
        check_export_metadata(source, missing)
    changed = dict(exported)
    changed[field] = "changed"
    with pytest.raises(ValueError, match="changed source field"):
        check_export_metadata(source, changed)


def test_export_rejects_conflicting_and_banned_aliases():
    source = source_arguments()["source_config"]
    source["num_local_experts"] = 1
    with pytest.raises(ValueError, match="Conflicting export aliases"):
        source_export_overrides(source)
    del source["num_local_experts"]
    exported = source_export_overrides(source)
    exported["max_seq_len"] = 262144
    with pytest.raises(ValueError, match="banned aliases"):
        check_export_metadata(source, exported)


def test_standard_adaptor_callback_uses_source_overrides(monkeypatch, tmp_path):
    source = source_arguments()
    source["source_config"].update(global_every=4, router_z_loss_coef=0.0)
    config = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=1,
        dataset_uri="data",
        cache_uri="cache",
        output_uri=str(tmp_path),
        run_id="export",
        seed=42,
        **source,
    )
    converter = HFCheckpointConverter(SnowballConfig, HfConfigClass=GrugMoeHfConfig)
    saved = []
    hooks = []

    def capture_save(bound_converter, model, path, **kwargs):
        saved.append((json.loads(json.dumps(bound_converter._build_hf_config_dict(model))), kwargs))

    monkeypatch.setattr(HFCheckpointConverter, "save_pretrained", capture_save)
    trainer = SimpleNamespace(
        config=SimpleNamespace(id="export", checkpointer=None), add_hook=lambda hook, every: hooks.append((hook, every))
    )
    config.adapter.install_export_hooks(
        trainer=trainer,
        converter=converter,
        tokenizer=None,
        export=AdaptorExportConfig(hf_save_path=str(tmp_path), hf_save_steps=16, hf_save_dtype="bfloat16"),
    )
    model = SimpleNamespace(config=config.model, Vocab=SimpleNamespace(size=128256))
    hook, interval = hooks[0]
    hook(SimpleNamespace(step=15, eval_model=model))
    assert interval == 16
    check_export_metadata(source["source_config"], saved[0][0])
    assert saved[0][1]["dtype_overrides"] == config.adapter.hf_save_dtype_overrides
    assert converter.config_overrides is None


def source_arguments():
    source = json.loads(SnowballConfig(max_seq_len=262144, qk_mult=1.75).to_hf_config(128256).to_json_string())
    source.update(max_seq_len=262144, bos_token_id=128000, eos_token_id=128001, pad_token_id=128001)
    return {
        "source_config": source,
        "source_generation_config": {"bos_token_id": 128000, "eos_token_id": [128001, 128009]},
    }


class TinyModel(eqx.Module):
    weight: jax.Array
    router_bias: jax.Array


def tiny_loss(model):
    return jnp.square(model.weight) + jnp.sum(model.router_bias), {}


def test_factory_optimizer_native_resume_preserves_progress(tmp_path):
    config = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=1,
        dataset_uri="data",
        cache_uri="cache",
        output_uri=str(tmp_path),
        run_id="identity",
        seed=42,
        **source_arguments(),
    )
    base = TinyModel(jnp.asarray(1.0), jnp.full((26,), 0.1234567, dtype=jnp.float32))
    state = TrainerState.init(
        model=base,
        optimizer=config.optimizer.build(16),
        key=jax.random.PRNGKey(42),
        mp=config.trainer.mp,
        is_trainable=config.adapter.trainable_filter(base),
        frozen_parameter_storage=config.trainer.frozen_parameter_storage,
    )
    gradient = jax.grad(lambda model: tiny_loss(model)[0])
    for _ in range(16):
        state, _ = state.take_step(gradient(state.model), key=state.training_key)
    native = str(tmp_path / "checkpoints/step-15")
    save_checkpoint(state.saveable_state, step=15, checkpoint_path=native)
    expected_bias = np.asarray(base.router_bias).copy()
    expected_weight = np.asarray(state.model.weight).copy()
    expected_optimizer = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state.opt_state)]
    assert (tmp_path / "checkpoints/step-15/metadata.json").exists()
    continuation = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=1,
        dataset_uri="data",
        cache_uri="cache",
        output_uri=str(tmp_path),
        run_id="identity",
        seed=42,
        stop_after_updates=64,
        resume_checkpoint=native,
        **source_arguments(),
    )
    cpu_trainer = dataclasses.replace(continuation.trainer, mesh=MeshConfig(), tracker=(), require_accelerator=False)
    trainer = Trainer(cpu_trainer, continuation.optimizer.build(64), tiny_loss, add_default_hooks=False)
    with cpu_trainer.use_device_mesh():
        restored = trainer.initial_state(
            jax.random.PRNGKey(42), model=base, is_trainable=config.adapter.trainable_filter(base)
        )
    check_restored_progress(continuation, restored, completed_updates=16)
    np.testing.assert_array_equal(restored.model.router_bias, expected_bias)
    np.testing.assert_array_equal(restored.model.weight, expected_weight)
    assert float(expected_weight) != 1.0
    for saved, loaded in zip(expected_optimizer, jax.tree.leaves(restored.opt_state), strict=True):
        np.testing.assert_array_equal(saved, loaded)
    with pytest.raises(AssertionError):
        check_restored_progress(continuation, dataclasses.replace(restored, step=0), completed_updates=16)


def test_continuation_keeps_schedule_and_explicit_native_restore(tmp_path):
    arguments = dict(
        hosts=1,
        dataset_uri="data.jsonl",
        cache_uri="cache",
        output_uri=str(tmp_path),
        run_id="identity",
        seed=42,
        **source_arguments(),
    )
    initial = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        **arguments,
    )
    native = str(tmp_path / "checkpoints/step-15")
    continuation = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        **arguments,
        stop_after_updates=64,
        resume_checkpoint=native,
    )
    assert continuation.trainer.checkpoint_search_paths("identity") == [native]
    assert continuation.trainer.load_checkpoint is True
    assert continuation.initialize_from_checkpoint_path is None
    assert continuation.initialize_model_from_checkpoint_path is None
    assert continuation.initialize_from_hf == initial.initialize_from_hf == BASE_MODEL
    assert continuation.data == initial.data
    assert continuation.data_seed == initial.data_seed
    assert continuation.trainer.num_train_steps == 64
    assert initial.trainer.num_train_steps == 16
    assert initial.trainer.checkpointer.keep is None
    assert initial.trainer.checkpointer.save_interval is None
    assert not initial.trainer.checkpointer.delete_old_temp_checkpoints
    short_schedule = initial.optimizer.lr_scheduler(16)
    long_schedule = continuation.optimizer.lr_scheduler(64)
    for step in range(16):
        np.testing.assert_equal(short_schedule(step), long_schedule(step))
    np.testing.assert_allclose([long_schedule(step) for step in (2, 15, 16, 63)], 2e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="no HF reset"):
        identity_sft_config(
            moe_capacity_factor=4.0,
            router_combine_fp32=False,
            fp32_activations=False,
            attention_implementation="torch_order",
            **arguments,
            stop_after_updates=64,
        )


@pytest.mark.timeout(120)
def test_global_microbatch_unequal_token_masks_matches_unsharded_reference():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from levanter.grad_accum import microbatched
        from levanter.models.loss import maybe_fused_next_token_loss
        from experiments.post_training.identity_context_distill.sft import identity_sft_config
        from experiments.post_training.identity_context_distill.test_sft import source_arguments

        config = identity_sft_config(moe_capacity_factor=4.0, router_combine_fp32=False, fp32_activations=False, attention_implementation="torch_order", hosts=1,
                                     dataset_uri="data", cache_uri="cache", output_uri="out", run_id="test",
                                     seed=42, **source_arguments())
        trainer = config.trainer
        assert trainer.microbatch_size == trainer.train_batch_size == 8
        mesh = trainer.device_mesh
        mapping = trainer.compute_axis_mapping
        Batch, Pos, Vocab = hax.Axis("batch", 8), hax.Axis("position", 8), hax.Axis("vocab", 3)
        Embed = hax.Axis("features", 3)
        raw_inputs = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3) / 50
        raw_inputs[..., 0] *= -1
        raw_ids = np.arange(64, dtype=np.int32).reshape(8, 8) % 3
        raw_mask = (np.arange(8)[None, :] < np.arange(1, 9)[:, None]).astype(np.float32)
        raw_mask[:, -1] = 0

        def reference(weight, inputs, ids, mask):
            log_probabilities = jax.nn.log_softmax(weight * inputs, axis=-1)
            target = jnp.roll(ids, -1, axis=1)
            nll = -jnp.take_along_axis(log_probabilities, target[..., None], axis=-1)[..., 0]
            return jnp.sum(nll * mask) / jnp.sum(mask)

        expected_loss, expected_grad = jax.value_and_grad(reference)(
            jnp.asarray(0.7), jnp.asarray(raw_inputs), jnp.asarray(raw_ids), jnp.asarray(raw_mask)
        )
        grouped = [jax.value_and_grad(reference)(0.7, raw_inputs[start:start+2],
                   raw_ids[start:start+2], raw_mask[start:start+2]) for start in range(0, 8, 2)]
        assert abs(float(np.mean([value for value, _ in grouped]) - expected_loss)) > 1e-3
        assert abs(float(np.mean([grad for _, grad in grouped]) - expected_grad)) > 1e-3

        with jax.set_mesh(mesh), hax.axis_mapping(mapping):
            inputs = hax.shard(hax.named(jnp.asarray(raw_inputs), (Batch, Pos, Embed)))
            ids = hax.shard(hax.named(jnp.asarray(raw_ids), (Batch, Pos)))
            mask = hax.shard(hax.named(jnp.asarray(raw_mask), (Batch, Pos)))
            head = hax.shard(hax.named(jnp.eye(3), (Embed, Vocab)))
            def objective(weight, inputs, ids, mask):
                return maybe_fused_next_token_loss(
                    Pos, Embed, Vocab, weight * inputs, head, ids, loss_weight=mask,
                    precision=jax.lax.Precision.HIGHEST,
                ).scalar(), {}
            gradient = eqx.filter_value_and_grad(objective, has_aux=True)
            wrapped = microbatched(gradient, Batch, trainer.microbatch_size,
                                  trainer.parameter_axis_mapping, mapping)
            assert wrapped is gradient
            (actual_loss, _), actual_grad = eqx.filter_jit(wrapped)(jnp.asarray(0.7), inputs, ids, mask)
            np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
        """,
        device_count=32,
    )


@pytest.mark.parametrize("router_dot_fp32", [False, True])
def test_runtime_router_dot_exception_preserves_source_architecture(router_dot_fp32):
    sources = source_arguments()
    source = sources["source_config"]
    before = json.dumps(source, sort_keys=True)
    model = dataclasses.replace(source_model_config(source), router_dot_fp32=router_dot_fp32)
    check_runtime_model(model, source)
    check_runtime_model(dataclasses.replace(model, moe_capacity_factor=4.0), source)
    check_runtime_model(dataclasses.replace(model, router_combine_fp32=True), source)
    check_runtime_model(dataclasses.replace(model, fp32_activations=True), source)
    for changes in ({"qk_mult": 1.7}, {"max_seq_len": 8192}, {"num_experts": 128}):
        with pytest.raises(ValueError, match="architecture differs"):
            check_runtime_model(dataclasses.replace(model, **changes), source)
    assert json.dumps(source, sort_keys=True) == before
    assert source_model_config(source).router_dot_fp32 is False


def test_four_host_recipe_requires_the_real_slice_layout(monkeypatch):
    sources = source_arguments()
    arguments = dict(dataset_uri="data", cache_uri="cache", output_uri="out", run_id="mesh", seed=42, **sources)
    production = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=4,
        **arguments,
    )
    cpu = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=1,
        **arguments,
    )
    ici, dcn = production.trainer.mesh.axis_shapes(32, 4)
    assert ici == {"data": 2, "expert": 4, "replica": 1, "model": 1}
    assert dcn == {"replica_dcn": 1, "context": 4}
    assert production.trainer.parameter_axis_mapping["embed"] == ("data", "context")
    assert production.trainer.compute_axis_mapping["batch"] == ("replica_dcn", "data", "expert")
    cpu.trainer.mesh.axis_shapes(32, 1)
    with pytest.raises(ValueError, match="DCN product"):
        cpu.trainer.mesh.axis_shapes(32, 4)
    with pytest.raises(ValueError, match="DCN product"):
        production.trainer.mesh.axis_shapes(32, 1)
    with pytest.raises(ValueError, match="hosts"):
        identity_sft_config(
            moe_capacity_factor=4.0,
            router_combine_fp32=False,
            fp32_activations=False,
            attention_implementation="torch_order",
            hosts=2,
            **arguments,
        )
    monkeypatch.setattr(jax, "device_count", lambda: 32)
    assert cpu.trainer.num_slices == 1
    assert set(cpu.trainer.mesh_axis_specs) == {"data", "expert", "replica", "model", "context", "replica_dcn"}
    with pytest.raises(ValueError, match="DCN product 4 must equal num_slices 1"):
        _ = production.trainer.mesh_axis_specs
    monkeypatch.setattr(TrainerConfig, "num_slices", property(lambda self: 4))
    assert set(production.trainer.mesh_axis_specs) == {"data", "expert", "replica", "model", "context", "replica_dcn"}
    check_runtime_recipe(production, **sources)
    with pytest.raises(AssertionError):
        check_runtime_recipe(cpu, **sources)
    assert isinstance(production.model, SnowballConfig)
    enabled = dataclasses.replace(production, model=dataclasses.replace(production.model, router_dot_fp32=True))
    with pytest.raises(AssertionError):
        check_runtime_recipe(enabled, **sources)
    enabled = dataclasses.replace(
        enabled, trainer=dataclasses.replace(enabled.trainer, compute_dtype_overrides={"router_bias": "float32"})
    )
    check_runtime_recipe(enabled, **sources)
    assert isinstance(enabled.model, SnowballConfig) and enabled.model.moe_capacity_factor == 4.0
    with pytest.raises(AssertionError):  # capacity 1.0 drops assignments
        check_runtime_recipe(
            dataclasses.replace(enabled, model=dataclasses.replace(enabled.model, moe_capacity_factor=1.0)), **sources
        )
    flags = dict(fp32_activations=False, attention_implementation="torch_order", hosts=4)
    with pytest.raises(ValueError, match="dropless"):
        identity_sft_config(**dict(arguments, moe_capacity_factor=1.0, router_combine_fp32=False), **flags)
    with pytest.raises(ValueError, match="router_combine_fp32"):
        identity_sft_config(**dict(arguments, moe_capacity_factor=4.0, router_combine_fp32=None), **flags)
    with pytest.raises(TypeError, match="fp32_activations"):
        identity_sft_config(
            **dict(arguments, moe_capacity_factor=4.0, router_combine_fp32=True),
            attention_implementation="torch_order",
            hosts=4,
        )
    for value in (None, 0):
        with pytest.raises(ValueError, match="fp32_activations"):
            identity_sft_config(
                **dict(arguments, moe_capacity_factor=4.0, router_combine_fp32=True, fp32_activations=value),
                attention_implementation="torch_order",
                hosts=4,
            )
    on = dict(arguments, moe_capacity_factor=4.0, router_combine_fp32=True, fp32_activations=True)
    with pytest.raises(TypeError, match="attention_implementation"):
        identity_sft_config(**on, hosts=4)
    for value in ("reference", None, ""):
        with pytest.raises(ValueError, match="attention_implementation"):
            identity_sft_config(**on, attention_implementation=value, hosts=4)
    # Config only: an explicit gpu_fa4_cute has no CPU forward.
    for value in ("torch_order", "gpu_fa4_cute"):
        assert identity_sft_config(**on, attention_implementation=value, hosts=4).model.attention_implementation == value
    combined = identity_sft_config(**on, attention_implementation="torch_order", hosts=4)
    assert isinstance(combined.model, SnowballConfig) and combined.model.router_combine_fp32
    assert combined.model.fp32_activations
    check_runtime_recipe(combined, **sources)
    monkeypatch.setattr(TrainerConfig, "num_slices", property(lambda self: 1))
    with pytest.raises(AssertionError):
        check_runtime_recipe(production, **sources)


def test_source_derived_context_scaling_and_export_round_trip():
    sources = source_arguments()
    config = identity_sft_config(
        moe_capacity_factor=4.0,
        router_combine_fp32=False,
        fp32_activations=False,
        attention_implementation="torch_order",
        hosts=1,
        dataset_uri="data",
        cache_uri="cache",
        output_uri="out",
        run_id="test",
        seed=42,
        **sources,
    )
    assert config.train_seq_len == 8192
    assert config.model.max_seq_len == 262144
    assert config.model.qk_mult == 1.75
    assert config.model.qk_mult != SnowballConfig().qk_mult
    assert config.use_hf_model_config
    assert config.hf_generation_eos_token_ids == [128001, 128009]
    exported = config.model.to_hf_config(128256).to_dict()
    assert exported["max_position_embeddings"] == 262144
    assert "max_seq_len" not in exported
    # The capacity and the attention implementation are runtime settings: export drops them and the HF reload
    # restores them.
    assert "moe_capacity_factor" not in exported and "attention_implementation" not in exported
    restored = dataclasses.replace(
        source_model_config(exported), moe_capacity_factor=4.0, attention_implementation="torch_order"
    )
    assert restored == config.model
    receipt = source_metadata_receipt(
        json.dumps(sources["source_config"]).encode(),
        json.dumps(sources["source_generation_config"]).encode(),
    )
    assert receipt["model_config"]["qk_mult"] == 1.75
    assert receipt["model_config"]["max_seq_len"] == 262144
    changed = dict(sources["source_config"], qk_mult=1.7)
    changed_receipt = source_metadata_receipt(
        json.dumps(changed).encode(),
        json.dumps(sources["source_generation_config"]).encode(),
    )
    assert changed_receipt["config_sha256"] != receipt["config_sha256"]
    assert changed_receipt["model_config"] != receipt["model_config"]
    check_source_metadata_receipt(
        receipt,
        json.dumps(sources["source_config"]).encode(),
        json.dumps(sources["source_generation_config"]).encode(),
    )
    with pytest.raises(ValueError, match="Regional"):
        check_source_metadata_receipt(
            receipt,
            json.dumps(changed).encode(),
            json.dumps(sources["source_generation_config"]).encode(),
        )


@pytest.mark.parametrize("change", ["context_alias", "missing_scale", "generation_eos", "model_pad", "global_period"])
def test_ambiguous_or_changed_source_metadata_requires_review(change):
    sources = source_arguments()
    if change == "context_alias":
        sources["source_config"]["max_seq_len"] = 8192
    elif change == "missing_scale":
        del sources["source_config"]["qk_mult"]
    elif change == "generation_eos":
        sources["source_generation_config"]["eos_token_id"] = [128001]
    elif change == "model_pad":
        sources["source_config"]["pad_token_id"] = None
    else:
        sources["source_config"]["global_every"] = 8
    with pytest.raises(ValueError):
        identity_sft_config(
            moe_capacity_factor=4.0,
            router_combine_fp32=False,
            fp32_activations=False,
            attention_implementation="torch_order",
            hosts=1,
            dataset_uri="data",
            cache_uri="cache",
            output_uri="out",
            run_id="test",
            seed=42,
            **sources,
        )
