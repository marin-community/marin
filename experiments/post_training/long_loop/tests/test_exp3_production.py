# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import draccus
import jax.numpy as jnp
import jmp
import pytest
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.checkpoint import Checkpointer, load_checkpoint, load_checkpoint_metadata
from levanter.main.train_lm import TrainLmConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import StepContext

from experiments.post_training.long_loop import exp3_production
from experiments.post_training.long_loop.execution_policy import SFT_TRAINING_DURATION
from experiments.post_training.long_loop.exp3_production import GEOMETRY, production_sft_recipe, read_data_release
from experiments.post_training.long_loop.exp3_recipe import SftLrSchedule, exp3_sft_spec, whole_run_lr_receipt
from experiments.post_training.long_loop.round_runner import load_round
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.shared_sft import select_shared_sft, shared_sft_spec
from experiments.post_training.long_loop.tests.test_initializer_stage import prepared_sources as prepared_sources
from experiments.post_training.long_loop.tests.test_initializer_stage import staged_initializer as staged_initializer
from experiments.sft.launcher import levanter_train_request, sft_step


@pytest.fixture
def preview_release(request):
    root = request.getfixturevalue("prepared_sources")
    data = json.loads((root / "data.json").read_text())
    data["cache_catalog"] = str(root / "cache_catalog.json")
    payloads = {
        "data.json": json.dumps(data).encode(),
        "cache_catalog.json": b'{"version":1,"splits":{}}',
        "source_loss.json": (root / "source-loss.json").read_bytes(),
        "train_weights.json": json.dumps(data["train_weights"]).encode(),
        "retention-by-source.csv": b"source,retained\nsource,64\n",
        "retention-by-transform.csv": b"transform,retained\nnone,64\n",
        "mixture-rationale.json": b"{}",
    }
    pins = {}
    for name, content in payloads.items():
        path = root / name
        path.write_bytes(content)
        pins[name] = {"uri": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    release = root / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_ready": False,
                "gpu_training_approved": False,
                "artifacts": pins,
                "geometry": asdict(GEOMETRY),
                "num_validation_sequences": data["num_validation_sequences"],
            }
        )
    )
    return release


def test_preview_metadata_can_be_resolved_but_not_admitted_and_member_changes_are_rejected(preview_release):
    loaded = read_data_release(str(preview_release), require_ready=False)
    assert loaded.data["num_validation_sequences"] == {"source": 8}
    with pytest.raises(ValueError, match="not admitted"):
        read_data_release(str(preview_release), require_ready=True)
    retention = preview_release.parent / "retention-by-source.csv"
    retention.write_bytes(b"source,retained\nsource,63\n")
    with pytest.raises(ValueError, match="member changed"):
        read_data_release(str(preview_release), require_ready=False)


@pytest.fixture
def production_recipe(preview_release, monkeypatch):
    root = preview_release.parent
    initializer = json.loads((root / "initializer.json").read_text())
    monkeypatch.setattr(exp3_production, "REFERENCE_INPUTS", str(root))
    monkeypatch.setattr(exp3_production, "INITIALIZER_RECEIPT", initializer["initializer_receipt"])
    monkeypatch.setattr(exp3_production, "INITIALIZER_SHA256", initializer["initializer_receipt_sha256"])
    release = read_data_release(str(preview_release), require_ready=False)
    return production_sft_recipe(release, initial_eval_inputs=exp3_production.REFERENCE_INPUTS)


@pytest.mark.parametrize("precision", ["p=f32,c=bf16", "p=bf16,c=bf16"])
@pytest.mark.parametrize("schedule_horizon,warmup_steps", [(None, 0), (360, 0), (360, 30)])
def test_full_recipe_survives_the_native_worker_boundary_without_smoke_geometry(
    preview_release, production_recipe, schedule_horizon, warmup_steps, precision
):
    root = preview_release.parent
    release = read_data_release(str(preview_release), require_ready=False)
    recipe = production_recipe
    recipe["trainer"]["mp"] = precision
    steps = schedule_horizon or 746
    if schedule_horizon is not None:
        recipe["stop_after_steps"] = steps
        recipe["trainer"]["num_train_steps"] = steps
        recipe["optimizer_schedule"] = {
            "horizon": steps,
            "warmup": warmup_steps,
            "decay": steps - warmup_steps,
            "min_lr_ratio": 0.1,
            "lr_schedule": "linear",
        }
    if warmup_steps:
        round_spec = load_round(Path(__file__).parents[1] / "rounds/exp5-setup.yaml")
        recipe["mesh"] = round_spec.sft.mesh
        recipe["trainer"]["mesh"] = round_spec.sft.mesh
        recipe["model"]["expert_weight_sharding"] = round_spec.sft.expert_weight_sharding
    spec = exp3_sft_spec("exp3-production-test", "2026.11.16.48", recipe)
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    step = sft_step(spec, resources)
    context = StepContext.for_run(str(root / "trained"), str(root), runtime_args=step.runtime_args)
    pod = step.build_config(context)
    request = levanter_train_request(pod, task_timeout_seconds=72000, priority=priority_band_value("production"))
    [worker] = request.entrypoint.callable_entrypoint.args
    config = draccus.decode(TrainLmConfig, json.loads(json.dumps(draccus.encode(worker.train_config))))
    assert config.trainer.num_train_steps == steps
    assert config.trainer.train_batch_size * config.train_seq_len == 67_108_864
    ici, dcn = config.trainer.mesh.axis_shapes(num_devices=256, num_slices=32)
    if warmup_steps:
        assert ici["context"] == 8 and "data" not in ici and dcn == {"replica_dcn": 1, "data": 32}
        assert ici.get("expert", 1) == 1
    else:
        assert ici["expert"] == 8 and dcn["context"] == 32
    assert request.resources.device.count * request.resources.replicas == 256
    assert request.timeout.to_seconds() == SFT_TRAINING_DURATION.total_seconds()
    assert config.trainer.training_budget.duration == SFT_TRAINING_DURATION
    assert config.trainer.checkpointer.keep_last_temporary_checkpoints == 3
    assert config.trainer.mp == jmp.get_policy(precision)
    assert config.optimizer_num_train_steps == (schedule_horizon or 4750)
    assert config.trainer.seed == config.data_seed == 0
    assert draccus.encode(config.data)["train_weights"] == release.data["train_weights"]
    assert config.hf_save_steps > steps
    if schedule_horizon is not None:
        for peak in (config.optimizer.learning_rate, config.optimizer.adam_lr):
            scheduler = config.optimizer.lr_scheduler(config.optimizer_num_train_steps, override_lr=peak)
            for index in (0, 15, 30, 180, 359):
                ratio = (
                    index / warmup_steps
                    if index < warmup_steps
                    else 1 - 0.9 * (index - warmup_steps) / (360 - warmup_steps)
                )
                assert float(scheduler(index)) == pytest.approx(peak * ratio, rel=1e-6)


def test_sft_lr_receipt_rejects_a_mismatch_with_the_declared_schedule(production_recipe):
    spec = exp3_sft_spec("lr-admission", "fixture", production_recipe)
    declared = SftLrSchedule(spec.num_train_steps, 0, spec.num_train_steps, 0.1, "linear")
    with pytest.raises(ValueError):
        whole_run_lr_receipt(spec, schedule=declared)
    flat = replace(spec, optimizer_num_train_steps=spec.num_train_steps)
    with pytest.raises(ValueError):
        whole_run_lr_receipt(flat, schedule=declared)


@pytest.mark.integration
@pytest.mark.parametrize("final_step", [4, 5])
@pytest.mark.parametrize("profile", ["snowball_exp3", "snowball_h100"])
def test_production_checkpoints_obey_two_hours_and_select_the_forced_final_state(
    tmp_path, production_recipe, final_step, profile
):
    if profile == "snowball_exp3":
        policy = draccus.decode(TrainerConfig, production_recipe["trainer"]).checkpointer
    else:
        resolved = resolve_spec(Path(__file__).parents[1] / "specs/round1.yaml", smoke=False)
        policy = shared_sft_spec("future-sft", "fixture", resolved["sft"]).checkpointer
    now = datetime(2026, 1, 1)
    permanent, temporary = tmp_path / "checkpoints", tmp_path / "resume"
    checkpointer = Checkpointer(
        permanent,
        policy.save_interval,
        [],
        temporary_base_path=temporary,
        keep_last_temporary_checkpoints=policy.keep_last_temporary_checkpoints,
        dt_now_injection=lambda: now,
    )
    checkpointer.on_step(tree={"model": jnp.asarray(0.0)}, step=0)
    now += timedelta(hours=2, seconds=-1)
    checkpointer.on_step(tree={"model": jnp.asarray(1.0)}, step=1)
    checkpointer.wait_until_finished()
    assert not list(tmp_path.rglob("metadata.json"))

    now += timedelta(seconds=1)
    checkpointer.on_step(tree={"model": jnp.asarray(2.0)}, step=2)
    checkpointer.wait_until_finished()
    assert load_checkpoint_metadata(temporary / "step-2")["is_temporary"]

    now += timedelta(hours=2, seconds=-1)
    checkpointer.on_step(tree={"model": jnp.asarray(3.0)}, step=3)
    checkpointer.wait_until_finished()
    assert not (temporary / "step-3/metadata.json").exists()
    now += timedelta(seconds=1)
    checkpointer.on_step(tree={"model": jnp.asarray(4.0)}, step=4)
    checkpointer.wait_until_finished()
    assert load_checkpoint_metadata(temporary / "step-4")["step"] == 4

    now += timedelta(minutes=1)
    final_state = {"model": jnp.asarray(float(final_step))}
    checkpointer.save_final_checkpoint(tree=final_state, step=final_step)
    checkpointer.wait_until_finished()
    select_shared_sft(output=str(tmp_path), completed_updates=final_step + 1)
    selected = json.loads((tmp_path / "selected.json").read_text())
    assert selected["checkpoint"] == str(permanent / f"step-{final_step}")
    restored = load_checkpoint({"model": jnp.asarray(0.0)}, selected["checkpoint"])
    assert restored["model"].item() == final_step
    assert load_checkpoint_metadata(selected["checkpoint"])["is_temporary"] is False


def test_frozen_packing_field_constructs_native_dataset_without_changing_publication():
    frozen = {"components": {"c4": {"cache_dir": "/cache/c4", "packing_lookahead": 128}}}
    translated = exp3_production.native_sft_data(frozen)
    native = draccus.decode(exp3_production.LmDataConfig, translated)
    assert native.components["c4"].pack_lookahead_window == 128
    assert frozen["components"]["c4"] == {"cache_dir": "/cache/c4", "packing_lookahead": 128}
    translated["components"]["c4"]["unknown_packing_field"] = 4
    with pytest.raises(draccus.utils.DecodingError, match="unknown_packing_field"):
        draccus.decode(exp3_production.LmDataConfig, translated)
