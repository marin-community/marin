# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import draccus
import jax.numpy as jnp
import pytest
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.checkpoint import Checkpointer, load_checkpoint, load_checkpoint_metadata
from levanter.main.train_lm import TrainLmConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import StepContext

from experiments.post_training.long_loop import exp3_production
from experiments.post_training.long_loop.exp3_production import GEOMETRY, production_sft_recipe, read_data_release
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec, whole_run_lr_receipt
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
    return production_sft_recipe(release)


@pytest.mark.parametrize("schedule_horizon", [None, 360])
def test_full_recipe_survives_the_native_worker_boundary_without_smoke_geometry(
    preview_release, production_recipe, schedule_horizon
):
    root = preview_release.parent
    release = read_data_release(str(preview_release), require_ready=False)
    recipe = production_recipe
    steps = schedule_horizon or 746
    if schedule_horizon is not None:
        recipe["stop_after_steps"] = steps
        recipe["trainer"]["num_train_steps"] = steps
        recipe["optimizer_schedule"] = {
            "horizon": steps,
            "warmup": 0,
            "decay": steps,
            "min_lr_ratio": 0.1,
            "lr_schedule": "linear",
        }
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
    assert ici["expert"] == 8 and dcn["context"] == 32
    assert request.resources.device.count * request.resources.replicas == 256
    assert request.timeout.to_seconds() == 72000
    assert config.trainer.training_budget.duration == timedelta(hours=20)
    assert config.trainer.checkpointer.keep_last_temporary_checkpoints == 3
    assert config.trainer.mp.param_dtype == jnp.float32
    assert config.optimizer_num_train_steps == (schedule_horizon or 4750)
    assert config.trainer.seed == config.data_seed == 0
    assert draccus.encode(config.data)["train_weights"] == release.data["train_weights"]
    assert config.hf_save_steps > steps
    if schedule_horizon is not None:
        for peak in (config.optimizer.learning_rate, config.optimizer.adam_lr):
            scheduler = config.optimizer.lr_scheduler(config.optimizer_num_train_steps, override_lr=peak)
            for index in (0, 90, 180, 270, 359):
                assert float(scheduler(index)) == pytest.approx(peak * (1 - 0.9 * index / 360), rel=1e-6)


def test_whole_run_preflight_rejects_the_inherited_flat_prefix_and_wrong_decay(production_recipe):
    spec = exp3_sft_spec("lr-admission", "fixture", production_recipe)
    with pytest.raises(ValueError, match="horizon must equal num_train_steps"):
        whole_run_lr_receipt(spec)
    flat = replace(spec, optimizer_num_train_steps=spec.num_train_steps)
    with pytest.raises(ValueError, match="entire horizon"):
        whole_run_lr_receipt(flat)


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
