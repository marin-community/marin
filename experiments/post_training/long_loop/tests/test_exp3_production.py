# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from datetime import timedelta

import draccus
import jax.numpy as jnp
import pytest
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import StepContext

from experiments.post_training.long_loop import exp3_production
from experiments.post_training.long_loop.exp3_production import GEOMETRY, production_sft_recipe, read_data_release
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec
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


def test_full_recipe_survives_the_native_worker_boundary_without_smoke_geometry(preview_release, monkeypatch):
    root = preview_release.parent
    initializer = json.loads((root / "initializer.json").read_text())
    monkeypatch.setattr(exp3_production, "REFERENCE_INPUTS", str(root))
    monkeypatch.setattr(exp3_production, "INITIALIZER_RECEIPT", initializer["initializer_receipt"])
    monkeypatch.setattr(exp3_production, "INITIALIZER_SHA256", initializer["initializer_receipt_sha256"])
    release = read_data_release(str(preview_release), require_ready=False)
    recipe = production_sft_recipe(release)
    spec = exp3_sft_spec("exp3-production-test", "2026.11.16.48", recipe)
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    step = sft_step(spec, resources)
    context = StepContext.for_run(str(root / "trained"), str(root), runtime_args=step.runtime_args)
    pod = step.build_config(context)
    request = levanter_train_request(pod, task_timeout_seconds=72000, priority=priority_band_value("production"))
    [worker] = request.entrypoint.callable_entrypoint.args
    config = draccus.decode(TrainLmConfig, json.loads(json.dumps(draccus.encode(worker.train_config))))
    assert config.trainer.num_train_steps == 746
    assert config.trainer.train_batch_size * config.train_seq_len == 67_108_864
    ici, dcn = config.trainer.mesh.axis_shapes(num_devices=256, num_slices=32)
    assert ici["expert"] == 8 and dcn["context"] == 32
    assert request.resources.device.count * request.resources.replicas == 256
    assert request.timeout.to_seconds() == 72000
    assert config.trainer.training_budget.duration == timedelta(hours=20)
    assert config.trainer.checkpointer.keep_last_temporary_checkpoints == 3
    assert config.trainer.mp.param_dtype == jnp.float32
    assert config.optimizer_num_train_steps == 4750
    assert config.trainer.seed == config.data_seed == 0
    assert draccus.encode(config.data)["train_weights"] == release.data["train_weights"]
    assert config.hf_save_steps > 746
