# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import draccus
import pytest
import yaml
from fray.types import ResourceConfig
from levanter.checkpoint import save_checkpoint
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.sft import SMOKE_CONFIG
from experiments.post_training.long_loop.shared_sft import (
    export_shared_sft,
    legacy_sft_step,
    select_shared_sft,
    shared_sft_spec,
)
from experiments.sft.configs.snowball_h100 import SPEC as SNOWBALL_SFT
from experiments.sft.launcher import sft_step

SPEC = Path(__file__).parents[1] / "specs/round1.yaml"


def test_native_sft_retains_sources_and_token_schedule_while_using_shared_defaults(tmp_path):
    resolved = resolve_spec(SPEC, smoke=False)
    spec = shared_sft_spec("test/sft", "2026.11.13.1", resolved["sft"])
    step = sft_step(spec, draccus.decode(ResourceConfig, resolved["sft"]["resources"]))
    context = StepContext.for_run(str(tmp_path), str(tmp_path), runtime_args=step.runtime_args)
    pod = step.build_config(context)
    config = pod.train_config
    assert pod.env_vars["XLA_FLAGS"].split() == resolved["sft"]["xla_flags"]
    assert config.optimizer == SNOWBALL_SFT.optimizer
    assert config.train_seq_len == resolved["sft"]["seq_len"]
    assert config.trainer.train_batch_size == resolved["sft"]["batch_size"]
    assert config.trainer.per_device_parallelism == resolved["sft"]["per_device_parallelism"]
    assert config.trainer.num_train_steps * config.train_seq_len * config.trainer.train_batch_size == 50063212544
    before = resolved["datasets"]["sft"]["data"]
    after = config.data
    assert list(after.components) == list(before["components"])
    for (old_step, old_weights), (new_step, new_weights) in zip(
        before["train_weights"], after.train_weights, strict=True
    ):
        assert old_weights == new_weights
        assert old_step * 67108864 == new_step * config.train_seq_len * config.trainer.train_batch_size
    encoded = draccus.encode(after)
    for name, original in before["components"].items():
        children = original["children"] if original["type"] == "concat" else {name: original}
        updated = encoded["components"][name]
        updated_children = updated["children"] if original["type"] == "concat" else {name: updated}
        for child_name, child in children.items():
            assert updated_children[child_name]["cache_dir"] == child["cache_dir"]
            expected = "serialized_chat" if name.startswith("sft/") else "text"
            assert updated_children[child_name]["format"]["type"] == expected


def test_legacy_grug_requires_acknowledgement_and_keeps_original_recipe(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    base = ArtifactStep.adopt("test/base", "2026.11.13.1", source=str(tmp_path / "base"))
    with pytest.raises(ValueError, match="i-know-i-am-using-a-deprecated-config-format"):
        legacy_sft_step("test/sft", "2026.11.13.1", recipe, base, acknowledge=False)
    step = legacy_sft_step("test/sft", "2026.11.13.1", recipe, base, acknowledge=True)
    context = StepContext.for_run(str(tmp_path / "run"), str(tmp_path), deps=step.deps, runtime_args=step.runtime_args)
    actual = step.build_config(context)
    assert actual.recipe["data"] == recipe["data"]
    assert actual.recipe["optimizer"] == recipe["optimizer"]
    assert actual.recipe["trainer"] == recipe["trainer"]
    assert actual.recipe["model"] == recipe["model"]


def test_native_sft_validates_real_multinode_layout_before_submitting(tmp_path):
    resolved = resolve_spec(SPEC, smoke=True)
    recipe = resolved["sft"]
    broken = copy.deepcopy(recipe)
    broken["mesh"]["axes"]["context"] = 32
    broken["mesh"]["dcn_axes"] = {"replica_dcn": -1}
    with pytest.raises(ValueError, match="ICI product 256 must equal devices_per_slice 8"):
        shared_sft_spec("test/sft", "2026.11.13.4", broken)
    spec = shared_sft_spec("test/sft", "2026.11.13.4", recipe)
    step = sft_step(spec, draccus.decode(ResourceConfig, recipe["resources"]))
    config = step.build_config(StepContext.for_run(str(tmp_path), str(tmp_path), runtime_args=step.runtime_args))
    ici, dcn = config.train_config.trainer.mesh.axis_shapes(num_devices=256, num_slices=32)
    assert ici["expert"] == 8
    assert dcn["context"] == 32
    assert dcn["replica_dcn"] == 1
    assert config.train_config.trainer.compute_axis_mapping["position"] == "context"


def test_native_handoff_preserves_exported_weights_and_declares_rl_context(tmp_path):
    root = tmp_path / "trained"
    hf = root / "hf/step-2"
    hf.mkdir(parents=True)
    save_checkpoint(
        {"model": 1, "step": 3}, step=2, checkpoint_path=str(root / "checkpoints/step-2"), is_temporary=False
    )
    select_shared_sft(output=str(root), completed_updates=3)
    selection = json.loads((root / "selected.json").read_text())
    assert selection["step"] == 2 and selection["completed_updates"] == 3
    (root / "consumed-config.json").write_text('{"format":"SFTSpec"}')
    (hf / "config.json").write_text('{"max_position_embeddings":32768}')
    (hf / "model.safetensors.index.json").write_text('{"weight_map":{"model.weight":"weight.safetensors"}}')
    (hf / "weight.safetensors").write_bytes(b"exact-selected-weights")
    export_shared_sft(
        output=str(tmp_path / "handoff"), trained=str(root), context_limit=262144, source_revision="a" * 40
    )
    destination = tmp_path / "handoff/hf"
    assert (destination / "weight.safetensors").read_bytes() == b"exact-selected-weights"
    assert json.loads((destination / "config.json").read_text())["max_position_embeddings"] == 262144
    assert json.loads((tmp_path / "handoff/handoff.json").read_text())["native_checkpoint"] == str(
        root / "checkpoints/step-2"
    )


@pytest.mark.parametrize("per_device_parallelism", [1, 2])
def test_large_global_batch_preserves_explicit_microbatch_and_native_mesh(tmp_path, per_device_parallelism):
    resolved = resolve_spec(SPEC, smoke=False)
    recipe = resolved["sft"]
    recipe.update(
        batch_size=256,
        per_device_parallelism=per_device_parallelism,
        target_positions=204 * 256 * 262144,
        stop_after_steps=204,
    )
    spec = shared_sft_spec("test/sft", "2026.11.13.11", recipe)
    step = sft_step(spec, draccus.decode(ResourceConfig, recipe["resources"]))
    pod = step.build_config(StepContext.for_run(str(tmp_path), str(tmp_path), runtime_args=step.runtime_args))
    trainer = pod.train_config.trainer
    assert trainer.train_batch_size == 256
    assert trainer.per_device_parallelism == per_device_parallelism
    assert trainer.num_train_steps == 204
    assert pod.train_config.optimizer == SNOWBALL_SFT.optimizer
    ici, dcn = trainer.mesh.axis_shapes(num_devices=256, num_slices=32)
    assert ici["expert"] == 8 and dcn["context"] == 32 and dcn["replica_dcn"] == 1
    assert trainer.compute_axis_mapping["batch"] == ("replica_dcn", "data", "expert")
    assert trainer.compute_axis_mapping["position"] == "context"
    assert pod.train_config.data.train_weights[1][0] == 250
