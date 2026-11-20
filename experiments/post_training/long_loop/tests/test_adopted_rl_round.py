# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles

from experiments.post_training.long_loop.archive import assert_best_rl_checkpoint
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.round_runner import adopted_round_rl, load_round

ROUND_DIR = Path(__file__).parents[1] / "rounds"


@pytest.fixture
def round_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "output"))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = tmp_path / "model"
    model.mkdir()
    manifest = json.dumps({"request": {"config": {"checkpoint": "s3://source/checkpoints/step-400407"}}}).encode()
    config = json.dumps(
        {"grugmoe_artifact_schema_version": 3, "num_hidden_layers": 96, "num_attention_heads": 96, "num_experts": 384}
    ).encode()
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"pinned training fixture")
    (data / "validation.parquet").write_bytes(b"pinned validation fixture")
    return model, data


@pytest.fixture
def local_round(round_inputs, profile):
    model, data = round_inputs
    spec = load_round(ROUND_DIR / f"mochi-{profile}.yaml")
    pins = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("config.json", "export-manifest.json")
    }
    train = {
        "uri": str(data),
        "file": "train.parquet",
        "sha256": hashlib.sha256((data / "train.parquet").read_bytes()).hexdigest(),
    }
    validation = {
        "uri": str(data),
        "file": "validation.parquet",
        "sha256": hashlib.sha256((data / "validation.parquet").read_bytes()).hexdigest(),
    }
    return dataclasses.replace(
        spec,
        adopted_model=dataclasses.replace(
            spec.adopted_model,
            uri=str(model),
            config_sha256=pins["config.json"],
            export_manifest_sha256=pins["export-manifest.json"],
        ),
        rl=dataclasses.replace(
            spec.rl,
            train_data={"mixed": train},
            validation_data=validation,
            max_steps=15,
        ),
    )


@pytest.mark.parametrize("profile,gpus", [("one-rack", 60), ("three-rack", 192)])
@pytest.mark.parametrize("resume", [False, True])
def test_round_adopts_sft_or_native_rl_without_sft_and_keeps_absolute_steps(
    round_inputs, local_round, tmp_path, gpus, resume
):
    _, data = round_inputs
    spec = dataclasses.replace(
        local_round,
        rl=dataclasses.replace(local_round.rl, native_checkpoint=str(tmp_path / "global_step_13") if resume else None),
    )
    terminal = adopted_round_rl(spec)
    stages = graph_handles([terminal])
    assert all(step.adopt_source for step in stages if step is not terminal)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    assert launch["skyrl"]["trainer"]["max_steps"] == 15
    assert launch["iris"]["allocation"]["num_nodes"] * launch["iris"]["allocation"]["gpus_per_node"] == gpus
    assert launch["iris"]["timeout"] == 48 * 3600
    callbacks = {cb["type"]: cb for cb in launch["skyrl"]["trainer"]["callbacks"]}
    assert callbacks["checkpoint"]["save_steps"] == 2
    assert callbacks["wall_time_budget"]["seconds"] == 36000
    assert callbacks["best_checkpoint"]["minimum_scored_tasks"] == 108
    assert callbacks["best_checkpoint"]["selection_eligibility"] == "covered"
    if resume:
        assert launch["inputs"]["starting_checkpoint"]["uri"].endswith("/global_step_13")
    else:
        assert "starting_checkpoint" not in launch["inputs"]
    (data / "train.parquet").write_bytes(b"changed after adoption")
    with pytest.raises(ValueError, match="Round data changed"):
        terminal.build_config(context)


@pytest.mark.parametrize("profile", ["one-rack", "three-rack"])
def test_round_best_and_final_native_checkpoints_use_permanent_storage(
    local_round, tmp_path, permanent_path, monkeypatch
):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    terminal = adopted_round_rl(local_round)
    output = permanent_path / "round"
    context = StepContext.for_run(
        str(output), str(output.parent), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    config = terminal.build_config(context)
    launch = yaml.safe_load(config.launch_config_yaml)
    assert config.output.checkpoint_root == str(output / "checkpoints")
    assert launch["artifacts"]["checkpoint_root"] == config.output.checkpoint_root
    assert str(tmp_path / "temporary") in config.output.attempts_root

    validate_launch(config.launch_config_yaml, tmp_path / "composed", model_config=b"{}")
    compiled = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())
    checkpoint_root = Path(compiled["skyrl"]["trainer"]["ckpt_path"])
    inventory = {"selected": str(checkpoint_root / "global_step_14"), "final": str(checkpoint_root / "global_step_15")}
    for uri in inventory.values():
        path = Path(uri)
        (path / "policy").mkdir(parents=True)
        (path / "trainer_state.pt").write_bytes(b"native optimizer state fixture")
        (path / "policy" / "weights.distcp").write_bytes(b"native policy fixture")
    receipt = assert_best_rl_checkpoint(str(tmp_path / "archive"), inventory)
    assert receipt["selected"]["uri"] == str(output / "checkpoints" / "global_step_14")
    assert receipt["final"]["uri"] == str(output / "checkpoints" / "global_step_15")
