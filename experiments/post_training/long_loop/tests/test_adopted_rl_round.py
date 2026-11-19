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


@pytest.mark.parametrize("profile,gpus", [("one-rack", 60), ("three-rack", 192)])
@pytest.mark.parametrize("resume", [False, True])
def test_round_adopts_sft_or_native_rl_without_sft_and_keeps_absolute_steps(
    round_inputs, tmp_path, profile, gpus, resume
):
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
    spec = dataclasses.replace(
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
            native_checkpoint=str(tmp_path / "global_step_13") if resume else None,
        ),
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
