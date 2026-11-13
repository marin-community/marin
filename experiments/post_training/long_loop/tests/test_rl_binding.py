# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit run-spec RL bindings preserve supplied data, runtime and training settings."""

from dataclasses import replace
from pathlib import Path

import draccus
import pytest
import yaml
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRolePlan

from experiments.post_training.long_loop import rl
from experiments.post_training.long_loop.spec_pipeline import runtime_execution, stage_execution


@pytest.mark.parametrize("export_selection,finalization", [("best", "any_terminal"), ("latest", "success")])
def test_real_builder_binds_selected_model_and_attempt_trace_paths(monkeypatch, export_selection, finalization):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    recipe = yaml.safe_load((Path(__file__).parents[1] / "specs/rl.yaml").read_text())
    recipe["config"]["trainer"]["hf_export_checkpoint"] = export_selection
    model = ArtifactStep.adopt("tests/adopted-handoff", "2026.11.10", "s3://test/committed-sft-export")
    data = ArtifactStep.adopt("tests/mixed", "2026.11.10", "s3://test/verified-data")
    step = rl.configured_rl_step(
        version="2026.11.10.1",
        model=ArtifactHfModel(model, relative_path="hf", tokenizer_uri="test/tokenizer", tokenizer_revision="a" * 40),
        train_data=(ArtifactDataSource(data, relative_path="agentic-train.parquet"),),
        validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
        config=recipe["config"],
        plan=draccus.decode(SkyRLRolePlan, recipe["role_plan"]),
        execution=stage_execution(
            recipe["execution"],
            runtime_execution(
                yaml.safe_load((Path(__file__).parents[1] / "specs/execution.yaml").read_text())["coordinator"]
            ),
        ),
        name=recipe["name"],
        retained=recipe["retained"],
        seed=recipe["seed"],
        ray_timeouts=recipe["ray_timeouts"],
    )
    context = StepContext.for_run(step.path(), "s3://test/output", runtime_args=step.runtime_args, deps=step.deps)
    built = step.build_config(replace(context, is_fingerprint=True))
    document = yaml.safe_load(built.launch_config_yaml)
    assert document["run"]["finalization"] == finalization
    config = document["skyrl"]
    assert config["trainer"]["hf_export_checkpoint"] == export_selection
    selection = next(c for c in config["trainer"]["callbacks"] if c["type"] == "best_checkpoint")
    assert selection["initial_model_uri"] == document["inputs"]["model"]["uri"]
    assert "PENDING" not in built.launch_config_yaml
    assert config["generator"]["trajectory_retention"]["run_id"] == document["run"]["id"]
    assert config["terminal_bench"]["installed_agent"]["output_path"].startswith(document["artifacts"]["attempts_root"])
