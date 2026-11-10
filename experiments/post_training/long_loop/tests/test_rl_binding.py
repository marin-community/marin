# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit run-spec RL bindings preserve supplied data, runtime and training settings."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import draccus
import pytest
import yaml
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, IrisSkyRLExecution, SkyRLRolePlan

from experiments.post_training.long_loop import rl
from experiments.post_training.long_loop.spec_pipeline import runtime_execution


def test_explicit_binding_preserves_configuration_and_admitted_data(monkeypatch):
    model_step = ArtifactStep.adopt("tests/model", "2026.11.08", "s3://test/model/v1")
    data_step = ArtifactStep.adopt("tests/admitted-mix", "2026.11.09", "s3://test/data/v2")
    model = ArtifactHfModel(model_step, tokenizer_uri="test/tokenizer", tokenizer_revision="a" * 40)
    train = (
        ArtifactDataSource(data_step, relative_path="agentic.parquet"),
        ArtifactDataSource(data_step, relative_path="nonagentic.parquet"),
    )
    validation = (ArtifactDataSource(data_step, relative_path="heldout.parquet"),)
    config = rl.smoke_recipe()
    config["trainer"].update(max_steps=17, dump_training_order=True)
    config["generator"]["trajectory_retention"] = {"output_path": "s3://test/retained", "sample_fraction": 0.1}
    original = deepcopy(config)
    execution = IrisSkyRLExecution(
        cluster="cw-rno2a",
        cluster_config="test/cluster.yaml",
        cpu=24,
        memory="900GB",
        disk="1TB",
        priority="production",
        max_retries=0,
        target_cluster=None,
        parent_cluster_config=None,
        coordinator_timeout_hours=9,
        coordinator_name="explicit-run",
        job_timeout_seconds=32400,
        required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY"),
    )
    plan = rl.SMOKE_PLAN
    calls = []

    def capture(spec, runtime, *, export_hf):
        calls.append((spec, runtime, export_hf))
        return model_step

    monkeypatch.setattr(rl, "skyrl_step", capture)
    rl.configured_rl_step(
        version="2026.11.09.2",
        model=model,
        train_data=train,
        validation_data=validation,
        config=config,
        plan=plan,
        execution=execution,
        name="explicit-rl",
        retained=-1,
        seed=81,
    )
    spec, runtime, exported = calls[0]
    assert yaml.safe_load(spec.config_yaml) == original
    assert config == original
    assert spec.train_data == train and spec.validation_data == validation
    assert spec.model == model and spec.seed == 81
    assert spec.topology.role_plan == plan and spec.topology.num_nodes == plan.num_nodes
    assert spec.retention.resume_checkpoint_count == -1
    assert runtime is execution and exported

    config["trainer"]["rollout_buffer"].update(
        consumed_route_quotas={"agentic": 32, "nonagentic": 32},
        source_routes={},
    )
    with pytest.raises(ValueError, match="route admission is incomplete"):
        rl.configured_rl_step(
            version="2026.11.09.2",
            model=model,
            train_data=train,
            validation_data=validation,
            config=config,
            plan=plan,
            execution=execution,
            name="explicit-rl",
            retained=-1,
            seed=81,
        )
    assert len(calls) == 1


def test_real_builder_binds_selected_model_and_attempt_trace_paths(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    recipe = yaml.safe_load((Path(__file__).parents[1] / "specs/rl.yaml").read_text())
    model = ArtifactStep.adopt("tests/adopted-handoff", "2026.11.10", "s3://test/committed-sft-export")
    data = ArtifactStep.adopt("tests/mixed", "2026.11.10", "s3://test/verified-data")
    step = rl.configured_rl_step(
        version="2026.11.10.1",
        model=ArtifactHfModel(model, relative_path="hf", tokenizer_uri="test/tokenizer", tokenizer_revision="a" * 40),
        train_data=(ArtifactDataSource(data, relative_path="agentic-train.parquet"),),
        validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
        config=recipe["config"],
        plan=draccus.decode(SkyRLRolePlan, recipe["role_plan"]),
        execution=runtime_execution(recipe["execution"]),
        name=recipe["name"],
        retained=recipe["retained"],
        seed=recipe["seed"],
    )
    context = StepContext.for_run(step.path(), "s3://test/output", runtime_args=step.runtime_args, deps=step.deps)
    built = step.build_config(replace(context, is_fingerprint=True))
    document = yaml.safe_load(built.launch_config_yaml)
    config = document["skyrl"]
    selection = next(c for c in config["trainer"]["callbacks"] if c["type"] == "best_checkpoint")
    assert selection["initial_model_uri"] == document["inputs"]["model"]["uri"]
    assert "PENDING" not in built.launch_config_yaml
    assert config["generator"]["trajectory_retention"]["run_id"] == document["run"]["id"]
    assert config["terminal_bench"]["installed_agent"]["output_path"].startswith(document["artifacts"]["attempts_root"])
