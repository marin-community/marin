# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit run-spec RL bindings preserve supplied data, runtime and training settings."""

from copy import deepcopy

import pytest
import yaml
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, IrisSkyRLExecution

from experiments.post_training.long_loop import rl


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
