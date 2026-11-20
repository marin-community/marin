# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adopted exports enter the standard round graph without replaying training."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext, lower
from marin.experiment.cli import graph_handles

from experiments.evaluation.micro210 import serving_config
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.round_runner import adopted_round_evaluation, load_round
from tests.rl.micro210_fixture import policy_fixture_reference

ROUND_PATH = Path(__file__).parents[1] / "rounds" / "mochi-sft-eval.yaml"


@pytest.fixture
def adopted_round(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", policy_fixture_reference())
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = tmp_path / "model"
    model.mkdir()
    manifest = json.dumps({"request": {"config": {"checkpoint": "s3://fixture/checkpoints/step-400407"}}}).encode()
    config = b'{"num_experts": 384, "max_position_embeddings": 262144}'
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    spec = load_round(ROUND_PATH)
    return dataclasses.replace(
        spec,
        adopted_model=dataclasses.replace(
            spec.adopted_model,
            uri=str(model),
            export_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            config_sha256=hashlib.sha256(config).hexdigest(),
        ),
    )


def test_adopted_round_lowers_to_eval_with_pinned_model_and_standard_ep_placement(adopted_round, tmp_path):
    terminal = adopted_round_evaluation(adopted_round)
    stages = graph_handles([terminal])
    assert {step.name for step in stages} == {
        adopted_round.adopted_model.name,
        "evaluation/micro210/prepared",
        terminal.name,
    }
    assert next(step for step in stages if step.name == adopted_round.adopted_model.name).adopt_source == (
        adopted_round.adopted_model.uri
    )
    context = StepContext.for_run(
        str(tmp_path / "evaluation"), str(tmp_path), runtime_args=terminal.runtime_args, deps=terminal.deps
    )
    run = terminal.build_config(context)
    launch = yaml.safe_load(run.skyrl.launch_config_yaml)
    config = launch["skyrl"]
    profile = run.serving_profile
    assert profile.role_plan.uses_expert_parallel_placement
    assert profile.role_plan.num_nodes == 8
    assert profile.role_plan.policy_num_gpus_per_node == 4
    assert config["trainer"]["placement"]["colocate_all"] is False
    for field in ("inference_engine_data_parallel_size", "inference_engine_expert_parallel_size"):
        assert config["generator"][field] == 16
    assert config["generator"]["enforce_eager"] is False
    assert config["generator"]["engine_init_kwargs"]["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert run.evaluation.model.uri == adopted_round.adopted_model.uri
    assert run.evaluation.model.identity == "sha256:" + adopted_round.adopted_model.export_manifest_sha256
    assert run.model_manifest_filename == "export-manifest.json"
    frozen = json.loads(Path(policy_fixture_reference().uri).read_text())["evaluation"]
    for field in ("sampling_params", "eval_sampling_params"):
        assert config["generator"][field] == frozen["config"]["generator"][field]
    assert config["context_budget"] == frozen["config"]["context_budget"]
    assert config["terminal_bench"]["installed_agent"] == frozen["config"]["terminal_bench"]["installed_agent"]
    protocol = json.loads(Path(policy_fixture_reference().uri).read_text())
    data = next(step for step in stages if step.name == "evaluation/micro210/prepared")
    assert data.adopt_config["sha256"] == protocol["prepared_rollouts_sha256"]
    assert adopted_round.adopted_model.export_manifest_sha256 in lower(terminal).fingerprint_payload
    # Regression: schema validation previously accepted a launch that lost the
    # qualified cross-host IB transport and inference-actor NCCL buffer.
    validate_launch(run.skyrl.launch_config_yaml, tmp_path / "compiled")
    compiled = yaml.safe_load((tmp_path / "compiled/compiled.yaml").read_text())
    assert compiled["skyrl"]["trainer"]["policy"]["nccl_buffer_size_bytes"] == 262144
    environment = compiled["runtime"]["task_env"]
    assert environment["NCCL_NET"] == "IB"
    assert environment["NCCL_IB_DISABLE"] == "0"
    assert environment["NCCL_IB_HCA"] == "ibp"


def test_changed_adopted_metadata_is_rejected_before_building_a_gpu_job(adopted_round):
    (Path(adopted_round.adopted_model.uri) / "export-manifest.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="digest changed"):
        adopted_round_evaluation(adopted_round)


def test_serving_profile_cannot_change_frozen_sampling(adopted_round, tmp_path):
    terminal = adopted_round_evaluation(adopted_round)
    context = StepContext.for_run(
        str(tmp_path / "evaluation"), str(tmp_path), runtime_args=terminal.runtime_args, deps=terminal.deps
    )
    run = terminal.build_config(dataclasses.replace(context, is_fingerprint=True))
    profile = dataclasses.replace(run.serving_profile, generator={"eval_sampling_params": {"temperature": 0}})
    frozen = json.loads(Path(policy_fixture_reference().uri).read_text())
    with pytest.raises(ValueError, match=r"protocol fields.*eval_sampling_params"):
        serving_config(frozen, profile)
