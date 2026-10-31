# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cohort bindings and environment validation gates on Mini840 GPU submission."""

import json

import pytest
from click.testing import CliRunner

from experiments.evaluation.mini_policy_runtime import launch
from experiments.evaluation.mini_policy_runtime.cohort import cohort_model


@pytest.mark.parametrize("defect", ["missing_environment", "wrong_dataset_scope"])
def test_incomplete_release_cannot_submit_gpu_jobs(tmp_path, monkeypatch, defect):
    manifest = {
        "required_source_keys": ["dockerfile:one", "dockerfile:two"],
        "expected_source_keys": 2,
        "validated_source_keys": ["dockerfile:one", "dockerfile:two"],
        "failures": [],
        "data_uri": "s3://example/full",
    }
    if defect == "missing_environment":
        manifest["validated_source_keys"].remove("dockerfile:two")
    else:
        manifest["data_uri"] = "s3://example/smoke"
    path = tmp_path / "environments.json"
    path.write_text(json.dumps(manifest))

    def forbidden_gpu_artifact(*args, **kwargs):
        pytest.fail("A GPU artifact was constructed for an unusable environment map")

    monkeypatch.setattr(launch, "skyrl_step", forbidden_gpu_artifact)
    result = CliRunner().invoke(
        launch.main,
        [
            "--version",
            "2026.10.31",
            "--run",
            "--model",
            "qwen36-35b-a3b",
            "--model-uri",
            "s3://example/model",
            "--data-uri",
            "s3://example/full",
            "--environment-manifest",
            str(path),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    "key,tool_parser",
    [
        ("qwen36-35b-a3b", "qwen3_coder"),
        ("qwen35-35b-a3b", "qwen3_coder"),
        ("qwen3-30b-a3b", "hermes"),
        ("nemotron3-nano-30b-a3b", "qwen3_coder"),
        ("snowball-67b-a2b", "hermes"),
    ],
)
def test_cohort_bindings_reach_the_generator(key, tool_parser):
    config, _ = launch.recipe(cohort_model(key), {}, {}, {}, "http://judge.invalid/v1", "DAYTONA_EVAL_API_KEY")
    generator = config["generator"]
    assert generator["engine_init_kwargs"] == {
        "enable_auto_tool_choice": True,
        "tool_call_parser": tool_parser,
    }
    default_kwargs = generator["chat_template_kwargs"]
    template = cohort_model(key).chat_template_kwargs or {}
    assert default_kwargs == template
    # The frozen execution profile is model-independent.
    assert generator["sampling_params"] == {"temperature": 0.6, "top_p": 0.95, "logprobs": 0}
    assert config["context_budget"] == {
        "request_window_tokens": 65536,
        "max_new_tokens_per_turn": 16384,
        "max_turns": 1024,
    }
