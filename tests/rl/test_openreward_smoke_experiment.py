# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from experiments.openreward_qwen3_8b_smoke import (
    QWEN3_8B_OPENREWARD,
    build_experiment_config,
    build_openreward_curriculum,
    required_env_var_names,
)


def _args(**overrides) -> SimpleNamespace:
    defaults = {
        "experiment_name_suffix": "openreward-smoke",
        "project_name": "marin_openreward",
        "num_train_steps": 50,
        "n_prompts": 8,
        "num_rollout_workers": 1,
        "train_tpu_type": "v5p-8",
        "inference_tpu_type": "v5p-8",
        "train_ram": "400g",
        "inference_ram": "400g",
        "zone": None,
        "max_input_tokens": 4096,
        "max_output_tokens": 1024,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_required_env_var_names_validates_presence_and_deduplicates(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")

    secret_names = required_env_var_names(["OPENAI_API_KEY", "OPENAI_API_KEY"])

    assert secret_names == ["OPENAI_API_KEY"]


def test_required_env_var_names_raises_for_missing_values(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variable"):
        required_env_var_names(["OPENAI_API_KEY"])


def test_build_experiment_config_uses_rl_dependency_group():
    config = build_experiment_config(_args())

    assert config.model_config == QWEN3_8B_OPENREWARD
    assert config.model_config.type == "qwen"
    assert config.model_config.pip_dependency_groups == ["rl"]
    assert config.inflight_weight_updates is False


def test_build_openreward_curriculum_reuses_train_manifest_for_eval():
    experiment_config = build_experiment_config(_args())

    curriculum = build_openreward_curriculum(
        "run-123",
        experiment_config,
        train_manifest_path="/tmp/train-manifest.json",
        eval_manifest_path=None,
        base_url="https://openreward.example",
        variant="math",
        api_key_env_var="OPENREWARD_API_KEY",
        secret_env_vars=["OPENAI_API_KEY"],
        eval_frequency=3,
    )

    lesson = curriculum.lessons["openreward"]

    assert curriculum.actor_name == "curriculum-run-123"
    assert curriculum.eval_frequency == 3
    assert lesson.env_config.env_class == "marin.rl.environments.openreward_env.OpenRewardEnv"
    assert lesson.env_config.env_args["train_manifest_path"] == "/tmp/train-manifest.json"
    assert lesson.env_config.env_args["eval_manifest_path"] == "/tmp/train-manifest.json"
    assert lesson.env_config.env_args["base_url"] == "https://openreward.example"
    assert lesson.env_config.env_args["variant"] == "math"
    assert lesson.env_config.env_args["api_key_env_var"] == "OPENREWARD_API_KEY"
    assert lesson.env_config.env_args["secret_env_vars"] == ["OPENAI_API_KEY"]
