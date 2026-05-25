# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from experiments.openreward_qwen3_8b_smoke import (
    build_experiment_config,
    build_openreward_curriculum,
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

    assert lesson.env_config.env_class == "marin.rl.integrations.openreward.env.OpenRewardEnv"
    assert lesson.env_config.env_args["train_manifest_path"] == "/tmp/train-manifest.json"
    assert lesson.env_config.env_args["eval_manifest_path"] == "/tmp/train-manifest.json"
