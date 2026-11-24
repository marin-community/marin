"""Credential selection for task backends and remote graders."""

import pytest
from omegaconf import OmegaConf

import cloud.iris.iris_backend as launcher


@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("backend", ["remote_judge", "daytona"])
def test_backend_secret_is_resolved_for_training_without_mutating_launch_config(monkeypatch, export, backend):
    config = OmegaConf.create(
        {
            "runtime": {
                "entrypoint": launcher.CHECKPOINT_EXPORT_ENTRYPOINT if export else "train",
                "task_env": {"SETTING": 1},
            },
            "skyrl": {"environment": {"task_sessions": {backend: {"api_key_env": "JUDGE_TEST_KEY"}}}},
        }
    )
    serialized = OmegaConf.to_yaml(config)
    monkeypatch.setenv("JUDGE_TEST_KEY", "secret-at-submission")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")
    expected = {"SETTING": "1"}
    if not export:
        expected["JUDGE_TEST_KEY"] = "secret-at-submission"
    assert launcher.config_task_environment(config) == expected
    assert OmegaConf.to_yaml(config) == serialized
    monkeypatch.delenv("JUDGE_TEST_KEY")
    if export:
        assert launcher.config_task_environment(config) == expected
    else:
        with pytest.raises(ValueError, match="JUDGE_TEST_KEY"):
            launcher.config_task_environment(config)


def test_task_rollouts_forward_separate_training_and_evaluation_credentials(monkeypatch):
    config = OmegaConf.create(
        {
            "runtime": {"entrypoint": "train", "task_env": {}},
            "skyrl": {
                "environment": {
                    "task_sessions": {
                        "daytona": {"api_key_env": "DAYTONA_RL_API_KEY", "eval_api_key_env": "EVAL_ORG_KEY"}
                    }
                },
            },
        }
    )
    serialized = OmegaConf.to_yaml(config)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-credential")
    monkeypatch.setenv("EVAL_ORG_KEY", "eval-credential")
    monkeypatch.setenv("DAYTONA_API_KEY", "generic-credential")
    assert launcher.config_task_environment(config) == {
        "DAYTONA_RL_API_KEY": "rl-credential",
        "EVAL_ORG_KEY": "eval-credential",
    }
    assert OmegaConf.to_yaml(config) == serialized
    monkeypatch.delenv("EVAL_ORG_KEY")
    with pytest.raises(ValueError, match="EVAL_ORG_KEY"):
        launcher.config_task_environment(config)
