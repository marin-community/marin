"""Credential selection for task backends and remote graders."""

import pytest
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import OmegaConf
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaImageBuild, DaytonaMachineFactory
from shellbox.machine import Backend
from skyrlrecipe.declared import DAYTONA_EVAL_BACKEND

import cloud.iris.iris_backend as launcher


def test_evaluation_provisioning_can_share_training_credentials(monkeypatch):
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "shared-test-key")
    monkeypatch.setattr(DaytonaConnection, "client_factory", lambda connection: lambda: connection)
    sessions = OmegaConf.create(
        {
            "daytona": {
                "api_key_env": "DAYTONA_RL_API_KEY",
                "ttl_minutes": 360,
                "image_build": "snapshot",
                "eval_image_build": "declarative",
            }
        }
    )

    factories = task_machine_factories(sessions)

    monkeypatch.setenv("DAYTONA_RL_API_KEY", "changed-after-construction")
    training, evaluation = factories[Backend.DAYTONA], factories[DAYTONA_EVAL_BACKEND]
    assert isinstance(training, DaytonaMachineFactory) and isinstance(evaluation, DaytonaMachineFactory)
    assert (training.image_build, evaluation.image_build) == (DaytonaImageBuild.SNAPSHOT, DaytonaImageBuild.DECLARATIVE)
    assert training.client_factory().api_key == "shared-test-key"
    assert evaluation.client_factory().api_key == "shared-test-key"


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
