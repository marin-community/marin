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
        with pytest.raises(KeyError, match="JUDGE_TEST_KEY"):
            launcher.config_task_environment(config)


def test_resolve_daytona_rl_api_key_prefers_explicit_rl_key(monkeypatch):
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "generic-key")
    monkeypatch.setattr(launcher, "_daytona_rl_api_key_from_secret_manager", lambda: "secret-manager-key")

    assert launcher._resolve_daytona_rl_api_key() == "rl-key"


def test_inline_validation_forwards_separate_org_credentials_without_serializing_secrets(monkeypatch):
    config = OmegaConf.create(
        {
            "runtime": {"entrypoint": "train", "task_env": {}},
            "skyrl": {
                "environment": {"task_sessions": {"daytona": {"api_key_env": "DAYTONA_RL_API_KEY"}}},
                "trainer": {"callbacks": [{"type": "harbor_evaluation", "daytona_api_key_env": "EVAL_ORG_KEY"}]},
            },
        }
    )
    serialized = OmegaConf.to_yaml(config)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-credential")
    monkeypatch.setenv("EVAL_ORG_KEY", "eval-credential")
    assert launcher.config_task_environment(config) == {
        "DAYTONA_RL_API_KEY": "rl-credential",
        "EVAL_ORG_KEY": "eval-credential",
    }
    assert OmegaConf.to_yaml(config) == serialized
    monkeypatch.delenv("EVAL_ORG_KEY")
    with pytest.raises(KeyError, match="EVAL_ORG_KEY"):
        launcher.config_task_environment(config)


def test_resolve_daytona_rl_api_key_rejects_generic_key(monkeypatch):
    monkeypatch.delenv("DAYTONA_RL_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "generic-key")
    monkeypatch.setattr(launcher, "_daytona_rl_api_key_from_secret_manager", lambda: None)

    with pytest.raises(SystemExit, match="no Daytona RL key available"):
        launcher._resolve_daytona_rl_api_key()


@pytest.mark.parametrize(
    ("entrypoint", "terminal_bench_data", "backend", "expected"),
    [
        ("skyrl_train.entrypoints.main_base", [], "daytona", False),
        ("skyrl_train.entrypoints.terminal_bench", [], "daytona", True),
        ("skyrl_train.entrypoints.terminal_bench_generate", [], "daytona", True),
        ("skyrl_train.entrypoints.terminal_bench", [], "docker", False),
        ("skyrl_train.entrypoints.main_base", ["s3://bucket/tasks.parquet"], "docker", False),
        # Nemotron Ultra routing sends its SWE rows to Harbor from the standard entrypoint.
        ("skyrl_train.entrypoints.main_base", ["s3://bucket/tasks.parquet"], "daytona", True),
    ],
)
def test_daytona_preflight_follows_task_backend(entrypoint, terminal_bench_data, backend, expected):
    config = OmegaConf.create(
        {
            "runtime": {"entrypoint": entrypoint},
            "skyrl": {
                "data": {"terminal_bench_data": terminal_bench_data},
                "terminal_bench_config": {"harbor": {"environment_type": backend}},
            },
        }
    )

    assert launcher._rl_config_uses_daytona(config) is expected
