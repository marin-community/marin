"""Credential selection for task backends and remote graders."""

import pytest
from omegaconf import OmegaConf
from rigging.secrets import SecretResolutionError
from marinskyrl.task_machine_factories import task_machine_factories

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
        with pytest.raises(SecretResolutionError, match="JUDGE_TEST_KEY"):
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
    with pytest.raises(SecretResolutionError, match="EVAL_ORG_KEY"):
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


@pytest.mark.parametrize("backend", ["remote_judge", "daytona", "validation_callback"])
def test_mounted_credentials_override_environment_and_leave_launch_document_unchanged(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.setenv("CAMPAIGN_BACKEND_KEY", "stale-environment-key")
    (tmp_path / "CAMPAIGN_BACKEND_KEY").write_text("mounted-credential")
    (tmp_path / "UNRELATED_KEY").write_text("must-not-forward")
    sessions = {} if backend == "validation_callback" else {backend: {"api_key_env": "CAMPAIGN_BACKEND_KEY"}}
    callbacks = (
        [{"type": "harbor_evaluation", "daytona_api_key_env": "CAMPAIGN_BACKEND_KEY"}]
        if backend == "validation_callback" else []
    )
    config = OmegaConf.create({
        "runtime": {"entrypoint": "train", "task_env": {}},
        "skyrl": {"environment": {"task_sessions": sessions}, "trainer": {"callbacks": callbacks}},
    })
    original = OmegaConf.to_yaml(config)
    assert launcher.config_task_environment(config) == {"CAMPAIGN_BACKEND_KEY": "mounted-credential"}
    monkeypatch.delenv("CAMPAIGN_BACKEND_KEY")
    assert launcher.config_task_environment(config) == {"CAMPAIGN_BACKEND_KEY": "mounted-credential"}
    assert OmegaConf.to_yaml(config) == original


@pytest.mark.parametrize("phase,selected", [("train", "RL_TEST_KEY"), ("eval", "EVAL_TEST_KEY")])
def test_real_worker_factory_reads_the_selected_org_from_mounted_files(tmp_path, monkeypatch, phase, selected):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    for name in ("RL_TEST_KEY", "EVAL_TEST_KEY"):
        monkeypatch.delenv(name, raising=False)
        (tmp_path / name).write_text("mounted-" + name)
    sessions = OmegaConf.create({"daytona": {
        "api_key_env": "RL_TEST_KEY", "eval_api_key_env": "EVAL_TEST_KEY", "ttl_minutes": 90,
    }})
    original = OmegaConf.to_yaml(sessions)
    factory = task_machine_factories(sessions, phase=phase)["daytona"]
    assert factory.client_factory.args[0].api_key == "mounted-" + selected
    assert OmegaConf.to_yaml(sessions) == original
    (tmp_path / selected).unlink()
    with pytest.raises(SecretResolutionError, match=selected):
        task_machine_factories(sessions, phase=phase)


def test_submission_forwards_both_declared_daytona_org_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    expected = {"RL_TEST_KEY": "mounted-rl", "EVAL_TEST_KEY": "mounted-eval"}
    for name, value in expected.items():
        (tmp_path / name).write_text(value)
        monkeypatch.setenv(name, "stale-env")
    config = OmegaConf.create({
        "runtime": {"entrypoint": "train", "task_env": {}},
        "skyrl": {"environment": {"task_sessions": {"daytona": {
            "api_key_env": "RL_TEST_KEY", "eval_api_key_env": "EVAL_TEST_KEY",
        }}}},
    })
    original = OmegaConf.to_yaml(config)
    assert launcher.config_task_environment(config) == expected
    assert OmegaConf.to_yaml(config) == original
