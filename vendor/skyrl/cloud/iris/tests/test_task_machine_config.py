"""Task backend admission must use the composed launch configuration."""

import os
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.machine import Backend
from marinskyrl.task_machine_factories import task_machine_factories
from skyrl_train.rollouts.task_worker import RemoteJudgeSettings

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config


def composed_config(tmp_path: Path, settings: dict):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["environment"] = {"task_sessions": settings}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    return compose_skyrl_config(
        parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config


def test_explicit_empty_local_settings_reach_the_host_capability_check(tmp_path, monkeypatch):
    config = composed_config(tmp_path, {"local": {}})
    assert OmegaConf.to_container(config.environment.task_sessions.local) == {}
    # A host without the supported local sandbox must reject admission. A dropped
    # empty mapping previously silently omitted this backend instead.
    monkeypatch.setattr("shellbox.backends.local.machine.sys.platform", "darwin")
    with pytest.raises(SandboxUnavailable):
        task_machine_factories(config.environment.task_sessions)


@pytest.mark.parametrize("settings", [{}, {"local": None}])
def test_omitted_or_null_local_settings_keep_the_backend_disabled(tmp_path, monkeypatch, settings):
    config = composed_config(tmp_path, settings)
    monkeypatch.setattr("marinskyrl.task_machine_factories.shutil.which", lambda name: None)
    factories = task_machine_factories(config.environment.task_sessions)
    assert Backend.LOCAL not in factories
    assert "shellsim" in factories


def test_empty_settings_do_not_clear_existing_mapping_defaults(tmp_path):
    baseline = composed_config(tmp_path, {})
    config = composed_config(tmp_path, {"session": {}})
    assert config.environment.task_sessions.session == baseline.environment.task_sessions.session


@pytest.mark.parametrize(
    "key,error",
    [
        ("seed", "trainer.seed.*not a mapping"),
        ("unknown_backend", "unknown key.*trainer.unknown_backend"),
    ],
)
def test_empty_mapping_preserves_config_merge_contracts(tmp_path, key, error):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["trainer"][key] = {}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=error):
        compose_skyrl_config(
            parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
        )


def test_daytona_factory_reads_named_credential_without_exporting_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("DAYTONA_TEST_KEY", raising=False)
    (tmp_path / "DAYTONA_TEST_KEY").write_text("synthetic-daytona-file")
    config = OmegaConf.create({"daytona": {"api_key_env": "DAYTONA_TEST_KEY", "ttl_minutes": 60}})
    observed = []
    def client_boundary(settings):
        observed.append(settings.api_key)
    monkeypatch.setattr("marinskyrl.task_machine_factories.AsyncDaytona", client_boundary)
    factories = task_machine_factories(config)
    factories[Backend.DAYTONA].client_factory()
    assert observed == ["synthetic-daytona-file"]
    assert "DAYTONA_TEST_KEY" not in os.environ


def test_remote_judge_reads_named_credential_at_connection_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("JUDGE_TEST_KEY", raising=False)
    (tmp_path / "JUDGE_TEST_KEY").write_text("synthetic-judge-file")
    settings = RemoteJudgeSettings("https://example.invalid", "JUDGE_TEST_KEY")
    connection = settings.connection()
    assert connection.api_key == "synthetic-judge-file"
    assert connection.base_url == "https://example.invalid"
    assert "JUDGE_TEST_KEY" not in os.environ
