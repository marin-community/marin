"""Task backend admission must use the composed launch configuration."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from shellbox.machine import Backend
from marinskyrl.task_machine_factories import task_machine_factories

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
    monkeypatch.setattr("marinskyrl.task_machine_factories.sys.platform", "darwin")
    with pytest.raises(ValueError, match="Local task execution requires a Linux host"):
        task_machine_factories(config.environment.task_sessions)


@pytest.mark.parametrize("settings", [{}, {"local": None}])
def test_omitted_or_null_local_settings_keep_the_backend_disabled(tmp_path, monkeypatch, settings):
    config = composed_config(tmp_path, settings)
    monkeypatch.setattr("marinskyrl.task_machine_factories.shutil.which", lambda name: None)
    factories = task_machine_factories(config.environment.task_sessions)
    assert Backend.LOCAL not in factories
    assert "shellsim" in factories


def test_empty_settings_do_not_clear_existing_mapping_defaults(tmp_path):
    config = composed_config(tmp_path, {"session": {}})
    assert config.environment.task_sessions.session.command_timeout == 120


def test_empty_mapping_cannot_replace_a_scalar_config_value(tmp_path):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["trainer"]["seed"] = {}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="trainer.seed.*not a mapping"):
        compose_skyrl_config(
            parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
        )


def test_empty_mapping_keeps_unknown_config_keys_rejected(tmp_path):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["trainer"]["unknown_backend"] = {}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown key.*trainer.unknown_backend"):
        compose_skyrl_config(
            parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
        )
