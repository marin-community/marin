"""Task backend admission must use the composed launch configuration."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.machine import Backend
from marinskyrl.task_machine_factories import task_machine_factories

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config


def composed_config(settings: dict):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["environment"] = {"task_sessions": settings}
    return compose_skyrl_config(
        parse_rl_config(raw), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config


def test_explicit_empty_local_settings_reach_the_host_capability_check(monkeypatch):
    config = composed_config({"local": {}})
    assert OmegaConf.to_container(config.environment.task_sessions.local) == {}
    # A host without the supported local sandbox must reject admission. A dropped
    # empty mapping previously silently omitted this backend instead.
    monkeypatch.setattr("shellbox.backends.local.machine.sys.platform", "darwin")
    with pytest.raises(SandboxUnavailable):
        task_machine_factories(config.environment.task_sessions)


@pytest.mark.parametrize("settings", [{}, {"local": None}])
def test_omitted_or_null_local_settings_keep_the_backend_disabled(monkeypatch, settings):
    config = composed_config(settings)
    monkeypatch.setattr("marinskyrl.task_machine_factories.shutil.which", lambda name: None)
    factories = task_machine_factories(config.environment.task_sessions)
    assert Backend.LOCAL not in factories
    assert "shellsim" in factories


def test_empty_settings_do_not_clear_existing_mapping_defaults():
    baseline = composed_config({})
    config = composed_config({"session": {}})
    assert config.environment.task_sessions.session == baseline.environment.task_sessions.session


def test_unknown_trainer_keys_are_rejected():
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["trainer"]["unknown_backend"] = {}
    with pytest.raises(ValueError, match="trainer.unknown_backend.*\\n.*Extra inputs"):
        compose_skyrl_config(parse_rl_config(raw), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100"))
