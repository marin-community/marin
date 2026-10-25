"""Task backend admission must use the composed launch configuration."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.machine import Backend
from marinskyrl.task_machine_factories import task_machine_factories
from skyrl_gym.rollout_specs import rollout_spec, shell_harness
from skyrl_train.dataset.tasks import source_row_task

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


def test_empty_session_settings_preserve_published_limits_and_shell_timeout(tmp_path):
    config = composed_config(tmp_path, {"session": {}})
    config.generator.max_turns = 3
    config.trajectory_runner.command_timeout = 17
    task = source_row_task(
        {
            "env_class": "gsm8k",
            "prompt": [{"role": "user", "content": "1 + 1?"}],
            "reward_model": {"ground_truth": "2"},
        },
        0,
        source_name="empty-session-merge",
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
    ).task
    spec = rollout_spec(task, {**dict(config.environment.task_sessions.session), "task_session": "shellbox"})

    assert spec.limits.max_turns == 3
    assert shell_harness(spec.harness).command_timeout == 17
