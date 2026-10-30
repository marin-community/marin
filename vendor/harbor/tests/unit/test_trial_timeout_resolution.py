import json
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import AgentInfo
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer


def _write_task(tmp_path: Path, task_toml: str) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "instruction.md").write_text("Complete the task.\n")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    for step in tomllib.loads(task_toml).get("steps", []):
        step_dir = task_dir / "steps" / step["name"]
        step_dir.mkdir(parents=True)
        (step_dir / "instruction.md").write_text("Complete this step.\n")
    return task_dir


async def _persisted_timeout_resolution(
    tmp_path: Path,
    task_toml: str,
    **config_kwargs,
) -> dict:
    task_dir = _write_task(tmp_path, task_toml)
    agent_config = config_kwargs.pop("agent", AgentConfig(name="nop"))
    verifier_config = config_kwargs.pop("verifier", VerifierConfig())
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name="trial",
        trials_dir=tmp_path / "trials",
        agent=agent_config,
        environment=EnvironmentConfig(type="docker"),
        verifier=verifier_config,
        **config_kwargs,
    )
    environment = MagicMock()
    environment.capabilities.mounted = False
    agent = MagicMock()
    agent.to_agent_info.return_value = AgentInfo(name="nop", version="1")

    with (
        patch(
            "harbor.trial.trial.EnvironmentFactory.create_environment_from_config",
            return_value=environment,
        ),
        patch(
            "harbor.trial.trial.AgentFactory.create_agent_from_config",
            return_value=agent,
        ),
    ):
        trial = await Trial.create(config)

    await trial._init_result()
    # config.json is persisted by the background artifact writer.
    get_artifact_writer().flush(timeout=30)
    persisted_config = json.loads(trial.paths.config_path.read_text())
    reloaded_config = TrialConfig.model_validate(persisted_config)
    assert reloaded_config.resolved_timeouts is not None
    return reloaded_config.resolved_timeouts.model_dump(mode="json")


_SINGLE_STEP_TASK_TOML = (
    "[agent]\n"
    "timeout_sec = 900\n"
    "[verifier]\n"
    "timeout_sec = 1200\n"
    "[environment]\n"
    "build_timeout_sec = 300\n"
)


@pytest.mark.parametrize(
    ("task_toml", "config_kwargs", "expected"),
    [
        pytest.param(
            _SINGLE_STEP_TASK_TOML,
            {
                "timeout_multiplier": 16,
                "agent": AgentConfig(name="nop", max_timeout_sec=7200),
                "verifier": VerifierConfig(max_timeout_sec=14400),
            },
            {
                "agent": 7200,
                "verifier": 14400,
                "agent_setup": 5760,
                "environment_build": 4800,
                "trial_attempt": None,
                "trial_cleanup_grace": 30.0,
                "steps": [],
            },
            id="hard-cap-after-shared-multiplier",
        ),
        pytest.param(
            _SINGLE_STEP_TASK_TOML,
            {
                "timeout_multiplier": 16,
                "agent_timeout_multiplier": 3,
                "verifier_timeout_multiplier": 4,
                "agent_setup_timeout_multiplier": 5,
                "environment_build_timeout_multiplier": 6,
                "trial_attempt_timeout_sec": 20000,
                "trial_cleanup_grace_sec": 45,
            },
            {
                "agent": 2700,
                "verifier": 4800,
                "agent_setup": 1800,
                "environment_build": 1800,
                "trial_attempt": 20000.0,
                "trial_cleanup_grace": 45.0,
                "steps": [],
            },
            id="uncapped-phase-multipliers",
        ),
        pytest.param(
            _SINGLE_STEP_TASK_TOML + "[[steps]]\n"
            'name = "short"\n'
            "[steps.agent]\n"
            "timeout_sec = 100\n"
            "[steps.verifier]\n"
            "timeout_sec = 100\n"
            "[[steps]]\n"
            'name = "above-cap"\n'
            "[steps.agent]\n"
            "timeout_sec = 10000\n"
            "[steps.verifier]\n"
            "timeout_sec = 2400\n",
            {
                "timeout_multiplier": 16,
                "agent": AgentConfig(name="nop", max_timeout_sec=7200),
                "verifier": VerifierConfig(max_timeout_sec=14400),
            },
            {
                "agent": 7200,
                "verifier": 14400,
                "agent_setup": 5760,
                "environment_build": 4800,
                "trial_attempt": None,
                "trial_cleanup_grace": 30.0,
                "steps": [
                    {"name": "short", "agent": 1600, "verifier": 1600},
                    {"name": "above-cap", "agent": 7200, "verifier": 14400},
                ],
            },
            id="multi-step-hard-cap",
        ),
    ],
)
@pytest.mark.asyncio
async def test_timeout_resolution_is_persisted_and_round_trips(
    tmp_path: Path,
    task_toml: str,
    config_kwargs: dict,
    expected: dict,
) -> None:
    resolved = await _persisted_timeout_resolution(
        tmp_path,
        task_toml,
        **config_kwargs,
    )

    assert resolved == expected
