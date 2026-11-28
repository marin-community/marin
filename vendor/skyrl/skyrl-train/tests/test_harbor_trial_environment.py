"""Harbor-trial environment acceptance against a real SWE-rebench V2.1@repaired row.

The fixture is the literal TaskSpec of one c55add0e pool row (the pinned, actor-image
re-seal of SWE-rebench V2.1). Its verifier resources embed the package task.toml, whose
[environment] deliberately declares the SAME docker image and env vars the TaskSpec
declares; the harbor_trial gate must accept that identity and fail closed on any drift.
"""

import json
from pathlib import Path

import pytest
from taskcompendium.models import EnvironmentRequirements, TaskSpec

from skyrl_train.rollouts.agent_runtime import harbor_trial_engine_requirements

FIXTURE = Path(__file__).parent / "fixtures" / "swe_rebench_c55add0e_task.json"


def real_task() -> TaskSpec:
    return TaskSpec.model_validate_json(FIXTURE.read_text())


def test_real_row_declares_the_package_environment_and_is_accepted():
    task = real_task()
    engine = harbor_trial_engine_requirements(task)
    # Harbor consumes the verified package environment, not the recording engine.
    assert engine == EnvironmentRequirements()
    assert task.environment_requirements.docker_image.startswith(
        "ghcr.io/marin-community/rl-data-swe-rebench-v2-1-actor@sha256:"
    )
    assert task.environment_requirements.setup_commands != ()
    assert task.environment_requirements.working_directory == "/testbed"


def test_drifted_docker_image_fails_closed():
    task = real_task()
    drifted = json.loads(FIXTURE.read_text())
    drifted["environment_requirements"]["docker_image"] = (
        "ghcr.io/marin-community/rl-data-swe-rebench-v2-1-actor@sha256:" + "0" * 64
    )
    task = TaskSpec.model_validate_json(json.dumps(drifted))
    with pytest.raises(ValueError, match="disagree with the Harbor package"):
        harbor_trial_engine_requirements(task)


def test_extra_env_var_fails_closed():
    task = real_task()
    drifted = json.loads(FIXTURE.read_text())
    drifted["environment_requirements"]["environment_variables"]["EXTRA"] = "undeclared"
    task = TaskSpec.model_validate_json(json.dumps(drifted))
    with pytest.raises(ValueError, match="disagree with the Harbor package"):
        harbor_trial_engine_requirements(task)


def test_added_tool_provider_fails_closed():
    task = real_task()
    drifted = json.loads(FIXTURE.read_text())
    drifted["environment_requirements"]["tool_providers"] = {
        "shell": {"action_interface": "shell", "initial_state": {"cwd": "/"}}
    }
    task = TaskSpec.model_validate_json(json.dumps(drifted))
    with pytest.raises(ValueError, match="disagree with the Harbor package"):
        harbor_trial_engine_requirements(task)


def test_unbound_capability_is_not_discarded_with_the_recording_environment():
    task = real_task()
    task = task.model_copy(
        update={"environment_requirements": task.environment_requirements.model_copy(update={"capabilities": ("gpu",)})}
    )
    with pytest.raises(ValueError, match="disagree with the Harbor package"):
        harbor_trial_engine_requirements(task)
