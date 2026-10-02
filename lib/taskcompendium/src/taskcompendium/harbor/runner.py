# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from pathlib import Path

from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field
from tasktrove_verify.spec import PredictedActionSpec

from taskcompendium.grading import resolve_verifier, validate_verifier
from taskcompendium.harbor.docker import docker_control_plane
from taskcompendium.lowering import (
    DOCKER_ENVIRONMENT,
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_environment_config,
    validate_exported_task,
)
from taskcompendium.models import VerifierSpec
from taskcompendium.submission import chat_request, submission_compatibility

DEFAULT_CHAT_TIMEOUT = 120


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    request_timeout: float = Field(default=DEFAULT_CHAT_TIMEOUT, gt=0, allow_inf_nan=False)
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    parallel_tool_calls: bool | None = None


def validate_launch_parallel_tool_calls(specification: VerifierSpec, parallel_tool_calls: bool | None) -> None:
    verifier = resolve_verifier(specification)
    if parallel_tool_calls is False and isinstance(verifier, PredictedActionSpec) and len(verifier.expected_calls) > 1:
        raise ValueError("Launch disables parallel calls required by the task")


def chat_agent_config(task_dir: Path, launch: ChatLaunch) -> AgentConfig:
    """Select the direct-chat agent independently of task semantics."""
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    request = chat_request(specification, convention)
    if launch.temperature is not None:
        request["temperature"] = launch.temperature
    if launch.parallel_tool_calls is not None:
        if "parallel_tool_calls" in request and request["parallel_tool_calls"] != launch.parallel_tool_calls:
            raise ValueError("Launch parallel-tool policy conflicts with the submission convention")
        request["parallel_tool_calls"] = launch.parallel_tool_calls
    validate_launch_parallel_tool_calls(specification.verifier, request.get("parallel_tool_calls"))
    return AgentConfig.model_validate(
        {
            "import_path": "taskcompendium.harbor.adapter:ChatAgent",
            "model_name": launch.model,
            "kwargs": {
                **launch.model_dump(exclude={"model", "temperature", "parallel_tool_calls"}),
                "request": request,
            },
        }
    )


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    agent: AgentConfig,
    trials_dir: Path,
    trial_name: str,
    *,
    trial_timeout: float | None = None,
) -> TrialResult:
    """Run a lowered task using the caller-selected Harbor agent."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    validate_environment_config(specification, environment_config, convention)
    validate_exported_task(specification, environment_config, task_dir)
    validate_verifier(specification.verifier)
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention differs from task contract: {'; '.join(compatibility.reasons)}")
    environment_import = (
        "taskcompendium.harbor.docker:DockerWorkspaceEnvironment"
        if environment_config.environment == DOCKER_ENVIRONMENT
        else "taskcompendium.harbor.adapter:NoToolEnvironment"
    )
    environment_kwargs = {}
    if environment_config.environment == DOCKER_ENVIRONMENT:
        control = await docker_control_plane()
        environment_kwargs = {"archive_socket": control.socket, "archive_api_version": control.api_version}
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": {"import_path": environment_import, "kwargs": environment_kwargs},
            "agent": agent.model_dump(),
            **({"trial_attempt_timeout_sec": trial_timeout} if trial_timeout is not None else {}),
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
