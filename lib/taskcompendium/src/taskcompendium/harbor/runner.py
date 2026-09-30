# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field
from tasktrove_verify.spec import JudgeRuntimeConfig

from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_environment_config,
)
from taskcompendium.submission import chat_request

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


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    launch: ChatLaunch,
    trials_dir: Path,
    trial_name: str,
    judge_runtime: JudgeRuntimeConfig | None = None,
) -> TrialResult:
    """Run a lowered task and return Harbor's trial result."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_environment_config(specification, environment_config)
    convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    request = chat_request(specification, convention)
    if launch.temperature is not None:
        request["temperature"] = launch.temperature
    if launch.parallel_tool_calls is not None:
        if "parallel_tool_calls" in request and request["parallel_tool_calls"] != launch.parallel_tool_calls:
            raise ValueError("Launch parallel-tool policy conflicts with the submission convention")
        request["parallel_tool_calls"] = launch.parallel_tool_calls
    agent = {
        "import_path": "taskcompendium.harbor.adapter:ChatAgent",
        "model_name": launch.model,
        "kwargs": {
            **launch.model_dump(exclude={"model", "temperature", "parallel_tool_calls"}),
            "request": request,
        },
    }
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"},
            "agent": agent,
            "verifier": {
                "import_path": "taskcompendium.harbor.adapter:SemanticVerifier",
                "kwargs": {"judge_runtime": judge_runtime.__dict__} if judge_runtime is not None else {},
            },
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
