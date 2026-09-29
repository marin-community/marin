# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field

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
from taskcompendium.submission import AnswerFormat, submission_instruction

DEFAULT_CHAT_TIMEOUT = 120


class ReplayLaunch(BaseModel):
    """A fixed response for exercising the Harbor trial path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: str


class ActionReplayLaunch(BaseModel):
    """A fixed final assistant action for exercising the Harbor trial path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: dict[str, Any]


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    request_timeout: float = Field(default=DEFAULT_CHAT_TIMEOUT, gt=0, allow_inf_nan=False)


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    launch: ReplayLaunch | ActionReplayLaunch | ChatLaunch,
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Run a lowered task and return Harbor's trial result."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_environment_config(specification, environment_config)
    try:
        convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    except ValueError:
        if isinstance(launch, ChatLaunch):
            raise
        convention = None  # Replay still runs so the verifier can record invalid private metadata.
    answer_format = convention.answer_format if convention is not None else None
    if isinstance(launch, ReplayLaunch):
        if answer_format in {AnswerFormat.ANSWER_CALL, AnswerFormat.FINAL_ACTION}:
            raise ValueError("Text replay requires a plain or JSON task")
        agent: dict[str, Any] = {
            "import_path": "taskcompendium.harbor.adapter:ReplayAgent",
            "kwargs": launch.model_dump(),
        }
    elif isinstance(launch, ActionReplayLaunch):
        if answer_format in {AnswerFormat.PLAIN, AnswerFormat.JSON}:
            raise ValueError("Action replay requires an action-submission task")
        agent = {
            "import_path": "taskcompendium.harbor.adapter:ActionReplayAgent",
            "kwargs": launch.model_dump(),
        }
    else:
        if convention is None:
            raise ValueError("Chat launch requires a readable convention")
        agent_path = "taskcompendium.harbor.adapter:DirectChatAgent"
        kwargs = launch.model_dump(exclude={"model"})
        kwargs["events"] = [event.model_dump(mode="json") for event in specification.context.events]
        kwargs["submission_instruction"] = submission_instruction(convention)
        if answer_format == AnswerFormat.FINAL_ACTION:
            agent_path = "taskcompendium.harbor.adapter:NativeActionAgent"
            kwargs["functions"] = [function.model_dump(mode="json") for function in specification.tools.functions]
            kwargs["tool_choice"] = specification.tools.tool_choice
            kwargs["parallel_tool_calls"] = specification.tools.parallel_tool_calls
        elif answer_format == AnswerFormat.ANSWER_CALL:
            agent_path = "taskcompendium.harbor.adapter:AnswerCallAgent"
        agent = {
            "import_path": agent_path,
            "model_name": launch.model,
            "kwargs": kwargs,
        }
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"},
            "agent": agent,
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
