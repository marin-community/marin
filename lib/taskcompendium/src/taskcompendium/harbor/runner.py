# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from taskcompendium.lowering import (
    DIRECT_CHAT_ENVIRONMENT,
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    STATEFUL_ENVIRONMENT,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    provider_class,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_environment_config,
    validate_exported_resources,
)
from taskcompendium.submission import AnswerFormat

DEFAULT_CHAT_TIMEOUT = 120


class AgentStrategy(StrEnum):
    DIRECT_CHAT = "direct_chat"
    STATEFUL_TOOLS = "stateful_tools"


class ReplayLaunch(BaseModel):
    """A fixed response for exercising the Harbor trial path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: str


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    strategy: AgentStrategy = AgentStrategy.DIRECT_CHAT
    request_timeout: float = Field(default=DEFAULT_CHAT_TIMEOUT, gt=0, allow_inf_nan=False)
    max_turns: int = Field(default=16, gt=0)
    trial_timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @property
    def agent_kwargs(self) -> dict[str, Any]:
        values = self.model_dump(exclude={"model", "trial_timeout", "strategy"})
        if "max_turns" in values:
            values.pop("max_turns")
        return values


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    launch: ReplayLaunch | ChatLaunch,
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Run a lowered task and return Harbor's trial result."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_environment_config(specification, environment_config)
    validate_exported_resources(specification, task_dir)
    try:
        convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    except (ValueError, ValidationError):
        if not isinstance(launch, ReplayLaunch) or environment_config.environment != DIRECT_CHAT_ENVIRONMENT:
            raise
        convention = None  # The verifier records invalid private metadata as an ungraded outcome.
    stateful = environment_config.environment == STATEFUL_ENVIRONMENT
    if convention is not None and stateful != (convention.answer_format == AnswerFormat.STATE):
        raise ValueError("Submission convention differs from environment binding")
    if isinstance(launch, ReplayLaunch):
        if stateful:
            raise ValueError("Stateful tasks require a model endpoint")
        agent: dict[str, Any] = {
            "import_path": "taskcompendium.harbor.adapter:ReplayAgent",
            "kwargs": launch.model_dump(),
        }
    else:
        expected_strategy = AgentStrategy.STATEFUL_TOOLS if stateful else AgentStrategy.DIRECT_CHAT
        if launch.strategy != expected_strategy:
            raise ValueError("Agent strategy differs from Harbor environment binding")
        agent_path = "taskcompendium.harbor.adapter:DirectChatAgent"
        kwargs = launch.agent_kwargs
        if stateful:
            agent_path = "taskcompendium.harbor.adapter:StatefulToolAgent"
            kwargs["max_turns"] = launch.max_turns
        agent = {
            "import_path": agent_path,
            "model_name": launch.model,
            "kwargs": kwargs,
        }
    if environment_config.environment == DIRECT_CHAT_ENVIRONMENT:
        environment = {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"}
    else:
        environment = {
            "import_path": (
                f"{provider_class(environment_config).__module__}:{provider_class(environment_config).__name__}"
            ),
            "kwargs": {
                "seed_sha256": environment_config.seed_sha256,
                "action_interface": environment_config.action_interface,
            },
        }
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": environment,
            "agent": agent,
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
            **(
                {"trial_attempt_timeout_sec": launch.trial_timeout}
                if isinstance(launch, ChatLaunch) and launch.trial_timeout
                else {}
            ),
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
