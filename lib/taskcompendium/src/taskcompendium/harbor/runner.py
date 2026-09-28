# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict

from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    HarborEnvironmentConfig,
    read_environment_config,
    read_specification,
    validate_environment_config,
)

DEFAULT_CHAT_TIMEOUT = 120


class ReplayLaunch(BaseModel):
    """A fixed response for exercising the Harbor trial path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: str


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    api_base: str
    api_key_env: str | None = None
    request_timeout: float = DEFAULT_CHAT_TIMEOUT


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
    validate_environment_config(read_specification(task_dir / SPECIFICATION_FILE), environment_config)
    if isinstance(launch, ReplayLaunch):
        agent: dict[str, Any] = {
            "import_path": "taskcompendium.harbor.adapter:ReplayAgent",
            "kwargs": launch.model_dump(),
        }
    else:
        agent = {
            "import_path": "taskcompendium.harbor.adapter:DirectChatAgent",
            "model_name": launch.model,
            "kwargs": launch.model_dump(exclude={"model"}),
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
