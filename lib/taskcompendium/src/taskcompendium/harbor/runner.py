# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor binding."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from taskcompendium.lowering import HarborTaskBinding, read_binding, read_specification, validate_binding


@dataclass(frozen=True)
class HarborLaunch:
    """Agent and model choices made when a lowered task is run."""

    agent: str
    model: str | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


async def run_trial(task_dir: Path, binding: HarborTaskBinding, launch: HarborLaunch, trials_dir: Path, trial_name: str):
    """Run a lowered task through Harbor's agent, environment, and custom verifier."""
    if binding != read_binding(task_dir / "binding.json"):
        raise ValueError("Launch binding differs from the exported task binding")
    validate_binding(read_specification(task_dir / "specification.json"), binding)
    agents = {
        "replay": "taskcompendium.harbor.adapter:ReplayAgent",
        "chat": "taskcompendium.harbor.adapter:DirectChatAgent",
    }
    if launch.agent not in agents:
        raise ValueError(f"Unsupported launch agent: {launch.agent}")
    if launch.agent == "chat" and launch.model is None:
        raise ValueError("Chat launch requires a model")
    agent: dict[str, Any] = {"import_path": agents[launch.agent], "kwargs": launch.agent_kwargs}
    if launch.model is not None:
        agent["model_name"] = launch.model
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
