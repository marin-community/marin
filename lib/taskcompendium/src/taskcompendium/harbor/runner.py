# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor binding."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from taskcompendium.lowering import HarborTaskBinding, read_binding, read_rendering, read_specification, validate_binding
from taskcompendium.rendering import AnswerFormat


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
        "action_replay": "taskcompendium.harbor.adapter:ActionReplayAgent",
        "chat": "taskcompendium.harbor.adapter:DirectChatAgent",
    }
    if launch.agent not in agents:
        raise ValueError(f"Unsupported launch agent: {launch.agent}")
    if launch.agent == "chat" and launch.model is None:
        raise ValueError("Chat launch requires a model")
    if "api_key" in launch.agent_kwargs:
        raise ValueError("Use api_key_env so credentials stay out of Harbor trial artifacts")
    kwargs = dict(launch.agent_kwargs)
    if launch.agent == "chat":
        rendering = read_rendering(task_dir / "rendering.json")
        if rendering.answer_format == AnswerFormat.FINAL_ACTION:
            agents["chat"] = "taskcompendium.harbor.adapter:NativeActionAgent"
            kwargs["functions"] = [dataclasses.asdict(function) for function in rendering.functions]
            kwargs["messages"] = [dataclasses.asdict(message) for message in rendering.messages]
            kwargs["tool_choice"] = rendering.tool_choice
            kwargs["parallel_tool_calls"] = rendering.parallel_tool_calls
    agent: dict[str, Any] = {"import_path": agents[launch.agent], "kwargs": kwargs}
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
