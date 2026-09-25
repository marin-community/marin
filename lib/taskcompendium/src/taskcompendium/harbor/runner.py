# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor binding."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from taskcompendium.harbor.adapter import DEFAULT_REQUEST_TIMEOUT
from taskcompendium.lowering import (
    BINDING_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborTaskBinding,
    read_binding,
    read_specification,
    read_submission_convention,
    validate_binding,
)
from taskcompendium.submission import AnswerFormat


@dataclass(frozen=True)
class HarborLaunch:
    """Agent and model choices made when a lowered task is run."""

    agent: str
    model: str | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


def _validate_launch(launch: HarborLaunch, answer_format: AnswerFormat | None) -> None:
    if "api_key" in launch.agent_kwargs:
        raise ValueError("Use api_key_env so credentials stay out of Harbor trial artifacts")
    if launch.agent == "chat":
        if not isinstance(launch.model, str) or not launch.model:
            raise ValueError("Chat launch requires a model")
        if answer_format is None:
            raise ValueError("Chat launch requires a readable convention")
        if set(launch.agent_kwargs) - {"api_base", "api_key_env", "request_timeout"}:
            raise ValueError("Unsupported chat launch arguments")
        if not isinstance(launch.agent_kwargs.get("api_base"), str) or not launch.agent_kwargs["api_base"]:
            raise ValueError("Chat launch requires api_base")
        api_key_env = launch.agent_kwargs.get("api_key_env")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise ValueError("api_key_env must be a nonempty string")
        timeout = launch.agent_kwargs.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        return
    if launch.agent not in {"replay", "action_replay"}:
        raise ValueError(f"Unsupported launch agent: {launch.agent}")
    if launch.model is not None:
        raise ValueError("Replay launch cannot select a model")
    if set(launch.agent_kwargs) != {"response"}:
        raise ValueError("Replay launch requires only a response")
    response = launch.agent_kwargs["response"]
    if launch.agent == "replay":
        if answer_format == AnswerFormat.FINAL_ACTION or not isinstance(response, str):
            raise ValueError("Text replay requires a plain or JSON task and string response")
    elif answer_format in {AnswerFormat.PLAIN, AnswerFormat.JSON} or not isinstance(response, dict):
        raise ValueError("Action replay requires a final-action task and object response")


async def run_trial(task_dir: Path, binding: HarborTaskBinding, launch: HarborLaunch, trials_dir: Path, trial_name: str):
    """Run a lowered task through Harbor's agent, environment, and custom verifier."""
    if binding != read_binding(task_dir / BINDING_FILE):
        raise ValueError("Launch binding differs from the exported task binding")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_binding(specification, binding)
    try:
        convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    except ValueError:
        if launch.agent == "chat":
            raise
        convention = None  # Replay still runs so the verifier can record invalid private metadata.
    answer_format = convention.answer_format if convention is not None else None
    _validate_launch(launch, answer_format)
    agents = {
        "replay": "taskcompendium.harbor.adapter:ReplayAgent",
        "action_replay": "taskcompendium.harbor.adapter:ActionReplayAgent",
        "chat": "taskcompendium.harbor.adapter:DirectChatAgent",
    }
    kwargs = dict(launch.agent_kwargs)
    if launch.agent == "chat" and convention is not None and convention.answer_format == AnswerFormat.FINAL_ACTION:
        request = specification.native_action_request
        if request is None:
            raise ValueError("Final-action task requires a source request")
        agents["chat"] = "taskcompendium.harbor.adapter:NativeActionAgent"
        kwargs["functions"] = [function.model_dump(mode="json") for function in request.functions]
        kwargs["messages"] = [message.model_dump(mode="json") for message in request.messages]
        kwargs["tool_choice"] = request.tool_choice
        kwargs["parallel_tool_calls"] = request.parallel_tool_calls
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
