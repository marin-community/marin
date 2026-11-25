"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field
from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness, ToolProvider, task_prompt
from rolloutengine.harness.workplace import (
    NATIVE_MODEL_STEPS,
    WORKPLACE_ACTION_INTERFACE,
    WorkplaceProvider,
    workplace_prompt,
)
from rolloutengine.models import MachineRuntimeSpec, Prompt, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.fetch import FETCH_ACTION_INTERFACE
from taskcompendium.models import FunctionDefinition, TaskSpec
from taskcompendium.runtime.shell import BoundShellTool, ShellToolConfig

SHELLBOX_HARNESS = "shellbox"
_SHELLBOX_PREFIX = SHELLBOX_HARNESS + ":"


class ShellHarnessSettings(BaseModel):
    """Per-task Shellbox settings serialized in the harness identifier."""

    command_timeout: float | None
    shell_tool: ShellToolConfig
    source_turn_limit: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)


def rollout_spec(
    task: TaskSpec,
    session: Mapping[str, Any],
    *,
    task_machine: MachineRuntimeSpec | None = None,
    verifier_machine: MachineRuntimeSpec | None = None,
) -> TaskRolloutSpec:
    """Separate harness selection from rollout limits in the published task spec."""
    settings = dict(session)
    harness = settings.pop("task_session")
    shell_tool = settings.pop("shell_tool", {})
    command_timeout = settings.pop("command_timeout", None)
    trace = TraceMode(settings.pop("trace", TraceMode.EXACT_TOKENS))
    if harness == SHELLBOX_HARNESS:
        workplace = any(
            p.action_interface == WORKPLACE_ACTION_INTERFACE
            for p in task.environment_requirements.tool_providers.values()
        )
        if workplace:
            workplace_prompt(task)
            turns = settings["max_turns"]
            settings["max_turns"] = NATIVE_MODEL_STEPS if turns is None else min(turns, NATIVE_MODEL_STEPS)
        shell_settings = ShellHarnessSettings(
            command_timeout=command_timeout,
            shell_tool=ShellToolConfig.model_validate(shell_tool),
            source_turn_limit=NATIVE_MODEL_STEPS if workplace else None,
        )
        harness += ":" + json.dumps(shell_settings.model_dump(), sort_keys=True, separators=(",", ":"))
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=RolloutLimits.model_validate(settings),
        trace=trace,
    )


TOOL_PROVIDERS: Mapping[str, ToolProvider] = MappingProxyType(
    {FETCH_ACTION_INTERFACE: FetchProvider(), WORKPLACE_ACTION_INTERFACE: WorkplaceProvider()}
)
"""Host tools every Shellbox harness serves, by action interface."""


def shell_harness(name: str) -> ShellHarness:
    """Restore the shell configuration encoded in a task's harness name."""
    if not name.startswith(_SHELLBOX_PREFIX):
        raise ValueError(f"Not a Shellbox harness: {name!r}")
    settings = ShellHarnessSettings.model_validate_json(name[len(_SHELLBOX_PREFIX) :])
    return ShellHarness(
        command_timeout=settings.command_timeout,
        shell_tool=settings.shell_tool,
        tools=TOOL_PROVIDERS,
        prompt=source_prompt,
        source_turn_limit=settings.source_turn_limit,
    )


def source_prompt(task: TaskSpec, shell_tool: BoundShellTool | None, tools: tuple[FunctionDefinition, ...]) -> Prompt:
    """Select the declared native protocol without changing other shell tasks."""
    if any(
        p.action_interface == WORKPLACE_ACTION_INTERFACE
        for p in task.environment_requirements.tool_providers.values()
    ):
        return workplace_prompt(task)
    return task_prompt(task, shell_tool, tools)


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
