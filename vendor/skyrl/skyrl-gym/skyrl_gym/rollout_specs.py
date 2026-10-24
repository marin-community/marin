"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel
from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness, ToolProvider
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.fetch import FETCH_ACTION_INTERFACE
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.shell import ShellToolConfig

SHELLBOX_HARNESS = "shellbox"
_SHELLBOX_PREFIX = SHELLBOX_HARNESS + ":"


class ShellHarnessSettings(BaseModel):
    """Per-task Shellbox settings serialized in the harness identifier."""

    command_timeout: float | None
    shell_tool: ShellToolConfig


def rollout_spec(
    task: TaskSpec,
    session: Mapping[str, Any],
    *,
    task_machine: MachineRuntimeSpec | None = None,
    verifier_machine: MachineRuntimeSpec | None = None,
) -> TaskRolloutSpec:
    """Separate harness selection from rollout limits in the published task spec.

    The task's ``turn_budget`` is the turn limit, so a session cannot set one.
    """
    settings = dict(session)
    if "max_turns" in settings:
        raise ValueError("The task's turn budget is the turn limit; the session cannot set max_turns")
    harness = settings.pop("task_session")
    shell_tool = settings.pop("shell_tool", {})
    command_timeout = settings.pop("command_timeout", None)
    trace = TraceMode(settings.pop("trace", TraceMode.EXACT_TOKENS))
    if harness == SHELLBOX_HARNESS:
        shell_settings = ShellHarnessSettings(
            command_timeout=command_timeout,
            shell_tool=ShellToolConfig.model_validate(shell_tool),
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


TOOL_PROVIDERS: Mapping[str, ToolProvider] = MappingProxyType({FETCH_ACTION_INTERFACE: FetchProvider()})
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
    )


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
