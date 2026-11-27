"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel
from rolloutengine.harness.retrieval_service import OpenResearcherServiceProvider
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE
from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness, ToolProvider
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.fetch import FETCH_ACTION_INTERFACE
from taskcompendium.models import TaskSpec, ExecutionRequirements, EXECUTION_RESOURCE_FIELDS
from taskcompendium.runtime.shell import ShellToolConfig

from skyrl_gym.nemotron_python_tools import NEMOTRON_PYTHON_ACTION_INTERFACE, NemotronPythonProvider

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
    """Separate harness selection from rollout limits in the published task spec."""
    settings = dict(session)
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
    declared = task.execution_requirements
    if declared is not None:
        settings["total_turn_timeout"] = min(
            settings.get("total_turn_timeout") or declared.agent_timeout_sec, declared.agent_timeout_sec
        )
        settings["verifier_timeout"] = max(settings.get("verifier_timeout") or 0, declared.verifier_timeout_sec)
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=_declared_machine(task_machine, task.execution_requirements),
        verifier_machine=_declared_machine(verifier_machine, task.verifier_execution_requirements),
        limits=RolloutLimits.model_validate(settings),
        trace=trace,
    )


def _declared_machine(
    runtime: MachineRuntimeSpec | None, requirements: ExecutionRequirements | None
) -> MachineRuntimeSpec | None:
    if runtime is None or requirements is None:
        return runtime
    return runtime.model_copy(
        update={name: max(getattr(runtime, name) or 0, getattr(requirements, name)) for name in EXECUTION_RESOURCE_FIELDS}
    )


TOOL_PROVIDERS: Mapping[str, ToolProvider] = MappingProxyType(
    {FETCH_ACTION_INTERFACE: FetchProvider(), NEMOTRON_PYTHON_ACTION_INTERFACE: NemotronPythonProvider()}
)
"""Host tools every Shellbox harness serves, by action interface."""


def shell_harness(name: str, *, retrieval_service: str | None = None) -> ShellHarness:
    """Restore the shell configuration encoded in a task's harness name."""
    if not name.startswith(_SHELLBOX_PREFIX):
        raise ValueError(f"Not a Shellbox harness: {name!r}")
    settings = ShellHarnessSettings.model_validate_json(name[len(_SHELLBOX_PREFIX) :])
    tools = dict(TOOL_PROVIDERS)
    if retrieval_service is not None:
        tools[OPENRESEARCHER_ACTION_INTERFACE] = OpenResearcherServiceProvider(retrieval_service)
    return ShellHarness(
        command_timeout=settings.command_timeout,
        shell_tool=settings.shell_tool,
        tools=tools,
    )


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
