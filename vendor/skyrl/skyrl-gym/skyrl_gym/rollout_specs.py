"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel
from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.browsecomp import BrowseCompProvider
from rolloutengine.harness.retrieval_service import OpenResearcherServiceProvider
from taskcompendium.browsecomp import BROWSECOMP_ACTION_INTERFACE
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE
from rolloutengine.harness.shell import ShellHarness, ToolProvider
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.fetch import FETCH_ACTION_INTERFACE
from taskcompendium.models import EXECUTION_RESOURCE_FIELDS, ExecutionRequirements, TaskSpec, declared_verifier_timeout
from taskcompendium.runtime.shell import ShellToolConfig

from skyrl_gym.nemotron_tools import NEMOTRON_PYTHON_INTERFACE, NemotronPythonProvider

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
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=_machine_with_declared_minimums(task_machine, task.execution_requirements),
        verifier_machine=_machine_with_declared_minimums(verifier_machine, task.verifier_execution_requirements),
        limits=rollout_limits(task, RolloutLimits.model_validate(settings), minimum_verifier_timeout=None),
        trace=trace,
    )


def rollout_limits(task: TaskSpec, limits: RolloutLimits, *, minimum_verifier_timeout: float | None) -> RolloutLimits:
    """Apply declared phase deadlines and preserve finite attempt bounds.

    Finite actor and verifier budgets extend an existing attempt deadline when their
    combined budget needs more time. Without a separate actor limit, preserve the attempt bound.
    """
    declared = task.execution_requirements
    actor = limits.total_turn_timeout
    if declared is not None:
        actor = declared.agent_timeout_sec if actor is None else min(actor, declared.agent_timeout_sec)
    deadlines = [
        value
        for value in (minimum_verifier_timeout, limits.verifier_timeout, declared_verifier_timeout(task))
        if value is not None
    ]
    timeout = max(deadlines, default=None)
    attempt = limits.attempt_timeout
    if attempt is not None and timeout is not None and actor is not None:
        attempt = max(attempt, actor + timeout + limits.cleanup_timeout)
    return limits.model_copy(
        update={"total_turn_timeout": actor, "verifier_timeout": timeout, "attempt_timeout": attempt}
    )


def _machine_with_declared_minimums(
    runtime: MachineRuntimeSpec | None, requirements: ExecutionRequirements | None
) -> MachineRuntimeSpec | None:
    if runtime is None or requirements is None:
        return runtime
    return runtime.model_copy(
        update={
            name: max(getattr(runtime, name) or 0, getattr(requirements, name)) for name in EXECUTION_RESOURCE_FIELDS
        }
    )


TOOL_PROVIDERS: Mapping[str, ToolProvider] = MappingProxyType(
    {
        FETCH_ACTION_INTERFACE: FetchProvider(),
        NEMOTRON_PYTHON_INTERFACE: NemotronPythonProvider(),
        BROWSECOMP_ACTION_INTERFACE: BrowseCompProvider(command_timeout=300),
    }
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
    return ShellHarness(command_timeout=settings.command_timeout, shell_tool=settings.shell_tool, tools=tools)


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
