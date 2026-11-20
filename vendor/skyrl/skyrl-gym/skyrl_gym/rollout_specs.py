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
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.shell import ShellToolConfig

from skyrl_gym.nemotron_tools import NEMOTRON_PYTHON_INTERFACE, NemotronPythonProvider

SHELLBOX_HARNESS = "shellbox"
_SHELLBOX_PREFIX = SHELLBOX_HARNESS + ":"


class ShellHarnessSettings(BaseModel):
    """Per-task Shellbox settings serialized in the harness identifier."""

    command_timeout: float | None
    retrieval_service: str | None = None
    shell_tool: ShellToolConfig


def load_task_spec(serialized: str) -> TaskSpec:
    """Parse a frozen canonical task without dropping fields or changing its contract."""
    return TaskSpec.model_validate_json(serialized)


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
    retrieval_service = settings.pop("retrieval_service", None)
    trace = TraceMode(settings.pop("trace", TraceMode.EXACT_TOKENS))
    if harness == SHELLBOX_HARNESS:
        shell_settings = ShellHarnessSettings(
            command_timeout=command_timeout,
            retrieval_service=retrieval_service,
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


TOOL_PROVIDERS: Mapping[str, ToolProvider] = MappingProxyType({
    FETCH_ACTION_INTERFACE: FetchProvider(),
    NEMOTRON_PYTHON_INTERFACE: NemotronPythonProvider(),
    BROWSECOMP_ACTION_INTERFACE: BrowseCompProvider(command_timeout=300),
    OPENRESEARCHER_ACTION_INTERFACE: OpenResearcherServiceProvider(endpoint=None),
})
"""Host tools every Shellbox harness serves, by action interface."""


def shell_harness(name: str) -> ShellHarness:
    """Restore the shell configuration encoded in a task's harness name."""
    if not name.startswith(_SHELLBOX_PREFIX):
        raise ValueError(f"Not a Shellbox harness: {name!r}")
    settings = ShellHarnessSettings.model_validate_json(name[len(_SHELLBOX_PREFIX) :])
    return ShellHarness(
        command_timeout=settings.command_timeout,
        shell_tool=settings.shell_tool,
        tools={**TOOL_PROVIDERS, OPENRESEARCHER_ACTION_INTERFACE: OpenResearcherServiceProvider(settings.retrieval_service)},
    )


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
