"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from rolloutengine.harness.core import Harness
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.harness.tools import ToolHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.shell import ShellToolConfig

PREFILL_HARNESS = "prefill"
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
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=RolloutLimits.model_validate(settings),
        trace=trace,
    )


def shell_harness(name: str) -> ToolHarness:
    """Restore the shell configuration encoded in a task's harness name."""
    if not name.startswith(_SHELLBOX_PREFIX):
        raise ValueError(f"Not a Shellbox harness: {name!r}")
    settings = ShellHarnessSettings.model_validate_json(name[len(_SHELLBOX_PREFIX) :])
    return ToolHarness(
        command_timeout=settings.command_timeout,
        shell_tool=settings.shell_tool,
    )


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)


def restore_harness(name: str) -> Harness:
    """Restore a text prefill or the shell configuration encoded in its name."""
    if name == PREFILL_HARNESS:
        return PrefillHarness()
    if is_shell_harness(name):
        return shell_harness(name)
    raise ValueError(f"Unknown rollout harness: {name}")
