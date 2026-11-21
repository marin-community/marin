# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool sets composed by experiment wiring."""

from collections.abc import Mapping
from types import MappingProxyType

from shellbox.machine import MachineFactory

from taskforge.harness.protocol import Tool
from taskforge.harness.tools.files import ReadFileTool, WriteFileTool
from taskforge.harness.tools.finish import FinishTool
from taskforge.harness.tools.grade import GradeTool
from taskforge.harness.tools.shell import ShellTool
from taskforge.harness.tools.submit import Gate, SubmitTool
from taskforge.harness.tools.web import WebFetchTool, WebSearchTool
from taskforge.lowering import LoweringSettings
from taskforge.policy import Policy


def _named(*tools: Tool[object]) -> Mapping[str, Tool[object]]:
    return MappingProxyType({tool.schema.name: tool for tool in tools})


def adversary_tools(
    *, task_path: str, policy: Policy, factories: Mapping[str, MachineFactory], settings: LoweringSettings
) -> Mapping[str, Tool[object]]:
    shell = ShellTool(policy.setting("solver.command_timeout", float), policy.setting("solver.tool_timeout", float))
    return _named(shell, ReadFileTool(), GradeTool(task_path, factories, settings), FinishTool())


def standard_tools(
    *,
    task_path: str,
    controls_path: str,
    policy: Policy,
    factories: Mapping[str, MachineFactory],
    settings: LoweringSettings,
    gate: Gate,
) -> Mapping[str, Tool[object]]:
    """The author's tools: the adversary's, with write, submit and web tools at their places in the order."""
    shared = adversary_tools(task_path=task_path, policy=policy, factories=factories, settings=settings)
    shell, read, grade, finish = shared.values()
    submit = SubmitTool(gate, task_path, controls_path)
    return _named(shell, read, WriteFileTool(), grade, submit, finish, WebSearchTool(), WebFetchTool())
