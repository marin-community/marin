# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool sets composed by experiment wiring."""

from collections.abc import Mapping
from types import MappingProxyType

from shellbox.machine import MachineFactory

from forge.harness.protocol import Tool
from forge.harness.tools.files import ReadFileTool, WriteFileTool
from forge.harness.tools.finish import FinishTool
from forge.harness.tools.grade import GradeTool
from forge.harness.tools.shell import ShellTool
from forge.harness.tools.submit import Gate, SubmitTool
from forge.harness.tools.web import WebFetchTool, WebSearchTool
from forge.lowering import LoweringSettings
from forge.policy import Policy


def standard_tools(
    *,
    task_path: str,
    controls_path: str,
    policy: Policy,
    factories: Mapping[str, MachineFactory],
    settings: LoweringSettings,
    gate: Gate,
) -> Mapping[str, Tool[object]]:
    tools: tuple[Tool[object], ...] = (
        ShellTool(policy.setting("solver.command_timeout", float), policy.setting("solver.tool_timeout", float)),
        ReadFileTool(),
        WriteFileTool(),
        GradeTool(task_path, factories, settings),
        SubmitTool(gate, task_path, controls_path),
        FinishTool(),
        WebSearchTool(),
        WebFetchTool(),
    )
    return MappingProxyType({tool.schema.name: tool for tool in tools})


def adversary_tools(
    *, task_path: str, policy: Policy, factories: Mapping[str, MachineFactory], settings: LoweringSettings
) -> Mapping[str, Tool[object]]:
    tools: tuple[Tool[object], ...] = (
        ShellTool(policy.setting("solver.command_timeout", float), policy.setting("solver.tool_timeout", float)),
        ReadFileTool(),
        GradeTool(task_path, factories, settings),
        FinishTool(),
    )
    return MappingProxyType({tool.schema.name: tool for tool in tools})
