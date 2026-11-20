"""Serve the NeMo math tool with SkyRL's task-local persistent Python kernel."""

import json
from dataclasses import dataclass

from pydantic import JsonValue, TypeAdapter
from rolloutengine.harness.shell import ToolCallResult, ToolContext
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from skyrl_gym.python_execution import PythonKernel

NEMOTRON_PYTHON_INTERFACE = "nemotron-agent:ns_tools_simple_agent"
PYTHON_TOOL_NAME = "stateful_python_code_exec"
TOOL_DEFINITIONS = TypeAdapter(tuple[FunctionDefinition, ...])


@dataclass
class NemotronPythonSession:
    kernel: PythonKernel

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        if name != PYTHON_TOOL_NAME:
            return ToolCallResult(json.dumps({"error": f"Unknown tool: {name}"}))
        code = arguments.get("code")
        if not isinstance(code, str):
            return ToolCallResult(json.dumps({"error": "stateful_python_code_exec requires a string code argument"}))
        result = await self.kernel.execute(code, timeout=10)
        return ToolCallResult((result.stdout + result.stderr).decode().removesuffix("\n"))

    async def close(self) -> None:
        await self.kernel.close()


class NemotronPythonProvider:
    """Keep the source tool schema and execute calls in one task-local namespace."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        tools = TOOL_DEFINITIONS.validate_python(requirement.initial_state["tools"])
        if len(tools) != 1 or tools[0].name != PYTHON_TOOL_NAME:
            raise ValueError("NeMo Python provider requires its source stateful Python tool schema")
        return tools

    async def open(self, context: ToolContext) -> NemotronPythonSession:
        if context.machine is None:
            raise ValueError("NeMo Python provider requires the source task machine")
        self.tools(context.requirement)
        kernel = PythonKernel(context.machine)
        await kernel.start()
        return NemotronPythonSession(kernel)
