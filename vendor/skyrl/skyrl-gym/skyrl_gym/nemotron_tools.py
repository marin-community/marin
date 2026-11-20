"""Serve the NeMo math tool with SkyRL's task-local persistent Python kernel."""

from dataclasses import dataclass

from pydantic import JsonValue, TypeAdapter
from rolloutengine.harness.shell import ToolCallResult, ToolContext
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from skyrl_gym.python_execution import PYTHON_TOOL_NAME, PythonKernel, python_tool_output

NEMOTRON_PYTHON_INTERFACE = "nemotron-agent:ns_tools_simple_agent"
TOOL_DEFINITIONS = TypeAdapter(tuple[FunctionDefinition, ...])


@dataclass
class NemotronPythonSession:
    kernel: PythonKernel

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        return ToolCallResult(await python_tool_output(self.kernel, name, arguments))

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
