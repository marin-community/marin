"""Serve the NeMo Gym ``ns_tools_simple_agent`` stateful Python tool to TaskSpec shell harnesses.

Retained Nemotron Ultra Math-TIR tasks declare the ``nemotron-agent:ns_tools_simple_agent`` provider
with the tool definitions it advertises. Each attempt runs one persistent interpreter on the task
machine and executes every ``stateful_python_code_exec`` call in it, with the source agent's
10-second per-call limit, exactly as the native Nemotron task session does.
"""

import json
from dataclasses import dataclass

from pydantic import JsonValue
from rolloutengine.harness.shell import ToolCallResult, ToolContext
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from skyrl_gym.python_execution import PythonKernel

NEMOTRON_PYTHON_ACTION_INTERFACE = "nemotron-agent:ns_tools_simple_agent"
PYTHON_TOOL = "stateful_python_code_exec"
PYTHON_CALL_TIMEOUT = 10.0


@dataclass
class NemotronPythonSession:
    kernel: PythonKernel

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        if name != PYTHON_TOOL:
            return ToolCallResult(json.dumps({"error": f"Unknown tool: {name}"}))
        code = arguments.get("code")
        if not isinstance(code, str):
            return ToolCallResult(json.dumps({"error": f"{PYTHON_TOOL} requires a string code argument"}))
        result = await self.kernel.execute(code, timeout=PYTHON_CALL_TIMEOUT)
        return ToolCallResult((result.stdout + result.stderr).decode(errors="replace").removesuffix("\n"))

    async def close(self) -> None:
        await self.kernel.close()


class NemotronPythonProvider:
    """Advertise the requirement's declared tools and run them in a task-machine interpreter."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        state = requirement.initial_state
        declared = state.get("tools") if isinstance(state, dict) else None
        if not isinstance(declared, list) or not declared:
            raise ValueError(f"{NEMOTRON_PYTHON_ACTION_INTERFACE} requires declared tools")
        tools = tuple(FunctionDefinition.model_validate(tool) for tool in declared)
        if [tool.name for tool in tools] != [PYTHON_TOOL]:
            raise ValueError(f"{NEMOTRON_PYTHON_ACTION_INTERFACE} serves only {PYTHON_TOOL}")
        return tools

    async def open(self, context: ToolContext) -> NemotronPythonSession:
        if context.machine is None:
            raise ValueError(f"{PYTHON_TOOL} requires a task machine")
        kernel = PythonKernel(context.machine)
        await kernel.start()
        return NemotronPythonSession(kernel)
