# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the pinned Workplace state machine without a shell fallback."""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue, TypeAdapter
from shellbox.machine import Command, Machine
from taskcompendium.models import FunctionDefinition, ProviderRequirement, TaskSpec
from taskcompendium.submission import conversation_messages

from rolloutengine.harness.shell import ToolCallResult, ToolContext
from rolloutengine.models import Prompt

WORKPLACE_ACTION_INTERFACE = "workplace-native-tools:v1"
NATIVE_MODEL_STEPS = 6
STATE_OBJECT = TypeAdapter(dict[str, Any])


def workplace_request(requirement: ProviderRequirement) -> dict[str, Any]:
    """Validate the source's model protocol before advertising its tools."""
    state = STATE_OBJECT.validate_python(requirement.initial_state)
    if requirement.action_interface != WORKPLACE_ACTION_INTERFACE:
        raise ValueError("Unexpected Workplace action interface")
    if state["native_max_steps"] != NATIVE_MODEL_STEPS or state["native_model_interface"] != "responses_api":
        raise ValueError("Workplace model protocol differs from the native six-step loop")
    request = STATE_OBJECT.validate_python(state["native_request"])
    if request["parallel_tool_calls"] is not False or request["temperature"] != 1.0:
        raise ValueError("Workplace native sampling options differ")
    if len(request["tools"]) != 27:
        raise ValueError("Workplace requires its original 27 tool schemas")
    return request


def workplace_prompt(task: TaskSpec) -> Prompt:
    """Keep the native conversation, function schemas and sampling options exact."""
    if set(task.environment_requirements.tool_providers) != {"workplace"} or task.final_tools or task.tool_bindings:
        raise ValueError("Workplace requires its native provider without additional tools")
    requirement = task.environment_requirements.tool_providers["workplace"]
    request = workplace_request(requirement)
    definitions = WorkplaceProvider().tools(requirement)
    if definitions != task.interaction_tools or conversation_messages(task.context.events) != request["input"]:
        raise ValueError("Workplace public task differs from its native request")
    tools = [
        {"type": "function", "function": {k: v for k, v in item.items() if k != "type"}} for item in request["tools"]
    ]
    return Prompt(tuple(request["input"]), {"tools": tools, "parallel_tool_calls": False, "temperature": 1.0})


@dataclass
class WorkplaceSession:
    machine: Machine
    requirement: ProviderRequirement

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        request = workplace_request(self.requirement)
        if name not in {tool["name"] for tool in request["tools"]}:
            raise ValueError(f"Undeclared native Workplace tool: {name}")
        state = STATE_OBJECT.validate_python(self.requirement.initial_state)
        payload = json.dumps({"name": name, "arguments": json.dumps(arguments)}).encode()
        result = await self.machine.run(
            Command(
                argv=("/opt/workplace-env/bin/python", state["runtime"], "call", "--root", state["state_root"]),
                stdin=payload,
                timeout=None,
            )
        )
        if result.exit_code != 0 or result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(f"Native Workplace tool failed: {result.stderr.decode(errors='replace')}")
        content = result.stdout.decode().strip()
        json.loads(content)
        return ToolCallResult(content)

    async def close(self) -> None:
        pass


class WorkplaceProvider:
    """Run the original task-declared runtime on the task's own machine."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        request = workplace_request(requirement)
        return tuple(
            FunctionDefinition.model_validate({k: v for k, v in item.items() if k != "type"})
            for item in request["tools"]
        )

    async def open(self, context: ToolContext) -> WorkplaceSession:
        workplace_request(context.requirement)
        if context.machine is None:
            raise ValueError("Workplace native tools require their state machine")
        return WorkplaceSession(context.machine, context.requirement)
