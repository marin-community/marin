# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat, shell-tool, and deterministic replay agents for Harbor."""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

import msgspec
from harbor.agents.base import BaseAgent, TurnCapExhaustedError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from taskcompendium.execution import HarnessToolBinding, ShellToolBinding
from taskcompendium.harbor.providers import ProviderActionEnvironment
from taskcompendium.models import FinalActionSubmission


def shell_tool_definition(binding: ShellToolBinding) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": binding.name,
            "description": "Execute a command in the task's persistent shell environment.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def _record(
    logs_dir, transcript: list[dict], response: str, context: AgentContext, tools: list[dict] | None = None
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "response.txt").write_text(response)
    (logs_dir / "transcript.json").write_text(json.dumps(transcript))
    (logs_dir / "tools.json").write_text(json.dumps(tools or []))
    context.metadata = {
        "assistant_final": response,
        "turns": len(transcript),
        "all_messages": list(transcript),
        "summarization_count": 0,
        "tools": tools or [],
    }


class ReplayAgent(BaseAgent):
    """Replay specified shell actions and a raw final answer through a real trial."""

    def __init__(
        self,
        *args,
        response: str = "",
        commands: list[str] | None = None,
        steps: list[dict[str, Any]] | None = None,
        tool_binding: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response = response
        self.commands = commands or []
        self.steps = steps
        self.step_index = 0
        self.history: list[dict[str, Any]] = []
        self.tool_binding = msgspec.convert(tool_binding, type=HarnessToolBinding) if tool_binding is not None else None
        if self.tool_binding is not None and self.tool_binding.interface != "terminal":
            raise ValueError("Replay requires a terminal tool interface")

    @staticmethod
    def name() -> str:
        return "taskcompendium-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript = self.history
        transcript.append({"role": "user", "content": instruction})
        if self.steps is None:
            response, commands = self.response, self.commands
        else:
            attempt = self.steps[self.step_index]
            self.step_index += 1
            response, commands = attempt["response"], attempt["commands"]
        if commands and self.tool_binding is None:
            raise ValueError("Replay commands require a declared tool binding")
        tools = (
            [shell_tool_definition(ShellToolBinding("terminal", self.tool_binding.backend))] if self.tool_binding else []
        )
        for command in commands:
            result = await environment.exec(command)
            call_id = f"replay-{len(transcript)}"
            transcript.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": json.dumps({"command": command})},
                        }
                    ],
                }
            )
            transcript.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result.model_dump())})
        transcript.append({"role": "assistant", "content": response})
        _record(self.logs_dir, transcript, response, context, tools)


class DirectChatAgent(BaseAgent):
    """Use an OpenAI-compatible chat endpoint without exposing execution tools."""

    def __init__(
        self,
        *args,
        api_base: str,
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        request_timeout: float = 120,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.history: list[dict[str, Any]] = []
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.chat_template_kwargs = chat_template_kwargs
        if max_tokens <= 0 or temperature < 0 or request_timeout <= 0:
            raise ValueError("Token and request budgets must be positive and temperature must be nonnegative")
        if self.model_name is None:
            raise ValueError("Chat agent requires an explicit model")

    @staticmethod
    def name() -> str:
        return "taskcompendium-chat"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    def _completion(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools is not None:
            body["tools"] = tools
        if self.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as result:
                payload = json.load(result)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion HTTP {error.code}: {detail}") from error
        return payload["choices"][0]["message"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript = self.history
        transcript.append({"role": "user", "content": instruction})
        message = await asyncio.to_thread(self._completion, transcript)
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            raise ValueError("Direct chat requires a textual assistant final response")
        transcript.append(message)
        _record(self.logs_dir, transcript, message["content"], context)


class ShellToolAgent(DirectChatAgent):
    """Chat with explicit shell tool calls, supported by ShellSim and Docker."""

    def __init__(self, *args, tool_binding: dict[str, Any], max_turns: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.tool_binding = msgspec.convert(tool_binding, type=ShellToolBinding)

    @staticmethod
    def name() -> str:
        return "taskcompendium-shell-tool"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript = self.history
        transcript.append({"role": "user", "content": instruction})
        tools = [shell_tool_definition(self.tool_binding)]
        for _ in range(self.max_turns):
            message = await asyncio.to_thread(self._completion, transcript, tools)
            transcript.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("Final response must contain text")
                _record(self.logs_dir, transcript, content, context, tools)
                return
            for call in calls:
                if call["function"]["name"] != self.tool_binding.name:
                    raise ValueError("Model called an undeclared tool")
                arguments = json.loads(call["function"]["arguments"])
                if set(arguments) != {"command"} or not isinstance(arguments["command"], str):
                    raise ValueError("Shell tool requires exactly one string command")
                result = await environment.exec(arguments["command"])
                transcript.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result.model_dump())}
                )
            _record(self.logs_dir, transcript, "", context, tools)
        raise TurnCapExhaustedError("Shell-tool agent exhausted its turn budget")


def action_output_definitions(contract: FinalActionSubmission) -> list[dict[str, Any]]:
    """Translate the public Responses-style definitions to Chat Completions wire shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": function.name,
                "parameters": function.parameters,
                **({"description": function.description} if function.description is not None else {}),
                **({"strict": function.strict} if function.strict is not None else {}),
            },
        }
        for function in contract.functions
    ]


class ActionOutputAgent(DirectChatAgent):
    """Request one native final action and retain it without dispatching a tool."""

    def __init__(self, *args, output_contracts: list[dict[str, Any]], **kwargs):
        super().__init__(*args, **kwargs)
        self.output_contracts = [msgspec.convert(contract, type=FinalActionSubmission) for contract in output_contracts]
        self.step_index = 0

    @staticmethod
    def name() -> str:
        return "taskcompendium-final-action"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.step_index >= len(self.output_contracts):
            raise ValueError("No final-action contract for Harbor step")
        contract = self.output_contracts[self.step_index]
        self.step_index += 1
        transcript = self.history
        transcript.append({"role": "user", "content": instruction})
        message = await asyncio.to_thread(self._completion, transcript, action_output_definitions(contract))
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("Model action must use native tool-call records")
        allowed = {function.name for function in contract.functions}
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if (
                not isinstance(function, dict)
                or function.get("name") not in allowed
                or not isinstance(function.get("arguments"), str)
            ):
                raise ValueError("Model emitted an action outside the public output contract")
        if not calls and (not contract.allow_message or not isinstance(message.get("content"), str)):
            raise ValueError("Model action must be a permitted message or function call")
        transcript.append(message)
        _record(self.logs_dir, transcript, message.get("content") or "", context, action_output_definitions(contract))


class ActionOutputReplayAgent(BaseAgent):
    """Replay a native final action for Harbor validation without invoking its domain tool."""

    def __init__(self, *args, output_contracts: list[dict[str, Any]], actions: list[dict[str, Any]], **kwargs):
        super().__init__(*args, **kwargs)
        self.output_contracts = [msgspec.convert(contract, type=FinalActionSubmission) for contract in output_contracts]
        self.actions = actions
        self.step_index = 0
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def name() -> str:
        return "taskcompendium-final-action-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.step_index >= len(self.output_contracts) or self.step_index >= len(self.actions):
            raise ValueError("No replay action for Harbor step")
        contract = self.output_contracts[self.step_index]
        action = self.actions[self.step_index]
        self.step_index += 1
        calls = action.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("Replay action must use a native tool-call list")
        allowed = {function.name for function in contract.functions}
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if (
                not isinstance(function, dict)
                or function.get("name") not in allowed
                or not isinstance(function.get("arguments"), str)
            ):
                raise ValueError("Replay action is outside the public output contract")
        if not calls and (not contract.allow_message or not isinstance(action.get("content"), str)):
            raise ValueError("Replay action must be a permitted message or function call")
        message = {"role": "assistant", **action}
        transcript = self.history
        transcript.extend(({"role": "user", "content": instruction}, message))
        _record(self.logs_dir, transcript, action.get("content") or "", context, action_output_definitions(contract))


class ProviderToolAgent(DirectChatAgent):
    """Dispatch only the declared provider interface; domain behavior stays in its adapter."""

    def __init__(self, *args, max_turns: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "taskcompendium-provider-tool"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not isinstance(environment, ProviderActionEnvironment):
            raise ValueError("Provider tool agent requires a provider action environment")
        transcript = self.history
        transcript.append({"role": "user", "content": instruction})
        tools = await environment.native_tool_definitions()
        for _ in range(self.max_turns):
            message = await asyncio.to_thread(self._completion, transcript, tools)
            transcript.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("Final response must contain text")
                _record(self.logs_dir, transcript, content, context, tools)
                return
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(call_id, str):
                    raise ValueError("Provider tool call is malformed")
                name, arguments = function.get("name"), function.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, str):
                    raise ValueError("Provider tool call is malformed")
                result = await environment.dispatch_action(name, arguments, call_id)
                transcript.append({"role": "tool", "tool_call_id": call_id, "content": result})
        raise TurnCapExhaustedError("Provider-tool agent exhausted its turn budget")
