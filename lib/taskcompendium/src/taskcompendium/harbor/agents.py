# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat, shell-tool, and deterministic replay agents for Harbor."""

import asyncio
import json
import urllib.request
from typing import Any

from harbor.agents.base import BaseAgent, TurnCapExhaustedError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a command in the task's persistent shell environment.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def _record(logs_dir, transcript: list[dict], response: str, context: AgentContext) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "response.txt").write_text(response)
    (logs_dir / "transcript.json").write_text(json.dumps(transcript))
    context.metadata = {"assistant_final": response, "turns": len(transcript)}


class ReplayAgent(BaseAgent):
    """Replay specified shell actions and a raw final answer through a real trial."""

    def __init__(self, *args, response: str = "", commands: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        self.commands = commands or []

    @staticmethod
    def name() -> str:
        return "taskcompendium-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        for command in self.commands:
            result = await environment.exec(command)
            transcript.append({"role": "assistant", "command": command})
            transcript.append({"role": "tool", "content": result.model_dump()})
        transcript.append({"role": "assistant", "content": self.response})
        _record(self.logs_dir, transcript, self.response, context)


class DirectChatAgent(BaseAgent):
    """Use an OpenAI-compatible chat endpoint without exposing execution tools."""

    def __init__(
        self, *args, api_base: str, api_key: str = "", max_tokens: int = 4096, request_timeout: float = 120, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        if max_tokens <= 0 or request_timeout <= 0:
            raise ValueError("Token and request budgets must be positive")
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
        body = {"model": self.model_name, "messages": messages, "max_tokens": self.max_tokens, "temperature": 0}
        if tools is not None:
            body["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as result:
            payload = json.load(result)
        return payload["choices"][0]["message"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        message = await asyncio.to_thread(self._completion, transcript)
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            raise ValueError("Direct chat requires a textual assistant final response")
        transcript.append(message)
        _record(self.logs_dir, transcript, message["content"], context)


class ShellToolAgent(DirectChatAgent):
    """Chat with explicit shell tool calls, supported by ShellSim and Docker."""

    def __init__(self, *args, max_turns: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "taskcompendium-shell-tool"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        transcript: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        for _ in range(self.max_turns):
            message = await asyncio.to_thread(self._completion, transcript, [SHELL_TOOL])
            transcript.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("Final response must contain text")
                _record(self.logs_dir, transcript, content, context)
                return
            for call in calls:
                if call["function"]["name"] != "shell":
                    raise ValueError("Only the shell tool is supported")
                arguments = json.loads(call["function"]["arguments"])
                if set(arguments) != {"command"} or not isinstance(arguments["command"], str):
                    raise ValueError("Shell tool requires exactly one string command")
                result = await environment.exec(arguments["command"])
                transcript.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result.model_dump())}
                )
            _record(self.logs_dir, transcript, "", context)
        raise TurnCapExhaustedError("Shell-tool agent exhausted its turn budget")
