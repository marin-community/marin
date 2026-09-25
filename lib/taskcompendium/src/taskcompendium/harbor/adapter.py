# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal Harbor runtime for direct-chat answer tasks."""

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from taskcompendium.grading import GradeResult, Outcome, grade_answer
from taskcompendium.lowering import RENDERING_FILE, SPECIFICATION_FILE, read_rendering, read_specification
from taskcompendium.rendering import AnswerFormat

RESPONSE_FILE = "response.txt"
ACTION_FILE = "action.json"
CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_REQUEST_TIMEOUT = 120
AGENT_LOGS_PATH = "/logs/agent"
VERIFIER_LOGS_PATH = "/logs/verifier"
ARTIFACTS_LOGS_PATH = "/logs/artifacts"
TESTS_PATH = "/tests"


def _record_answer(logs_dir: Path, instruction: str, answer: str, context: AgentContext) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / RESPONSE_FILE).write_text(answer)
    context.metadata = {
        "assistant_final": answer,
        "turns": 1,
        "all_messages": [{"role": "user", "content": instruction}, {"role": "assistant", "content": answer}],
        "summarization_count": 0,
        "tools": [],
    }


def _record_action(
    logs_dir: Path, messages: list[dict[str, str]], action: dict[str, Any], context: AgentContext
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / ACTION_FILE).write_text(json.dumps(action))
    context.metadata = {
        "assistant_final": action,
        "turns": 1,
        "all_messages": [*messages, action],
        "summarization_count": 0,
        "tools": [],
    }


def _chat_completion(
    api_base: str, api_key_env: str | None, request_timeout: float, body: dict[str, Any]
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key_env is not None:
        headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
    request = urllib.request.Request(
        f"{api_base}{CHAT_COMPLETIONS_PATH}", data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            message = json.load(response)["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat completion HTTP {error.code}: {detail}") from error
    if not isinstance(message, dict):
        raise ValueError("Chat completion requires an assistant message object")
    return message


class NoToolEnvironment(BaseEnvironment):
    """A Harbor environment with no agent filesystem or execution tools."""

    @staticmethod
    def type() -> str:
        return "taskcompendium-direct-chat"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        if (self.environment_dir / "inputs").exists():
            raise ValueError("Direct chat cannot expose filesystem inputs")

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if command == "pwd":
            return ExecResult(stdout="/app\n", stderr="", return_code=0)
        raise ValueError("Direct chat has no shell")

    async def empty_dirs(self, dirs, *, chmod: bool = True) -> None:
        if not set(map(str, dirs)).issubset({AGENT_LOGS_PATH, VERIFIER_LOGS_PATH, ARTIFACTS_LOGS_PATH, TESTS_PATH}):
            raise ValueError("Direct chat has no filesystem")

    async def upload_file(self, source_path, target_path) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def upload_dir(self, source_dir, target_dir) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def download_file(self, source_path, target_path) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def download_dir(self, source_dir, target_dir) -> None:
        if source_dir not in {AGENT_LOGS_PATH, ARTIFACTS_LOGS_PATH}:
            raise ValueError("Direct chat has no filesystem")


class ReplayAgent(BaseAgent):
    """Supply a fixed final response to exercise the real Harbor trial path."""

    def __init__(self, *args, response: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response

    @staticmethod
    def name() -> str:
        return "taskcompendium-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _record_answer(self.logs_dir, instruction, self.response, context)


class ActionReplayAgent(BaseAgent):
    """Replay a native final action without invoking its advertised function."""

    def __init__(self, *args, response: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response

    @staticmethod
    def name() -> str:
        return "taskcompendium-action-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _record_action(self.logs_dir, [{"role": "user", "content": instruction}], self.response, context)


class DirectChatAgent(BaseAgent):
    """Send the rendered request to an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *args,
        api_base: str,
        api_key_env: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Direct chat requires a model name")
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.request_timeout = request_timeout

    @staticmethod
    def name() -> str:
        return "taskcompendium-chat"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    def _completion(self, instruction: str) -> str:
        body = {"model": self.model_name, "messages": [{"role": "user", "content": instruction}]}
        message = _chat_completion(self.api_base, self.api_key_env, self.request_timeout, body)
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            raise ValueError("Direct chat requires a textual final answer")
        return message["content"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._completion, instruction)
        _record_answer(self.logs_dir, instruction, response, context)


class NativeActionAgent(DirectChatAgent):
    """Request one native final action and retain it without tool dispatch."""

    def __init__(
        self,
        *args,
        functions: list[dict[str, Any]],
        messages: list[dict[str, str]],
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.functions = functions
        self.messages = messages
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

    @staticmethod
    def name() -> str:
        return "taskcompendium-final-action"

    def _action_completion(self) -> dict[str, Any]:
        tools = []
        for function in self.functions:
            definition = {"name": function["name"], "parameters": function["parameters"]}
            for key in ("description", "strict"):
                if function.get(key) is not None:
                    definition[key] = function[key]
            tools.append({"type": "function", "function": definition})
        body = {"model": self.model_name, "messages": self.messages, "tools": tools}
        if self.tool_choice is not None:
            body["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.parallel_tool_calls
        return _chat_completion(self.api_base, self.api_key_env, self.request_timeout, body)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._action_completion)
        _record_action(self.logs_dir, self.messages, response, context)


class SemanticVerifier(BaseVerifier):
    """Grade private task metadata with access to Harbor's verifier environment."""

    async def verify(self) -> VerifierResult:
        try:
            root = self.task.paths.task_dir
            specification = read_specification(root / SPECIFICATION_FILE)
            rendering = read_rendering(root / RENDERING_FILE)
            action = rendering.answer_format == AnswerFormat.FINAL_ACTION
            response_path = self.trial_paths.agent_dir / (ACTION_FILE if action else RESPONSE_FILE)
            response = response_path.read_text() if response_path.exists() else None
            result = grade_answer(specification, rendering, response, self.environment)
        except Exception as error:
            result = GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")
            self._write_result(result)
            raise RuntimeError(result.error) from error
        self._write_result(result)
        if result.status != Outcome.GRADED or result.reward is None:
            raise RuntimeError(result.error or result.status.value)
        return VerifierResult(rewards={"reward": result.reward})

    def _write_result(self, result: GradeResult) -> None:
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_text(
            json.dumps({"status": result.status.value, "reward": result.reward, "error": result.error}) + "\n"
        )
