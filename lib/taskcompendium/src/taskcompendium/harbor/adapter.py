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
from upath import UPath

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.lowering import (
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    read_specification,
    read_submission_convention,
)
from taskcompendium.submission import AnswerFormat, answer_call_tool
from taskcompendium.verifier_registry import grade_answer

RESPONSE_FILE = "response.txt"
ACTION_FILE = "action.json"
CHAT_COMPLETIONS_PATH = "/chat/completions"
AGENT_LOGS_PATH = "/logs/agent"
VERIFIER_LOGS_PATH = "/logs/verifier"
ARTIFACTS_LOGS_PATH = "/logs/artifacts"
TESTS_PATH = "/tests"
HARBOR_DOWNLOAD_DIRS = frozenset({AGENT_LOGS_PATH, ARTIFACTS_LOGS_PATH})
HARBOR_EMPTY_DIRS = HARBOR_DOWNLOAD_DIRS | {VERIFIER_LOGS_PATH, TESTS_PATH}


def _record_submission(
    logs_dir: Path | UPath,
    messages: list[dict[str, Any]],
    assistant_final: object,
    final_message: dict[str, Any],
    filename: str,
    payload: str,
    context: AgentContext,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / filename).write_text(payload)
    context.metadata = {
        "assistant_final": assistant_final,
        "turns": 1,
        "all_messages": [*messages, final_message],
        "summarization_count": 0,
        "tools": [],
    }


def _record_answer(logs_dir: Path | UPath, messages: list[dict[str, Any]], answer: str, context: AgentContext) -> None:
    _record_submission(
        logs_dir, messages, answer, {"role": "assistant", "content": answer}, RESPONSE_FILE, answer, context
    )


def _record_action(
    logs_dir: Path | UPath, messages: list[dict[str, Any]], action: dict[str, Any], context: AgentContext
) -> None:
    _record_submission(logs_dir, messages, action, action, ACTION_FILE, json.dumps(action), context)


def _chat_completion(api_base: str, api_key: str | None, request_timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
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
        if not set(map(str, dirs)).issubset(HARBOR_EMPTY_DIRS):
            raise ValueError("Direct chat has no filesystem")

    async def upload_file(self, source_path, target_path) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def upload_dir(self, source_dir, target_dir) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def download_file(self, source_path, target_path) -> None:
        raise ValueError("Direct chat has no filesystem")

    async def download_dir(self, source_dir, target_dir) -> None:
        if source_dir not in HARBOR_DOWNLOAD_DIRS:
            raise ValueError("Direct chat has no filesystem")


class ReplayAgent(BaseAgent):
    """Submit a caller-provided final response without a model request."""

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
        _record_answer(self.logs_dir, [{"role": "user", "content": instruction}], self.response, context)


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
        request_timeout: float,
        events: list[dict[str, Any]],
        submission_instruction: str,
        api_key_env: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Direct chat requires a model name")
        self.api_base = api_base.rstrip("/")
        self.api_key = os.environ[api_key_env] if api_key_env is not None else None
        self.request_timeout = request_timeout
        self.events = events
        self.submission_instruction = submission_instruction

    @staticmethod
    def name() -> str:
        return "taskcompendium-chat"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    def _chat_messages(self) -> list[dict[str, Any]]:
        messages = []
        for event in self.events:
            if event["type"] == "message":
                messages.append({"role": event["role"], "content": event["content"]})
            elif event["type"] == "assistant_tool_calls":
                messages.append(
                    {
                        "role": "assistant",
                        "content": event["content"],
                        "tool_calls": [
                            {
                                "id": call["call_id"],
                                "type": "function",
                                "function": {"name": call["name"], "arguments": call["arguments"]},
                            }
                            for call in event["calls"]
                        ],
                    }
                )
            else:
                messages.append({"role": "tool", "tool_call_id": event["call_id"], "content": event["content"]})
        if self.submission_instruction:
            messages.append({"role": "user", "content": self.submission_instruction})
        return messages

    def _completion(self) -> str:
        body = {"model": self.model_name, "messages": self._chat_messages()}
        message = _chat_completion(self.api_base, self.api_key, self.request_timeout, body)
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            raise ValueError("Direct chat requires a textual final answer")
        return message["content"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._completion)
        _record_answer(self.logs_dir, self._chat_messages(), response, context)


class NativeActionAgent(DirectChatAgent):
    """Request one native final action and retain it without tool dispatch."""

    def __init__(
        self,
        *args,
        functions: list[dict[str, Any]],
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.functions = functions
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
        body: dict[str, Any] = {"model": self.model_name, "messages": self._chat_messages(), "tools": tools}
        if self.tool_choice is not None:
            body["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.parallel_tool_calls
        return _chat_completion(self.api_base, self.api_key, self.request_timeout, body)

    # Harbor passes instruction by keyword; the source messages are the native-action prompt.
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._action_completion)
        _record_action(self.logs_dir, self._chat_messages(), response, context)


class AnswerCallAgent(DirectChatAgent):
    """Collect a final answer function call without dispatching it."""

    @staticmethod
    def name() -> str:
        return "taskcompendium-answer-call"

    def _answer_completion(self) -> dict[str, Any]:
        messages = self._chat_messages()
        body = {
            "model": self.model_name,
            "messages": messages,
            "tools": [answer_call_tool()],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        return _chat_completion(self.api_base, self.api_key, self.request_timeout, body)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._answer_completion)
        _record_action(self.logs_dir, self._chat_messages(), response, context)


class SemanticVerifier(BaseVerifier):
    """Grade the submitted answer against the task's private reference."""

    async def verify(self) -> VerifierResult:
        try:
            root = self.task.paths.task_dir
            specification = read_specification(root / SPECIFICATION_FILE)
            convention = read_submission_convention(root / SUBMISSION_CONVENTION_FILE)
            action = convention.answer_format in {AnswerFormat.ANSWER_CALL, AnswerFormat.FINAL_ACTION}
            response_path = self.trial_paths.agent_dir / (ACTION_FILE if action else RESPONSE_FILE)
            response = response_path.read_text() if response_path.exists() else None
            result = grade_answer(specification, convention, response, self.environment)
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
