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
from taskcompendium.lowering import (
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    read_specification,
    read_submission_convention,
)

RESPONSE_FILE = "response.txt"
AGENT_LOGS_PATH = "/logs/agent"
ARTIFACTS_LOGS_PATH = "/logs/artifacts"


def _record_response(logs_dir: Path, instruction: str, response: str, context: AgentContext) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / RESPONSE_FILE).write_text(response)
    context.metadata = {
        "assistant_final": response,
        "turns": 1,
        "all_messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "summarization_count": 0,
        "tools": [],
    }


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
        if not set(map(str, dirs)).issubset({AGENT_LOGS_PATH, "/logs/verifier", ARTIFACTS_LOGS_PATH, "/tests"}):
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
        _record_response(self.logs_dir, instruction, self.response, context)


class DirectChatAgent(BaseAgent):
    """Send the rendered request to an OpenAI-compatible chat endpoint."""

    def __init__(self, *args, api_base: str, api_key_env: str | None = None, request_timeout: float = 120, **kwargs):
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
        headers = {"Content-Type": "application/json"}
        if self.api_key_env is not None:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                message: dict[str, Any] = json.load(response)["choices"][0]["message"]
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion HTTP {error.code}: {detail}") from error
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            raise ValueError("Direct chat requires a textual final answer")
        return message["content"]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(self._completion, instruction)
        _record_response(self.logs_dir, instruction, response, context)


class SemanticVerifier(BaseVerifier):
    """Grade the submitted answer against the task's private reference."""

    async def verify(self) -> VerifierResult:
        try:
            root = self.task.paths.task_dir
            specification = read_specification(root / SPECIFICATION_FILE)
            convention = read_submission_convention(root / SUBMISSION_CONVENTION_FILE)
            response_path = self.trial_paths.agent_dir / RESPONSE_FILE
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
