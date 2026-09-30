# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exported tasks through Harbor's agent and verifier lifecycle.

ChatAgent sends one prepared model request and saves typed conversation evidence.
Advertised function calls are submissions and are never executed.
NoToolEnvironment satisfies Harbor's lifecycle without exposing shell or file
access to the agent. SemanticVerifier reads the private task specification and
saved conversation, extracts the answer using its convention, and grades it.
Agent messages and grading outcomes are written to host-side trial logs.
"""

import asyncio
import json
import os
import tempfile
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
from taskcompendium.harbor.protocol import chat_conversation
from taskcompendium.harbor.script_runtime import ScriptSubmission, run_script_verifier
from taskcompendium.harbor.workspace import UnsafeWorkspaceError, capture_workspace
from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    PRIVATE_RESOURCES_DIR,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    WORKSPACE_DOCKER_ENVIRONMENT,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_exported_private_resources,
)
from taskcompendium.models import AnswerType, ConversationTrace, TaskSpec
from taskcompendium.submission import SubmissionConvention, extract_answer
from taskcompendium.verifier_registry import grade_answer, resolve_verifier
from taskcompendium.verifiers.script import ScriptVerifier

SUBMISSION_FILE = "submission.json"
CHAT_RESPONSE_FILE = "chat-response.json"
CHAT_COMPLETIONS_PATH = "/chat/completions"
# Harbor normally downloads agent logs and task-produced artifacts from these paths.
AGENT_LOGS_PATH = "/logs/agent"
ARTIFACTS_LOGS_PATH = "/logs/artifacts"
# Harbor uses these paths for verifier output and private verifier inputs.
VERIFIER_LOGS_PATH = "/logs/verifier"
TESTS_PATH = "/tests"
# Direct chat writes logs on the host, so these bookkeeping operations are no-ops.
# Only Harbor's standard paths are accepted; other filesystem operations fail.
HARBOR_DOWNLOAD_DIRS = frozenset({AGENT_LOGS_PATH, ARTIFACTS_LOGS_PATH})
HARBOR_EMPTY_DIRS = HARBOR_DOWNLOAD_DIRS | {VERIFIER_LOGS_PATH, TESTS_PATH}


def _record_submission(
    logs_dir: Path | UPath,
    messages: list[dict[str, Any]],
    response: dict[str, Any],
    context: AgentContext,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / CHAT_RESPONSE_FILE).write_text(json.dumps(response))
    context.metadata = {
        "assistant_final": response,
        "turns": 1,
        "all_messages": [*messages, response],
        "summarization_count": 0,
        "tools": [],
    }
    conversation = chat_conversation([*messages, response])
    (logs_dir / SUBMISSION_FILE).write_text(conversation.model_dump_json())


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
    """Run standalone direct-chat Harbor trials without agent filesystem or tools."""

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


class ChatAgent(BaseAgent):
    """Send one prepared chat request and retain the final assistant message."""

    def __init__(
        self,
        *args,
        api_base: str,
        request_timeout: float,
        request: dict[str, Any],
        api_key_env: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            raise ValueError("Chat requires a model name")
        self.api_base = api_base.rstrip("/")
        self.api_key = os.environ[api_key_env] if api_key_env is not None else None
        self.request_timeout = request_timeout
        self.request = request

    @staticmethod
    def name() -> str:
        return "taskcompendium-chat"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        response = await asyncio.to_thread(
            _chat_completion,
            self.api_base,
            self.api_key,
            self.request_timeout,
            {**self.request, "model": self.model_name},
        )
        _record_submission(self.logs_dir, self.request["messages"], response, context)


class ReplayAgent(BaseAgent):
    """Submit a fixed text response and optionally prepare a workspace."""

    def __init__(self, *args, response: str, request: dict[str, Any], workspace_command: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        self.request = request
        self.workspace_command = workspace_command

    @staticmethod
    def name() -> str:
        return "taskcompendium-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self.workspace_command is not None:
            completed = await environment.exec(self.workspace_command, cwd="/app")
            if completed.return_code != 0:
                raise RuntimeError(
                    f"Workspace replay command failed ({completed.return_code}): "
                    f"{(completed.stderr or completed.stdout or '')[-1000:]}"
                )
        _record_submission(
            self.logs_dir, self.request["messages"], {"role": "assistant", "content": self.response}, context
        )


class SemanticVerifier(BaseVerifier):
    """Grade the final submission with the task's private verifier."""

    async def verify(self) -> VerifierResult:
        script_task = False
        try:
            root = self.task.paths.task_dir
            specification = read_specification(root / SPECIFICATION_FILE)
            convention = read_submission_convention(root / SUBMISSION_CONVENTION_FILE)
            conversation_path = self.trial_paths.agent_dir / SUBMISSION_FILE
            conversation = ConversationTrace.model_validate_json(conversation_path.read_text())
            verifier = resolve_verifier(specification.verifier)
            if isinstance(verifier, ScriptVerifier):
                script_task = True
                result = await self._grade_script(specification, convention, conversation, verifier)
            else:
                result = grade_answer(specification, convention, conversation, self.environment)
        except UnsafeWorkspaceError as error:
            result = GradeResult(Outcome.INVALID_TASK, None, "Final workspace cannot be captured safely")
            self._write_result(result)
            raise RuntimeError(result.error) from error
        except Exception as error:
            if script_task:
                self.logger.exception("Script verifier failed")
                result = GradeResult(Outcome.INFRA_ERROR, None, "Script verifier infrastructure failure")
            else:
                result = GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")
            self._write_result(result)
            raise RuntimeError(result.error) from error
        self._write_result(result)
        if result.status != Outcome.GRADED or result.reward is None:
            raise RuntimeError(result.error or result.status.value)
        return VerifierResult(rewards={"reward": result.reward})

    async def _grade_script(
        self,
        specification: TaskSpec,
        convention: SubmissionConvention,
        conversation: ConversationTrace,
        verifier: ScriptVerifier,
    ) -> GradeResult:
        if specification.answer_type in (AnswerType.TEXT, AnswerType.NUMBER):
            try:
                answer = extract_answer(conversation.events[-1], convention)
            except (ValueError, TypeError) as error:
                return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        else:
            answer = None
        root = self.task.paths.task_dir
        try:
            validate_exported_private_resources(specification, root)
        except ValueError:
            return GradeResult(Outcome.INVALID_TASK, None, "Pinned private resources are unavailable or changed")
        environment_config = read_environment_config(root / ENVIRONMENT_CONFIG_FILE)
        submission = ScriptSubmission(
            protocol_version=verifier.protocol_version,
            answer_type=specification.answer_type,
            convention_id=convention.id,
            answer=answer,
        )
        with tempfile.TemporaryDirectory(prefix="taskcompendium-snapshot-") as temporary:
            scratch = Path(temporary)
            workspace = scratch / "workspace"
            if environment_config.environment == WORKSPACE_DOCKER_ENVIRONMENT:
                await capture_workspace(self.environment, workspace)
            else:
                workspace.mkdir()

            def resolve_exported(uri: str) -> bytes:
                for resource in verifier.resources:
                    if resource.uri == uri:
                        return (root / PRIVATE_RESOURCES_DIR / resource.path).read_bytes()
                raise ValueError("Unknown private resource reference")

            result = await asyncio.to_thread(
                run_script_verifier,
                verifier,
                submission,
                workspace,
                scratch / "result",
                resolve_exported,
            )
            if result.status == Outcome.GRADED:
                return GradeResult(Outcome.GRADED, result.reward)
            self.logger.warning("Script verification failed: %s", result.error)
            return GradeResult(result.status, None, "Script verification did not produce a score")

    def _write_result(self, result: GradeResult) -> None:
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_text(
            json.dumps({"status": result.status.value, "reward": result.reward, "error": result.error}) + "\n"
        )
