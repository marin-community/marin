# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exported chat tasks through Harbor's agent and verifier lifecycle.

ChatAgent sends the prepared conversation to a model endpoint. It dispatches
bound provider calls across turns and saves the full model-visible conversation
as typed grading evidence. Terminal calls end the interaction without dispatch.
The Harbor environment owns lifecycle and grading access; tool providers are
separate services. Trial logs retain raw model messages and tool observations.
"""

import asyncio
import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.harbor.protocol import assistant_message, chat_conversation
from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    ToolBinding,
    provider_class,
    read_environment_config,
    read_specification,
    read_submission_convention,
    selected_tool_definitions,
    validate_provider_surface,
)
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, ConversationTrace
from taskcompendium.provider_sources import PROVIDER_SOURCES_DIR, parse_git_provider
from taskcompendium.submission import GradingAttempt
from taskcompendium.verifier_registry import grade_answer

SUBMISSION_FILE = "submission.json"
CHAT_RESPONSE_FILE = "chat-response.json"
CHAT_COMPLETIONS_PATH = "/chat/completions"
# Harbor normally downloads agent logs and task-produced artifacts from these paths.
AGENT_LOGS_PATH = "/logs/agent"
ARTIFACTS_LOGS_PATH = "/logs/artifacts"
# Harbor uses these paths for verifier output and private verifier inputs.
VERIFIER_LOGS_PATH = "/logs/verifier"
TESTS_PATH = "/tests"
# Host chat writes logs on the host for direct and provider-backed trials.
# Only Harbor's standard paths are accepted; other filesystem operations fail.
HARBOR_DOWNLOAD_DIRS = frozenset({AGENT_LOGS_PATH, ARTIFACTS_LOGS_PATH})
HARBOR_EMPTY_DIRS = HARBOR_DOWNLOAD_DIRS | {VERIFIER_LOGS_PATH, TESTS_PATH}


@runtime_checkable
class ToolProvider(Protocol):
    """A callable tool service whose state is scoped to one trial."""

    async def native_tool_definitions(self) -> list[dict[str, Any]]: ...

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str: ...


@runtime_checkable
class ManagedToolProvider(Protocol):
    """Optional tool-provider lifecycle, independent of Harbor's lifecycle."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


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


class CompositeToolEnvironment(BaseEnvironment):
    """Run host-managed chat with zero or more trial-scoped tool providers."""

    def __init__(self, *args, tool_providers: dict[str, dict[str, Any]] | None = None, **kwargs):
        self.bindings = {name: ToolBinding.model_validate(item) for name, item in (tool_providers or {}).items()}
        self.providers: dict[str, ToolProvider] = {}
        self.tool_owners: dict[str, str] = {}
        super().__init__(*args, **kwargs)
        for name, binding in self.bindings.items():
            source = (
                self.environment_dir / PROVIDER_SOURCES_DIR / name
                if parse_git_provider(binding.provider) is not None
                else None
            )
            validate_provider_surface(binding, source)
            provider_type = provider_class(binding, source)
            if issubclass(provider_type, BaseEnvironment):
                raise TypeError(f"Tool provider {name!r} must not be a Harbor environment")
            provider = provider_type(**self.provider_kwargs(binding))
            if not isinstance(provider, ToolProvider):
                raise TypeError(f"Provider {name!r} does not expose tool methods")
            self.providers[name] = provider

    @staticmethod
    def type() -> str:
        return "taskcompendium-composite-tools"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        # Harbor requires this hook; provider bindings are validated before the trial starts.
        pass

    def provider_kwargs(self, binding: ToolBinding) -> dict[str, Any]:
        """Supply constructor arguments for a provider bound to this environment."""
        return {"seed_sha256": binding.seed_sha256, "action_interface": binding.action_interface}

    async def start(self, force_build: bool) -> None:
        started: list[ManagedToolProvider] = []
        try:
            for provider in self.providers.values():
                if isinstance(provider, ManagedToolProvider):
                    started.append(provider)
                    await provider.start()
        except Exception as error:
            cleanup_errors = []
            for provider in reversed(started):
                try:
                    await provider.stop()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise ExceptionGroup("Provider start and cleanup failed", [error, *cleanup_errors]) from error
            raise

    async def stop(self, delete: bool) -> None:
        errors = []
        for provider in reversed(tuple(self.providers.values())):
            if isinstance(provider, ManagedToolProvider):
                try:
                    await provider.stop()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Provider cleanup failed", errors)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if command == "pwd":
            return ExecResult(stdout="/app\n", stderr="", return_code=0)
        raise ValueError("No shell provider is bound")

    async def empty_dirs(self, dirs, *, chmod: bool = True) -> None:
        if not set(map(str, dirs)).issubset(HARBOR_EMPTY_DIRS):
            raise ValueError("No filesystem provider is bound")

    async def upload_file(self, source_path, target_path) -> None:
        raise ValueError("No filesystem provider is bound")

    async def upload_dir(self, source_dir, target_dir) -> None:
        raise ValueError("No filesystem provider is bound")

    async def download_file(self, source_path, target_path) -> None:
        raise ValueError("No filesystem provider is bound")

    async def download_dir(self, source_dir, target_dir) -> None:
        if source_dir not in HARBOR_DOWNLOAD_DIRS:
            raise ValueError("No filesystem provider is bound")

    async def native_tool_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        owners: dict[str, str] = {}
        for provider_name, binding in self.bindings.items():
            provider_tools = selected_tool_definitions(
                await self.providers[provider_name].native_tool_definitions(), binding.tools
            )
            digest = hashlib.sha256(
                json.dumps(provider_tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
            if digest != binding.tools_sha256:
                raise ValueError(f"Runtime tool surface differs for provider {provider_name!r}")
            for definition in provider_tools:
                name = definition["function"]["name"]
                if name in owners:
                    raise ValueError(f"Duplicate provider tool name: {name}")
                owners[name] = provider_name
            definitions.extend(provider_tools)
        self.tool_owners = owners
        return definitions

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        if not self.tool_owners:
            await self.native_tool_definitions()
        owner = self.tool_owners.get(name)
        if owner is None:
            raise ValueError(f"Unknown provider tool: {name}")
        return await self.providers[owner].dispatch_action(name, arguments, call_id)


class ChatAgent(BaseAgent):
    """Run chat with optional tool calls and retain the final assistant message."""

    def __init__(
        self,
        *args,
        api_base: str,
        request_timeout: float,
        request: dict[str, Any],
        max_turns: int,
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
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "taskcompendium-chat"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def _dispatch_provider_calls(
        self,
        calls: tuple[ConversationToolCall, ...],
        environment: CompositeToolEnvironment,
        messages: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        seen_call_ids: set[str],
    ) -> None:
        call_ids = [call.call_id for call in calls]
        if len(set(call_ids)) != len(call_ids) or seen_call_ids.intersection(call_ids):
            raise ValueError("Tool call IDs must be unique")
        seen_call_ids.update(call_ids)
        for call in calls:
            arguments = json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False)
            observation = await environment.dispatch_action(call.name, arguments, call.call_id)
            actions.append(
                {"call_id": call.call_id, "name": call.name, "arguments": arguments, "observation": observation}
            )
            messages.append({"role": "tool", "tool_call_id": call.call_id, "content": observation})

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not isinstance(environment, CompositeToolEnvironment):
            raise TypeError("Chat requires a composite Harbor environment")
        bindings = read_environment_config(environment.environment_dir.parent / ENVIRONMENT_CONFIG_FILE).tool_providers
        provider_tools = await environment.native_tool_definitions()
        terminal_tools = self.request.get("tools", [])
        terminal_names = {tool["function"]["name"] for tool in terminal_tools}
        provider_names = {tool["function"]["name"] for tool in provider_tools}
        if terminal_names & provider_names:
            raise ValueError("Provider and terminal tool names overlap")
        tools = [*provider_tools, *terminal_tools]
        messages = list(self.request["messages"])
        actions: list[dict[str, Any]] = []
        seen_call_ids = {call["id"] for message in messages for call in message.get("tool_calls", ())}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for turn in range(1, self.max_turns + 1):
            request = {**self.request, "model": self.model_name, "messages": messages}
            if tools:
                request["tools"] = tools
            message = await asyncio.to_thread(
                _chat_completion,
                self.api_base,
                self.api_key,
                self.request_timeout,
                request,
            )
            (self.logs_dir / CHAT_RESPONSE_FILE).write_text(json.dumps(message))
            assistant_turn = assistant_message(message)
            calls = assistant_turn.calls if isinstance(assistant_turn, AssistantToolCalls) else ()
            messages.append(message)
            context.metadata = {
                "assistant_final": message if not calls else None,
                "turns": turn,
                "all_messages": messages,
                "summarization_count": 0,
                "tools": actions,
                "tool_definitions": tools,
            }
            if not calls:
                (self.logs_dir / SUBMISSION_FILE).write_text(chat_conversation(messages).model_dump_json())
                return
            # A completion containing only terminal calls is the submission. Do not dispatch it to providers.
            if any(call.name not in provider_names for call in calls):
                if any(call.name in provider_names for call in calls):
                    raise ValueError("Terminal and provider calls cannot share a completion")
                if bindings and not terminal_names:
                    raise ValueError("Unknown provider tool call")
                context.metadata["assistant_final"] = message
                (self.logs_dir / SUBMISSION_FILE).write_text(chat_conversation(messages).model_dump_json())
                return
            # Provider observations become the next model-visible turn; call IDs and action order are retained.
            await self._dispatch_provider_calls(calls, environment, messages, actions, seen_call_ids)
        raise RuntimeError(f"Tool agent exhausted {self.max_turns} turns")


class SemanticVerifier(BaseVerifier):
    """Grade the submitted answer or authoritative state."""

    async def verify(self) -> VerifierResult:
        try:
            root = self.task.paths.task_dir
            specification = read_specification(root / SPECIFICATION_FILE)
            convention = read_submission_convention(root / SUBMISSION_CONVENTION_FILE)
            response_path = self.trial_paths.agent_dir / SUBMISSION_FILE
            conversation = ConversationTrace.model_validate_json(response_path.read_text())
            if not isinstance(self.environment, CompositeToolEnvironment):
                raise TypeError("Chat verification requires a composite Harbor environment")
            attempt = GradingAttempt(
                conversation=conversation,
                tool_providers=self.environment.providers,
                workspace=self.environment,
            )
            result = await grade_answer(specification, convention, attempt)
        except Exception as error:
            result = GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")
            self._write_result(result)
            raise RuntimeError(result.error) from error
        self._write_result(result)
        if result.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE) or result.reward is None:
            raise RuntimeError(result.error or result.status.value)
        return VerifierResult(rewards={"reward": result.reward})

    def _write_result(self, result: GradeResult) -> None:
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_text(
            json.dumps({"status": result.status.value, "reward": result.reward, "error": result.error}) + "\n"
        )
