# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate and control agents for unchanged TaskSpec trials."""

import json
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rolloutengine.chat import ChatModel
from rolloutengine.harness.shell import ShellHarness, task_prompt
from rolloutengine.harness.repository_shell import RepositoryShellProvider
from rolloutengine.models import (
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TraceMode,
)
from rolloutengine.recorder import harness_rollout
from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Command, ExitReason, Machine
from taskcompendium.chat import assistant_message, chat_conversation, chat_input
from taskcompendium.harbor.oracle import read_oracle_command
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    GradingAttempt,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import conversation_messages
from taskcompendium.runtime.resources import has_oracle_solution
from taskcompendium.runtime.oracle import (
    OracleAnswerTimeout,
    oracle_answer_attempt,
    read_oracle_answer,
    run_oracle_command,
)
from taskcompendium.runtime.shell import upload_resources
from taskcompendium.repository_shell import SHELL_ACTION_INTERFACE

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError
from harbor_config.model_name import split_provider_model_name


def empty_attempt(task: TaskSpec) -> GradingAttempt:
    return GradingAttempt(ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))))


def original_prompt(task, shell_tool, provider_tools):
    prompt = task_prompt(task, shell_tool, provider_tools)
    return replace(prompt, messages=tuple(conversation_messages(task.context.events)))


def control_attempt(task: TaskSpec, payload: list[dict]) -> GradingAttempt:
    if len(payload) != 1 or payload[0].get("role") != "assistant":
        raise ValueError("A TaskSpec control must provide exactly one assistant response")
    response = assistant_message(payload[0])
    return GradingAttempt(ConversationTrace(events=(*task.context.events, response)))


class TaskSpecAgent(BaseAgent):
    """A Harbor agent whose input is a TaskSpec, rather than an instruction string."""

    attempt: GradingAttempt | None = None

    @staticmethod
    def name() -> str:
        return "taskcompendium"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        raise ValueError("TaskSpec agents require Harbor's TaskSpec trial workload")

    @abstractmethod
    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt: ...


class TaskSpecChatAgent(TaskSpecAgent):
    model_input: ConversationInput | None = None

    @staticmethod
    def name() -> str:
        return "task-spec-chat"

    def __init__(
        self, *args, api_base: str, max_turns: int, max_tokens: int, request_parameters: dict[str, Any], **kwargs
    ):
        sampling = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "reasoning_effort", "extra_body"}
        extra = request_parameters.get("extra_body", {})
        if request_parameters.keys() - sampling or extra.keys() - {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}:
            raise ValueError("Candidate request parameters may change only sampling and template controls")
        if extra.get("chat_template_kwargs", {}).keys() - {"enable_thinking", "reasoning_effort"}:
            raise ValueError("Candidate template controls cannot change the task interface")
        super().__init__(*args, **kwargs)
        self.api_base = api_base
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.request_parameters = request_parameters

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
        self.attempt = empty_attempt(task)
        async with AsyncOpenAI(
            base_url=self.api_base,
            api_key=self._model_authentication.get(self.api_base, "EMPTY"),
            max_retries=0,
        ) as client:
            if self.model_alias is None:
                raise ValueError("TaskSpecChatAgent requires a model identity")
            model = ChatModel(
                client, split_provider_model_name(self.model_alias)[1], self.max_tokens,
            )

            async def record(request: ModelRequest) -> ModelTurn:
                self.model_input = chat_input(list(request.messages))
                parameters = {**request.options, **self.request_parameters}
                extra_body = {
                    **request.options.get("extra_body", {}),
                    **self.request_parameters.get("extra_body", {}),
                }
                parameters["extra_body"] = extra_body
                request = replace(request, options=parameters)
                turn = await model(request)
                self.attempt = GradingAttempt(
                    chat_conversation([*request.messages, turn.message]),
                    stop_reason=turn.stop_reason,
                )
                return turn

            rollout = await harness_rollout(
                ShellHarness(
                    command_timeout=None,
                    prompt=original_prompt,
                    tools={SHELL_ACTION_INTERFACE: RepositoryShellProvider(command_timeout=None)},
                ),
                task,
                record,
                limits=limits.model_copy(update={"max_turns": self.max_turns}),
                machine=machine,
                trace=TraceMode.MESSAGES,
            )
            if rollout.conversations:
                self.attempt = GradingAttempt(
                    chat_conversation(list(rollout.conversations[-1].messages)),
                    stop_reason=rollout.stop_reason,
                )
        assert self.attempt is not None
        return self.attempt


class TaskSpecResponseAgent(TaskSpecAgent):
    """Run an explicitly supplied control response without exposing private source data."""

    @staticmethod
    def name() -> str:
        return "task-spec-response"

    def __init__(self, *args, response_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_path = Path(response_path)

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
        self.attempt = control_attempt(task, json.loads(self.response_path.read_text()))
        return self.attempt


class TaskSpecEmptyAgent(TaskSpecAgent):
    """Submit an empty TaskSpec attempt for liveness control."""

    @staticmethod
    def name() -> str:
        return "nop"

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
        self.attempt = empty_attempt(task)
        return self.attempt


class TaskSpecOracleAgent(TaskSpecAgent):
    """Execute a preserved oracle script, or an explicitly supplied calibration-only response."""

    @staticmethod
    def name() -> str:
        return "oracle"

    def __init__(self, *args, task_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir = task_dir
        self.exit_code: int | None = None
        self.output = b""
        self.answer: bytes | None = None
        self.command = read_oracle_command(task_dir)

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
        if self.command is not None:
            self.command.require_task(task)
            if machine is None:
                raise ValueError("The declared OracleCommand requires its original execution environment")
            self.attempt = empty_attempt(task)
            await upload_resources(machine, task.resources.oracle, limits.total_turn_timeout, user=None)
            result = await run_oracle_command(
                machine, self.command.submission, cwd=None, timeout=limits.total_turn_timeout,
                output_limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
            )
            self.exit_code = result.exit_code
            self.output = result.stdout + result.stderr
            if result.reason == ExitReason.TIMED_OUT:
                raise AgentTimeoutError("TaskSpec oracle command timed out")
            answer_file = self.command.submission.answer_file
            assert answer_file is not None
            try:
                self.answer = await read_oracle_answer(
                    machine, answer_file, cwd=None, timeout=limits.total_turn_timeout,
                    output_limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
                )
            except OracleAnswerTimeout as error:
                raise AgentTimeoutError(str(error)) from error
            self.attempt = oracle_answer_attempt(task, self.answer)
            return self.attempt
        if not has_oracle_solution(task):
            response = json.loads((self.task_dir / "solution/response.json").read_text())
            self.attempt = control_attempt(task, response)
            self.exit_code = 0
            return self.attempt
        resources = (*task.resources.all, *task.resources.worker, *task.resources.oracle)
        paths = {resource.path for resource in resources}
        if task.answer_type != AnswerType.FILE or "solution/solve.sh" not in paths or machine is None:
            raise ValueError("The original TaskSpec requires an explicit oracle delivery contract for this answer type")
        await upload_resources(machine, task.resources.oracle, limits.total_turn_timeout, user=None)
        result = await machine.run(Command(("bash", "/solution/solve.sh"), timeout=limits.total_turn_timeout))
        self.exit_code = result.exit_code
        self.output = result.stdout + result.stderr
        self.attempt = GradingAttempt(
            ConversationTrace(
                events=(
                    *task.context.events,
                    TextMessage(role="assistant", content="Done."),
                )
            )
        )
        if result.reason == ExitReason.TIMED_OUT:
            raise AgentTimeoutError("TaskSpec oracle script timed out")
        return self.attempt


TASK_SPEC_AGENTS = {agent.name(): agent for agent in (TaskSpecChatAgent, TaskSpecResponseAgent)}
