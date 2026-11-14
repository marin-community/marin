# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate and control agents for unchanged TaskSpec trials."""

import asyncio
import json
import re
from abc import abstractmethod
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer
from rolloutengine.chat import ChatModel
from rolloutengine.harness.shell import ShellHarness, task_prompt
from rolloutengine.models import (
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TraceMode,
)
from rolloutengine.recorder import harness_rollout
from shellbox.machine import Command, ExitReason, Machine
from taskcompendium.chat import assistant_message, chat_conversation, chat_input
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
from taskcompendium.runtime.shell import upload_resources

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError
from harbor.llms.vllm_tokenization import completion_budget_from_count
from harbor.llms.base import ContextBudgetExceededError
from harbor.utils.artifact_writer import get_artifact_writer
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


@cache
def candidate_tokenizer(model: str, revision: str):
    return AutoTokenizer.from_pretrained(model, revision=revision)


class TaskSpecChatAgent(TaskSpecAgent):
    model_input: ConversationInput | None = None

    @staticmethod
    def name() -> str:
        return "task-spec-chat"

    def __init__(
        self, *args, api_base: str, max_turns: int, max_tokens: int, context_window: int,
        max_input_tokens: int, request_parameters: dict[str, Any], tokenizer_model: str, tokenizer_revision: str, **kwargs
    ):
        sampling = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "reasoning_effort", "extra_body"}
        extra = request_parameters.get("extra_body", {})
        if request_parameters.keys() - sampling or extra.keys() - {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}:
            raise ValueError("Candidate request parameters may change only sampling and template controls")
        if extra.get("chat_template_kwargs", {}).keys() - {"enable_thinking", "reasoning_effort"}:
            raise ValueError("Candidate template controls cannot change the task interface")
        if re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision) is None:
            raise ValueError("Candidate tokenizer requires an immutable Hugging Face commit pin")
        super().__init__(*args, **kwargs)
        self.api_base = api_base
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.max_input_tokens = max_input_tokens
        self.request_parameters = request_parameters
        self.request_number = 0
        self.tokenizer_model = tokenizer_model
        self.tokenizer_revision = tokenizer_revision

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
        self.attempt = empty_attempt(task)
        tokenizer = await asyncio.to_thread(candidate_tokenizer, self.tokenizer_model, self.tokenizer_revision)
        async with AsyncOpenAI(
            base_url=self.api_base,
            api_key=self._model_authentication.get(self.api_base, "EMPTY"),
            max_retries=0,
        ) as client:
            if self.model_alias is None:
                raise ValueError("TaskSpecChatAgent requires a model identity")
            model = ChatModel(
                client, split_provider_model_name(self.model_alias)[1], self.max_tokens,
                context_window=self.context_window,
            )

            async def record(request: ModelRequest) -> ModelTurn:
                self.model_input = chat_input(list(request.messages))
                parameters = {**request.options, **self.request_parameters}
                extra_body = {
                    **request.options.get("extra_body", {}),
                    **self.request_parameters.get("extra_body", {}),
                }
                parameters["extra_body"] = extra_body
                prompt_ids = await asyncio.to_thread(
                    tokenizer.apply_chat_template, list(request.messages),
                    tools=parameters.get("tools") or None, add_generation_prompt=True,
                    tokenize=True, return_dict=False, **extra_body.get("chat_template_kwargs", {}),
                )
                if len(prompt_ids) > self.max_input_tokens:
                    writer = get_artifact_writer()
                    payload = json.dumps({"prompt_tokens": len(prompt_ids), "max_input_tokens": self.max_input_tokens,
                                          "reason": "input_budget_unserved"}).encode()
                    future = await writer.submit_required(
                        self.logs_dir / "candidate-budget-rejection.json", lambda: payload,
                        description="Unserved candidate input budget rejection",
                    )
                    await writer.await_write(future)
                    raise ContextBudgetExceededError("Candidate input exceeds the configured request budget before dispatch")
                parameters["max_tokens"] = completion_budget_from_count(
                    prompt_tokens=len(prompt_ids), context_window=self.context_window,
                    max_output_tokens=self.max_tokens, requested_max_tokens=parameters.get("max_tokens"),
                )
                request = replace(request, options=parameters)
                number = self.request_number
                self.request_number += 1
                writer = get_artifact_writer()
                request_payload = json.dumps(asdict(request)).encode()
                future = await writer.submit_required(
                    self.logs_dir / "candidate-requests" / f"{number:06d}-request.json",
                    lambda: request_payload, description="TaskSpec candidate request",
                )
                await writer.await_write(future)
                turn = await model(request)
                response_payload = json.dumps(asdict(turn)).encode()
                future = await writer.submit_required(
                    self.logs_dir / "candidate-requests" / f"{number:06d}-response.json",
                    lambda: response_payload, description="TaskSpec candidate response",
                )
                await writer.await_write(future)
                if turn.usage is not None and (
                    turn.usage.prompt_tokens > self.max_input_tokens
                    or turn.usage.completion_tokens > self.max_tokens
                    or turn.usage.prompt_tokens + turn.usage.completion_tokens > self.context_window
                ):
                    raise ValueError("Server usage exceeds the configured candidate token budget")
                self.attempt = GradingAttempt(
                    chat_conversation([*request.messages, turn.message]),
                    stop_reason=turn.stop_reason,
                )
                return turn

            rollout = await harness_rollout(
                ShellHarness(command_timeout=None, prompt=original_prompt),
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

    async def run_task(self, task: TaskSpec, machine: Machine | None, limits: RolloutLimits) -> GradingAttempt:
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
