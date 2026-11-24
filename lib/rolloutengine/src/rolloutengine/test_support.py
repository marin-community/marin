# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task, model, and runtime helpers for RolloutEngine tests."""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from shellbox.machine import Command, ExitReason, Machine, NetworkPolicy, Result
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

from rolloutengine.chat import ChatModel
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.core import Harness
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import (
    GradedRollout,
    MachineRuntimeSpec,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutLimits,
    TaskRolloutSpec,
    TraceMode,
)
from rolloutengine.recorder import harness_rollout

TWELVE = NumericSpec("12", tolerance_abs=0, tolerance_rel=0)
SHELL = "shell"
REPLAY_MODEL = "fixture-model"
# The fields of a recorded machine exchange that the agent sent, rather than the machine's result.
SENT_FIELDS = ("command", "user", "cwd", "env", "timeout", "upload", "content")


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.continuation.served_token_ids, 90, 91) if request.continuation is not None else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


def arithmetic_task(**update) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "id": "arithmetic",
            "context": ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
            "environment_requirements": EnvironmentRequirements(),
            "answer_type": AnswerType.NUMBER,
            "answer_format": PlainText(),
            "grader": verifyit_package(TWELVE).grader,
            "source": Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            **update,
        }
    )


def machine_runtime(**overrides) -> MachineRuntimeSpec:
    return MachineRuntimeSpec.model_validate(
        {
            "backend": "local",
            "network": NetworkPolicy.DENY,
            "cpus": None,
            "memory_mb": None,
            "storage_mb": None,
            "gpus": 0,
            "user": None,
            "startup_timeout": None,
            "cleanup_timeout": None,
            **overrides,
        }
    )


def rollout_limits(**overrides) -> RolloutLimits:
    return RolloutLimits.model_validate(
        {
            "max_turns": 3,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": None,
            "attempt_timeout": None,
            "verifier_timeout": 5,
            "cleanup_timeout": 5,
            **overrides,
        }
    )


def rollout_spec(
    task: TaskSpec,
    *,
    harness=SHELL,
    machine=None,
    verifier_machine=None,
    trace=None,
    judge_connection=None,
    **limits,
) -> TaskRolloutSpec:
    return TaskRolloutSpec.model_validate(
        {
            "task": task,
            "harness": harness,
            "task_machine": machine,
            "verifier_machine": verifier_machine,
            "judge_connection": judge_connection,
            **({} if trace is None else {"trace": trace}),
            "limits": rollout_limits(**limits),
        }
    )


@dataclass(frozen=True)
class ModelEngine:
    """A rollout engine and the one model that a test runs through it."""

    engine: RolloutEngine
    model: Model

    def check(self, spec: TaskRolloutSpec) -> None:
        self.engine.check(spec)

    async def run_task(self, spec: TaskRolloutSpec, run_id: str | None = None) -> GradedRollout:
        return await self.engine.run_task(spec, self.model, run_id=run_id)


def engine(model, machine_factories=None, *, harnesses=None, observer=None) -> ModelEngine:
    """Run ``model.complete`` with a shell harness under ``SHELL``, unless ``harnesses`` replaces it."""
    rollout_engine = RolloutEngine(
        {} if machine_factories is None else machine_factories,
        {SHELL: ShellHarness(command_timeout=None), **({} if harnesses is None else harnesses)},
        observer=observer,
    )
    return ModelEngine(rollout_engine, model.complete)


def tool_call(name: str, arguments: dict, *, call_id: str) -> dict:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": function}]}


def shell_call(command: str, *, call_id: str = "write") -> dict:
    return tool_call("shell", {"command": command}, call_id=call_id)


def tool_observations(graded: GradedRollout) -> list[dict]:
    """The JSON content of each tool message in the first conversation of ``graded``."""
    messages = graded.rollout.conversations[0].messages
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


def replay_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """The scenarios of a replay transcript: a JSON object with ``recorded_from`` and ``scenarios``.

    ``recorded_from`` names the agent and revision that the transcript records. Each scenario has the agent's
    ``instruction`` and ``options``, and, where the agent uses them, the ``model`` exchanges of a ``ReplayEndpoint``
    and the ``machine`` exchanges of a ``ReplayMachine``.
    """
    return json.loads(path.read_text())["scenarios"]


@dataclass
class ReplayEndpoint:
    """An OpenAI-compatible server that answers each request with the next recorded response.

    Each of ``exchanges`` has the ``path`` and JSON ``request`` that the recorded agent sent, and the ``status`` and
    JSON ``response`` that it got. ``requests`` collects the ``path`` and ``request`` of each request.
    """

    exchanges: list[dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"path": request.url.path, "request": json.loads(request.content)})
        assert len(self.requests) <= len(self.exchanges), f"The transcript ends before {self.requests[-1]}"
        exchange = self.exchanges[len(self.requests) - 1]
        return httpx.Response(exchange["status"], json=exchange["response"])

    def client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://replay.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )


def recorded_requests(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The requests of model ``exchanges``, as ``ReplayEndpoint.requests`` collects them."""
    return [{"path": exchange["path"], "request": exchange["request"]} for exchange in exchanges]


def without_stream(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``ReplayEndpoint.requests`` without ``stream``, which ``ChatModel`` sends as ``false`` and Harbor leaves out."""
    return [
        {**request, "request": {key: value for key, value in request["request"].items() if key != "stream"}}
        for request in requests
    ]


@dataclass
class ReplayMachine:
    """A task machine that answers each command with the next recorded result, and records commands and uploads.

    A command exchange has the script that the recorded agent ran inside ``wrapper`` as ``command``, any of
    ``user``, ``cwd``, ``env`` and ``timeout``, and its ``exit_code``, ``stdout``, and optional ``stderr`` and
    ``timed_out``. An upload exchange has the ``upload`` target and its ``content``. ``commands`` collects each
    command and upload with the fields that its exchange records. ``fixed_results`` answers commands outside the
    transcript, such as a system probe. The transcript records a ``nonce`` that the agent generates for each run as
    ``placeholder``; the machine puts this run's nonce in place of ``placeholder`` in the recorded output.
    """

    exchanges: list[dict[str, Any]]
    wrapper: tuple[str, ...]
    nonce: re.Pattern[str] | None = None
    placeholder: str = ""
    fixed_results: Mapping[tuple[str, ...], Result] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)

    def _exchange(self, sent: dict[str, Any], kind: str) -> dict[str, Any]:
        index = len(self.commands)
        exchange = self.exchanges[index] if index < len(self.exchanges) else {}
        if kind not in exchange:
            self.commands.append(sent)
            raise AssertionError(f"The transcript has {exchange or 'nothing'} where the agent sent {sent}")
        self.commands.append({key: value for key, value in sent.items() if key in exchange})
        return exchange

    async def run(self, command: Command) -> Result:
        if command.argv in self.fixed_results:
            return self.fixed_results[command.argv]
        assert command.argv[:-1] == self.wrapper, f"{command.argv} does not run in {self.wrapper}"
        script = command.argv[-1]
        nonce = None
        if self.nonce is not None and (nonce := self.nonce.search(script)) is not None:
            script = self.nonce.sub(lambda _: self.placeholder, script)
        sent = {
            "command": script,
            "user": command.user,
            "cwd": command.cwd,
            "env": command.env,
            "timeout": command.timeout,
        }
        exchange = self._exchange(sent, "command")
        stdout = exchange["stdout"] if nonce is None else exchange["stdout"].replace(self.placeholder, nonce[0])
        reason = ExitReason.TIMED_OUT if exchange.get("timed_out") else ExitReason.EXITED
        stderr = exchange.get("stderr", "")
        return Result(exchange["exit_code"], stdout.encode(), stderr.encode(), False, False, reason)

    async def upload(self, source: Path, target: str) -> None:
        self._exchange({"upload": target, "content": source.read_text()}, "upload")

    async def download(self, source: str, target: Path) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


def recorded_commands(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The commands and uploads of machine ``exchanges``, as ``ReplayMachine.commands`` collects them."""
    return [{key: exchange[key] for key in SENT_FIELDS if key in exchange} for exchange in exchanges]


async def replay_rollout(
    harness: Callable[[openai.AsyncOpenAI], Harness],
    instruction: str,
    model: ReplayEndpoint,
    machine: Machine | None = None,
) -> Rollout:
    """Run a harness on one user instruction, with ``model`` serving its chat model, and no turn limit.

    ``harness`` builds the harness from the endpoint's client, for a harness that also calls the server directly.
    """
    task = arithmetic_task(context=ConversationInput(events=(TextMessage(role="user", content=instruction),)))
    async with model.client() as client:
        return await harness_rollout(
            harness(client),
            task,
            ChatModel(client, REPLAY_MODEL, max_tokens=None),
            limits=rollout_limits(max_turns=None),
            machine=machine,
            trace=TraceMode.MESSAGES,
        )
