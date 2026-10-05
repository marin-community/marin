# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay builder controls through RolloutEngine and compare each grade with its expectation.

A control runs as an ordinary trial whose model is a ``ScriptedModel``: it answers each request
with the control's next assistant turn. A workspace control is delivered the way an agent would
leave it: one scripted shell call per file (``workspace_turn``: decode the bytes into the path and
set its mode, run by the shell tool as the task's ``agent_user``, after environment setup and the
healthcheck), then a final reply (``WORKSPACE_REPLY``). The task's own grader then runs on that
machine state. Grading is the engine's, unchanged.

The engine accepts a turn only with exact token ids, so the scripted model gets them from the
server's chat template. For each scripted turn it sends two ``max_tokens=1`` requests with
``return_token_ids``:

1. ``Tokenize.prompt_ids``: the request's messages with the generation prompt; its
   ``prompt_token_ids`` are the turn's prompt;
2. ``Tokenize.rendered_ids``: the same messages plus the scripted assistant turn, without the
   generation prompt; its ``prompt_token_ids`` extend the first, and the extension is the turn's
   response.

Cost per scripted turn: two requests, each one decoded token (discarded) plus a prefill of the
whole conversation, most of which the server's prefix cache already holds from the previous turn.
A typical control (one to three turns) costs two to six such requests. The response ids lack the
stop token a sampled turn ends with; the next turn's prompt supplies it as an observation token.
Responses carry no log probabilities.

Staged tasks are not replayed yet: a control for stage ``n`` needs the earlier stages passed first.
"""

import asyncio
import base64
import json
import math
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutContractError
from taskcompendium.environment import EnvironmentFile
from taskcompendium.grading import GradeResult
from taskcompendium.grading import Outcome as GradeStatus
from taskcompendium.models import AssistantToolCalls, TaskSpec, TextMessage

from taskforge.ledger.records import Ledger
from taskforge.llm.client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import TOKEN_FIELDS, served_tokens
from taskforge.spec.controls import Control, Expectation, Workspace, reply, shell_turn, validate_controls
from taskforge.validate.outcome import Graded, Outcome, TrialKind
from taskforge.validate.trials import EngineSettings, TrialPlan, run_trial

WORKSPACE_REPLY = "The workspace is ready for grading."
NO_GENERATION_PROMPT: dict[str, object] = {"add_generation_prompt": False}


class Tokenize(Protocol):
    """The served chat template's token ids for a conversation."""

    async def prompt_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        """``messages`` rendered with the generation prompt, as a request for the next turn is."""
        ...

    async def rendered_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        """``messages`` rendered as a finished conversation, without the generation prompt."""
        ...


@dataclass(frozen=True)
class ServerTokenizer:
    """``Tokenize`` through one-token GLM completions.

    ``policy`` must match the solver policy's template inputs (``reasoning_effort``).
    """

    client: GlmClient
    policy: LLMPolicy

    async def prompt_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        return await self._served_prompt(messages, {**TOKEN_FIELDS, **options})

    async def rendered_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        return await self._served_prompt(messages, {**TOKEN_FIELDS, **options, **NO_GENERATION_PROMPT})

    async def _served_prompt(self, messages: Sequence[Mapping[str, Any]], fields: dict[str, Any]) -> tuple[int, ...]:
        probe = replace(self.policy, max_tokens=1, max_continuations=0)
        completion = await self.client.complete([dict(message) for message in messages], probe, fields)
        return served_tokens(completion).prompt


def wire_message(turn: TextMessage | AssistantToolCalls) -> dict[str, Any]:
    """A transcript turn as the chat message a GLM rollout model would have produced."""
    if isinstance(turn, TextMessage):
        return {"role": "assistant", "content": turn.content, "reasoning_content": ""}
    return {
        "role": "assistant",
        "content": turn.content or "",
        "reasoning_content": "",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in turn.calls
        ],
    }


@dataclass(frozen=True)
class ScriptedModel:
    """A RolloutEngine model that replays fixed assistant turns.

    The turn served is the number of assistant messages the engine has already appended, so the
    model is stateless and a retried attempt replays from the start.
    """

    turns: tuple[dict[str, Any], ...]
    context_assistant_turns: int
    tokenize: Tokenize

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages) - self.context_assistant_turns
        message = self.turns[served]
        prompt = await self.tokenize.prompt_ids(request.messages, request.options)
        rendered = await self.tokenize.rendered_ids((*request.messages, message), request.options)
        if len(rendered) <= len(prompt) or rendered[: len(prompt)] != prompt:
            raise RolloutContractError("The rendered scripted turn does not extend the rendered prompt")
        return ModelTurn(
            message=message,
            prompt_token_ids=prompt,
            response_token_ids=rendered[len(prompt) :],
            logprobs=None,
            stop_reason="tool_calls" if "tool_calls" in message else "stop",
            text=message["content"],
        )


class ControlVerdict(StrEnum):
    MET = "met"
    VIOLATED = "violated"
    UNGRADED = "ungraded"
    """The replay produced no grade, so the control says nothing about the grader."""


@dataclass(frozen=True)
class ControlOutcome:
    control: Control
    outcome: Outcome
    verdict: ControlVerdict


@dataclass(frozen=True)
class ControlPlan:
    """Which item the controls belong to, how they retry, and where they are recorded.

    Each control is one ``CONTROL`` trial named by its id.
    """

    item_id: str
    round: int
    max_retries: int
    retry_backoff: ExponentialBackoff
    evidence_dir: Path
    ledger: Ledger

    def trial_plan(self) -> TrialPlan:
        return TrialPlan(
            item_id=self.item_id,
            round=self.round,
            kind=TrialKind.CONTROL,
            k=1,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            evidence_dir=self.evidence_dir,
            ledger=self.ledger,
        )


def expectation_met(expect: Expectation, grade: GradeResult) -> bool:
    if grade.status != expect.status:
        return False
    if expect.status != GradeStatus.GRADED:
        return True
    assert grade.reward is not None
    if expect.reward_min is not None and grade.reward < expect.reward_min:
        return False
    if expect.reward_max is not None and grade.reward > expect.reward_max:
        return False
    return all(
        name in grade.rewards and math.isclose(grade.rewards[name], value) for name, value in expect.components.items()
    )


def workspace_turn(index: int, file: EnvironmentFile) -> AssistantToolCalls:
    """One shell call that writes ``file`` as the agent would: its parent made, its bytes, its mode."""
    path = shlex.quote(file.path)
    directory = shlex.quote(str(PurePosixPath(file.path).parent))
    data = shlex.quote(base64.b64encode(file.content).decode())
    command = f"mkdir -p {directory} && printf %s {data} | base64 -d > {path} && chmod {file.mode:o} {path}"
    return shell_turn((f"workspace-{index}", command))


def control_turns(control: Control) -> tuple[dict[str, Any], ...]:
    """The assistant turns a control replays."""
    if isinstance(control.payload, Workspace):
        turns = [workspace_turn(index, file) for index, file in enumerate(control.payload.files)]
        return (*(wire_message(turn) for turn in turns), wire_message(reply(WORKSPACE_REPLY)))
    return tuple(wire_message(turn) for turn in control.payload.turns)


async def replay(
    task: TaskSpec, controls: Sequence[Control], plan: ControlPlan, settings: EngineSettings, tokenize: Tokenize
) -> list[ControlOutcome]:
    """Replay every control concurrently as a ``CONTROL`` trial named by its id.

    Raises:
        ValueError: ``controls`` is not a valid control set for ``task`` (``validate_controls``), a
            control needs more turns than ``settings.max_turns``, or ``task`` is staged.
    """
    validate_controls(task, controls)
    if task.stages:
        raise ValueError("Control replay does not support staged tasks yet")
    too_long = [control.id for control in controls if len(control_turns(control)) > settings.max_turns]
    if too_long:
        raise ValueError(f"Controls {too_long} need more than max_turns={settings.max_turns} turns")
    context_assistant_turns = sum(
        isinstance(event, AssistantToolCalls) or (isinstance(event, TextMessage) and event.role == "assistant")
        for event in task.context.events
    )
    trial_plan = plan.trial_plan()
    async with asyncio.TaskGroup() as group:
        runs = [
            group.create_task(
                run_trial(
                    task,
                    trial_plan,
                    settings,
                    ScriptedModel(control_turns(control), context_assistant_turns, tokenize),
                    control.id,
                )
            )
            for control in controls
        ]
    return [_control_outcome(control, run.result()) for control, run in zip(controls, runs, strict=True)]


def _control_outcome(control: Control, outcome: Outcome) -> ControlOutcome:
    if not isinstance(outcome, Graded):
        return ControlOutcome(control, outcome, ControlVerdict.UNGRADED)
    met = expectation_met(control.expect, outcome.grade)
    return ControlOutcome(control, outcome, ControlVerdict.MET if met else ControlVerdict.VIOLATED)
