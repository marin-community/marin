# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay builder controls through RolloutEngine and compare each grade with its expectation.

A control runs as an ordinary trial whose model is a ``ScriptedModel``: it answers each request
with the control's next assistant turn. A workspace control is delivered the way an agent would
leave it: one scripted shell call per file (``spec.controls.workspace_turn``: decode the bytes into
the path and set its mode, run by the shell tool as the task machine's ``MachineRuntimeSpec.user``,
after the setup commands), then a final reply (``WORKSPACE_REPLY``). The task's own grader then runs on that
machine state. Grading is the engine's, unchanged.

A control replays under the engine's messages trace, as every trial does (``EngineSettings.apply``):
a scripted turn carries no token ids, the engine skips its prefix, evidence and log-probability
checks, and the attempt records no token evidence, which a control never needs.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from rigging.timing import ExponentialBackoff
from rolloutengine.models import ModelRequest, ModelTurn, TaskRolloutSpec
from taskcompendium.models import AssistantToolCalls, TaskSpec, TextMessage

from taskforge.ledger.records import Ledger
from taskforge.spec.controls import Control, Workspace, candidate_turns, validate_controls, wire_message
from taskforge.stage50_validate.outcome import Graded, Outcome, TrialKind
from taskforge.stage50_validate.trials import Deadlines, EngineSettings, TrialPlan, run_trial

WORKSPACE_REPLY = "The workspace is ready for grading."


@dataclass(frozen=True)
class ScriptedModel:
    """A RolloutEngine model that replays fixed assistant turns.

    The turn served is the number of assistant messages the engine has already appended, so the
    model is stateless and a retried attempt replays from the start.
    """

    turns: tuple[dict[str, Any], ...]
    context_assistant_turns: int

    def for_step(self, step: str) -> Self:
        """This model: a scripted turn makes no server call, so the attempt's step changes nothing."""
        return self

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages) - self.context_assistant_turns
        message = self.turns[served]
        return ModelTurn(
            message=message,
            prompt_token_ids=(),
            response_token_ids=(),
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
    """Which workpiece the controls belong to, their deadlines, how they retry, and where they are recorded.

    Each control is one ``CONTROL`` trial named by its id. ``first_attempts`` maps a control id to
    the number of its first attempt file and ledger step, as ``TrialPlan.first_attempt`` does for a
    trial: a control re-entered after earlier attempts passes the count already on disk. A control
    it does not name starts at 0.
    """

    workpiece_id: str
    round: int
    deadlines: Deadlines
    max_retries: int
    retry_backoff: ExponentialBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempts: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(attempt < 0 for attempt in self.first_attempts.values()):
            raise ValueError("A control plan needs non-negative first attempts")

    def trial_plan(self, control_id: str) -> TrialPlan:
        """The ``CONTROL`` trial plan of the control named ``control_id``."""
        return TrialPlan(
            workpiece_id=self.workpiece_id,
            round=self.round,
            kind=TrialKind.CONTROL,
            k=1,
            deadlines=self.deadlines,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            evidence_dir=self.evidence_dir,
            ledger=self.ledger,
            first_attempt=self.first_attempts.get(control_id, 0),
        )


def control_turns(control: Control) -> tuple[dict[str, Any], ...]:
    """The assistant turns a control replays."""
    if isinstance(control.payload, Workspace):
        return candidate_turns(WORKSPACE_REPLY, control.payload.files)
    return tuple(wire_message(turn) for turn in control.payload.turns)


def check_replayable(task: TaskSpec, controls: Sequence[Control], max_turns: int) -> None:
    """Refuse a control set that cannot be replayed against ``task`` within ``max_turns`` turns.

    Raises:
        ValueError: ``controls`` is not a valid control set for ``task`` (``validate_controls``) or a
            control needs more than ``max_turns`` turns.
    """
    validate_controls(task, controls)
    too_long = [control.id for control in controls if len(control_turns(control)) > max_turns]
    if too_long:
        raise ValueError(f"Controls {too_long} need more than max_turns={max_turns} turns")


def context_assistant_turns(task: TaskSpec) -> int:
    """The assistant turns in ``task``'s context, which a ``ScriptedModel`` skips before its script."""
    return sum(
        isinstance(event, AssistantToolCalls) or (isinstance(event, TextMessage) and event.role == "assistant")
        for event in task.context.events
    )


async def replay(
    spec: TaskRolloutSpec,
    controls: Sequence[Control],
    plan: ControlPlan,
    settings: EngineSettings,
) -> list[ControlOutcome]:
    """Replay every control concurrently as a ``CONTROL`` trial of ``spec``, named by its id.

    Raises:
        ValueError: ``check_replayable`` refuses ``controls``, or ``plan.first_attempts`` names a control not
            in ``controls``.
    """
    task = spec.task
    check_replayable(task, controls, settings.max_turns)
    unknown = sorted(set(plan.first_attempts) - {control.id for control in controls})
    if unknown:
        raise ValueError(f"First attempts name controls {unknown} that are not replayed")
    context_turns = context_assistant_turns(task)
    async with asyncio.TaskGroup() as group:
        runs = [
            group.create_task(
                run_trial(
                    spec,
                    plan.trial_plan(control.id),
                    settings,
                    ScriptedModel(control_turns(control), context_turns),
                    control.id,
                )
            )
            for control in controls
        ]
    return [control_outcome(control, run.result()) for control, run in zip(controls, runs, strict=True)]


def control_outcome(control: Control, outcome: Outcome) -> ControlOutcome:
    """``outcome`` judged against ``control.expect``."""
    if not isinstance(outcome, Graded):
        return ControlOutcome(control, outcome, ControlVerdict.UNGRADED)
    met = control.expect.met_by(outcome.grade)
    return ControlOutcome(control, outcome, ControlVerdict.MET if met else ControlVerdict.VIOLATED)
