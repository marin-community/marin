# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builder-authored fixed controls and the rules a control set must meet.

A control is a candidate submission with a known label and the grade it must
receive. Its payload is either a scripted transcript (the assistant turns to
replay, shell calls included) or a workspace (files installed in the task
machine before grading).
``validate/`` replays controls through RolloutEngine; this module only checks
that a control set is well formed and complete for a task.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter
from rolloutengine.shell_tool import SHELL_TOOL_NAME
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    AnswerCall,
    AssistantToolCalls,
    ConversationToolCall,
    TaskResource,
    TaskSpec,
    TextMessage,
)

from taskforge.spec.draft import SHELL_CAPABILITY

CONTROL_ID = re.compile(r"[A-Za-z0-9._-]+")
REJECTION_CEILING = 0.2
"""Highest reward a negative or graded malformed control may demand."""


class ControlKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MALFORMED = "malformed"


class ControlCategory(StrEnum):
    KNOWN_CORRECT = "known_correct"
    PLAUSIBLE_WRONG = "plausible_wrong"
    TASK_SPECIFIC_SHORTCUT = "task_specific_shortcut"
    REWARD_HACK = "reward_hack"
    EMPTY_OR_MALFORMED = "empty_or_malformed"


CATEGORIES: dict[ControlKind, frozenset[ControlCategory]] = {
    ControlKind.POSITIVE: frozenset({ControlCategory.KNOWN_CORRECT}),
    ControlKind.NEGATIVE: frozenset(
        {ControlCategory.PLAUSIBLE_WRONG, ControlCategory.TASK_SPECIFIC_SHORTCUT, ControlCategory.REWARD_HACK}
    ),
    ControlKind.MALFORMED: frozenset({ControlCategory.EMPTY_OR_MALFORMED}),
}
REQUIRED_CATEGORIES = frozenset(
    {ControlCategory.KNOWN_CORRECT, ControlCategory.EMPTY_OR_MALFORMED, ControlCategory.PLAUSIBLE_WRONG}
)
ADVERSARIAL = frozenset({ControlCategory.TASK_SPECIFIC_SHORTCUT, ControlCategory.REWARD_HACK})


class ControlConcern(StrEnum):
    """The part of the grader a control exercises.

    Extraction controls are kept apart because answer extraction is expected to move to a cheap
    model: the controls that only pin today's parser must be identifiable so they can be dropped or
    rewritten then, and a task's required coverage may not rest on them.
    """

    REFERENCE = "reference"
    """A reference solution the grader must accept."""
    ACCEPTANCE = "acceptance"
    """The grader's acceptance rule: which answers count as right and which as wrong."""
    EXTRACTION = "extraction"
    """Answer extraction and parsing: empty, malformed, or unusually formatted submissions."""
    SHORTCUT = "shortcut"
    """A shortcut or prompt injection that earns credit without doing the task."""


CONCERNS: dict[ControlCategory, frozenset[ControlConcern]] = {
    ControlCategory.KNOWN_CORRECT: frozenset(
        {ControlConcern.REFERENCE, ControlConcern.ACCEPTANCE, ControlConcern.EXTRACTION}
    ),
    ControlCategory.PLAUSIBLE_WRONG: frozenset({ControlConcern.ACCEPTANCE, ControlConcern.EXTRACTION}),
    ControlCategory.TASK_SPECIFIC_SHORTCUT: frozenset({ControlConcern.SHORTCUT, ControlConcern.EXTRACTION}),
    ControlCategory.REWARD_HACK: frozenset({ControlConcern.SHORTCUT, ControlConcern.EXTRACTION}),
    ControlCategory.EMPTY_OR_MALFORMED: frozenset({ControlConcern.EXTRACTION, ControlConcern.ACCEPTANCE}),
}
REQUIRED_CONCERNS = frozenset({ControlConcern.REFERENCE, ControlConcern.ACCEPTANCE, ControlConcern.SHORTCUT})
EXPECTED_STATUSES = frozenset({Outcome.GRADED, Outcome.SUBMISSION_FAILURE})


@dataclass(frozen=True)
class Expectation:
    """The grade a control must receive.

    A graded expectation with only an upper bound is a no-credit expectation: a
    submission failure, which scores zero, meets it as well as a graded reward
    within the bound.
    """

    status: Outcome
    reward_min: float | None = None
    reward_max: float | None = None

    def __post_init__(self) -> None:
        if self.status not in EXPECTED_STATUSES:
            raise ValueError(f"A control expects a graded result or a submission failure, not {self.status}")
        bounds = (self.reward_min, self.reward_max)
        if self.status == Outcome.SUBMISSION_FAILURE and bounds != (None, None):
            raise ValueError("A submission failure has no reward bounds")
        for bound in bounds:
            if bound is not None and not (math.isfinite(bound) and 0 <= bound <= 1):
                raise ValueError(f"Reward bound {bound} is outside [0, 1]")
        if self.reward_min is not None and self.reward_max is not None and self.reward_min > self.reward_max:
            raise ValueError("Reward bounds are reversed")

    @property
    def no_credit(self) -> bool:
        """Whether the expectation asks only that the submission earn at most ``reward_max``."""
        return self.status == Outcome.GRADED and self.reward_min is None and self.reward_max is not None

    def met_by(self, grade: GradeResult) -> bool:
        """Whether ``grade`` meets this expectation."""
        if self.no_credit and grade.status == Outcome.SUBMISSION_FAILURE:
            return True
        if grade.status != self.status:
            return False
        if self.status != Outcome.GRADED:
            return True
        assert grade.reward is not None
        if self.reward_min is not None and grade.reward < self.reward_min:
            return False
        return self.reward_max is None or grade.reward <= self.reward_max


@dataclass(frozen=True)
class Transcript:
    """Assistant turns replayed in place of the model, in order.

    Tool results come from the environment during replay, so turns hold only
    assistant messages. Executable tasks receive ``shell`` calls.
    """

    turns: tuple[TextMessage | AssistantToolCalls, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("A transcript needs at least one assistant turn")
        if any(isinstance(turn, TextMessage) and turn.role != "assistant" for turn in self.turns):
            raise ValueError("A transcript holds only assistant turns")
        if any(isinstance(turn, TextMessage) for turn in self.turns[:-1]):
            raise ValueError("Only the final transcript turn may be a text reply")
        calls = self.calls()
        if len({call.call_id for call in calls}) != len(calls):
            raise ValueError("Transcript call identifiers must be unique")

    def calls(self) -> tuple[ConversationToolCall, ...]:
        return tuple(call for turn in self.turns if isinstance(turn, AssistantToolCalls) for call in turn.calls)


@dataclass(frozen=True)
class Workspace:
    """Files installed in the task machine in place of an agent's work, relative to the machine root."""

    files: tuple[TaskResource, ...]


@dataclass(frozen=True)
class Control:
    """One labeled candidate submission for a task."""

    id: str
    kind: ControlKind
    category: ControlCategory
    concern: ControlConcern
    author: str
    payload: Transcript | Workspace
    expect: Expectation

    def __post_init__(self) -> None:
        if not CONTROL_ID.fullmatch(self.id):
            raise ValueError(f"Control id {self.id!r} is not a safe portable name")
        if not self.author.strip():
            raise ValueError(f"Control {self.id} lacks an author")
        if self.category not in CATEGORIES[self.kind]:
            raise ValueError(f"Control {self.id}: {self.category} is not a {self.kind} category")
        if self.concern not in CONCERNS[self.category]:
            raise ValueError(f"Control {self.id}: a {self.category} control cannot exercise {self.concern}")
        _check_expectation(self)


def shell_turn(*commands: tuple[str, str]) -> AssistantToolCalls:
    """One assistant turn of ``(call_id, command)`` shell calls."""
    return AssistantToolCalls.model_validate(
        {
            "calls": [
                {"call_id": call_id, "name": SHELL_TOOL_NAME, "arguments": {"command": command}}
                for call_id, command in commands
            ]
        }
    )


def reply(content: str) -> TextMessage:
    """A final assistant text reply."""
    return TextMessage(role="assistant", content=content)


def validate_controls(task: TaskSpec, controls: Sequence[Control]) -> None:
    """Check that ``controls`` is a complete, replayable control set for ``task``.

    The set needs a known-correct, an empty-or-malformed, a plausible-wrong, and a
    task-specific-shortcut or reward-hack control, and among its controls a
    reference, an acceptance and a shortcut concern, so the task's coverage does
    not rest on extraction controls. Workspace payloads and shell calls need a
    task machine.

    Raises:
        ValueError: naming the first violated rule.
    """
    identifiers = [control.id for control in controls]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Control ids must be unique")
    for control in controls:
        _check_payload(task, control)
    categories = {control.category for control in controls}
    missing = sorted(REQUIRED_CATEGORIES - categories)
    if missing or not ADVERSARIAL & categories:
        raise ValueError(f"Controls lack {', '.join(missing) or 'task_specific_shortcut or reward_hack'}")
    uncovered = sorted(REQUIRED_CONCERNS - {control.concern for control in controls})
    if uncovered:
        raise ValueError(f"Controls lack concern: {', '.join(uncovered)}")


CONTROLS = TypeAdapter(tuple[Control, ...])


def controls_json(controls: Sequence[Control]) -> bytes:
    return CONTROLS.dump_json(tuple(controls))


def parse_controls(data: bytes | str) -> tuple[Control, ...]:
    """Parse controls written by ``controls_json`` or by a builder agent."""
    return CONTROLS.validate_json(data)


def _check_expectation(control: Control) -> None:
    expect = control.expect
    if control.kind == ControlKind.POSITIVE and (
        expect.status != Outcome.GRADED or expect.reward_min is None or expect.reward_min <= REJECTION_CEILING
    ):
        raise ValueError(f"Positive control {control.id} must demand a graded minimum reward above {REJECTION_CEILING}")
    if control.kind == ControlKind.NEGATIVE and (
        expect.status != Outcome.GRADED or expect.reward_max is None or expect.reward_max > REJECTION_CEILING
    ):
        raise ValueError(f"Negative control {control.id} must demand a graded reward <= {REJECTION_CEILING}")
    if (
        control.kind == ControlKind.MALFORMED
        and expect.status == Outcome.GRADED
        and (expect.reward_max is None or expect.reward_max > REJECTION_CEILING)
    ):
        raise ValueError(f"Graded malformed control {control.id} must demand reward <= {REJECTION_CEILING}")


def _check_payload(task: TaskSpec, control: Control) -> None:
    has_machine = SHELL_CAPABILITY in task.environment_requirements.capabilities
    if isinstance(control.payload, Workspace):
        if not has_machine:
            raise ValueError(f"Workspace control {control.id} requires a task machine")
        return
    final_names = {function.name for function in task.final_tools}
    if isinstance(task.answer_format, AnswerCall):
        final_names.add(ANSWER_CALL_NAME)
    *steps, final = control.payload.turns
    if any(call.name in final_names for turn in steps if isinstance(turn, AssistantToolCalls) for call in turn.calls):
        raise ValueError(f"Control {control.id} submits before its final turn")
    if isinstance(final, AssistantToolCalls) and not any(call.name in final_names for call in final.calls):
        raise ValueError(f"Control {control.id} does not end with a reply or a submission")
    for call in control.payload.calls():
        if call.name == SHELL_TOOL_NAME:
            if not has_machine:
                raise ValueError(f"Control {control.id} calls shell in a task without a machine")
            if set(call.arguments) != {"command"} or not isinstance(call.arguments["command"], str):
                raise ValueError(f"Control {control.id} calls shell without one string command")
        elif call.name not in final_names:
            raise ValueError(f"Control {control.id} calls {call.name!r}, which the task does not offer")
