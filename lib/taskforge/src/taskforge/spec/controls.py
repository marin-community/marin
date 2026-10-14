# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed controls for a task and the rules a control set must meet.

A control is a candidate submission with a known label and the grade it must
receive. Its payload is either a scripted transcript (the assistant turns to
replay, shell calls included) or a workspace (files installed in the task
machine before grading).
Validation replays controls through RolloutEngine. This module checks that a
control set is well formed and complete for a task, and turns a candidate's
files and reply into the assistant turns a replay serves.
"""

import base64
import json
import math
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import TypeAdapter
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
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import ShellToolConfig

from taskforge.sandbox.images import DEFAULT_MODE
from taskforge.spec.draft import SHELL_CAPABILITY

SHELL_TOOL = ShellToolConfig()
"""The shell tool RolloutEngine offers a task with the shell capability and no tool binding of its own."""

CONTROL_ID = re.compile(r"[A-Za-z0-9._-]+")


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
        _check_expectation_shape(self)


def shell_turn(*commands: tuple[str, str]) -> AssistantToolCalls:
    """One assistant turn of ``(call_id, command)`` shell calls."""
    return AssistantToolCalls.model_validate(
        {
            "calls": [
                {"call_id": call_id, "name": SHELL_TOOL.name, "arguments": {SHELL_TOOL.command_parameter: command}}
                for call_id, command in commands
            ]
        }
    )


def reply(content: str) -> TextMessage:
    """A final assistant text reply."""
    return TextMessage(role="assistant", content=content)


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


def workspace_turn(index: int, file: TaskResource) -> AssistantToolCalls:
    """One shell call that writes ``file`` as the agent would: its parent made, its bytes, its mode.

    ``file.path`` is relative to the machine root, as a task's own resources are.
    """
    target = PurePosixPath("/", file.path)
    path = shlex.quote(str(target))
    directory = shlex.quote(str(target.parent))
    data = shlex.quote(base64.b64encode(resource_bytes(file)).decode())
    mode = file.mode or DEFAULT_MODE
    command = f"mkdir -p {directory} && printf %s {data} | base64 -d > {path} && chmod {mode} {path}"
    return shell_turn((f"workspace-{index}", command))


def candidate_turns(final_reply: str, files: Sequence[TaskResource]) -> tuple[dict[str, Any], ...]:
    """The assistant turns of a candidate that writes ``files`` with one shell call each, then replies.

    A validation control with a workspace payload replays these turns, and so does
    ``grade_candidate`` when it prototypes a grader on a candidate, so both grade the state an agent
    would leave.
    """
    turns = [wire_message(workspace_turn(index, file)) for index, file in enumerate(files)]
    return (*turns, wire_message(reply(final_reply)))


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


def check_expectations(controls: Sequence[Control], rejection_ceiling: float) -> None:
    """Check each control's reward bounds against the grader's ``rejection_ceiling``.

    The ceiling is the highest reward the grader gives a submission that is wrong in substance: 0
    for an all-or-nothing grader, else the most partial credit a wrong answer can collect. A negative
    or graded malformed control may demand at most that much, and a positive control must demand
    more. ``validate_controls`` does not know the grader, so the writer calls this beside it.

    Raises:
        ValueError: naming the first control whose bounds cross the ceiling, or a ceiling outside
            [0, 1).
    """
    if not (math.isfinite(rejection_ceiling) and 0 <= rejection_ceiling < 1):
        raise ValueError(f"A rejection ceiling is a reward in [0, 1), not {rejection_ceiling}")
    for control in controls:
        expect = control.expect
        if control.kind == ControlKind.POSITIVE:
            assert expect.reward_min is not None
            if expect.reward_min <= rejection_ceiling:
                raise ValueError(
                    f"Positive control {control.id} must demand a minimum reward above the grader's rejection "
                    f"ceiling {rejection_ceiling}, not {expect.reward_min}"
                )
        elif expect.status == Outcome.GRADED:
            assert expect.reward_max is not None
            if expect.reward_max > rejection_ceiling:
                raise ValueError(
                    f"{control.kind.capitalize()} control {control.id} must demand a reward at most the grader's "
                    f"rejection ceiling {rejection_ceiling}, not {expect.reward_max}"
                )


CONTROLS = TypeAdapter(tuple[Control, ...])


def controls_json(controls: Sequence[Control]) -> bytes:
    return CONTROLS.dump_json(tuple(controls))


def parse_controls(data: bytes | str) -> tuple[Control, ...]:
    """Parse controls written by ``controls_json`` or by a builder agent."""
    return CONTROLS.validate_json(data)


def _check_expectation_shape(control: Control) -> None:
    """A positive control demands a graded minimum; a negative or graded malformed control, a graded maximum.

    The bounds themselves are checked against the grader's rejection ceiling by ``check_expectations``.
    """
    expect = control.expect
    if control.kind == ControlKind.POSITIVE and (expect.status != Outcome.GRADED or expect.reward_min is None):
        raise ValueError(f"Positive control {control.id} must demand a graded minimum reward")
    if control.kind == ControlKind.NEGATIVE and (expect.status != Outcome.GRADED or expect.reward_max is None):
        raise ValueError(f"Negative control {control.id} must demand a graded maximum reward")
    if control.kind == ControlKind.MALFORMED and expect.status == Outcome.GRADED and expect.reward_max is None:
        raise ValueError(f"Graded malformed control {control.id} must demand a maximum reward")


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
        arguments = call.arguments
        if not isinstance(arguments, dict):
            raise ValueError(f"Control {control.id} calls {call.name!r} with arguments that are not a JSON object")
        if call.name == SHELL_TOOL.name:
            if not has_machine:
                raise ValueError(f"Control {control.id} calls shell in a task without a machine")
            parameter = SHELL_TOOL.command_parameter
            if set(arguments) != {parameter} or not isinstance(arguments[parameter], str):
                raise ValueError(f"Control {control.id} calls shell without one string command")
        elif call.name not in final_names:
            raise ValueError(f"Control {control.id} calls {call.name!r}, which the task does not offer")
