# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builder-authored fixed controls and the rules a control set must meet.

A control is a candidate submission with a known label and the grade it must
receive. Its payload is either a scripted transcript (the assistant turns to
replay, shell calls included) or a workspace (files installed before grading).
``validate/`` replays controls through RolloutEngine; this module only checks
that a control set is well formed and complete for a task. The rules port the
controls gate of the capability_env_gen pipeline.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import TypeAdapter
from rolloutengine.task_session import SHELL_TOOL_NAME
from taskcompendium.environment import EnvironmentFile, EnvironmentKind
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, TaskSpec, TextMessage, VerifierKind
from taskcompendium.submission import ANSWER_CALL_NAME

from taskforge.spec.draft import emits_reward_components

CONTROL_ID = re.compile(r"[A-Za-z0-9._-]+")
REJECTION_CEILING = 0.2
"""Highest reward a negative or graded malformed control may demand."""


class ControlKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MALFORMED = "malformed"
    PARTIAL = "partial"


class ControlCategory(StrEnum):
    KNOWN_CORRECT = "known_correct"
    PLAUSIBLE_WRONG = "plausible_wrong"
    TASK_SPECIFIC_SHORTCUT = "task_specific_shortcut"
    REWARD_HACK = "reward_hack"
    EMPTY_OR_MALFORMED = "empty_or_malformed"
    CRITERION_MUTATION = "criterion_mutation"


CATEGORIES: dict[ControlKind, frozenset[ControlCategory]] = {
    ControlKind.POSITIVE: frozenset({ControlCategory.KNOWN_CORRECT}),
    ControlKind.NEGATIVE: frozenset(
        {ControlCategory.PLAUSIBLE_WRONG, ControlCategory.TASK_SPECIFIC_SHORTCUT, ControlCategory.REWARD_HACK}
    ),
    ControlKind.MALFORMED: frozenset({ControlCategory.EMPTY_OR_MALFORMED}),
    ControlKind.PARTIAL: frozenset({ControlCategory.CRITERION_MUTATION}),
}
REQUIRED_PER_STAGE = frozenset(
    {ControlCategory.KNOWN_CORRECT, ControlCategory.EMPTY_OR_MALFORMED, ControlCategory.PLAUSIBLE_WRONG}
)
ADVERSARIAL = frozenset({ControlCategory.TASK_SPECIFIC_SHORTCUT, ControlCategory.REWARD_HACK})
EXPECTED_STATUSES = frozenset({Outcome.GRADED, Outcome.SUBMISSION_FAILURE})


@dataclass(frozen=True)
class Expectation:
    """The grade a control must receive.

    ``components`` are exact reward components (``GradeResult.rewards``) that
    pin which criterion a partial control breaks.
    """

    status: Outcome
    reward_min: float | None = None
    reward_max: float | None = None
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in EXPECTED_STATUSES:
            raise ValueError(f"A control expects a graded result or a submission failure, not {self.status}")
        bounds = (self.reward_min, self.reward_max)
        if self.status == Outcome.SUBMISSION_FAILURE and (bounds != (None, None) or self.components):
            raise ValueError("A submission failure has no reward bounds")
        for bound in bounds:
            if bound is not None and not (math.isfinite(bound) and 0 <= bound <= 1):
                raise ValueError(f"Reward bound {bound} is outside [0, 1]")
        if self.reward_min is not None and self.reward_max is not None and self.reward_min > self.reward_max:
            raise ValueError("Reward bounds are reversed")
        if any(not name or not math.isfinite(value) for name, value in self.components.items()):
            raise ValueError("Reward components need names and finite values")


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
    """Files installed in the task machine in place of an agent's work."""

    files: tuple[EnvironmentFile, ...]


@dataclass(frozen=True)
class Control:
    """One labeled candidate submission for one stage of a task."""

    id: str
    kind: ControlKind
    category: ControlCategory
    author: str
    payload: Transcript | Workspace
    expect: Expectation
    stage: int = 0
    partial_credit_reason: str | None = None

    def __post_init__(self) -> None:
        if not CONTROL_ID.fullmatch(self.id):
            raise ValueError(f"Control id {self.id!r} is not a safe portable name")
        if not self.author.strip():
            raise ValueError(f"Control {self.id} lacks an author")
        if self.category not in CATEGORIES[self.kind]:
            raise ValueError(f"Control {self.id}: {self.category} is not a {self.kind} category")
        if self.stage < 0:
            raise ValueError(f"Control {self.id} has a negative stage")
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

    Every stage needs a known-correct, an empty-or-malformed, a plausible-wrong,
    and a task-specific-shortcut or reward-hack control. Partial controls need a
    stage grader that writes JSON reward files, the only graders that report
    reward components.

    Raises:
        ValueError: naming the first violated rule.
    """
    identifiers = [control.id for control in controls]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Control ids must be unique")
    stage_count = max(1, len(task.stages))
    for control in controls:
        if control.stage >= stage_count:
            raise ValueError(f"Control {control.id} names stage {control.stage}; the task has {stage_count}")
        _check_payload(task, control)
        grader = task.stages[control.stage].verifier if task.stages else task.verifier
        if grader.kind == VerifierKind.SHELL and control.expect.status == Outcome.SUBMISSION_FAILURE:
            raise ValueError(f"Control {control.id} expects a submission failure from a shell verifier")
        if grader.kind != VerifierKind.SHELL and isinstance(control.payload, Workspace):
            raise ValueError(f"Workspace control {control.id} requires a shell verifier")
        if control.kind == ControlKind.PARTIAL and not emits_reward_components(grader):
            raise ValueError(
                f"Partial control {control.id} needs reward components, which only a JSON reward file reports"
            )
    for stage in range(stage_count):
        categories = {control.category for control in controls if control.stage == stage}
        missing = sorted(REQUIRED_PER_STAGE - categories)
        if missing or not ADVERSARIAL & categories:
            raise ValueError(
                f"Stage {stage} lacks controls: {', '.join(missing) or 'task_specific_shortcut or reward_hack'}"
            )


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
    if control.kind == ControlKind.PARTIAL and (
        expect.status != Outcome.GRADED
        or expect.reward_min is None
        or expect.reward_max is None
        or expect.reward_max >= 1
        or not expect.components
        or not (control.partial_credit_reason or "").strip()
    ):
        raise ValueError(f"Partial control {control.id} needs reward bounds below one, reward components, and a reason")
    if control.kind != ControlKind.PARTIAL and control.partial_credit_reason is not None:
        raise ValueError(f"Only partial controls carry a partial-credit reason: {control.id}")


def _check_payload(task: TaskSpec, control: Control) -> None:
    executable = task.environment.kind != EnvironmentKind.NULL
    if isinstance(control.payload, Workspace):
        if not executable:
            raise ValueError(f"Workspace control {control.id} requires an executable task environment")
        return
    final_names = {function.name for function in task.final_tools} | {ANSWER_CALL_NAME}
    *steps, final = control.payload.turns
    if any(call.name in final_names for turn in steps if isinstance(turn, AssistantToolCalls) for call in turn.calls):
        raise ValueError(f"Control {control.id} submits before its final turn")
    if isinstance(final, AssistantToolCalls) and not any(call.name in final_names for call in final.calls):
        raise ValueError(f"Control {control.id} does not end with a reply or a submission")
    for call in control.payload.calls():
        if call.name == SHELL_TOOL_NAME:
            if not executable:
                raise ValueError(f"Control {control.id} calls shell in a task without a machine")
            if set(call.arguments) != {"command"} or not isinstance(call.arguments["command"], str):
                raise ValueError(f"Control {control.id} calls shell without one string command")
        elif call.name not in final_names:
            raise ValueError(f"Control {control.id} calls {call.name!r}, which the task does not offer")
