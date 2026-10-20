# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decision-model graders: the grader type, the state its decision model reads, and the host-side grade.

A decision-model grader (``DecisionGrader``) is its own grader type, not an LLM judge: a decision model of
fixed strength reads one state and answers one comparative question on five levels (``LEVELS``: missing,
contradicts, partially matches, equivalent, matches exactly). Its spec is what the authoring model writes,
the rubric and the gold reference, plus the state the decision model reads, a template whose default
(``SOLVER_TURN_STATE``) passes the visible output of the solver's final turn; a program changes it only by
passing another template. Which model answers is the host's choice behind ``DecisionClient``.

Grading stays out of the rollout. A trial runs with the grader swapped to ``NoGrader`` (``trial_form``), so
nothing of the grader enters the engine, and the host grades afterwards the conversation the engine would
have graded: the last of ``Rollout.conversations`` with a model turn, which under the shell harness is the
only one (``host_graded``, through ``grade_decision_answer``). ``run_decision_trial`` does both; validation
and the build's ``grade_candidate`` grade through it.

Every decision read is bounded by the trial's verifier deadline, which the caller passes to the client. A read
that fails transiently (``DecisionUnavailable``: a connection error, the deadline, a 5xx, the service reporting
itself unavailable) is read once more after ``DECISION_RETRY_BACKOFF`` seconds, on the same conversation, as a
grader command that timed out is graded once more on fresh machines; the solver's rollout is kept, never re-run.
A second transient failure, or any other error (a bad request, a schema mismatch), ends the grade as a
``GradedRolloutInterrupted`` during ``GRADE``, with the error as its cause and the engine's grade.

TaskSpec has no decision grader type yet (it is proposed upstream). Until it has, ``task_package`` is the thin
adapter where a TaskSpec is built: the grader travels as a ``NoGrader`` whose ``contract`` holds it, and
``task_decision_grader`` reads it back.

The acceptance level (the bar) is set at build time from the reads of the reference and its equivalent
variants by ``known_correct_check``, whose one caller is ``sandbox_effects.calibrate_decision_grader``.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import (
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    Rollout,
    RolloutFailure,
    RolloutOperation,
    TaskRolloutSpec,
)
from taskcompendium.chat import chat_conversation
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, NoGrader, PlainText, SubmissionFailure, TaskSpec

from taskforge.spec.rubric import Rubric, RubricItem

DECISION_GRADER_KIND = "decision_model"
REFERENCE_CONTROL = "reference"
"""The name of the reference's read in a calibration: validation replays the reference as this control."""
TASK_GRADER_REASON = "decision_model: graded on the host by the decision grader in contract"
"""The ``NoGrader.reason`` of a task whose grader is a ``DecisionGrader`` (``task_package``)."""
TRIAL_GRADER_REASON = "decision_model: graded on the host after the trial"
"""The ``NoGrader.reason`` a decision task's trial runs under (``trial_form``)."""

# The question, its levels and the rubric's view are WS10d v3's form DE verbatim (the verifier v3 measured).
LEVELS = (
    "missing: the candidate does not address this (empty, off-task, or only keywords)",
    "contradicts the reference: the candidate addresses it but its content disagrees with the reference",
    "partially matches: part agrees with the reference, part is wrong or missing",
    "equivalent to the reference: different notation, wording, units or order, same correct content",
    "matches exactly: the same content as the reference",
)
LEVEL_CREDIT = (0.0, 0.0, 0.5, 1.0, 1.0)
MAX_LEVEL = float(len(LEVELS) - 1)
PARTIAL_LEVEL = 2.0
""""partially matches": the least a known-correct candidate may read, and the bar of an uncalibrated grader."""
EQUIVALENT_LEVEL = 3.0
"""A calibrated bar never lies above "equivalent": an equivalent answer always passes."""
REJECTED_REWARD_MAX = 0.5
"""A candidate below the bar earns at most this; it is the rejection ceiling of every decision grader."""
DECISION_ATTEMPTS = 2
"""Reads of one grade: the first, and one retry after a transient failure."""
DECISION_RETRY_BACKOFF = 2.0
"""Seconds before the retry of a decision read that failed transiently."""
QUESTION_ID = "whole"
JUDGE_NOTE = (
    "Judge meaning, not wording: accept the listed equivalences; a negated, quoted, hypothetical or retracted mention "
    "does not count as asserting something; keyword lists without the corresponding claim earn nothing; when the answer "
    "states several values, the designated final block governs."
)
QUESTION = (
    "Compare the whole candidate answer with the reference answer (which earned full credit), against the whole "
    f"rubric. {JUDGE_NOTE}"
)


class StateSource(StrEnum):
    """What a key of the decision model's state holds."""

    DELIVERABLE = "deliverable"
    """The rubric's deliverable statement."""
    RUBRIC = "rubric"
    """The rubric's items."""
    REFERENCE = "reference"
    """The gold reference answer."""
    SOLVER_TURN_OUTPUT = "solver_turn_output"
    """The visible text of the solver's final turn: its last assistant message, without reasoning."""


SOLVER_TURN_STATE: Mapping[str, StateSource] = MappingProxyType(
    {
        "task_deliverable": StateSource.DELIVERABLE,
        "rubric": StateSource.RUBRIC,
        "y_reference_answer": StateSource.REFERENCE,
        "z_candidate_answer": StateSource.SOLVER_TURN_OUTPUT,
    }
)
"""The default state template: the deliverable, the rubric, then the reference and the solver's visible
output, which sort last."""


class DecisionGrader(BaseModel):
    """A decision-model grader.

    Attributes:
        rubric: What a correct answer delivers, written by the authoring model.
        reference: The gold reference answer, written by the authoring model; validation replays it as the
            reference control.
        state: The decision model's state, key by key (``SOLVER_TURN_STATE`` unless a program sets another).
        acceptance_level: The calibrated bar on the 0-4 level scale, or ``None`` for ``PARTIAL_LEVEL``.
        calibration: The level each known-correct candidate read when the bar was set, by name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["decision_model"] = DECISION_GRADER_KIND
    rubric: Rubric
    reference: str = Field(min_length=1)
    state: dict[str, StateSource] = Field(default_factory=lambda: dict(SOLVER_TURN_STATE))
    acceptance_level: float | None
    calibration: dict[str, float]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not self.reference.strip():
            raise ValueError("a decision grader needs a non-empty reference answer")
        if StateSource.SOLVER_TURN_OUTPUT not in self.state.values():
            raise ValueError(f"the decision model's state must include {StateSource.SOLVER_TURN_OUTPUT}")
        if any(not key for key in self.state):
            raise ValueError("the decision model's state keys must be non-empty")
        if self.acceptance_level is not None and not 0 <= self.acceptance_level <= EQUIVALENT_LEVEL:
            raise ValueError(f"acceptance_level lies in [0, {EQUIVALENT_LEVEL}], got {self.acceptance_level}")
        if any(not 0 <= level <= MAX_LEVEL for level in self.calibration.values()):
            raise ValueError(f"calibration levels lie in [0, {MAX_LEVEL}]")
        return self

    @property
    def bar(self) -> float:
        """The level a candidate must reach to be accepted."""
        return PARTIAL_LEVEL if self.acceptance_level is None else self.acceptance_level


@dataclass(frozen=True)
class ScoreQuestion:
    """A question whose answer is one of ``levels``, ordered from worst to best."""

    prompt: str
    levels: tuple[str, ...]


class DecisionUnavailable(Exception):
    """The decision model gave no answer for a reason a later read can fix.

    A ``DecisionClient`` raises it for a connection error, a timeout, a 5xx, or the service reporting itself
    unavailable. Any other exception from ``decide`` is permanent: a bad request or a schema mismatch.
    """


class DecisionClient(Protocol):
    """A decision model: for each question, the probability of each of its levels, in order.

    The host supplies one (``RLEnvBuildServices.decision``, ``EngineSettings.decision``); a build program and
    an author never construct one. ``decide`` answers within ``timeout`` seconds, the caller's deadline, and
    raises ``DecisionUnavailable`` for a transient failure. ``elastic`` is the state key the client may shorten
    to fit its model's window: the solver's output.
    """

    async def decide(
        self,
        state: Mapping[str, JsonValue],
        questions: Mapping[str, ScoreQuestion],
        *,
        elastic: str,
        timeout: float,
    ) -> Mapping[str, tuple[float, ...]]: ...


@dataclass(frozen=True)
class DecisionGrade:
    """A candidate's comparative read: its expected level, the bar, and the reward they give."""

    level: float
    bar: float
    reward: float

    @property
    def accepted(self) -> bool:
        return self.level >= self.bar


@dataclass(frozen=True)
class KnownCorrectCheck:
    """``known_correct_check``'s verdict: why the known-correct reads are refused, or the bar they set."""

    problem: str | None
    acceptance_level: float | None


def comparative_question() -> ScoreQuestion:
    return ScoreQuestion(QUESTION, LEVELS)


def level_credit(level: float) -> float:
    """``LEVEL_CREDIT`` interpolated at a fractional level."""
    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"level lies in [0, {MAX_LEVEL}], got {level}")
    low = min(int(level), len(LEVEL_CREDIT) - 1)
    high = min(low + 1, len(LEVEL_CREDIT) - 1)
    return LEVEL_CREDIT[low] * (1 - (level - low)) + LEVEL_CREDIT[high] * (level - low)


def expected_level(probabilities: Sequence[float]) -> float:
    """The probability-weighted level of a read over ``LEVELS``."""
    if (
        len(probabilities) != len(LEVELS)
        or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or not sum(probabilities) > 0
    ):
        raise ValueError(f"a read gives {len(LEVELS)} finite nonnegative probabilities, got {list(probabilities)}")
    return sum(index * p for index, p in enumerate(probabilities)) / sum(probabilities)


def decision_grade(level: float, bar: float) -> DecisionGrade:
    """1.0 at or above the bar; below it, the credit relative to the bar's, scaled to ``REJECTED_REWARD_MAX``."""
    if level >= bar:
        return DecisionGrade(level, bar, 1.0)
    relative = min(1.0, level_credit(level) / level_credit(bar)) if level_credit(bar) > 0 else 0.0
    return DecisionGrade(level, bar, REJECTED_REWARD_MAX * relative)


def known_correct_check(levels: Mapping[str, float]) -> KnownCorrectCheck:
    """Check 3: every known-correct candidate reads at least "partially matches", and the reads set the bar.

    ``levels`` are the reads of the reference and its equivalent variants. One below ``PARTIAL_LEVEL`` refuses
    the grader: either the rubric rejects a correct answer or the variant is not equivalent. Otherwise the bar
    is the lowest read, capped at "equivalent", so every known-correct candidate passes; the reference is in
    the set, so the bar never rejects the positive control.

    This check is not decided. It is kept in this one function, called only by
    ``sandbox_effects.calibrate_decision_grader``, so dropping or switching it touches nothing else: a
    replacement that returns ``KnownCorrectCheck(None, None)`` accepts every grader at the ``PARTIAL_LEVEL`` bar.
    """
    if not levels:
        return KnownCorrectCheck("no known-correct candidate was read", None)
    failing = {name: round(level, 2) for name, level in levels.items() if level < PARTIAL_LEVEL}
    if failing:
        return KnownCorrectCheck(
            f"known-correct candidates read below 'partially matches' (level {PARTIAL_LEVEL}): {failing}", None
        )
    return KnownCorrectCheck(None, min(min(levels.values()), EQUIVALENT_LEVEL))


def solver_turn_output(conversation: ConversationTrace) -> str:
    """The visible text of the solver's final turn: the final assistant message's content, without reasoning.

    Raises:
        SubmissionFailure: the conversation does not end with a non-empty assistant text message.
    """
    return PlainText().extract(GradingAttempt(conversation)).value


def rubric_view(item: RubricItem) -> dict[str, JsonValue]:
    """The item as WS10d v3's state shows it: what it tests and how it passes, never its weight or cap."""
    equivalences: list[JsonValue] = list(item.equivalences)
    view: dict[str, JsonValue] = {
        "id": item.id,
        "role": str(item.role),
        "tests": item.intent,
        "pass_criterion": item.pass_criterion,
        "equivalences": equivalences,
        "negation_handling": item.negation_handling,
    }
    return view


def solver_output_key(grader: DecisionGrader) -> str:
    """The state key that holds the solver's output (the first, if the template repeats it)."""
    return next(key for key, source in grader.state.items() if source is StateSource.SOLVER_TURN_OUTPUT)


def decision_state(grader: DecisionGrader, solver_output: str) -> dict[str, JsonValue]:
    """The decision model's state: ``grader.state``'s keys, in order, each filled from its source."""
    values: dict[StateSource, JsonValue] = {
        StateSource.DELIVERABLE: grader.rubric.deliverable,
        StateSource.RUBRIC: [rubric_view(item) for item in grader.rubric.items],
        StateSource.REFERENCE: grader.reference,
        StateSource.SOLVER_TURN_OUTPUT: solver_output,
    }
    return {key: values[source] for key, source in grader.state.items()}


async def grade_decision_answer(
    task: TaskSpec, attempt: GradingAttempt, client: DecisionClient, *, timeout: float
) -> GradeResult:
    """Grade one decision-model task: one comparative read of the solver's final turn against the bar.

    The read is bounded by ``timeout`` seconds, the caller's verifier deadline. A final turn without visible
    text is a ``SUBMISSION_FAILURE`` with reward 0, decided without a read. ``detail`` carries the expected
    ``level``, the ``bar`` and whether the answer was ``accepted``.

    Raises:
        TypeError: ``task`` does not carry a decision grader.
        DecisionUnavailable: the read failed transiently, or did not answer within ``timeout`` seconds.
    """
    grader = task_decision_grader(task)
    if grader is None:
        raise TypeError(f"decision-model grading needs a decision-model grader, not {task.grader.kind}")
    try:
        output = solver_turn_output(attempt.conversation)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    state, questions = decision_state(grader, output), {QUESTION_ID: comparative_question()}
    try:
        async with asyncio.timeout(timeout):
            reads = await client.decide(state, questions, elastic=solver_output_key(grader), timeout=timeout)
    except TimeoutError as error:
        raise DecisionUnavailable(f"no decision within the {timeout:.0f}-second verifier deadline") from error
    grade = decision_grade(expected_level(reads[QUESTION_ID]), grader.bar)
    detail = {"level": grade.level, "bar": grade.bar, "accepted": grade.accepted}
    return GradeResult(Outcome.GRADED, grade.reward, detail=detail, passed=grade.accepted)


def task_package(grader: DecisionGrader) -> GraderPackage:
    """The TaskSpec form of ``grader``: a ``NoGrader`` whose contract holds it, with no verifier resources."""
    return GraderPackage(NoGrader(reason=TASK_GRADER_REASON, contract=grader.model_dump(mode="json")))


def task_decision_grader(task: TaskSpec) -> DecisionGrader | None:
    """The decision grader ``task`` carries (``task_package``), or ``None`` for a task with another grader."""
    grader = task.grader
    if not isinstance(grader, NoGrader) or grader.reason != TASK_GRADER_REASON:
        return None
    return DecisionGrader.model_validate(grader.contract)


def trial_form(spec: TaskRolloutSpec) -> TaskRolloutSpec:
    """``spec`` with its grader swapped to a bare ``NoGrader``: what a decision task's trial runs."""
    task = spec.task.model_copy(update={"grader": NoGrader(reason=TRIAL_GRADER_REASON)})
    return spec.model_copy(update={"task": task})


def _graded_messages(rollout: Rollout) -> tuple[dict[str, Any], ...]:
    """The chat the engine grades: the last conversation with a model turn, through its last response.

    Raises:
        ValueError: no conversation has a model turn.
    """
    for conversation in reversed(rollout.conversations):
        if conversation.steps:
            return conversation.messages
    raise ValueError("the rollout has no model turn to grade")


async def host_graded(
    result: GradedRollout, task: TaskSpec, client: DecisionClient, *, deadline: float
) -> GradedRollout:
    """``result``, a ``trial_form`` trial, with the host's grade of the chat the engine grades (``_graded_messages``).

    Each read is bounded by ``deadline`` seconds. A transient failure is read once more after
    ``DECISION_RETRY_BACKOFF``, on the same conversation. A rollout the engine ended before its grading step (a
    deadline or the generation limit before the first reply) keeps the engine's grade.

    Raises:
        GradedRolloutInterrupted: the second read failed transiently (its cause a ``DecisionUnavailable``), or a
            read failed permanently; the rollout and the engine's grade are kept, and ``rollout.failure`` names
            the error.
    """
    rollout, grade = result.rollout, result.grade
    if grade.status is not Outcome.UNAVAILABLE or grade.error != TRIAL_GRADER_REASON:
        return result
    try:
        conversation = chat_conversation(list(_graded_messages(rollout)))
    except ValueError as error:
        raise _grade_interrupted(result, error) from error
    attempt = GradingAttempt(conversation, stop_reason=rollout.stop_reason)
    for read in range(1, DECISION_ATTEMPTS + 1):
        try:
            graded = await grade_decision_answer(task, attempt, client, timeout=deadline)
        except DecisionUnavailable as error:
            if read == DECISION_ATTEMPTS:
                raise _grade_interrupted(result, error) from error
            await asyncio.sleep(DECISION_RETRY_BACKOFF)
        except Exception as error:
            raise _grade_interrupted(result, error) from error
        else:
            return GradedRollout(rollout, graded)
    raise AssertionError("unreachable: the last read returns or raises")


def _grade_interrupted(result: GradedRollout, error: Exception) -> GradedRolloutInterrupted:
    """The host grade's failure as the engine reports one of its own: ``rollout.failure`` names the error."""
    failure = RolloutFailure(type(error).__name__, {"operation": RolloutOperation.GRADE.value, "error": str(error)})
    return GradedRolloutInterrupted(replace(result.rollout, failure=failure), RolloutOperation.GRADE, result.grade)


async def run_decision_trial(
    engine: RolloutEngine, spec: TaskRolloutSpec, model: Model, client: DecisionClient
) -> GradedRollout:
    """One trial of a decision task: the engine runs ``trial_form(spec)`` with ``model``, then the host grades it
    within the spec's verifier deadline.

    Raises:
        ValueError: the spec has no verifier deadline to bound the decision read with.
    """
    deadline = spec.limits.verifier_timeout
    if deadline is None:
        raise ValueError("a decision task's limits need a verifier_timeout: it bounds every decision read")
    return await host_graded(await engine.run_task(trial_form(spec), model), spec.task, client, deadline=deadline)
