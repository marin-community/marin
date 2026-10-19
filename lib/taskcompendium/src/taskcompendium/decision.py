# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decision-model graders: the grader, the state its decision model reads, the level scale and the reward.

A decision model is not an LLM judge. It reads one state and answers one comparative question on a fixed
five-level scale (``LEVELS``): the whole candidate answer against a gold reference, under a rubric. The
task record holds the rubric, the reference and the acceptance level (the bar); which model answers is the
host's choice behind ``DecisionClient``. Grading runs on the host after the attempt, with no environment.

The reward is 1.0 at or above the bar. Below it the reward is ``REJECTED_REWARD_MAX`` times the level's
credit relative to the bar's credit, so a rejected answer never earns more than 0.5. A grader without a
calibrated bar accepts at ``PARTIAL_LEVEL``.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_RUBRIC_ITEMS = 48
LEVELS = (
    "missing: the candidate does not address the deliverable (empty, off-task, or only keywords)",
    "contradicts the reference: the candidate addresses it but its content disagrees with the reference",
    "partially matches: part agrees with the reference, part is wrong or missing",
    "equivalent to the reference: different notation, wording, units or order, same correct content",
    "matches exactly: the same content as the reference",
)
LEVEL_CREDIT = (0.0, 0.0, 0.5, 1.0, 1.0)
MAX_LEVEL = float(len(LEVELS) - 1)
PARTIAL_LEVEL = 2.0
""""partially matches": the bar of a grader without a calibrated acceptance level."""
EQUIVALENT_LEVEL = 3.0
"""The highest acceptance level: an answer equivalent to the reference always passes."""
REJECTED_REWARD_MAX = 0.5
"""The most a candidate below the bar earns."""
QUESTION_ID = "whole"
QUESTION = (
    "Compare the whole candidate answer with the reference answer (which earned full credit), against the whole "
    "rubric. Judge meaning, not wording: accept the listed equivalences; a negated, quoted, hypothetical or "
    "retracted mention does not count as asserting something; keyword lists without the corresponding claim earn "
    "nothing; an answer that offers several alternatives is graded on the alternative it commits to, and is wrong "
    "if it commits to none."
)


class ItemRole(StrEnum):
    SCORED = "scored"
    GATE = "gate"
    CAP = "cap"
    DISQUALIFIER = "disqualifier"


class RubricItem(BaseModel):
    """One property a correct deliverable has, stated as its intent; "satisfied" is always the good outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1, description="A short snake_case name, unique in the rubric.")
    intent: str = Field(min_length=1, description="What property of the answer this item establishes.")
    pass_criterion: str = Field(
        min_length=1,
        description="When an expert marks the item satisfied, in plain language: the meaning, never a word or "
        "pattern ('the answer recommends design B', not 'contains recommend'), with the reference values, "
        "tolerances and units.",
    )
    equivalences: tuple[str, ...] = Field(
        description="Forms that also satisfy the item: notation, units, synonyms, order, equivalent expressions."
    )
    negation_handling: str = Field(
        description="How negated, quoted, hypothetical or retracted mentions count for this item ('' if none can arise)."
    )
    weight: float = Field(ge=0, description="The item's share of the reward; 0 for a pure gate or cap.")
    role: ItemRole = Field(
        description="scored; gate or disqualifier (failing it caps the whole reward); cap (failing it caps the reward)."
    )
    cap_value: float | None = Field(
        description="For gate, cap and disqualifier items, the most the reward can be when the item is not "
        "satisfied (0 zeroes it); null for a scored item."
    )

    @model_validator(mode="after")
    def validate_item(self) -> Self:
        if (self.role is ItemRole.SCORED) != (self.cap_value is None):
            raise ValueError(f"Rubric item {self.id!r}: cap_value is set exactly when the role is not scored")
        if self.cap_value is not None and not 0 <= self.cap_value < 1:
            raise ValueError(f"Rubric item {self.id!r}: cap_value lies in [0, 1)")
        return self


class Rubric(BaseModel):
    """What a correct answer delivers, item by item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deliverable: str = Field(
        min_length=1,
        description="The deliverable as the task states it to the solver, plus the given data a grader needs.",
    )
    items: tuple[RubricItem, ...] = Field(min_length=1, max_length=MAX_RUBRIC_ITEMS)

    @model_validator(mode="after")
    def validate_rubric(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Rubric item ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        if not any(item.role is ItemRole.SCORED and item.weight > 0 for item in self.items):
            raise ValueError("A rubric needs a scored item with a positive weight")
        return self


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
"""The default state: the deliverable, the rubric, then the reference and the solver's visible output, which
sort last."""


class DecisionGrader(BaseModel):
    """A decision model compares the solver's final visible turn with ``reference`` under ``rubric``.

    Attributes:
        rubric: What a correct answer delivers.
        reference: The gold reference answer.
        state: The decision model's state, key by key, in order; ``SOLVER_TURN_STATE`` by default.
        acceptance_level: The calibrated bar on the 0-4 level scale, at most ``EQUIVALENT_LEVEL``, or
            ``None`` for ``PARTIAL_LEVEL``.
        calibration: The level each known-correct candidate read when the bar was set, by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["decision_model"] = "decision_model"
    rubric: Rubric
    reference: str
    state: dict[str, StateSource] = Field(default_factory=lambda: dict(SOLVER_TURN_STATE))
    acceptance_level: float | None
    calibration: dict[str, float]

    @model_validator(mode="after")
    def validate_grader(self) -> Self:
        if not self.reference.strip():
            raise ValueError("A decision grader requires a nonempty reference answer")
        if any(not key for key in self.state):
            raise ValueError("Decision state keys must be nonempty")
        if StateSource.SOLVER_TURN_OUTPUT not in self.state.values():
            raise ValueError(f"The decision state must include {StateSource.SOLVER_TURN_OUTPUT}")
        if self.acceptance_level is not None and not 0 <= self.acceptance_level <= EQUIVALENT_LEVEL:
            raise ValueError(f"The acceptance level lies in [0, {EQUIVALENT_LEVEL}], got {self.acceptance_level}")
        if any(not 0 <= level <= MAX_LEVEL for level in self.calibration.values()):
            raise ValueError(f"Calibration levels lie in [0, {MAX_LEVEL}]")
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


class DecisionClient(Protocol):
    """A decision model: for each question, the probability of each of its levels, in order.

    The host supplies one; a task never names a model. ``decide`` raises ``RuntimeError`` or ``OSError``
    when the model cannot answer, which grading reports as an infrastructure error.
    """

    async def decide(
        self, state: Mapping[str, JsonValue], questions: Mapping[str, ScoreQuestion]
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


def comparative_question() -> ScoreQuestion:
    return ScoreQuestion(QUESTION, LEVELS)


def level_credit(level: float) -> float:
    """``LEVEL_CREDIT`` interpolated at a fractional level."""
    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"A level lies in [0, {MAX_LEVEL}], got {level}")
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
        raise ValueError(f"A read gives {len(LEVELS)} finite nonnegative probabilities, got {list(probabilities)}")
    return sum(index * p for index, p in enumerate(probabilities)) / sum(probabilities)


def decision_grade(level: float, bar: float) -> DecisionGrade:
    """1.0 at or above the bar; below it, the credit relative to the bar's, scaled to ``REJECTED_REWARD_MAX``."""
    if level >= bar:
        return DecisionGrade(level, bar, 1.0)
    relative = min(1.0, level_credit(level) / level_credit(bar)) if level_credit(bar) > 0 else 0.0
    return DecisionGrade(level, bar, REJECTED_REWARD_MAX * relative)


def decision_state(grader: DecisionGrader, solver_output: str) -> dict[str, JsonValue]:
    """The decision model's state: ``grader.state``'s keys, in order, each filled from its source."""
    values: dict[StateSource, JsonValue] = {
        StateSource.DELIVERABLE: grader.rubric.deliverable,
        StateSource.RUBRIC: [item.model_dump(mode="json") for item in grader.rubric.items],
        StateSource.REFERENCE: grader.reference,
        StateSource.SOLVER_TURN_OUTPUT: solver_output,
    }
    return {key: values[source] for key, source in grader.state.items()}
