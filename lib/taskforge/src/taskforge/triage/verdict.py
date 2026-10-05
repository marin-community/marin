# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage verdict on one proposal: decision, structural results, rubric samples, and reasons."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter

from taskforge.llm.client import FinishReason, Usage
from taskforge.triage.checks import CheckResult


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    REJECT = "reject"


class RubricAxis(StrEnum):
    """The seven proposal-review axes, each scored 1 (invalid) to 5 (excellent)."""

    REALISM = "realism"
    ALIGNMENT = "alignment"
    SPECIFICITY = "specificity"
    REWARD_VALIDITY = "reward_validity"
    ENVIRONMENT_FIT = "environment_fit"
    DIVERSITY = "diversity"
    SOURCE_HONESTY = "source_honesty"


@dataclass(frozen=True)
class RubricResult:
    """One rubric sample; ``recommendation`` is the model's own verdict, not the decision.

    ``scores`` holds one ``(axis, score)`` pair per ``RubricAxis``, in ``RubricAxis`` order.
    """

    scores: tuple[tuple[RubricAxis, int], ...]
    critical_failures: tuple[str, ...]
    issues: tuple[str, ...]
    required_changes: tuple[str, ...]
    recommendation: TriageDecision

    def __post_init__(self):
        axes = tuple(axis for axis, _ in self.scores)
        if axes != tuple(RubricAxis):
            raise ValueError(f"scores must cover {tuple(RubricAxis)} in order, got {axes}")

    def score(self, axis: RubricAxis) -> int:
        return dict(self.scores)[axis]


@dataclass(frozen=True)
class ModelCall:
    """Cost of one model completion that contributed to the verdict."""

    usage: Usage
    wall_time: float
    finish_reason: FinishReason


@dataclass(frozen=True)
class Verdict:
    """Triage outcome for the proposal with ``proposal_digest``.

    ``rubric`` holds the independent rubric samples whose majority set ``decision``. It is empty,
    and so is ``calls``, when a fatal structural failure or a null proposal decided the verdict
    without a model call.
    """

    proposal_id: str
    proposal_digest: str
    decision: TriageDecision
    structural: tuple[CheckResult, ...]
    rubric: tuple[RubricResult, ...]
    reasons: tuple[str, ...]
    calls: tuple[ModelCall, ...]

    def to_json(self) -> str:
        return _VERDICT.dump_json(self, indent=2).decode()

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        return _VERDICT.validate_json(text)


_VERDICT = TypeAdapter(Verdict)
