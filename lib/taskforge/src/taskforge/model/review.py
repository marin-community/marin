# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, stage60_review/{decision,rules}.py and stage50_validate/outcome.py.

"""Completed validation evidence and the decisions made from it."""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter
from taskcompendium.grading_result import GradeResult

from taskforge.model.controls import Candidate

OMIT_NONE, OMIT_EMPTY = Field(exclude_if=lambda value: value is None), Field(exclude_if=lambda value: value == "")
OMIT_NOTHING = Field(exclude_if=lambda value: value == ())
"""Persistence rule: a new optional persisted field is omitted from its serialised form when it holds its ONE declared
default (``None``, ``""`` or ``()``, marked ``Annotated[T, OMIT_NONE]``, ``OMIT_EMPTY`` or ``OMIT_NOTHING``), so
adding it changes no existing serialised bytes, and no other valid value encodes like it. Canonical content digests
still visit every field; a change of meaning bumps the journal format."""


def nonempty(digest: str | None) -> None:
    if digest == "":
        raise ValueError("An empty string is never a digest")


class Cause(StrEnum):
    """Why a trial has no grade. Incomplete evaluation cannot support acceptance."""

    AGENT_TIMEOUT = "agent_timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GENERATION_LIMIT = "generation_limit"
    GRADER_TIMEOUT = "grader_timeout"
    GRADER_MISSING_REWARD = "grader_missing_reward"
    GRADER_EMPTY_REWARD = "grader_empty_reward"
    GRADER_INVALID_REWARD = "grader_invalid_reward"
    GRADER_INVALID_CHECKS = "grader_invalid_checks"
    GRADER_REWARD_RANGE = "grader_reward_range"
    GRADER_EXECUTION = "grader_execution"
    GRADER_INFRA = "grader_infra"
    INVALID_TASK = "invalid_task"
    VERIFIER_SKIPPED = "verifier_skipped"
    NO_GRADE = "no_grade"
    HELD = "held"
    """A solver slot whose effect held: typed missing evidence, never a failed trial."""


class BandOutcome(StrEnum):
    IN_BAND = "in_band"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"


class FindingKind(StrEnum):
    """What a solve-review finding says is wrong with the task. Difficulty is never one."""

    AMBIGUOUS_PROMPT = "ambiguous_prompt"
    GRADER_DEFECT = "grader_defect"
    ANSWER_LEAK = "answer_leak"
    ENVIRONMENT_PROBLEM = "environment_problem"


type Text = Annotated[str, Field(pattern=r"\S")]
"""Prose with at least one character beyond whitespace, kept as entered."""


@dataclass(frozen=True, slots=True)
class Citation:
    """``quote``, verbatim from parts ``start`` to ``end`` (inclusive) of one evidence source: a trial's messages, its
    grade as the last part, or a task file's lines."""

    source: str
    start: int
    end: int
    quote: Text


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """A task defect the solver trials show, the concrete ``edit`` that would remove it, and its quoted evidence."""

    kind: FindingKind
    description: Text
    edit: Text
    citations: Annotated[tuple[Citation, ...], Field(min_length=1)]


class Attribution(StrEnum):
    """What a trial that did not pass reflects."""

    SOLVER = "solver"
    DEFECT = "defect"
    UNINFORMATIVE = "uninformative"


@dataclass(frozen=True, slots=True)
class TrialAccount:
    trial: int
    attribution: Attribution
    reason: Text


class EvidenceKind(StrEnum):
    """Validation evidence an acceptance went without."""

    SOLVE_REVIEW = "solve_review"
    TRANSCRIPTS = "transcripts"
    CONFIRMATION_TRIALS = "confirmation_trials"


@dataclass(frozen=True, slots=True)
class MissingEvidence:
    kind: EvidenceKind
    detail: str


class ReviewEnd(StrEnum):
    REPORTED = "reported"
    NO_REPORT = "no_report"
    """The session ended without a report (window or output exhausted): missing evidence, named on acceptance."""
    SKIPPED = "skipped"
    """No trial had a transcript to review: missing evidence, named on acceptance."""


@dataclass(frozen=True, slots=True)
class SolveReview:
    """One solve-review of the evidence index whose digest is ``evidence``. ``findings`` passed their citation checks
    and ``dropped`` did not, each beside why; ``missing`` is the evidence it went without (its own report, or complete
    transcripts). A final review sorts its surviving findings: ``findings`` send the task back, ``notes`` ship with it."""

    evidence: str
    end: ReviewEnd
    findings: tuple[ReviewFinding, ...] = ()
    dropped: tuple[tuple[ReviewFinding, str], ...] = ()
    accounts: tuple[TrialAccount, ...] = ()
    missing: tuple[MissingEvidence, ...] = ()
    notes: Annotated[tuple[ReviewFinding, ...], OMIT_NOTHING] = ()


@dataclass(frozen=True, slots=True)
class Decline:
    """A review finding the author declined to act on, with its reason."""

    finding: ReviewFinding
    reason: Text


class RejectKind(StrEnum):
    TASK = "task"
    BUDGET = "budget"
    HOST = "host"


@dataclass(frozen=True, slots=True)
class Accept:
    """``restore`` names, by digest, the earlier submission accepted in place of the current one; ``missing`` the
    validation evidence the acceptance went without, and ``notes`` the review findings it is accepted with."""

    band: BandOutcome
    reasons: tuple[str, ...] = ()
    restore: Annotated[str | None, OMIT_NONE] = None
    kind: Literal["accept"] = "accept"
    missing: Annotated[tuple[MissingEvidence, ...], OMIT_NOTHING] = ()
    notes: Annotated[tuple[ReviewFinding, ...], OMIT_NOTHING] = ()

    def __post_init__(self) -> None:
        nonempty(self.restore)


@dataclass(frozen=True, slots=True)
class Reject:
    category: RejectKind
    reasons: tuple[str, ...]
    kind: Literal["reject"] = "reject"


@dataclass(frozen=True, slots=True)
class Modify:
    """``fallback`` names, by digest, the submission the next round falls back to; ``review`` the solve-review findings
    the author revises on and may decline."""

    findings: tuple[str, ...]
    placement: BandOutcome | None = None
    fallback: Annotated[str | None, OMIT_NONE] = None
    kind: Literal["modify"] = "modify"
    review: Annotated[tuple[ReviewFinding, ...], OMIT_NOTHING] = ()

    def __post_init__(self) -> None:
        nonempty(self.fallback)


@dataclass(frozen=True, slots=True)
class Revise:
    """The final review's one revision, outside the shared modification budget: ``findings`` is the band line of every
    trial of the confirmed submission, ``review`` the final review's revise findings. It keeps no fallback."""

    findings: tuple[str, ...]
    placement: BandOutcome
    review: tuple[ReviewFinding, ...]
    kind: Literal["revise"] = "revise"


@dataclass(frozen=True, slots=True)
class Retry:
    cause: Cause
    count: int
    reasons: tuple[str, ...] = ()
    kind: Literal["retry"] = "retry"


@dataclass(frozen=True, slots=True)
class Incomplete:
    reasons: tuple[str, ...]
    kind: Literal["incomplete"] = "incomplete"
    awaits: Annotated[Literal["decision_model"] | None, OMIT_NONE] = field(default=None, kw_only=True)
    """The one service whose recovery resumes this hold without a person; None holds for review."""


@dataclass(frozen=True, slots=True)
class Abandon:
    reasons: Annotated[tuple[str, ...], Field(min_length=1)]
    kind: Literal["abandon"] = "abandon"

    def __post_init__(self) -> None:
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("Abandon requires non-empty reasons")


type Decision = Annotated[Accept | Reject | Modify | Revise | Retry | Incomplete | Abandon, Field(discriminator="kind")]


@dataclass(frozen=True, slots=True)
class BandRules:
    """Inclusive solve-rate band, copied from Taskforge's review rules."""

    low: float
    high: float
    accepted: frozenset[BandOutcome]

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("the solve-rate band must satisfy 0 <= low <= high <= 1")

    def place(self, solve_rate: float) -> BandOutcome:
        if solve_rate < self.low:
            return BandOutcome.TOO_HARD
        if solve_rate > self.high:
            return BandOutcome.TOO_EASY
        return BandOutcome.IN_BAND


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: int
    grade: GradeResult | None
    cause: Cause | None = None
    detail: Annotated[str, OMIT_EMPTY] = ""


@dataclass(frozen=True, slots=True)
class ControlResult:
    index: int
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None


class Defect(StrEnum):
    """How a grader mis-scores a candidate, relative to the reward the candidate deserves."""

    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"
    GRADER_CRASH = "grader_crash"


class ObservedStatus(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    GRADER_ERROR = "grader_error"


REWARD_TOLERANCE = 1e-6
"""Rewards a grader prints and a model copies back agree to this tolerance."""


@dataclass(frozen=True, slots=True)
class Observation:
    """What a grader did to a candidate: a finite reward exactly when it graded."""

    status: ObservedStatus
    reward: float | None = None

    def __post_init__(self) -> None:
        if (self.status is ObservedStatus.GRADED) != (self.reward is not None and math.isfinite(self.reward)):
            raise ValueError("an observed reward is given, and finite, exactly when the status is graded")

    def matches(self, other: "Observation") -> bool:
        if self.reward is None or other.reward is None:
            return self.status is other.status
        return abs(self.reward - other.reward) <= REWARD_TOLERANCE


@dataclass(frozen=True, slots=True)
class DefectClaim:
    """What the grader DID to a candidate (``observed``), kept apart from what the candidate DESERVES."""

    defect: Defect
    observed: Observation
    deserved_reward: float | None

    def shown_by(self, observation: Observation) -> bool:
        """Whether ``observation`` is wrong in the claimed direction; a correct grade never is."""
        if self.defect is Defect.GRADER_CRASH:
            return observation.status is ObservedStatus.GRADER_ERROR
        deserved = self.deserved_reward
        if deserved is None or observation.status is ObservedStatus.GRADER_ERROR:
            return False
        if observation.reward is None:  # a submission failure: no credit, wrong only for a deserving candidate
            return self.defect is Defect.FALSE_REJECT and deserved > REWARD_TOLERANCE
        if self.defect is Defect.FALSE_ACCEPT:
            return observation.reward > deserved + REWARD_TOLERANCE
        return observation.reward < deserved - REWARD_TOLERANCE


@dataclass(frozen=True, slots=True)
class AdversaryFinding:
    """A reported claim and its replay; ``met`` when a fresh verifier's grade also showed the claimed defect."""

    claim: str
    candidate: Candidate
    defect: DefectClaim
    grade: GradeResult | None
    met: bool
    cause: Cause | None = None


class AdversaryStatus(StrEnum):
    """How an adversary search ended.

    ``EXHAUSTED`` means the search budget ran out before a report. Review accepts on it, because a
    full session without a demonstrated defect is evidence of a strong task, but the record says the
    budget ran out, not that the adversary validated the task.
    """

    REPORTED = "reported"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


class SearchLimit(StrEnum):
    """The budget that ended an ``EXHAUSTED`` adversary search."""

    CONTEXT_WINDOW = "context window"
    OUTPUT_BUDGET = "output budget"


class SearchEnd(StrEnum):
    """Why an adversary session stopped."""

    REPORTED = "reported"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HELD = "held"
    """It raised a hold, or ended without its report: the workpiece cannot accept on it."""


@dataclass(frozen=True, slots=True)
class Continuation:
    """Where an adversary session stopped, for a consumer that continues it; the default workflow never does.

    ``owner`` is the sandbox owner whose journal rows rebuild the session's machine; ``conversation`` is the content
    digest of its conversation at the exit, which replaying that owner's session reproduces (None when the session
    raised before an exit), and ``turns`` its assistant turns.
    """

    owner: str
    stopped: SearchEnd
    conversation: str | None = None
    turns: int = 0


@dataclass(frozen=True, slots=True)
class AdversaryResult:
    """One adversary's search; ``limit`` is set exactly when ``status`` is ``EXHAUSTED``.

    A search without a reproduced finding is evidence of a strong task, and its ``reasons`` say it ran out of
    search budget without a finding, never that it validated the task.
    """

    name: str
    status: AdversaryStatus
    findings: tuple[AdversaryFinding, ...]
    reasons: tuple[str, ...] = ()
    limit: SearchLimit | None = None
    continuation: Continuation | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    task_digest: str
    solver: tuple[TrialResult, ...]
    controls: tuple[ControlResult, ...]
    adversary: tuple[AdversaryResult, ...]
    lowering_error: str | None = None
    lowering_cause: Cause | None = None
    review: Annotated[SolveReview | None, OMIT_NONE] = None
    confirms: Annotated[int | None, OMIT_NONE] = None
    """A confirmation record: the evidence index of the validation round whose submission it confirms. Its ``solver``
    holds that round's scored trials and the new ones, its ``review`` the final review."""


@dataclass(frozen=True, slots=True)
class Budget:
    max_modify_rounds: int
    max_validation_retries: int
    max_missing_trials: int
    """Solver slots a round may lack and still be placed over its scored ones; more hold it."""


VALIDATION_RESULT = TypeAdapter(ValidationResult)
