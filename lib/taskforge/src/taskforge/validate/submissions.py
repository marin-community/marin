# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The records of an adversary trial: what each verifier submission graded, and the verdict line it ended on.

An adversary trial is an agent loop with a ``submit`` tool (``validate.adversary``). Every call that
reached the verifier is one ``Submission``: the ``Candidate`` it graded (the final reply and the files
captured from the adversary's workspace), the full ``GradeResult`` and whether it passed. The trial's
final text reply ends on a verdict line that ``parse_claim`` reads into a ``Claim``. ``attempts``
persists both beside the attempt's outcome; ``calibration`` tiers a trial from them.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import TypeAdapter
from rolloutengine.contracts import LENGTH_STOP_REASON, RolloutData
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import TaskResource

from taskforge.validate.outcome import GRADED_STATUSES, Outcome

NO_SHORTCUT_LINE = "NO_SHORTCUT"
SHORTCUT_PREFIX = "SHORTCUT:"


@dataclass(frozen=True)
class Candidate:
    """What one ``submit`` call graded: the final reply and the files captured from the adversary's workspace."""

    reply: str
    files: tuple[TaskResource, ...]
    """Relative to the machine root and sorted by path; ``()`` on a task without a machine or when none were
    listed."""

    @property
    def paths(self) -> tuple[str, ...]:
        """The files' absolute paths in the machine."""
        return tuple(str(PurePosixPath("/", f.path)) for f in self.files)


@dataclass(frozen=True)
class Submission:
    """One verifier call of an adversary attempt.

    Attributes:
        ordinal: 1-based: the n-th verifier call of the attempt.
        turn: 0-based agent turn that issued it; ``None`` when the run was lost to the total-turn deadline or a
            model failure.
        candidate: What was graded.
        grade: The verifier's result in full, diagnostics included; the model saw only part of it.
        passed: ``passing(grade)``.
        wall_time: Seconds the grading took, machine preparation included.
    """

    ordinal: int
    turn: int | None
    candidate: Candidate
    grade: GradeResult
    passed: bool
    wall_time: float


@dataclass(frozen=True)
class AdversaryTrial:
    """One adversary trial's last attempt: its outcome, the system turn it ran under, and every verifier submission."""

    outcome: Outcome
    system: str
    submissions: tuple[Submission, ...]


class ClaimKind(StrEnum):
    SHORTCUT = "shortcut"
    NO_SHORTCUT = "no_shortcut"
    NONE = "none"
    """No verdict: the final turn is tool calls, was cut off, or the run ended on turns, context or a deadline."""


@dataclass(frozen=True)
class Claim:
    kind: ClaimKind
    why: str
    """The text after ``SHORTCUT_PREFIX``, stripped; ``""`` otherwise."""


def passing(grade: GradeResult) -> bool:
    """``grade.passed`` when the grader reports it, else ``reward >= score_max`` on a graded result; False for a
    result without a reward. The rule ``RewardStats.solved`` applies per trial, here per submission."""
    if grade.status not in GRADED_STATUSES or grade.reward is None:
        return False
    return grade.reward >= grade.score_max if grade.passed is None else grade.passed


def final_reply(rollout: RolloutData) -> str | None:
    """The content of the rollout's last model turn when it is a text reply without tool calls."""
    if not rollout.steps:
        return None
    message = rollout.steps[-1].turn.message
    return None if message.get("tool_calls") else str(message.get("content") or "")


def parse_claim(reply: str | None) -> Claim:
    """The last non-empty line of the final text reply: exactly ``NO_SHORTCUT``, or ``SHORTCUT:`` then the why on
    the same line. Anything else, or no text reply, is ``Claim(NONE, "")``."""
    lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
    if not lines:
        return Claim(ClaimKind.NONE, "")
    last = lines[-1]
    if last == NO_SHORTCUT_LINE:
        return Claim(ClaimKind.NO_SHORTCUT, "")
    if last.startswith(SHORTCUT_PREFIX):
        return Claim(ClaimKind.SHORTCUT, last.removeprefix(SHORTCUT_PREFIX).strip())
    return Claim(ClaimKind.NONE, "")


def trial_claim(outcome: Outcome) -> Claim:
    """The claim of an attempt: ``parse_claim`` over its final reply, ``NONE`` without a rollout or one that was cut
    off on the output budget."""
    rollout = outcome.rollout
    if rollout is None or not rollout.steps or rollout.steps[-1].turn.stop_reason == LENGTH_STOP_REASON:
        return Claim(ClaimKind.NONE, "")
    return parse_claim(final_reply(rollout))


SUBMISSIONS = TypeAdapter(tuple[Submission, ...])
