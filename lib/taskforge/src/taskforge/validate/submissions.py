# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The records of an adversary trial: what each verifier submission graded, and the verdict line it ended on.

``ClaimKind`` is the verdict line's kind, which the calibration summary counts per role. ``passing`` and
``final_reply`` read a grade and a rollout the same way for every trial kind.
"""

from enum import StrEnum

from rolloutengine.contracts import RolloutData
from taskcompendium.grading_result import GradeResult

from taskforge.validate.outcome import GRADED_STATUSES


class ClaimKind(StrEnum):
    SHORTCUT = "shortcut"
    NO_SHORTCUT = "no_shortcut"
    NONE = "none"
    """No verdict: the final turn is tool calls, was cut off, or the run ended on turns, context or a deadline."""


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
