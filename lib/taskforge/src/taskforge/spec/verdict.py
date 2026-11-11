# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The verdict file a Taskforge grader program writes, and the per-check results it reports.

A grader program writes one JSON object to ``GRADER_VERDICT_PATH``::

    {"reward": 0.5, "detail": {"checks": [{"name": "total", "passed": true, "weight": 1.0}, ...]}}

RolloutEngine reads it through ``VERDICT_REWARD``: TaskCompendium's ``FileReward`` reads ``reward`` and makes
``detail`` the ``GradeResult.detail``, so the checks travel with every grade.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from taskcompendium.grading_result import GradeResult
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat

GRADER_VERDICT_PATH = "/logs/verifier/reward.json"
REWARD_KEY = "reward"
CHECKS_KEY = "checks"
VERDICT_REWARD = FileReward(files=(RewardFile(path=GRADER_VERDICT_PATH, format=RewardFileFormat.JSON, key=REWARD_KEY),))


@dataclass(frozen=True)
class GraderCheck:
    """One independently gradable property of a submission and whether it held."""

    name: str
    passed: bool
    weight: float


def _check(entry: object) -> GraderCheck:
    if not isinstance(entry, dict):
        raise ValueError(f"A grader check must be an object, got {entry!r}")
    name, passed, weight = entry.get("name"), entry.get("passed"), entry.get("weight")
    if not isinstance(name, str) or not isinstance(passed, bool):
        raise ValueError(f"A grader check needs a string name and a boolean passed: {entry!r}")
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise ValueError(f"A grader check needs a numeric weight: {entry!r}")
    return GraderCheck(name, passed, float(weight))


def grader_checks(grade: GradeResult) -> tuple[GraderCheck, ...] | None:
    """The per-check results a Taskforge grader program wrote, or None when the grade has none.

    An in-process verifyit grade, or a grader that wrote no ``checks``, has none.

    Raises:
        ValueError: ``checks`` is not a list of ``{name, passed, weight}`` objects.
    """
    if grade.detail is None or CHECKS_KEY not in grade.detail:
        return None
    entries = grade.detail[CHECKS_KEY]
    if not isinstance(entries, list):
        raise ValueError(f"Grader checks must be a list, got {entries!r}")
    return tuple(_check(entry) for entry in entries)


def verdict_json(reward: float, checks: Sequence[GraderCheck]) -> str:
    """The verdict file text a grader program writes for ``reward`` and ``checks``."""
    return json.dumps({REWARD_KEY: reward, "detail": {CHECKS_KEY: [asdict(check) for check in checks]}})


def reward_in_range(grade: GradeResult) -> bool:
    """Whether a graded result's reward lies within the grader's declared ``[score_min, score_max]``.

    A grader that scores 0 to 100 against the default range of 0 to 1 is a grader defect: every
    reward it gives reads as full credit and its mean is off the scale.
    """
    if grade.reward is None:
        return True
    return grade.score_min <= grade.reward <= grade.score_max
