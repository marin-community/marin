# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, spec/verdict.py.

"""Parse per-check evidence from a grader's reward detail."""

from dataclasses import dataclass

from taskcompendium.grading_result import GradeResult


@dataclass(frozen=True, slots=True)
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
    if grade.detail is None or "checks" not in grade.detail:
        return None
    entries = grade.detail["checks"]
    if not isinstance(entries, list):
        raise ValueError(f"Grader checks must be a list, got {entries!r}")
    return tuple(_check(entry) for entry in entries)
