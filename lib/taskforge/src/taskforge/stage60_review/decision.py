# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision review makes on a validation round.

Review returns one of four decisions:

* ``Accept`` ends the workpiece with its task, labelled with where the solve rate fell against the band.
* ``Reject`` ends the workpiece for the reason in its ``RejectKind``.
* ``Retry`` sends the same draft through validation again.
* ``Modify`` sends the program back to the author for the next round, with the review's findings.
"""

from dataclasses import dataclass
from enum import StrEnum


class RejectKind(StrEnum):
    TASK = "task"
    """The evidence shows a defect in the task."""
    BUDGET = "budget"
    """The workpiece used up its build revisions, or review asked for a MODIFY round past its budget."""
    HOST = "host"
    """This host cannot run the task."""


class BandOutcome(StrEnum):
    """Where an accepted task's solve rate fell against the calibration band."""

    IN_BAND = "in_band"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"


@dataclass(frozen=True)
class Accept:
    band: BandOutcome


@dataclass(frozen=True)
class Reject:
    kind: RejectKind
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Retry:
    """The evidence is incomplete, and ``count`` trials went ungraded for ``cause``."""

    cause: str
    count: int


@dataclass(frozen=True)
class Modify:
    """The author revises the program to fix ``findings``."""

    findings: tuple[str, ...]


Decision = Accept | Reject | Retry | Modify
