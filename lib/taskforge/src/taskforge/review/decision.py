# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision review makes about one validation round, and its ``decision.json`` file.

A decision is one of four outcomes. ``Accept`` ends the item with its task: calibrated (``band`` is
``IN_BAND``), or outside the band where the consumer's band rule chose to accept it. ``Reject`` ends it
for a typed reason. ``Retry`` re-runs the trials that are not settled, rebuilding nothing. ``Modify``
sends the program back to the author for round + 1 with the findings; ``review.rules.decide`` never
returns it here, so the edge exists for a review that can revise a program.

The summary a decision was made from is ``calibration.json`` beside ``decision.json``.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import TypeAdapter

from taskforge.atomic_file import write_atomic
from taskforge.content_hash import pretty_json
from taskforge.validate.calibration import Finding, FindingKind
from taskforge.validate.outcome import Cause

DECISION_FILE = "decision.json"
TYPE_KEY = "type"


class RejectKind(StrEnum):
    TASK = "task"
    """The task cannot be made valid with the evidence at hand."""
    BUDGET = "budget"
    """Build revisions are exhausted."""
    HOST = "host"
    """This host's factories cannot run the task, or its answer format cannot carry its answer."""


class BandOutcome(StrEnum):
    """Where an accepted task's solve rate fell against the calibration band."""

    IN_BAND = "in_band"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"


BAND_OUTCOMES: Mapping[FindingKind, BandOutcome] = {
    FindingKind.TOO_EASY: BandOutcome.TOO_EASY,
    FindingKind.TOO_HARD: BandOutcome.TOO_HARD,
}
"""The ``BandOutcome`` of an accept outside the band, by the band finding's kind."""


@dataclass(frozen=True)
class Accept:
    band: BandOutcome
    """Where the solve rate fell: ``IN_BAND``, or the kind the consumer's ``BandRules`` accepted outside it."""


@dataclass(frozen=True)
class Reject:
    kind: RejectKind
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Retry:
    """The evidence is incomplete for a cause a fresh run of the unsettled trials can fix."""

    cause: Cause
    """The most frequent cause among the ungraded trials a re-run can change (``RERUNNABLE``)."""
    count: int
    """How many trials that cause left ungraded."""


@dataclass(frozen=True)
class Modify:
    """Send the program back to the author for round + 1; ``decide`` never returns it here."""

    findings: tuple[Finding, ...]
    """What the revised program must fix."""


type Decision = Accept | Reject | Retry | Modify

_ADAPTERS: dict[str, TypeAdapter] = {kind.__name__: TypeAdapter(kind) for kind in (Accept, Reject, Retry, Modify)}


def write_decision(path: Path, decision: Decision) -> None:
    """Write ``decision`` as JSON, tagged with its class name under ``"type"``."""
    name = type(decision).__name__
    fields = _ADAPTERS[name].dump_python(decision, mode="json")
    write_atomic(path, pretty_json({TYPE_KEY: name, **fields}).encode())


def load_decision(path: Path) -> Decision:
    """The inverse of ``write_decision``; an unknown ``"type"`` raises ``ValueError``."""
    fields = json.loads(path.read_bytes())
    name = fields.pop(TYPE_KEY)
    if name not in _ADAPTERS:
        raise ValueError(f"unknown decision type {name!r}; expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[name].validate_python(fields)
