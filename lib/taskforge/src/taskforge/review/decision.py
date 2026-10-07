# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision review makes about one validation round, and its ``decision.json`` file.

A decision is one of four outcomes. ``Accept`` ends the item with a calibrated task. ``Reject`` ends it
for a typed reason: the task cannot be made valid (``TASK``), the repair budget is spent (``BUDGET``), or
this host cannot run it (``HOST``). ``Repair`` sends the builder program back to the author with the
findings as the revision brief and names the memoized steps to recompute. ``Retry`` re-runs the trials
that are not settled, rebuilding nothing.

``Repair`` carries a brief, not a patch: the loop turns it into ``build.author.Revision(source,
brief.failure)``, so the author is the only model that writes builder code.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import TypeAdapter

from taskforge.canonical import pretty_json, write_atomic
from taskforge.validate.calibration import CalibrationSummary, Finding
from taskforge.validate.outcome import Cause

DECISION_FILE = "decision.json"
TYPE_KEY = "type"


class RejectKind(StrEnum):
    TASK = "task"
    """The task cannot be made valid with the evidence at hand."""
    BUDGET = "budget"
    """Repairs or build revisions are exhausted while findings remain."""
    HOST = "host"
    """This host's factories or conventions cannot run the task; another host may."""


@dataclass(frozen=True)
class Accept:
    summary: CalibrationSummary


@dataclass(frozen=True)
class Reject:
    kind: RejectKind
    reasons: tuple[str, ...]
    summary: CalibrationSummary | None
    """``None`` when the item was rejected before validation produced a summary."""


@dataclass(frozen=True)
class RepairBrief:
    """What the author receives as ``Revision.failure``: the findings rendered, the noted adversary
    passes after them as information, and the findings' new controls as JSON."""

    findings: tuple[Finding, ...]
    """What the author must fix: decisive or band findings."""
    notes: tuple[Finding, ...]
    """Adversary passes the calibration only noted: shown to the author, never required to fix."""
    failure: str


@dataclass(frozen=True)
class Repair:
    """Revise the builder program the findings condemn, then rebuild and re-validate.

    ``program_digest`` names the program the findings apply to (``draft.provenance.program_digest``).
    ``invalidate`` names the steps whose memoized output the evidence condemned even if the author
    leaves their code unchanged; they go to ``run_build(..., invalidate=...)`` so a model-driven step is
    resampled rather than replayed from the cache.
    """

    program_digest: str
    brief: RepairBrief
    invalidate: tuple[str, ...]


@dataclass(frozen=True)
class Retry:
    """The evidence is incomplete for a cause a fresh run of the unsettled trials can fix."""

    cause: Cause
    """The most frequent cause among the ungraded trials."""
    count: int
    """How many trials that cause left ungraded."""


type Decision = Accept | Reject | Repair | Retry

_ADAPTERS: dict[str, TypeAdapter] = {kind.__name__: TypeAdapter(kind) for kind in (Accept, Reject, Repair, Retry)}


def decision_json(decision: Decision) -> bytes:
    """``decision`` as JSON, tagged with its class name under ``"type"``."""
    name = type(decision).__name__
    fields = _ADAPTERS[name].dump_python(decision, mode="json")
    return pretty_json({TYPE_KEY: name, **fields}).encode()


def parse_decision(data: bytes | str) -> Decision:
    """The inverse of ``decision_json``; an unknown ``"type"`` raises ``ValueError``."""
    fields = json.loads(data)
    name = fields.pop(TYPE_KEY)
    if name not in _ADAPTERS:
        raise ValueError(f"unknown decision type {name!r}; expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[name].validate_python(fields)


def write_decision(path: Path, decision: Decision) -> None:
    write_atomic(path, decision_json(decision))


def load_decision(path: Path) -> Decision:
    return parse_decision(path.read_bytes())
