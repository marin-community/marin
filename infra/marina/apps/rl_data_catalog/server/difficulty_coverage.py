# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Difficulty coverage over the catalog the Atlas is being rebuilt from.

The coverage classifier is a pure function over the served catalog's current
declarations and the stored difficulty observations. Every Available or Pending
source resolves to exactly one of:

- ``MEASURED``: every solver carries a measurement bound to the source's
  current release and converter (the measurement IDs are retained).
- ``NEEDS_MEASUREMENT``: at least one solver has no measurement at the current
  pin — never measured, measured on a stale pin, or a gap whose cause has been
  cleared (for example a rebuilt environment artifact).
- ``BLOCKED``: at least one solver's current reason carries an open typed
  :class:`~finestore.rl.difficulty.DifficultyGap`, with the reason kept as
  evidence.

The rebuild publishes the computed document, the Atlas Difficulty view reads
the counts and the NEEDS_MEASUREMENT list from it, and the campaign queue CLI
emits sources straight from NEEDS_MEASUREMENT — no hand-maintained lists.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from finestore.rl.difficulty import (
    DifficultyGap,
    DifficultyObservation,
    MeasurementStatus,
    Solver,
    difficulty_gaps,
)
from pydantic import BaseModel, ConfigDict
from taskcompendium.identity import canonical_sha256

from .content_access import content_is_withheld
from .difficulty import raw_current_models
from .restrictions import PRIVATE_TEXT_MARKER
from .sealed_difficulty import sealed_difficulty_models
from .source_identity import canonical_source_id

COVERED_STATUSES = ("Available", "Pending")
"""The catalog rows the coverage classifier accounts for; the denominator."""


class CoverageStatus(StrEnum):
    MEASURED = "measured"
    NEEDS_MEASUREMENT = "needs_measurement"
    BLOCKED = "blocked"


class SolverCoverage(BaseModel):
    """One solver's measurement state at the source's current pin."""

    model_config = ConfigDict(frozen=True)

    solver: Solver
    status: MeasurementStatus
    measurement_id: str | None
    n: int
    solved: int
    gaps: tuple[DifficultyGap, ...] = ()
    reason: str | None = None


class SourceCoverage(BaseModel):
    """One source's Difficulty coverage at its current pin and converter."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    catalog_status: str
    status: CoverageStatus
    solvers: tuple[SolverCoverage, ...]
    measurement_ids: tuple[str, ...] = ()
    gaps: tuple[DifficultyGap, ...] = ()
    evidence: tuple[str, ...] = ()

    @property
    def measurable(self) -> bool:
        """Whether the source may enter the next campaign queue."""
        return self.status is CoverageStatus.NEEDS_MEASUREMENT


class DifficultyCoverage(BaseModel):
    """The computed coverage document the rebuild publishes."""

    model_config = ConfigDict(frozen=True)

    load_id: str | None
    snapshot_id: str | None
    generated_at: str
    sources: dict[str, SourceCoverage]
    counts: dict[str, int]
    queue: tuple[dict[str, Any], ...] = ()

    def queue_sources(self) -> list[str]:
        """The next campaign queue: every source that NEEDS_MEASUREMENT, protocol order."""
        return sorted(source_id for source_id, source in self.sources.items() if source.measurable)


def _solver_coverage(model: dict[str, Any], withheld: bool = False) -> SolverCoverage:
    status = MeasurementStatus(model["status"])
    gaps: tuple[DifficultyGap, ...] = ()
    reason = model.get("reason")
    if status is not MeasurementStatus.MEASURED and reason and model.get("measurement_id"):
        # A record-backed not-measured row carries typed gaps; the summary's
        # synthetic row for a solver with no record at this pin is the plain
        # needs-measurement state, never a gap.
        try:
            gaps = tuple(gap for gap, _ in difficulty_gaps(reason))
        except ValueError as error:
            raise ValueError(f"Difficulty observation carries an unmapped reason: {reason!r}") from error
    measurement_id = model.get("measurement_id")
    if withheld:
        reason = PRIVATE_TEXT_MARKER if gaps else None
    return SolverCoverage(
        solver=Solver(model["solver"]),
        status=status,
        measurement_id=measurement_id if status is MeasurementStatus.MEASURED else None,
        n=model["n"],
        solved=model["solved"],
        gaps=gaps,
        reason=reason if gaps else None,
    )


def source_coverage(
    declaration: dict[str, Any], records: list[DifficultyObservation], sealed_record: dict | None = None
) -> SourceCoverage:
    """Classify one source from its current declaration and stored observations.

    Classification reads the typed reasons verbatim; a PRIVATE_ALWAYS source's
    served reason text is still replaced by the private marker, so its content
    can never travel through the coverage document.
    """
    withheld = content_is_withheld(declaration)
    models = sealed_difficulty_models(declaration, raw_current_models(declaration, records), sealed_record)
    solvers = tuple(_solver_coverage(model, withheld=withheld) for model in models)
    measured = [model for model in solvers if model.status is MeasurementStatus.MEASURED]
    gaps = tuple(gap for model in solvers for gap in model.gaps)
    evidence = tuple(model.reason for model in solvers if model.reason and model.gaps)
    if gaps:
        status = CoverageStatus.BLOCKED
    elif len(measured) == len(Solver):
        status = CoverageStatus.MEASURED
    else:
        status = CoverageStatus.NEEDS_MEASUREMENT
    return SourceCoverage(
        source_id=canonical_source_id(declaration["id"]),
        catalog_status=declaration["status"],
        status=status,
        solvers=solvers,
        measurement_ids=tuple(model.measurement_id for model in measured),
        gaps=gaps,
        evidence=evidence,
    )


def difficulty_coverage(
    declarations: dict[str, dict[str, Any]],
    records: dict[str, list[DifficultyObservation]],
    sealed_records: dict[str, dict] | None = None,
) -> DifficultyCoverage:
    """Classify every Available or Pending source in the catalog being rebuilt."""
    sources: dict[str, SourceCoverage] = {}
    for source_id, declaration in declarations.items():
        if declaration.get("status") not in COVERED_STATUSES:
            continue
        coverage = source_coverage(
            declaration,
            records.get(canonical_source_id(source_id), []),
            (sealed_records or {}).get(canonical_source_id(source_id)),
        )
        sources[coverage.source_id] = coverage
    counts = {
        "sources": len(sources),
        "measured": sum(1 for source in sources.values() if source.status is CoverageStatus.MEASURED),
        "needs_measurement": sum(1 for source in sources.values() if source.status is CoverageStatus.NEEDS_MEASUREMENT),
        "blocked": sum(1 for source in sources.values() if source.status is CoverageStatus.BLOCKED),
        "solvers_measured": sum(
            1 for source in sources.values() for model in source.solvers if model.status is MeasurementStatus.MEASURED
        ),
        "solvers_gap_blocked": sum(1 for source in sources.values() for model in source.solvers if model.gaps),
        "solvers_needs_measurement": sum(
            1
            for source in sources.values()
            for model in source.solvers
            if model.status is not MeasurementStatus.MEASURED and not model.gaps
        ),
    }
    return DifficultyCoverage(
        load_id=None,
        snapshot_id=None,
        generated_at=datetime.now(UTC).isoformat(),
        sources=sources,
        counts=counts,
        queue=tuple(
            QueuedSource(declaration=declaration, coverage=sources[canonical_source_id(source_id)]).payload()
            for source_id, declaration in sorted(declarations.items())
            if canonical_source_id(source_id) in sources and sources[canonical_source_id(source_id)].measurable
        ),
    )


class QueuedSource(BaseModel):
    """One campaign-queue entry: the declaration verbatim plus its coverage."""

    model_config = ConfigDict(frozen=True)

    declaration: dict[str, Any]
    coverage: SourceCoverage

    def payload(self) -> dict[str, Any]:
        """The campaign source-queue document."""
        return {"declaration": self.declaration, "coverage": self.coverage.model_dump(mode="json")}


def campaign_queue(document: DifficultyCoverage, declarations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit the next Difficulty campaign source queue from NEEDS_MEASUREMENT."""
    queue = []
    for source_id in document.queue_sources():
        declaration = declarations.get(source_id)
        if declaration is None:
            raise ValueError(f"Queued source is absent from the catalog: {source_id}")
        queue.append(QueuedSource(declaration=declaration, coverage=document.sources[source_id]).payload())
    return queue


def document_digest(document: DifficultyCoverage) -> str:
    """The canonical digest binding a published coverage artifact to its content."""
    return canonical_sha256(document.model_dump(mode="json"))
