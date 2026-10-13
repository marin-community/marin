# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed curation catalog records shared by campaign writers and Atlas readers."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finestore.layout import BlobTables, OnConflict, parse_uri
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore


class Cohort(StrEnum):
    ADMITTED = "admitted"
    EXCLUDED = "excluded"


class Verdict(StrEnum):
    GOOD = "good"
    SOME_ISSUES = "some_issues"
    BAD = "bad"
    UNKNOWN = "unknown"


class JudgmentStatus(StrEnum):
    REVIEWED = "reviewed"
    UNAVAILABLE = "unavailable"


class AssessmentStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class AssessmentPaths:
    curated_root: str
    inputs_root: str
    judgments_root: str


class CatalogRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PipelineReference(CatalogRecord):
    name: str
    version: str
    source: str | None
    revision: str | None
    files: list[str]


class SourceDeclaration(CatalogRecord):
    id: str
    name: str
    display_name: str
    origin: str
    family: str
    tags: list[str]
    task_count: int | None
    notes: str
    dataset_id: str | None
    dataset_revision: str | None
    url: str | None
    type: str
    turns: str
    is_benchmark: bool
    kind: str
    status: str
    pipeline: PipelineReference | None


class AdmissionCounts(CatalogRecord):
    admitted: int = 0
    rejected: int = 0
    deferred: int = 0
    no_grader: int = 0
    unverified: int = 0


class ControlCounts(CatalogRecord):
    checked: int = 0
    passed: int = 0
    failed: int = 0
    defective: int = 0
    unsupported: int = 0
    infra_error: int = 0
    inconsistent: int = 0


class ControlSummary(CatalogRecord):
    status: str | None
    eligible_count: int | None
    sample_count: int | None
    counts: ControlCounts


class CurationSummary(CatalogRecord):
    status: str
    mode: str
    source_revision: str | None
    recipe_revision: str | None
    raw_population_count: int | None
    raw_sample_count: int | None
    processed_rows: int | None
    unprocessed_rows: int | None
    admission_counts: AdmissionCounts
    verification: ControlSummary | None


class ModelReference(CatalogRecord):
    model: str
    revision: str
    max_tokens: int
    max_attempts: int
    retry_max_tokens: int


class ArchivedFile(CatalogRecord):
    name: str
    uri: str


class AssessmentAttempt(CatalogRecord):
    run_id: str
    name: str
    source_id: str | None
    pipeline_key: str | None
    status: AssessmentStatus
    reason: str | None
    paths: AssessmentPaths | None
    curation_uri: str | None


class Assessment(CatalogRecord):
    pipeline_key: str
    source_id: str
    scope: str
    model: ModelReference
    rubric: str
    curation: CurationSummary
    curation_uri: str
    traces: list[ArchivedFile]


class AssessmentPopulation(CatalogRecord):
    pipeline_key: str
    cohort: Cohort
    population: int
    sample_count: int


class AssessmentItem(CatalogRecord):
    pipeline_key: str
    task_id: str
    cohort: Cohort
    status: JudgmentStatus
    verdict: Verdict
    explanation: str
    evidence_ids: list[str]
    packet_uri: str
    has_task: bool
    omissions: list[str]
    exclusion_reason: str | None


class SnapshotSource(CatalogRecord):
    snapshot_id: str
    id: str
    declaration: SourceDeclaration
    assessment_key: str | None
    latest_attempt_key: str | None
    latest_attempt_status: AssessmentStatus
    reason: str | None
    curation: CurationSummary | None = None
    diagnostics: list[ArchivedFile] = Field(default_factory=list)


class CatalogSnapshot(CatalogRecord):
    snapshot_id: str
    created_at: str
    source_count: int


class CurrentCatalog(CatalogRecord):
    name: str
    snapshot_id: str


ASSESSMENTS = "curation_assessments"
POPULATIONS = "curation_populations"
ITEMS = "curation_items"
SOURCES = "curation_inventory"
SNAPSHOTS = "curation_snapshots"
CURRENT = "curation_current"
ATTEMPTS = "curation_attempts"
TABLES: dict[str, tuple[type[CatalogRecord], tuple[str, ...]]] = {
    ATTEMPTS: (AssessmentAttempt, ("run_id", "name")),
    ASSESSMENTS: (Assessment, ("pipeline_key",)),
    POPULATIONS: (AssessmentPopulation, ("pipeline_key", "cohort")),
    ITEMS: (AssessmentItem, ("pipeline_key", "task_id")),
    SOURCES: (SnapshotSource, ("snapshot_id", "id")),
    SNAPSHOTS: (CatalogSnapshot, ("snapshot_id",)),
    CURRENT: (CurrentCatalog, ("name",)),
}


def register_catalog(store: DataStore) -> None:
    for name, (model, keys) in TABLES.items():
        store.table(
            name,
            primary_key=keys,
            schema=arrow_schema(model),
            on_conflict=OnConflict.SUPERSEDE if name == CURRENT else OnConflict.ERROR,
        )


def validate_items(populations: list[AssessmentPopulation], items: list[AssessmentItem]) -> None:
    expected = {row.cohort: row.sample_count for row in populations}
    if len(expected) != len(populations) or set(expected) != set(Cohort):
        raise ValueError("Assessment populations must contain each cohort exactly once")
    if len({row.task_id for row in items}) != len(items):
        raise ValueError("Assessment contains duplicate task IDs")
    for cohort in Cohort:
        if sum(row.cohort == cohort for row in items) != expected[cohort]:
            raise ValueError(f"Assessment item coverage differs for {cohort}")
    keys = {row.pipeline_key for row in [*populations, *items]}
    if len(keys) != 1:
        raise ValueError("Assessment records have different pipeline keys")


class CurationCatalog:
    """Read one FineStore view with typed source, assessment and packet access."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.view = ReadView(root)

    def current(self) -> CatalogSnapshot:
        current = self.view.point(CURRENT, name="latest")
        if current is None:
            raise FileNotFoundError(f"No published curation catalog at {self.root}")
        snapshot = self.view.point(SNAPSHOTS, snapshot_id=current["snapshot_id"])
        if snapshot is None:
            raise ValueError("Catalog pointer refers to a missing snapshot")
        return CatalogSnapshot.model_validate({key: snapshot[key] for key in CatalogSnapshot.model_fields})

    def sources(self, snapshot_id: str) -> Iterator[SnapshotSource]:
        for row in self.view.iter_rows(
            SOURCES, columns=list(SnapshotSource.model_fields), where=[("snapshot_id", "==", snapshot_id)]
        ):
            yield SnapshotSource.model_validate(row)

    def assessment(self, pipeline_key: str) -> Assessment:
        row = self.view.point(ASSESSMENTS, pipeline_key=pipeline_key)
        if row is None:
            raise FileNotFoundError(f"Assessment not found: {pipeline_key}")
        return Assessment.model_validate({key: row[key] for key in Assessment.model_fields})

    def attempts(self, run_id: str) -> Iterator[AssessmentAttempt]:
        for row in self.view.iter_rows(
            ATTEMPTS, columns=list(AssessmentAttempt.model_fields), where=[("run_id", "==", run_id)]
        ):
            yield AssessmentAttempt.model_validate(row)

    def populations(self, pipeline_key: str) -> list[AssessmentPopulation]:
        return [
            AssessmentPopulation.model_validate(row)
            for row in self.view.iter_rows(
                POPULATIONS,
                columns=list(AssessmentPopulation.model_fields),
                where=[("pipeline_key", "==", pipeline_key)],
            )
        ]

    def items(self, pipeline_key: str, *, cohort: Cohort | None = None) -> Iterator[AssessmentItem]:
        where: list[tuple[str, str, object]] = [("pipeline_key", "==", pipeline_key)]
        if cohort is not None:
            where.append(("cohort", "==", cohort.value))
        for row in self.view.iter_rows(ITEMS, columns=list(AssessmentItem.model_fields), where=where):
            yield AssessmentItem.model_validate(row)

    def packet(self, pipeline_key: str, task_id: str) -> dict[str, Any]:
        row = self.view.point(ITEMS, pipeline_key=pipeline_key, task_id=task_id)
        if row is None:
            raise FileNotFoundError(f"Task is outside the assessment sample: {task_id}")
        return self.document(row["packet_uri"])

    def packets(self, items: Sequence[AssessmentItem]) -> Iterator[tuple[AssessmentItem, dict[str, Any]]]:
        """Read a source's archived packets in one bounded blob request."""
        references = [parse_uri(item.packet_uri) for item in items]
        if any(ref is None or ref.table != BlobTables.DESCRIPTORS for ref in references):
            raise ValueError("Task packets must be archived in this catalog")
        keys = [ref.key for ref in references if ref is not None]
        payloads = self.view.read_blobs(keys)
        for item, key in zip(items, keys, strict=True):
            if key not in payloads:
                raise FileNotFoundError(f"Catalog packet is unavailable: {item.task_id}")
            yield item, json.loads(payloads[key])

    def document(self, uri: str) -> dict[str, Any]:
        data = self.view.resolve(uri)
        if data is None:
            raise FileNotFoundError(f"Catalog evidence is unavailable: {uri}")
        return json.loads(data)
