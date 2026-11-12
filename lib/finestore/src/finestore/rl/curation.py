# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed curation catalog records shared by campaign writers and Atlas readers."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# Assessments may publish with a small share of recorded unavailable judgments — reviewer
# refusals or unparseable answers on adversarial tasks — instead of failing the campaign.
# 2% matches the observed rate of pathological model answers on validation cohorts (1-2 of a
# 105-task sample); a larger share signals systemic reviewer or transport failure, so the
# campaign, publish, and load gates all fail closed above it. Defined once; every gate imports
# this constant.
MAX_UNAVAILABLE_JUDGMENT_SHARE: float = 0.02


class AssessmentStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class AssessmentPaths:
    curated_root: str
    inputs_root: str
    judgments_root: str


def _record_value(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    arguments = get_args(annotation)
    if type(None) in arguments:
        annotation = next(argument for argument in arguments if argument is not type(None))
        arguments = get_args(annotation)
    if get_origin(annotation) is dict:
        entries = value.items() if isinstance(value, dict) else value
        result = {}
        for key, item in entries:
            if key in result:
                raise ValueError(f"Duplicate catalog map key: {key}")
            result[key] = _record_value(item, arguments[1])
        return result
    if get_origin(annotation) is list:
        return [_record_value(item, arguments[0]) for item in value]
    return value


class CatalogRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _arrow_values(cls, value: Any) -> Any:
        """Restore map values and implicit defaults materialized as null by Arrow schema evolution."""
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for name, field in cls.model_fields.items():
            if name not in result:
                continue
            if result[name] is None and not field.is_required() and field.default is not None:
                result.pop(name)
                continue
            result[name] = _record_value(result[name], field.annotation)
        return result


class IndependentQC(CatalogRecord):
    """Quality checks reported by a pinned source release."""

    verdict: Verdict
    method: str = Field(min_length=1)
    reviewed: int = Field(ge=0)
    # Absent when the release retained its reviewed population without an admission claim.
    admitted: int | None = Field(default=None, ge=0)
    evidence_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def _admitted_within_reviewed(self) -> "IndependentQC":
        if self.admitted is not None and self.admitted > self.reviewed:
            raise ValueError(f"admitted {self.admitted} exceeds reviewed {self.reviewed}")
        return self


class PipelineReference(CatalogRecord):
    name: str
    version: str
    source: str | None
    revision: str | None
    files: list[str]


class DaytonaSafety(StrEnum):
    """Whether running a source fits the Daytona snapshot budget."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    NOT_APPLICABLE = "not_applicable"
    UNASSESSED = "unassessed"


DAYTONA_SAFE_SNAPSHOT_LIMIT = 10
"""The unique-snapshot budget a source must fit to count as Daytona-safe."""


def daytona_verdict(*, snapshot_count: int | None, execution_failures: int | None, agentic: bool) -> DaytonaSafety:
    """Derive a source's Daytona verdict from its measurement.

    Sources that provision no sandbox are safe regardless of any count. An agentic
    source without a measured count is unassessed: a count that was never taken
    cannot establish safety, and a declared value must not stand in for one.
    Otherwise the source is unsafe when its unique environment count exceeds
    ``DAYTONA_SAFE_SNAPSHOT_LIMIT`` or when the Daytona backend refuses a task's
    declared execution requirements.
    """
    if not agentic:
        return DaytonaSafety.SAFE
    if snapshot_count is None:
        return DaytonaSafety.UNASSESSED
    if snapshot_count > DAYTONA_SAFE_SNAPSHOT_LIMIT or execution_failures:
        return DaytonaSafety.UNSAFE
    return DaytonaSafety.SAFE


class CurationMode(StrEnum):
    """How much of a source a curation pass examined."""

    QUICK = "quick"
    """Inventory or a spot check. Never admits tasks and never selects for training."""

    SAMPLE = "sample"
    """A documented sample was executed end to end, including its verifiers."""

    FULL = "full"
    """Every task was executed end to end, including its verifier."""


class OversightLevel(StrEnum):
    """The weakest form of review a source's admission rests on."""

    NONE = "none"
    """Inventory only: no execution evidence, so the source is not selectable."""

    SAMPLE = "sample"
    """Execution was verified on a documented sample of tasks."""

    FULL = "full"
    """Execution was verified on every task."""


class OversightEvidence(CatalogRecord):
    """The pointers an oversight level rests on, so a reader can check it."""

    campaign: str | None = None
    recipe_revision: str | None = None
    verification: str
    level: OversightLevel = OversightLevel.NONE
    reviewed: int | None = None
    admitted: int | None = None


class OracleValidation(StrEnum):
    """How an admitted source's grader was qualified."""

    EXECUTED = "executed"
    """Every sampled task's oracle solution ran and scored its expected reward."""

    VERIFIER_EXECUTED = "verifier_executed"
    """The grader rejected an untouched submission; successful oracle execution is not certified."""

    SKIPPED = "skipped"
    """Neither qualification ran; the source's execution is not certified."""


class GradingEnvironmentEvidence(CatalogRecord):
    """Census derived from the published task specifications, including rejected mismatches."""

    assessed_tasks: int = Field(ge=0)
    required_counts: dict[str, int]
    declared_counts: dict[str, int]
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    evidence_url: str


class DaytonaUnsafeReason(StrEnum):
    TOO_MANY_SNAPSHOTS = "too_many_snapshots"
    UNBUILT_IMAGE = "unbuilt_image"
    HOST_IMAGE_REQUIRED = "host_image_required"
    EXECUTION_REQUIREMENT = "execution_requirement"


class RoleExecutionEvidence(CatalogRecord):
    machine_slots: int = Field(ge=0)
    failures: dict[str, int]


class DaytonaMeasurement(CatalogRecord):
    dataset_id: str
    dataset_revision: str
    daytona_snapshot_count: int = Field(ge=0)
    daytona_machine_slots: int = Field(ge=0)
    daytona_execution_failures: int = Field(ge=0)
    daytona_count_basis: str = Field(min_length=1)
    daytona_unsafe_reasons: list[DaytonaUnsafeReason]
    daytona_unsafe_details: dict[str, str]
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None

    @property
    def safety(self) -> DaytonaSafety:
        if self.daytona_machine_slots == 0:
            return DaytonaSafety.NOT_APPLICABLE
        if self.daytona_snapshot_count > DAYTONA_SAFE_SNAPSHOT_LIMIT or self.daytona_execution_failures:
            return DaytonaSafety.UNSAFE
        return DaytonaSafety.SAFE

    @model_validator(mode="after")
    def _measured_reasons(self) -> "DaytonaMeasurement":
        reasons = set(self.daytona_unsafe_reasons)
        if self.daytona_machine_slots == 0 and (self.daytona_snapshot_count or self.daytona_execution_failures):
            raise ValueError("A source without an actor cannot have actor snapshots or refusals")
        over_budget = self.daytona_snapshot_count > DAYTONA_SAFE_SNAPSHOT_LIMIT
        if (DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS in reasons) != over_budget:
            raise ValueError("Snapshot-budget reason disagrees with the measured count")
        refusal_reasons = reasons - {DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS}
        if bool(refusal_reasons) != bool(self.daytona_execution_failures):
            raise ValueError("Actor refusal reasons disagree with the refused-task count")
        if set(self.daytona_unsafe_details) != {reason.value for reason in reasons}:
            raise ValueError("Every unsafe reason requires its measured detail")
        if any(not detail.strip() for detail in self.daytona_unsafe_details.values()):
            raise ValueError("Unsafe details must describe the measured failure")
        return self


class ConvertedExamples(CatalogRecord):
    count: int = Field(ge=0)
    input_rows: int = Field(ge=0)
    rejections: dict[str, int]
    admission_rejections: dict[str, int]
    source_dataset_id: str
    source_revision: str
    converter_revision: str
    converter_version: str
    evidence_url: str
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _conversion_ledger(self) -> "ConvertedExamples":
        if self.count > self.input_rows:
            raise ValueError("Converted examples exceed selected input rows")
        if self.count == 0 and not self.failure_reason:
            raise ValueError("Zero converted examples require the ingestion failure reason")
        return self


class DeconReceiptStatus(StrEnum):
    CLEAN = "clean"
    REMEDIATED = "remediated"


class DeconAdmissionReceipt(CatalogRecord):
    private: bool
    format: str
    status: DeconReceiptStatus
    dataset_id: str
    dataset_revision: str
    retained_tasks: int = Field(ge=0)
    dropped_tasks: int = Field(ge=0)
    drop_manifest_uri: str | None
    original_private_dataset_id: str | None
    original_private_dataset_revision: str | None

    @model_validator(mode="after")
    def _remediation_provenance(self) -> "DeconAdmissionReceipt":
        if not self.private or self.format != "taskcompendium" or not self.dataset_id.startswith("open-athena/"):
            raise ValueError("Admission requires a private open-athena TaskCompendium release")
        if self.status is DeconReceiptStatus.CLEAN and self.dropped_tasks:
            raise ValueError("A clean receipt cannot drop contaminated tasks")
        if self.status is DeconReceiptStatus.REMEDIATED:
            if (
                not self.drop_manifest_uri
                or not self.original_private_dataset_id
                or not self.original_private_dataset_revision
            ):
                raise ValueError("Remediation requires the drop manifest and exact original private pin")
            if not self.dataset_id.startswith("open-athena/") or not self.original_private_dataset_id.startswith(
                "open-athena/"
            ):
                raise ValueError("Remediation inputs and outputs must be private open-athena TaskCompendium releases")
            if (self.dataset_id, self.dataset_revision) == (
                self.original_private_dataset_id,
                self.original_private_dataset_revision,
            ):
                raise ValueError("Remediation requires a new dataset pin")
        return self


class LexicalContaminationEvidence(CatalogRecord):
    version: int = Field(ge=1, le=1)
    input_dataset_id: str
    input_dataset_revision: str
    converter_revision: str
    text_renderer_version: str
    measured_tasks: int = Field(ge=0)
    contaminated_tasks: int = Field(ge=0)
    contaminated_fraction: float = Field(ge=0, le=1)
    max_overlap: float = Field(ge=0, le=1)
    mean_max_overlap: float = Field(ge=0, le=1)
    p95_max_overlap: float = Field(ge=0, le=1)
    matched_evals: dict[str, int]
    eval_corpus_version: str
    eval_bloom_uri: str
    eval_hash_index_uri: str
    feature_filter_version: int
    ngram_length: int = Field(ge=13, le=13)
    overlap_threshold: float = Field(ge=0.5, le=0.5)
    min_matched_features: int = Field(ge=2, le=2)
    drop_set_uris: list[str]
    task_flags_uri: str
    evidence_url: str
    receipt: DeconAdmissionReceipt | None = None

    @model_validator(mode="after")
    def _receipt_matches_measurement(self) -> "LexicalContaminationEvidence":
        if self.contaminated_tasks > self.measured_tasks:
            raise ValueError("Contaminated tasks exceed the measured population")
        expected = self.contaminated_tasks / self.measured_tasks if self.measured_tasks else 0.0
        if abs(self.contaminated_fraction - expected) > 1e-12:
            raise ValueError("Contamination fraction disagrees with its numerator and denominator")
        receipt = self.receipt
        if receipt is None:
            return self
        if receipt.retained_tasks + receipt.dropped_tasks != self.measured_tasks:
            raise ValueError("Decontamination receipt does not conserve the measured population")
        if receipt.status is DeconReceiptStatus.CLEAN and self.contaminated_tasks:
            raise ValueError("A flagged population cannot carry a clean receipt")
        if receipt.status is DeconReceiptStatus.REMEDIATED and receipt.dropped_tasks != self.contaminated_tasks:
            raise ValueError("Remediation must drop every flagged task")
        return self

    def admission_blocker(self, dataset_id: str | None, dataset_revision: str | None) -> str | None:
        if self.receipt is None:
            return "clean or remediated decontamination receipt is missing"
        if (self.receipt.dataset_id, self.receipt.dataset_revision) != (dataset_id, dataset_revision):
            return "decontamination receipt belongs to a different dataset pin"
        return None


def require_decontamination(
    evidence: LexicalContaminationEvidence | None, dataset_id: str | None, dataset_revision: str | None
) -> None:
    """Require a clean or remediated private release before source QA or admission."""
    if evidence is None:
        raise ValueError("Decontamination measurement and private-release admission receipt are missing")
    if blocker := evidence.admission_blocker(dataset_id, dataset_revision):
        raise ValueError(f"Decontamination admission blocked: {blocker}")


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
    independent_qc: IndependentQC | None = None
    # Non-agentic sources never provision Daytona sandboxes; an agentic source stays
    # UNASSESSED until its unique snapshot count is assessed against DAYTONA_SAFE_SNAPSHOT_LIMIT.
    daytona_safety: DaytonaSafety | None = DaytonaSafety.UNASSESSED
    # Measured distinct snapshot identities and the rule applied, required for any agentic
    # SAFE/UNSAFE verdict. A verdict without these stays UNASSESSED; counts are never inferred.
    daytona_snapshot_count: int | None = None
    daytona_snapshot_threshold: int | None = None
    daytona_count_basis: str | None = None
    # How execution was qualified when the source ships no oracle solutions.
    oracle_validation: OracleValidation | None = None
    # The weakest review behind the admission, with the pointers that establish it.
    oversight: OversightLevel = OversightLevel.NONE
    oversight_evidence: OversightEvidence | None = None
    grading_environment: GradingEnvironmentEvidence | None = None
    converted_examples: ConvertedExamples | None = None
    lexical_contamination: LexicalContaminationEvidence | None = None
    daytona_machine_slots: int | None = None
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None
    daytona_execution_failures: int | None = None
    daytona_unsafe_reasons: list[DaytonaUnsafeReason] = Field(default_factory=list)
    daytona_unsafe_details: dict[str, str] = Field(default_factory=dict)


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
    mode: CurationMode
    source_revision: str | None
    recipe_revision: str | None
    raw_population_count: int | None
    raw_sample_count: int | None
    processed_rows: int | None
    unprocessed_rows: int | None
    admission_counts: AdmissionCounts
    verification: ControlSummary | None

    @model_validator(mode="after")
    def _quick_curation_admits_nothing(self) -> "CurationSummary":
        """A spot check has no admitted tasks to report."""
        if self.mode is CurationMode.QUICK and self.admission_counts.admitted:
            raise ValueError("QUICK curation cannot carry admitted tasks")
        return self


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
INVENTORY = "curation_inventory"
SNAPSHOTS = "curation_snapshots"
CURRENT = "curation_current"
ATTEMPTS = "curation_attempts"
TABLES: dict[str, tuple[type[CatalogRecord], tuple[str, ...]]] = {
    ATTEMPTS: (AssessmentAttempt, ("run_id", "name")),
    ASSESSMENTS: (Assessment, ("pipeline_key",)),
    POPULATIONS: (AssessmentPopulation, ("pipeline_key", "cohort")),
    ITEMS: (AssessmentItem, ("pipeline_key", "task_id")),
    INVENTORY: (SnapshotSource, ("snapshot_id", "id")),
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

    def _records[T: CatalogRecord](
        self, table: str, model: type[T], where: list[tuple[str, str, object]]
    ) -> Iterator[T]:
        for row in self.view.iter_rows(table, columns=list(model.model_fields), where=where):
            yield model.model_validate(row)

    def sources(self, snapshot_id: str) -> Iterator[SnapshotSource]:
        return self._records(INVENTORY, SnapshotSource, [("snapshot_id", "==", snapshot_id)])

    def assessment(self, pipeline_key: str) -> Assessment:
        row = self.view.point(ASSESSMENTS, pipeline_key=pipeline_key)
        if row is None:
            raise FileNotFoundError(f"Assessment not found: {pipeline_key}")
        return Assessment.model_validate({key: row[key] for key in Assessment.model_fields})

    def attempts(self, run_id: str) -> Iterator[AssessmentAttempt]:
        return self._records(ATTEMPTS, AssessmentAttempt, [("run_id", "==", run_id)])

    def populations(self, pipeline_key: str) -> list[AssessmentPopulation]:
        return list(self._records(POPULATIONS, AssessmentPopulation, [("pipeline_key", "==", pipeline_key)]))

    def items(self, pipeline_key: str, *, cohort: Cohort | None = None) -> Iterator[AssessmentItem]:
        where: list[tuple[str, str, object]] = [("pipeline_key", "==", pipeline_key)]
        if cohort is not None:
            where.append(("cohort", "==", cohort.value))
        return self._records(ITEMS, AssessmentItem, where)

    def packet(self, pipeline_key: str, task_id: str) -> dict[str, Any]:
        row = self.view.point(ITEMS, pipeline_key=pipeline_key, task_id=task_id)
        if row is None:
            raise FileNotFoundError(f"Task is outside the assessment sample: {task_id}")
        return self.document(row["packet_uri"])

    def packets(self, items: Sequence[AssessmentItem]) -> Iterator[tuple[AssessmentItem, dict[str, Any]]]:
        """Read a bounded set of archived packets in one blob request."""
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
