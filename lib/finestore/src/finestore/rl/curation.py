# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed curation catalog records shared by campaign writers and Atlas readers."""

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, get_args, get_origin
from urllib.parse import urlsplit

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


class SourceOrigin(StrEnum):
    """Upstream provider; internal campaigns belong in build_path."""

    AI_MO = "AI-MO"
    ASDIV = "ASDiv"
    ALLENAI = "AllenAI"
    ANTHROPIC = "Anthropic"
    BYTEDTSINGHUA_SIA = "BytedTsinghua-SIA"
    CHILLED = "ChilleD"
    CODEPARROT = "CodeParrot"
    ELEUTHERAI = "EleutherAI"
    EVALCHEMY = "Evalchemy"
    GPQA = "GPQA"
    GRETEL = "Gretel"
    HARBOR_HUB = "Harbor Hub"
    HUGGINGFACEH4 = "HuggingFaceH4"
    LM_EVAL = "lm-eval"
    MARIN = "Marin"
    NEMOTRON = "Nemotron"
    OPEN_R1 = "Open-R1"
    OPENAI = "OpenAI"
    OPENATHENA = "OpenAthena"
    PRIME_RL = "PRIME-RL"
    REASONING_GYM = "Reasoning Gym"
    TRL = "TRL"
    TASKTROVE = "TaskTrove"
    AGENTICA_ORG = "agentica-org"
    DI_ZHANG_FDU = "di-zhang-fdu"
    PAFITIS = "pafitis"


class DatasetRelease(CatalogRecord):
    """Exact dataset used for training, independently of upstream provenance."""

    repo: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    url: str = Field(min_length=1)

    @model_validator(mode="after")
    def _immutable_hugging_face_pin(self) -> "DatasetRelease":
        host = urlsplit(self.url).hostname
        if host == "huggingface.co" or self.repo.startswith("open-athena/"):
            if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
                raise ValueError("Hugging Face current_release requires an immutable commit")
            if self.url != f"https://huggingface.co/datasets/{self.repo}/tree/{self.revision}":
                raise ValueError("Hugging Face current_release URL must match its repository and revision")
        return self


def source_identifier(origin: SourceOrigin, dataset: str, subset: str | None, variant: str | None) -> str:
    return f"{origin}:{dataset}" + (f"/{subset}" if subset else "") + (f"@{variant}" if variant else "")


def source_title(name: str, subset: str | None, variant: str | None) -> str:
    return " · ".join(part for part in (name, subset, variant) if part)


class ContentEvidenceReference(CatalogRecord):
    """Immutable content-evidence location and byte digest."""

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NoOracleExecutabilityPolicy(CatalogRecord):
    """Versioned ruling that records static retention without an execution claim."""

    version: Literal[1]
    executability_proof_required_for_no_oracle: Literal[False]
    decision: ContentEvidenceReference

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class NoOracleStaticBlocker(StrEnum):
    TOOL_PROVIDER_REQUIRED = "tool_provider_required"
    SHARED_EXECUTOR_REQUIRED = "shared_executor_required"
    UNPINNED_ENVIRONMENT = "unpinned_environment"
    ENVIRONMENT_CONTRACT_UNRESOLVED = "environment_contract_unresolved"
    BACKEND_CONTRACT_UNSUPPORTED = "backend_contract_unsupported"
    SOURCE_ENV_INCOMPATIBLE_WITH_GRADER = "SOURCE_ENV_INCOMPATIBLE_WITH_GRADER"


class OracleFiltrationEvidence(CatalogRecord):
    """An exact original population, canonical filter and read-back private output."""

    original_release: DatasetRelease
    original_cohort: ContentEvidenceReference
    original_count: int = Field(ge=0)
    oracle_filter: ContentEvidenceReference
    oracle_filter_version: int
    filtered_release: DatasetRelease
    private: bool
    tasks_parquet: ContentEvidenceReference
    admitted_count: int = Field(ge=0)
    exclusion_counts: dict[str, int]
    validation: ContentEvidenceReference

    @model_validator(mode="after")
    def _population_partition(self) -> "OracleFiltrationEvidence":
        if not self.private or not self.filtered_release.repo.startswith("open-athena/"):
            raise ValueError("Oracle filtration requires a private open-athena release")
        if any(count <= 0 for count in self.exclusion_counts.values()):
            raise ValueError("Filtration exclusion counts must describe observed tasks")
        if self.original_count != self.admitted_count + sum(self.exclusion_counts.values()):
            raise ValueError("Oracle filtration does not conserve the original population")
        return self


class QualityResultFinalization(StrEnum):
    """Explicit whole-source content Result finalization, independent of coverage."""

    COMPLETE = "complete"
    ACCEPTED_INCOMPLETE = "accepted_incomplete"


class SourceQualityResult(CatalogRecord):
    """Informational Result over conclusive reviewed tasks; exclusions are separate."""

    reviewed: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    issues: list[str]
    repair_recommendation: str
    unavailable_count: int = Field(default=0, ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    sample_count: int | None = None
    issue_rate: float | None = None
    verdict: Verdict = Verdict.UNKNOWN

    @model_validator(mode="before")
    @classmethod
    def _computed_result(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        reviewed, issues = result["reviewed"], result["issue_count"]
        if issues > reviewed:
            raise ValueError("Issue count exceeds reviewed tasks")
        rate = issues / reviewed if reviewed else None
        verdict = (
            Verdict.UNKNOWN
            if not reviewed
            else (
                Verdict.BAD
                if 2 * issues >= reviewed
                else Verdict.GOOD if 10 * issues <= reviewed else Verdict.SOME_ISSUES
            )
        )
        sample_count = reviewed + result.get("unavailable_count", 0) + result.get("uncertain_count", 0)
        for name, expected in (("issue_rate", rate), ("verdict", verdict), ("sample_count", sample_count)):
            if name in result and result[name] != expected:
                raise ValueError(f"{name} must be computed from reviewed issue counts")
            result[name] = expected
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


DAYTONA_SAFE_SNAPSHOT_LIMIT = 10
"""The unique-snapshot budget a source must fit to count as Daytona-safe."""


class CurationMode(StrEnum):
    """Population scope requested for a curation pass."""

    QUICK = "quick"
    """Conversion without execution qualification. Never admits tasks or selects for training."""

    SAMPLE = "sample"
    """A documented sample was executed end to end, including its verifiers."""

    FULL = "full"
    """Process the full population; execution coverage comes from separate oversight evidence."""


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
    daytona_unsafe_counts: dict[str, int] = Field(default_factory=dict)
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
        if set(self.daytona_unsafe_counts) != {reason.value for reason in reasons}:
            raise ValueError("Every unsafe reason requires its affected actor-task count")
        if any(count <= 0 or count > self.daytona_machine_slots for count in self.daytona_unsafe_counts.values()):
            raise ValueError("Unsafe counts must describe affected actor task slots")
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


class SourceUse(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class LexicalUse(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    BENCHMARK = "benchmark"


class DeconReceiptStatus(StrEnum):
    CLEAN = "clean"
    REMEDIATED = "remediated"
    BENCHMARK_MEASURED = "measured (benchmark, not remediated)"


class DecontaminatedRelease(CatalogRecord):
    private: bool
    format: str
    dataset_id: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    drop_manifest_uri: str | None
    original_private_dataset_id: str | None
    original_private_dataset_revision: str | None = Field(pattern=r"^[0-9a-f]{40}$")
    duplicate_manifest_uri: str | None = None
    dedup_census_uri: str | None = None
    original_dataset_id: str | None = None
    # Acquired files use a SHA-256 content pin; remote datasets use a Git commit.
    original_dataset_revision: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class LexicalReceipt(CatalogRecord):
    use: LexicalUse
    status: DeconReceiptStatus
    retained_tasks: int = Field(ge=0)
    dropped_tasks: int = Field(ge=0)
    duplicate_dropped_tasks: int = Field(default=0, ge=0)
    release: DecontaminatedRelease | None

    @model_validator(mode="after")
    def _remediation_provenance(self) -> "LexicalReceipt":
        if self.use in {LexicalUse.EVAL, LexicalUse.BENCHMARK}:
            if (
                self.status is not DeconReceiptStatus.BENCHMARK_MEASURED
                or self.release is not None
                or self.dropped_tasks
                or self.duplicate_dropped_tasks
            ):
                raise ValueError("Evaluation and benchmark receipts measure the unchanged population without a release")
            return self
        if self.status is DeconReceiptStatus.BENCHMARK_MEASURED or self.release is None:
            raise ValueError("Training receipts require a clean or remediated release")
        if self.duplicate_dropped_tasks and not (
            self.release.duplicate_manifest_uri
            and self.release.dedup_census_uri
            and self.release.drop_manifest_uri
            and self.release.original_dataset_id
            and self.release.original_dataset_revision
        ):
            raise ValueError("Combined retention requires the original pin, duplicate and census provenance")
        if self.duplicate_dropped_tasks and (self.release.dataset_id, self.release.dataset_revision) == (
            self.release.original_dataset_id,
            self.release.original_dataset_revision,
        ):
            raise ValueError("Combined retention requires a new dataset pin")
        if self.status is DeconReceiptStatus.CLEAN and self.dropped_tasks:
            raise ValueError("A clean receipt cannot drop contaminated tasks")
        if self.status is DeconReceiptStatus.REMEDIATED:
            release = self.release
            if (
                not release.drop_manifest_uri
                or not release.original_private_dataset_id
                or not release.original_private_dataset_revision
            ):
                raise ValueError("Remediation requires the drop manifest and exact original private pin")
            if (release.dataset_id, release.dataset_revision) == (
                release.original_private_dataset_id,
                release.original_private_dataset_revision,
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
    receipt: LexicalReceipt | None = None

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
        if receipt.retained_tasks + receipt.dropped_tasks + receipt.duplicate_dropped_tasks != self.measured_tasks:
            raise ValueError("Decontamination receipt does not conserve the measured population")
        if receipt.duplicate_dropped_tasks:
            release = receipt.release
            assert release is not None
            if (release.original_dataset_id, release.original_dataset_revision) != (
                self.input_dataset_id,
                self.input_dataset_revision,
            ):
                raise ValueError("Combined retention provenance belongs to a different measured input")
            if self.contaminated_tasks > receipt.dropped_tasks + receipt.duplicate_dropped_tasks:
                raise ValueError("Combined retention leaves flagged tasks outside its exclusions")
        if receipt.status is DeconReceiptStatus.CLEAN and self.contaminated_tasks:
            raise ValueError("A flagged population cannot carry a clean receipt")
        if receipt.status is DeconReceiptStatus.REMEDIATED:
            if receipt.duplicate_dropped_tasks:
                if receipt.dropped_tasks > self.contaminated_tasks:
                    raise ValueError("Unique lexical exclusions exceed the raw flagged population")
            elif receipt.dropped_tasks != self.contaminated_tasks:
                raise ValueError("Remediation must drop every flagged task")
        return self

    def admission_blocker(self, dataset_id: str | None, dataset_revision: str | None) -> str | None:
        if self.receipt is None:
            return "clean or remediated decontamination receipt is missing"
        if self.receipt.status is DeconReceiptStatus.BENCHMARK_MEASURED:
            return "benchmark measurement does not certify a training release"
        release = self.receipt.release
        assert release is not None
        if (release.dataset_id, release.dataset_revision) != (dataset_id, dataset_revision):
            return "decontamination receipt belongs to a different dataset pin"
        return None


def require_decontamination(
    evidence: LexicalContaminationEvidence | None, dataset_id: str | None, dataset_revision: str | None
) -> None:
    """Require a clean or remediated release receipt bound to the selected dataset pin."""
    if evidence is None:
        raise ValueError("Decontamination measurement and private-release admission receipt are missing")
    if blocker := evidence.admission_blocker(dataset_id, dataset_revision):
        raise ValueError(f"Decontamination admission blocked: {blocker}")


class IntegrityContractGroup(CatalogRecord):
    id: str | None
    version: int | None
    dependency_state: str | None
    native_family: str | None
    harness_policy_sha256: str | None
    present: bool
    count: int = Field(gt=0)


class IntegrityContractExample(CatalogRecord):
    task_id: str
    contract_json: str

    @model_validator(mode="after")
    def _bounded_payload(self) -> "IntegrityContractExample":
        if len(self.contract_json.encode()) > 16384:
            raise ValueError("Contract example exceeds 16 KiB")
        return self


class IntegrityContractSummary(CatalogRecord):
    """Bounded census metadata; full contract groups remain in the evidence artifact."""

    groups: list[IntegrityContractGroup] = Field(max_length=64)
    distinct_groups: int = Field(ge=0)
    absent_contract_tasks: int = Field(ge=0)
    required_shared_tasks: int = Field(ge=0)
    shared_missing_contract_tasks: int = Field(ge=0)
    example: IntegrityContractExample | None
    evidence_url: str
    dataset_id: str
    dataset_revision: str
    task_specs_sha256: str


class ActorCensusInput(CatalogRecord):
    dataset_id: str
    dataset_revision: str
    converter_version: str
    converter_revision: str
    measured_tasks: int = Field(gt=0)
    task_specs_sha256: str
    evidence_url: str


class NoOracleStaticRetentionEvidence(CatalogRecord):
    """Complete no-oracle population, with one typed exclusion per blocked task."""

    source_id: str
    source_release: DatasetRelease
    current_input: ActorCensusInput
    intended_use: SourceUse
    retained_cohort: ContentEvidenceReference
    retained_count: int = Field(ge=0)
    static_retained_count: int = Field(ge=0)
    exclusion_counts: dict[NoOracleStaticBlocker, int]
    oracle_filter: ContentEvidenceReference
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _static_partition(self):
        if any(value <= 0 for value in self.exclusion_counts.values()) or (
            self.retained_count != self.static_retained_count + sum(self.exclusion_counts.values())
        ):
            raise ValueError("Static retention must partition unique retained tasks exactly once")
        if (self.current_input.dataset_id, self.current_input.dataset_revision) != (
            self.source_release.repo,
            self.source_release.revision,
        ):
            raise ValueError("Static retention source pin differs from its converted input")
        return self


class WithinSourceDedupEvidence(CatalogRecord):
    """Exact converted-population census and its mechanical exclusion ledger."""

    raw_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    within_source_duplicates: int = Field(ge=0)
    source_dataset_id: str
    source_revision: str
    converter_revision: str
    converter_version: str
    drop_manifest_uri: str
    evidence_url: str

    @model_validator(mode="after")
    def _population_ledger(self) -> "WithinSourceDedupEvidence":
        if self.raw_count != self.unique_count + self.within_source_duplicates:
            raise ValueError("Raw task count must equal unique tasks plus within-source duplicates")
        return self


class SourceDeclaration(CatalogRecord):
    id: str
    name: str
    display_name: str
    origin: SourceOrigin
    build_path: str
    upstream_dataset: str
    upstream_name: str
    subset: str | None = None
    variant: str | None = None
    current_release: DatasetRelease
    family: str
    tags: list[str]
    task_count: int | None
    within_source_dedup: WithinSourceDedupEvidence
    notes: str
    dataset_id: str | None
    dataset_revision: str | None
    url: str | None
    type: str
    turns: str
    is_benchmark: bool
    intended_use: SourceUse | None = None
    kind: str
    status: str
    pipeline: PipelineReference | None
    independent_qc: IndependentQC | None = None
    daytona_safety: DaytonaSafety | None
    daytona_exemption: str | None = Field(default=None, min_length=1)
    daytona_snapshot_count: int | None = None
    daytona_snapshot_threshold: int | None = None
    daytona_count_basis: str | None = None
    oracle_validation: OracleValidation | None = None
    # The weakest review behind the admission, with the pointers that establish it.
    oversight: OversightLevel = OversightLevel.NONE
    oversight_evidence: OversightEvidence | None = None
    grading_environment: GradingEnvironmentEvidence | None = None
    integrity_contracts: IntegrityContractSummary | None = None
    actor_census_input: ActorCensusInput | None = None
    converted_examples: ConvertedExamples | None = None
    oracle_filtration: OracleFiltrationEvidence | None = Field(default=None, exclude_if=lambda value: value is None)
    no_oracle_executability_policy: NoOracleExecutabilityPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    no_oracle_static_retention: NoOracleStaticRetentionEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    lexical_contamination: LexicalContaminationEvidence | None = None
    daytona_machine_slots: int | None = None
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None
    daytona_execution_failures: int | None = None
    daytona_unsafe_reasons: list[DaytonaUnsafeReason] = Field(default_factory=list)
    daytona_unsafe_details: dict[str, str] = Field(default_factory=dict)
    daytona_unsafe_counts: dict[str, int] = Field(default_factory=dict)
    quality_result: SourceQualityResult | None = None
    quality_result_finalization: QualityResultFinalization | None = None
    quality_result_waiver: ContentEvidenceReference | None = None

    @model_validator(mode="after")
    def _canonical_identity(self) -> "SourceDeclaration":
        dedup = self.within_source_dedup
        if self.pipeline is None or (dedup.source_dataset_id, dedup.source_revision, dedup.converter_version) != (
            self.dataset_id,
            self.dataset_revision,
            self.pipeline.version,
        ):
            raise ValueError("Within-source deduplication belongs to another dataset pin or converter version")
        if self.task_count != dedup.unique_count:
            raise ValueError("Atlas task count must equal the post-deduplication unique count")
        filtration = self.oracle_filtration
        if filtration is not None:
            if filtration.filtered_release != self.current_release or self.task_count != filtration.admitted_count:
                raise ValueError("Oracle filtration differs from the configured release or Atlas count")
            if filtration.admitted_count == 0 and self.status == "Available":
                raise ValueError("A source filtered to zero cannot remain admitted")
        census = self.actor_census_input
        static = self.no_oracle_static_retention
        if static is not None and (
            self.no_oracle_executability_policy is None
            or static.source_id != self.id
            or static.source_release != self.current_release
            or static.current_input != census
            or static.intended_use != self.intended_use
            or static.retained_count != self.task_count
            or static.policy_sha256 != self.no_oracle_executability_policy.sha256
        ):
            raise ValueError("No-oracle static retention differs from the current source population or Use")
        if census is not None and (
            self.pipeline is None
            or (census.dataset_id, census.dataset_revision, census.converter_version)
            != (self.dataset_id, self.dataset_revision, self.pipeline.version)
        ):
            raise ValueError("Actor census input or converter version differs from the declaration")
        if self.display_name != source_title(self.upstream_name, self.subset, self.variant):
            raise ValueError("Source title differs from generated title")
        if self.daytona_safety is not None:
            measurement = DaytonaMeasurement.model_validate(
                {
                    "dataset_id": self.dataset_id,
                    "dataset_revision": self.dataset_revision,
                    "daytona_snapshot_count": self.daytona_snapshot_count,
                    "daytona_machine_slots": self.daytona_machine_slots,
                    "daytona_execution_failures": self.daytona_execution_failures,
                    "daytona_count_basis": self.daytona_count_basis,
                    "daytona_unsafe_reasons": self.daytona_unsafe_reasons,
                    "daytona_unsafe_details": self.daytona_unsafe_details,
                    "daytona_unsafe_counts": self.daytona_unsafe_counts,
                    "daytona_roles": self.daytona_roles,
                }
            )
            if measurement.safety is not self.daytona_safety:
                raise ValueError("Daytona safety must match the actor census")
        if self.origin is SourceOrigin.EVALCHEMY and "eval" not in self.tags:
            raise ValueError("Evalchemy benchmarks must be marked for evaluation")
        return self


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
    quality_result: SourceQualityResult | None = None

    @model_validator(mode="after")
    def _quick_curation_admits_nothing(self) -> "CurationSummary":
        """QUICK conversion has no admitted tasks to report."""
        if self.mode is CurationMode.QUICK and self.admission_counts.admitted:
            raise ValueError("QUICK curation cannot carry admitted tasks")
        return self


class ModelReference(CatalogRecord):
    reviewer_spec_json: str | None = None
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
